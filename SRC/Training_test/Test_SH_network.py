import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_network import SH_SIREN, SH_LINEAR, Scaler
from SRC.Linear.Linear_equivalent import LinearEquivalentGenerator


def main(run_path=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    runs_dir = os.path.join(base_dir, "Outputs", "Runs")
    data_dir = os.path.join(base_dir, "Data")

    test_path = os.path.join(data_dir, "Samples_2190-2_250k_r0_test.parquet")
    test_df = pd.read_parquet(test_path)

    mask = np.abs(test_df["lat"].values) < 89.9999
    test_df = test_df[mask].reset_index(drop=True)

    A = test_df.copy()

    g = A["dg_total_mGal"].values
    mean_g = g.mean()
    std_g = g.std()
    mask_F = np.abs(g - mean_g) > 2 * std_g

    F = A[mask_F].reset_index(drop=True)
    C = A[~mask_F].reset_index(drop=True)
    A_idx = np.arange(len(A))
    F_idx = np.where(mask_F)[0]
    C_idx = np.where(~mask_F)[0]

    if run_path is not None:
        latest_run = run_path
    else:
        run_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
                    if os.path.isdir(os.path.join(runs_dir, d))]
        if not run_dirs:
            raise FileNotFoundError("No run directories found in Outputs/Runs")
        latest_run = max(run_dirs, key=os.path.getmtime)
    print(f"Using latest run: {os.path.basename(latest_run)}")

    model_path = os.path.join(latest_run, "model.pth")
    config_path = os.path.join(latest_run, "config.json")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Missing model.pth or config.json in run folder")

    with open(config_path, "r") as f:
        config = json.load(f)
    checkpoint = torch.load(model_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["mode"]
    print(f"Loaded config: mode={mode}, lmax={config['lmax']}")

    scaler_data = checkpoint["scaler"]
    scaler = Scaler(mode=mode)
    scaler.U_mean = scaler_data.get("U_mean")
    scaler.U_std = scaler_data.get("U_std")

    g_mean = scaler_data.get("g_mean")
    g_std = scaler_data.get("g_std")

    if g_mean is not None:
        scaler.g_mean = torch.tensor(g_mean, dtype=torch.float32, device=device)
    if g_std is not None:
        scaler.g_std = torch.tensor(g_std, dtype=torch.float32, device=device)

    cache_path = os.path.join(base_dir, "Data", "cache_test.npy")
    arch = config["architecture"]

    if arch == "sirensh":
        ModelClass = SH_SIREN
    elif arch == "linearsh":
        ModelClass = SH_LINEAR
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Expected 'sirensh' or 'linearsh'.")

    model = ModelClass(
        lmax=config["lmax"],
        hidden_features=config["hidden_features"],
        hidden_layers=config["hidden_layers"],
        first_omega_0=config.get("first_omega_0"),
        hidden_omega_0=config.get("hidden_omega_0"),
        device=device,
        scaler=scaler,
        cache_path=cache_path,
        exclude_degrees=config.get("exclude_degrees"),
        mode=mode,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    def build_tensors_from_df(df, device):
        lon = torch.tensor(df["lon"].values, dtype=torch.float32, device=device)
        lat = torch.tensor(df["lat"].values, dtype=torch.float32, device=device)

        true_U = torch.tensor(df["dU_m2_s2"].values, dtype=torch.float32, device=device)
        true_theta = torch.tensor(df["dg_theta_mGal"].values, dtype=torch.float32, device=device)
        true_phi = torch.tensor(df["dg_phi_mGal"].values, dtype=torch.float32, device=device)
        true_mag = torch.tensor(df["dg_total_mGal"].values, dtype=torch.float32, device=device)

        return lon, lat, true_U, true_theta, true_phi, true_mag

    lon_A, lat_A, U_A, theta_A, phi_A, mag_A = build_tensors_from_df(A, device)

    lon_F, lat_F, U_F, theta_F, phi_F, mag_F = build_tensors_from_df(F, device)

    lon_C, lat_C, U_C, theta_C, phi_C, mag_C = build_tensors_from_df(C, device)

    def evaluate_and_save_dataset(
            model,
            mode,
            lon,
            lat,
            true_U,
            true_theta,
            true_phi,
            true_mag,
            scaler,
            device,
            latest_run,
            prefix_tag,
    ):
        """
        Runs a SINGLE forward pass for the given subset (A, F, or C),
        computes MSEs, saves predictions to .npy, and returns:
            results:      dict with mse_U, mse_g, mse_grad, mse_consistency (some may be None)
            pred_stats:   dict of stats for predicted fields
            true_stats:   dict of stats for true fields
        """

        lon = lon.to(device)
        lat = lat.to(device)

        true_theta = true_theta.to(device)
        true_phi = true_phi.to(device)
        true_mag = true_mag.to(device)

        if true_U is not None:
            true_U = true_U.to(device)

        if mode == "U":
            lon_req = lon.clone().detach().requires_grad_(True)
            lat_req = lat.clone().detach().requires_grad_(True)

            U_scaled, grads = model(lon_req, lat_req, return_gradients=True)
            U_pred = scaler.unscale_potential(U_scaled).detach()

            g_scaled = torch.stack(grads, dim=1)
            g_phys = scaler.unscale_gravity(g_scaled).detach()

            g_theta = g_phys[:, 0]
            g_phi = g_phys[:, 1]
            g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

            g_theta_grad = g_phi_grad = g_mag_grad = None

        elif mode == "g_direct":
            preds = model(lon, lat).detach()
            g_pred = scaler.unscale_gravity(preds)

            g_theta = g_pred[:, 0]
            g_phi = g_pred[:, 1]
            g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

            U_pred = None
            g_theta_grad = g_phi_grad = g_mag_grad = None

        elif mode == "g_indirect":
            lon_req = lon.clone().detach().requires_grad_(True)
            lat_req = lat.clone().detach().requires_grad_(True)

            U_scaled, grads = model(lon_req, lat_req)
            U_pred = scaler.unscale_potential(U_scaled).detach()

            g_scaled = torch.stack(grads, dim=1)
            g_phys = scaler.unscale_gravity(g_scaled).detach()

            g_theta = g_phys[:, 0]
            g_phi = g_phys[:, 1]
            g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

            g_theta_grad = g_phi_grad = g_mag_grad = None

        elif mode == "g_hybrid":
            out = model(lon, lat)

            U_pred = scaler.unscale_potential(out["U_pred"]).detach()
            g_pred = scaler.unscale_gravity(out["g_pred"]).detach()
            g_from_grad = scaler.unscale_gravity(out["g_from_gradU"]).detach()

            g_theta = g_pred[:, 0]
            g_phi = g_pred[:, 1]
            g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

            g_theta_grad = g_from_grad[:, 0]
            g_phi_grad = g_from_grad[:, 1]
            g_mag_grad = torch.sqrt(g_theta_grad ** 2 + g_phi_grad ** 2)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        mse_U = None
        mse_g = None
        mse_grad = None
        mse_consistency = None

        if U_pred is not None and true_U is not None:
            mse_U = torch.mean((U_pred.ravel() - true_U) ** 2).item()

        if g_theta is not None and g_phi is not None:
            mse_g = (
                    torch.mean((g_theta - true_theta) ** 2).item()
                    + torch.mean((g_phi - true_phi) ** 2).item()
            )

        if mode == "g_hybrid":
            mse_grad = (
                    torch.mean((g_theta_grad - true_theta) ** 2).item()
                    + torch.mean((g_phi_grad - true_phi) ** 2).item()
            )

            mse_consistency = (
                    torch.mean((g_theta_grad - g_theta) ** 2).item()
                    + torch.mean((g_phi_grad - g_phi) ** 2).item()
            )

        results = {
            "mse_U": mse_U,
            "mse_g": mse_g,
            "mse_grad": mse_grad,
            "mse_consistency": mse_consistency,
        }

        prefix = os.path.join(latest_run, f"test_results_{prefix_tag}")

        if U_pred is not None:
            np.save(f"{prefix}_U.npy", U_pred.cpu().numpy().ravel())

        np.save(f"{prefix}_g_theta.npy", g_theta.cpu().numpy())
        np.save(f"{prefix}_g_phi.npy", g_phi.cpu().numpy())
        np.save(f"{prefix}_g_mag.npy", g_mag.cpu().numpy())

        if g_theta_grad is not None:
            np.save(f"{prefix}_g_theta_grad.npy", g_theta_grad.cpu().numpy())
            np.save(f"{prefix}_g_phi_grad.npy", g_phi_grad.cpu().numpy())
            np.save(f"{prefix}_g_mag_grad.npy", g_mag_grad.cpu().numpy())

        def stats(arr):
            return {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }

        pred_stats = {
            "g_theta": stats(g_theta.cpu().numpy()),
            "g_phi": stats(g_phi.cpu().numpy()),
            "g_mag": stats(g_mag.cpu().numpy()),
        }

        if U_pred is not None:
            pred_stats["U"] = stats(U_pred.cpu().numpy())

        if g_theta_grad is not None:
            pred_stats["g_theta_grad"] = stats(g_theta_grad.cpu().numpy())
            pred_stats["g_phi_grad"] = stats(g_phi_grad.cpu().numpy())
            pred_stats["g_mag_grad"] = stats(g_mag_grad.cpu().numpy())

        true_stats = {
            "g_theta": stats(true_theta.cpu().numpy()),
            "g_phi": stats(true_phi.cpu().numpy()),
            "g_mag": stats(true_mag.cpu().numpy()),
        }

        if true_U is not None:
            true_stats["U"] = stats(true_U.cpu().numpy())

        return results, pred_stats, true_stats

    res_A, pred_A, true_A = evaluate_and_save_dataset(
        model, mode,
        lon_A, lat_A,
        U_A, theta_A, phi_A, mag_A,
        scaler, device,
        latest_run,
        prefix_tag="A"
    )

    res_F, pred_F, true_F = evaluate_and_save_dataset(
        model, mode,
        lon_F, lat_F,
        U_F, theta_F, phi_F, mag_F,
        scaler, device,
        latest_run,
        prefix_tag="F"
    )

    res_C, pred_C, true_C = evaluate_and_save_dataset(
        model, mode,
        lon_C, lat_C,
        U_C, theta_C, phi_C, mag_C,
        scaler, device,
        latest_run,
        prefix_tag="C"
    )

    gen = LinearEquivalentGenerator(run_path, test_path)

    lin_model = gen.evaluate_on_test(
        df_grid=gen.model["df_grid"],
        dU_grid=gen.model["dU_grid"],
        lats_grid=gen.model["lats"],
        lons_grid=gen.model["lons"],
        clm_full_g=gen.model["clm_full_g"],
        clm_low_g=gen.model["clm_low_g"],
        r0=gen.model["r0"],
        L=gen.model["L"],
        save=True,
        label="model",
        A_idx=A_idx, F_idx=F_idx, C_idx=C_idx
    )

    lin_equiv = gen.evaluate_on_test(
        df_grid=gen.equiv["df_grid"],
        dU_grid=gen.equiv["dU_grid"],
        lats_grid=gen.equiv["lats"],
        lons_grid=gen.equiv["lons"],
        clm_full_g=gen.equiv["clm_full_g"],
        clm_low_g=gen.equiv["clm_low_g"],
        r0=gen.equiv["r0"],
        L=gen.equiv["L"],
        save=True,
        label="equiv",
        A_idx=A_idx, F_idx=F_idx, C_idx=C_idx
    )

    def evaluate_linear_baseline(paths, subset, run_path,
                                 true_U, true_theta, true_phi):

        def stats(arr):
            return {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std())
            }

        subset_results = {}

        if "U" in paths:
            U_path = os.path.join(run_path, paths["U"].format(subset=subset))
            if not os.path.exists(U_path):
                return None

            lin_U = np.load(U_path)
            mse = float(np.mean((lin_U - true_U) ** 2))

            subset_results["mse_U"] = mse
            subset_results["stats"] = {"U": stats(lin_U)}
            return subset_results

        theta_path = os.path.join(run_path, paths["theta"].format(subset=subset))
        phi_path = os.path.join(run_path, paths["phi"].format(subset=subset))

        if not (os.path.exists(theta_path) and os.path.exists(phi_path)):
            return None

        lin_theta = np.load(theta_path)
        lin_phi = np.load(phi_path)

        mse_theta = np.mean((lin_theta - true_theta) ** 2)
        mse_phi = np.mean((lin_phi - true_phi) ** 2)

        subset_results["mse_g"] = float(mse_theta + mse_phi)
        subset_results["stats"] = {
            "theta": stats(lin_theta),
            "phi": stats(lin_phi)
        }
        return subset_results

    linear_paths = {
        "U_model": {"U": "linear_U_{subset}_model.npy"},
        "U_equiv": {"U": "linear_U_{subset}_equiv.npy"},
        "g_model": {"theta": "linear_g_theta_{subset}_model.npy",
                    "phi": "linear_g_phi_{subset}_model.npy"},
        "g_equiv": {"theta": "linear_g_theta_{subset}_equiv.npy",
                    "phi": "linear_g_phi_{subset}_equiv.npy"},
    }
    linear_results = {}

    for label, paths in linear_paths.items():
        linear_results[label] = {}

        linear_results[label]["A"] = evaluate_linear_baseline(
            paths=paths,
            subset="A",
            run_path=run_path,
            true_U=A["dU_m2_s2"].to_numpy() if "U" in paths else None,
            true_theta=A["dg_theta_mGal"].to_numpy(),
            true_phi=A["dg_phi_mGal"].to_numpy(),
        )

        linear_results[label]["F"] = evaluate_linear_baseline(
            paths=paths,
            subset="F",
            run_path=run_path,
            true_U=F["dU_m2_s2"].to_numpy() if "U" in paths else None,
            true_theta=F["dg_theta_mGal"].to_numpy(),
            true_phi=F["dg_phi_mGal"].to_numpy(),
        )

        linear_results[label]["C"] = evaluate_linear_baseline(
            paths=paths,
            subset="C",
            run_path=run_path,
            true_U=C["dU_m2_s2"].to_numpy() if "U" in paths else None,
            true_theta=C["dg_theta_mGal"].to_numpy(),
            true_phi=C["dg_phi_mGal"].to_numpy(),
        )

    meta = {
        "mode": mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "samples": {
            "A": len(A),
            "F": len(F),
            "C": len(C),
        },

        "nn": {
            "A": {
                "mse": res_A,
                "pred_stats": pred_A,
                "true_stats": true_A
            },
            "F": {
                "mse": res_F,
                "pred_stats": pred_F,
                "true_stats": true_F
            },
            "C": {
                "mse": res_C,
                "pred_stats": pred_C,
                "true_stats": true_C
            }
        },
        "linear": linear_results,
    }


    meta_file = os.path.join(latest_run, "test_results_report.json")

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Metadata saved to {meta_file}")
    return model, test_df


if __name__ == "__main__":
    mp.freeze_support()
    main()
