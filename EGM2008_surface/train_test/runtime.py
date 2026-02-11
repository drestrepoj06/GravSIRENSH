import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from EGM2008_surface.location_encoder.coordinates_network import SH_SIREN, SH_LINEAR, MANDS2022, MANDS2022Scaler, Scaler
from EGM2008_surface.linear_equivalent import LinearEquivalentGenerator


def main(run_path=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    runs_dir = os.path.join(base_dir, "outputs", "runs")
    data_dir = os.path.join(base_dir, "data")

    test_path = os.path.join(data_dir, "Samples_2190-2_250k_r0_test.parquet")
    test_df = pd.read_parquet(test_path)

    mask = np.abs(test_df["lat"].values) < 89.9999
    test_df = test_df[mask].reset_index(drop=True)

    sample = test_df.sample(n = 10000, random_state = 42)
    sample_idx = np.arange(len(sample))

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
    print(f"Loaded config: mode={mode}")

    scaler_data = checkpoint["scaler"]
    arch = config["architecture"]

    if arch == "mands2022":
        scaler = MANDS2022Scaler()
        U_min = scaler_data.get("U_min", None)
        U_max = scaler_data.get("U_max", None)

        if U_min is None or U_max is None:
            raise ValueError(
                "Checkpoint missing U_min/U_max for mands2022 scaler."
            )

        scaler.U_min = float(U_min)
        scaler.U_max = float(U_max)

        a_min = scaler_data.get("a_min", None)
        a_max = scaler_data.get("a_max", None)

        if a_min is None or a_max is None:
            raise ValueError(
                "Checkpoint missing a_min/a_max for mands2022 scaler."
            )

        scaler.a_min = np.asarray(a_min, dtype=float)
        scaler.a_max = np.asarray(a_max, dtype=float)

        lon_min = scaler_data.get("lon_min", None)
        lon_max = scaler_data.get("lon_max", None)
        lat_min = scaler_data.get("lat_min", None)
        lat_max = scaler_data.get("lat_max", None)

        if None in (lon_min, lon_max, lat_min, lat_max):
            raise ValueError(
                "Missing lon/lat min-max in checkpoint scaler_data. "
                "You must save lon_min, lon_max, lat_min, lat_max when training."
            )

        scaler.lon_min = float(lon_min)
        scaler.lon_max = float(lon_max)
        scaler.lat_min = float(lat_min)
        scaler.lat_max = float(lat_max)

    else:
        scaler = Scaler(mode=mode)
        scaler.U_mean = scaler_data.get("U_mean")
        scaler.U_std = scaler_data.get("U_std")

        g_mean = scaler_data.get("g_mean")
        g_std = scaler_data.get("g_std")
        if g_mean is not None:
            scaler.g_mean = torch.tensor(g_mean, dtype=torch.float32, device=device)
        if g_std is not None:
            scaler.g_std = torch.tensor(g_std, dtype=torch.float32, device=device)

    if arch == "sirensh":
        ModelClass = SH_SIREN
    elif arch == "linearsh":
        ModelClass = SH_LINEAR
    elif arch == "mands2022":
        ModelClass = MANDS2022
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Expected 'sirensh', 'linearsh', or 'mands2022'.")

    cache_path = None

    if arch == "mands2022":
        model = ModelClass(
            hidden_features=config["hidden_features"],
            hidden_layers=config["hidden_layers"],
            device=device,
            scaler=scaler,
            mode=mode,
        )
    elif arch == "sirensh":
        model = ModelClass(
            lmax=config["lmax"],
            hidden_features=config["hidden_features"],
            hidden_layers=config["hidden_layers"],
            device=device,
            scaler=scaler,
            cache_path=cache_path,
            exclude_degrees=config.get("exclude_degrees"),
            mode=mode,
            first_omega_0=config.get("first_omega_0") if arch == "sirensh" else None,
            hidden_omega_0=config.get("hidden_omega_0") if arch == "sirensh" else None,
        )
    else:
        model = ModelClass(
            lmax=config["lmax"],
            hidden_features=config["hidden_features"],
            hidden_layers=config["hidden_layers"],
            device=device,
            scaler=scaler,
            cache_path=cache_path,
            exclude_degrees=config.get("exclude_degrees"),
            mode=mode
        )

    state = checkpoint["state_dict"]

    if arch == "mands2022":
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and model_state[k].shape == v.shape}

        model.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    def build_tensors_from_df(df, device):
        lon = torch.tensor(df["lon"].values, dtype=torch.float32, device=device)
        lat = torch.tensor(df["lat"].values, dtype=torch.float32, device=device)

        true_U = torch.tensor(df["dU_m2_s2"].values, dtype=torch.float32, device=device)
        true_theta = torch.tensor(df["dg_theta_mGal"].values, dtype=torch.float32, device=device)
        true_phi = torch.tensor(df["dg_phi_mGal"].values, dtype=torch.float32, device=device)
        true_rad = torch.tensor(df["dg_r_mGal"].values, dtype=torch.float32, device=device)
        true_mag = torch.tensor(df["dg_total_mGal"].values, dtype=torch.float32, device=device)

        return lon, lat, true_U, true_theta, true_phi, true_rad, true_mag

    lon, lat, U, theta, phi, rad, mag = build_tensors_from_df(sample, device)

    def evaluate_and_save_dataset(
            model,
            mode,
            lon,
            lat,
            true_U,
            true_theta,
            true_phi,
            true_rad,
            true_mag,
            scaler,
            device,
            latest_run,
            prefix_tag,
    ):
        """
        Runs a SINGLE forward pass for the given subset (A, F, or C),
        computes RMSEs, saves predictions to .npy, and returns:
            results:      dict with rmse_U, rmse_g (some may be None)
            pred_stats:   dict of stats for predicted fields
            true_stats:   dict of stats for true fields
        """

        def unscale_g(scaler, g_scaled):
            if hasattr(scaler, "unscale_gravity"):
                return scaler.unscale_gravity(g_scaled)

            if hasattr(scaler, "unscale_accel_uniform"):
                return scaler.unscale_accel_uniform(g_scaled)

            raise AttributeError(
                f"Scaler of type {type(scaler).__name__} has no unscale_gravity or unscale_accel_uniform."
            )
        lon = lon.to(device)
        lat = lat.to(device)

        true_theta = true_theta.to(device)
        true_phi = true_phi.to(device)
        true_rad = true_rad.to(device)
        true_mag = true_mag.to(device)

        if true_U is not None:
            true_U = true_U.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if mode == "g_direct":
            with torch.no_grad():
                preds = model(lon, lat)
            U_scaled, grads = None, None

        elif mode == "U":
            lon_req = lon.detach().clone().requires_grad_(True)
            lat_req = lat.detach().clone().requires_grad_(True)
            U_scaled = model(lon_req, lat_req)
            preds = None

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_pred = time.perf_counter() - t0

        timing_nn = {
            "n_points": int(lon.shape[0]),
            "t_pred_s": float(t_pred),
            "mode": mode,
            "device": str(device),
        }

        if mode == "g_direct":
            g_pred = unscale_g(scaler, preds).detach()
            g_theta = g_pred[:, 0]
            g_phi = g_pred[:, 1]
            g_rad = g_pred[:, 2]
            g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2 + g_rad ** 2)

            U_pred = None

        elif mode == "U":
            U_pred = scaler.unscale_potential(U_scaled).detach()
            g_theta = None
            g_phi = None
            g_rad = None
            g_mag = None

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        rmse_U = None
        rmse_g = None

        if U_pred is not None and true_U is not None:
            rmse_U = torch.sqrt(torch.mean((U_pred.ravel() - true_U) ** 2)).item()

        if g_theta is not None and g_phi is not None and g_rad is not None:
            rmse_g = ((
                    torch.mean((g_theta - true_theta) ** 2).item()
                    + torch.mean((g_phi - true_phi) ** 2).item()
                    + torch.mean((g_rad - true_rad) ** 2).item()
            )**0.5)/1e5

        results = {
            "rmse_U": rmse_U,
            "rmse_g": rmse_g
        }

        prefix = os.path.join(latest_run, f"test_results_{prefix_tag}")

        if U_pred is not None:
            np.save(f"{prefix}_U.npy", U_pred.cpu().numpy().ravel())
        if g_theta is not None and g_phi is not None and g_rad is not None:
            np.save(f"{prefix}_g_theta.npy", g_theta.cpu().numpy())
            np.save(f"{prefix}_g_phi.npy", g_phi.cpu().numpy())
            np.save(f"{prefix}_g_rad.npy", g_rad.cpu().numpy())
            np.save(f"{prefix}_g_mag.npy", g_mag.cpu().numpy())

        def stats(arr):
            return {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }

        pred_stats = {}

        if g_theta is not None and g_phi is not None and g_rad is not None:
            pred_stats = {
                "g_theta": stats(g_theta.cpu().numpy()),
                "g_phi": stats(g_phi.cpu().numpy()),
                "g_rad": stats(g_rad.cpu().numpy()),
                "g_mag": stats(g_mag.cpu().numpy()),
            }

        if U_pred is not None:
            pred_stats["U"] = stats(U_pred.cpu().numpy())

        true_stats = {
            "g_theta": stats(true_theta.cpu().numpy()),
            "g_phi": stats(true_phi.cpu().numpy()),
            "g_rad": stats(true_rad.cpu().numpy()),
            "g_mag": stats(true_mag.cpu().numpy()),
        }

        if true_U is not None:
            true_stats["U"] = stats(true_U.cpu().numpy())

        return results, pred_stats, true_stats, timing_nn

    res, pred, true, timing = evaluate_and_save_dataset(
        model, mode,
        lon, lat,
        U, theta, phi, rad, mag,
        scaler, device,
        latest_run,
        prefix_tag="Runtime dataset"
    )


    gen = LinearEquivalentGenerator(run_path, sample)

    lin_model = None
    lin_model_timing = None
    lin_equiv_timing = None

    if gen.model is not None:
        lin_model, lin_model_timing = gen.evaluate_on_test(
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
            A_idx=sample_idx, F_idx=None, C_idx=None
        )

    lin_equiv, lin_equiv_timing = gen.evaluate_on_test(
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
        A_idx=sample_idx, F_idx=None, C_idx=None
    )

    def evaluate_linear_baseline(paths, subset, run_path,
                                 true_U, true_theta, true_phi, true_rad):

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
            rmse = float(np.sqrt(np.mean((lin_U - true_U)**2)))

            subset_results["rmse_U"] = rmse
            subset_results["stats"] = {"U": stats(lin_U)}
            return subset_results

        theta_path = os.path.join(run_path, paths["theta"].format(subset=subset))
        phi_path = os.path.join(run_path, paths["phi"].format(subset=subset))
        rad_path = os.path.join(run_path, paths["r"].format(subset=subset))

        if not (os.path.exists(theta_path) and os.path.exists(phi_path)):
            return None

        lin_theta = np.load(theta_path)
        lin_phi = np.load(phi_path)
        lin_rad = np.load(rad_path)


        subset_results["rmse_g"] = float(np.sqrt(np.mean((lin_theta-true_theta)**2) + np.mean((lin_phi-true_phi)**2) + np.mean((lin_rad-true_rad)**2)) / 1e5)
        subset_results["stats"] = {
            "theta": stats(lin_theta),
            "phi": stats(lin_phi),
            "rad": stats(lin_rad)
        }

        return subset_results

    linear_paths = {
        "U_model": {"U": "linear_U_{subset}_model.npy"},
        "U_equiv": {"U": "linear_U_{subset}_equiv.npy"},
        "g_model": {"theta": "linear_g_theta_{subset}_model.npy",
                    "phi": "linear_g_phi_{subset}_model.npy",
                    "r": "linear_g_r_{subset}_model.npy"},
        "g_equiv": {"theta": "linear_g_theta_{subset}_equiv.npy",
                    "phi": "linear_g_phi_{subset}_equiv.npy",
                    "r": "linear_g_r_{subset}_equiv.npy"},
    }
    linear_results = {}

    for label, paths in linear_paths.items():
        linear_results[label] = {}

        linear_results[label]["sample"] = evaluate_linear_baseline(
            paths=paths,
            subset="sample",
            run_path=run_path,
            true_U=sample["dU_m2_s2"].to_numpy() if "U" in paths else None,
            true_theta=sample["dg_theta_mGal"].to_numpy(),
            true_phi=sample["dg_phi_mGal"].to_numpy(),
            true_rad=sample["dg_r_mGal"].to_numpy(),
        )

    meta = {
        "mode": mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "samples": len(sample),

        "nn": {
            "Sample": {"rmse": res, "pred_stats": pred, "true_stats": true},
        },

        "linear": linear_results,

        "timing": {
            "linear": {
                "Samples": {
                    "equiv": lin_equiv_timing,
                    "model": lin_model_timing if lin_model is not None else None
                }
            },
            "nn": {
                "Samples": timing
            }
        }
    }


    meta_file = os.path.join(latest_run, "test_results_report.json")

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Metadata saved to {meta_file}")
    return model, test_df


if __name__ == "__main__":
    mp.freeze_support()
    main()
