import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_network import SH_SIREN, SH_LINEAR, Scaler


def main(run_path=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    runs_dir = os.path.join(base_dir, "Outputs", "Runs")
    data_dir = os.path.join(base_dir, "Data")
    mse_consistency = None
    mse_grad = None

    # === Load test dataset ===
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

    # === Find the latest run ===
    if run_path is not None:
        latest_run = run_path
    else:
        run_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
                    if os.path.isdir(os.path.join(runs_dir, d))]
        if not run_dirs:
            raise FileNotFoundError("❌ No run directories found in Outputs/Runs")
        latest_run = max(run_dirs, key=os.path.getmtime)
    print(f"📂 Using latest run: {os.path.basename(latest_run)}")

    model_path = os.path.join(latest_run, "model.pth")
    config_path = os.path.join(latest_run, "config.json")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError("❌ Missing model.pth or config.json in run folder")

    # === Load config and checkpoint ===
    with open(config_path, "r") as f:
        config = json.load(f)
    checkpoint = torch.load(model_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["mode"]
    print(f"🧩 Loaded config: mode={mode}, lmax={config['lmax']}")

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

    # Instantiate the correct model
    model = ModelClass(
        lmax=config["lmax"],
        hidden_features=config["hidden_features"],
        hidden_layers=config["hidden_layers"],
        first_omega_0=config.get("first_omega_0"),  # None for SH_LINEAR (ignored)
        hidden_omega_0=config.get("hidden_omega_0"),  # None for SH_LINEAR (ignored)
        device=device,
        scaler=scaler,
        cache_path=cache_path,
        exclude_degrees=config.get("exclude_degrees"),
        mode=mode,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print("✅ Model and scaler loaded successfully.")

    def build_tensors_from_df(df, device):
        lon = torch.tensor(df["lon"].values, dtype=torch.float32, device=device)
        lat = torch.tensor(df["lat"].values, dtype=torch.float32, device=device)

        true_U = torch.tensor(df["dU_m2_s2"].values, dtype=torch.float32, device=device)
        true_theta = torch.tensor(df["dg_theta_mGal"].values, dtype=torch.float32, device=device)
        true_phi = torch.tensor(df["dg_phi_mGal"].values, dtype=torch.float32, device=device)
        true_mag = torch.tensor(df["dg_total_mGal"].values, dtype=torch.float32, device=device)

        return lon, lat, true_U, true_theta, true_phi, true_mag

    # A: full set
    lon_A, lat_A, U_A, theta_A, phi_A, mag_A = build_tensors_from_df(A, device)

    # F: high-acceleration subset (> mean + 2σ)
    lon_F, lat_F, U_F, theta_F, phi_F, mag_F = build_tensors_from_df(F, device)

    # C: complement
    lon_C, lat_C, U_C, theta_C, phi_C, mag_C = build_tensors_from_df(C, device)

    print("⚙️ Running model inference...")

    start = time.time()

    def to_np(t):
        """Detach and convert tensor to NumPy."""
        return t.detach().cpu().numpy()

    def evaluate_dataset(model, mode, lon, lat,
                         true_U, true_theta, true_phi,
                         scaler, device):

        # All tensors must be on device
        lon = lon.to(device)
        lat = lat.to(device)

        true_U = true_U.to(device) if true_U is not None else None
        true_theta = true_theta.to(device)
        true_phi = true_phi.to(device)

        # -----------------------------
        # MODE: U
        # -----------------------------
        if mode == "U":
            lon_req = lon.clone().detach().requires_grad_(True)
            lat_req = lat.clone().detach().requires_grad_(True)

            U_scaled, grads = model(lon_req, lat_req, return_gradients=True)
            U_pred = scaler.unscale_potential(U_scaled).detach()

            g_scaled = torch.stack(grads, dim=1)
            g_phys = scaler.unscale_gravity(g_scaled).detach()

            g_theta = g_phys[:, 0]
            g_phi = g_phys[:, 1]

            mse_U = torch.mean((U_pred.ravel() - true_U) ** 2).item()
            mse_g = torch.mean((g_theta - true_theta) ** 2).item() + \
                    torch.mean((g_phi - true_phi) ** 2).item()

            return {
                "mse_U": mse_U,
                "mse_g": mse_g
            }

        # -----------------------------
        # MODE: g_direct
        # -----------------------------
        elif mode == "g_direct":

            preds = model(lon, lat).detach()
            g_pred = scaler.unscale_gravity(preds)

            g_theta = g_pred[:, 0]
            g_phi = g_pred[:, 1]

            mse_g = torch.mean((g_theta - true_theta) ** 2).item() + \
                    torch.mean((g_phi - true_phi) ** 2).item()

            return {
                "mse_U": None,
                "mse_g": mse_g
            }

        # -----------------------------
        # MODE: g_indirect
        # -----------------------------
        elif mode == "g_indirect":

            lon_req = lon.clone().detach().requires_grad_(True)
            lat_req = lat.clone().detach().requires_grad_(True)

            U_scaled, grads = model(lon_req, lat_req)
            U_pred = scaler.unscale_potential(U_scaled).detach()

            g_scaled = torch.stack(grads, dim=1)
            g_phys = scaler.unscale_gravity(g_scaled).detach()

            g_theta = g_phys[:, 0]
            g_phi = g_phys[:, 1]

            mse_U = torch.mean((U_pred.ravel() - true_U) ** 2).item()
            mse_g = torch.mean((g_theta - true_theta) ** 2).item() + \
                    torch.mean((g_phi - true_phi) ** 2).item()

            return {
                "mse_U": mse_U,
                "mse_g": mse_g
            }

        # -----------------------------
        # MODE: g_hybrid
        # -----------------------------
        elif mode == "g_hybrid":

            out = model(lon, lat)

            U_pred_scaled = out["U_pred"]
            g_pred_scaled = out["g_pred"]
            g_from_gradU_scaled = out["g_from_gradU"]

            U_pred = scaler.unscale_potential(U_pred_scaled).detach()
            g_pred = scaler.unscale_gravity(g_pred_scaled).detach()
            g_from_gradU = scaler.unscale_gravity(g_from_gradU_scaled).detach()

            g_theta = g_pred[:, 0]
            g_phi = g_pred[:, 1]
            g_theta_grad = g_from_gradU[:, 0]
            g_phi_grad = g_from_gradU[:, 1]

            mse_U = torch.mean((U_pred.ravel() - true_U) ** 2).item()

            mse_g = torch.mean((g_theta - true_theta) ** 2).item() + \
                    torch.mean((g_phi - true_phi) ** 2).item()

            mse_grad = torch.mean((g_theta_grad - true_theta) ** 2).item() + \
                       torch.mean((g_phi_grad - true_phi) ** 2).item()

            mse_consistency = torch.mean((g_theta_grad - g_theta) ** 2).item() + \
                              torch.mean((g_phi_grad - g_phi) ** 2).item()

            return {
                "mse_U": mse_U,
                "mse_g": mse_g,
                "mse_grad": mse_grad,
                "mse_consistency": mse_consistency
            }

    def predict_and_save(model, mode, lon, lat,
                         true_U, true_theta, true_phi, true_mag,
                         scaler, device, latest_run,
                         prefix_tag):
        """
        prefix_tag must be: 'A', 'F', or 'C'
        """

        # ===============================
        # Run model evaluation (reuse previous function)
        # ===============================
        results = evaluate_dataset(
            model, mode,
            lon, lat,
            true_U, true_theta, true_phi,
            scaler, device
        )

        # Recompute predictions (this ensures output tensors exist)
        # --------------------------------------------------------
        # MODE: U
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

            # Hybrid-only fields
            g_theta_grad = None
            g_phi_grad = None
            g_mag_grad = None

        # MODE: g_direct
        elif mode == "g_direct":
            preds = model(lon, lat).detach()
            g_pred = scaler.unscale_gravity(preds)

            g_theta = g_pred[:, 0]
            g_phi = g_pred[:, 1]
            g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

            U_pred = None
            g_theta_grad = None
            g_phi_grad = None
            g_mag_grad = None

        # MODE: g_indirect
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

            g_theta_grad = None
            g_phi_grad = None
            g_mag_grad = None

        # MODE: g_hybrid
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

        # ===============================
        # Saving predictions
        # ===============================
        prefix = os.path.join(latest_run, f"test_results_{prefix_tag}")

        # Save U
        if U_pred is not None:
            np.save(f"{prefix}_U.npy", U_pred.cpu().numpy().ravel())

        # Save gravity predictions
        np.save(f"{prefix}_g_theta.npy", g_theta.cpu().numpy())
        np.save(f"{prefix}_g_phi.npy", g_phi.cpu().numpy())
        np.save(f"{prefix}_g_mag.npy", g_mag.cpu().numpy())

        # Save gradient gravity if exists
        if g_theta_grad is not None:
            np.save(f"{prefix}_g_theta_grad.npy", g_theta_grad.cpu().numpy())
            np.save(f"{prefix}_g_phi_grad.npy", g_phi_grad.cpu().numpy())
            np.save(f"{prefix}_g_mag_grad.npy", g_mag_grad.cpu().numpy())

        # ===============================
        # Stats helper
        # ===============================
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

        # True stats
        true_stats = {
            "g_theta": stats(true_theta.cpu().numpy()),
            "g_phi": stats(true_phi.cpu().numpy()),
            "g_mag": stats(true_mag.cpu().numpy())
        }

        if true_U is not None:
            true_stats["U"] = stats(true_U.cpu().numpy())

        # Return everything
        return results, pred_stats, true_stats

    res_A, pred_A, true_A = predict_and_save(
        model, mode,
        lon_A, lat_A,
        U_A, theta_A, phi_A, mag_A,
        scaler, device,
        latest_run,
        prefix_tag="A"
    )

    # F
    res_F, pred_F, true_F = predict_and_save(
        model, mode,
        lon_F, lat_F,
        U_F, theta_F, phi_F, mag_F,
        scaler, device,
        latest_run,
        prefix_tag="F"
    )

    # C
    res_C, pred_C, true_C = predict_and_save(
        model, mode,
        lon_C, lat_C,
        U_C, theta_C, phi_C, mag_C,
        scaler, device,
        latest_run,
        prefix_tag="C"
    )

    def evaluate_linear_baseline(label, paths,
                                 true_U, true_theta, true_phi,
                                 idx_subset):

        def stats(arr):
            return {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std())
            }

        subset_results = {}

        # ==== Potential baseline ====
        if "U" in paths:
            U_path = paths["U"]

            if not os.path.exists(U_path):
                return None

            lin_U_full = np.load(U_path)

            # Select subset
            lin_U = lin_U_full[idx_subset]

            mse = float(np.mean((lin_U - true_U) ** 2))

            subset_results["mse_U"] = mse
            subset_results["stats"] = {"U": stats(lin_U)}

            return subset_results

        # ==== Gravity baseline ====
        theta_path = paths["theta"]
        phi_path = paths["phi"]

        if not (os.path.exists(theta_path) and os.path.exists(phi_path)):
            return None

        lin_theta_full = np.load(theta_path)
        lin_phi_full = np.load(phi_path)

        # Select subset
        lin_theta = lin_theta_full[idx_subset]
        lin_phi = lin_phi_full[idx_subset]

        mse_theta = np.mean((lin_theta - true_theta) ** 2)
        mse_phi = np.mean((lin_phi - true_phi) ** 2)

        subset_results["mse_g"] = float(mse_theta + mse_phi)
        subset_results["stats"] = {
            "theta": stats(lin_theta),
            "phi": stats(lin_phi)
        }

        return subset_results

    idx_A = np.arange(len(A))  # full set
    idx_F = F.index.to_numpy()  # indices inside A
    idx_C = C.index.to_numpy()  # complement indices
    linear_paths = {"U_model": {"U": os.path.join(run_path, "linear_U_model.npy")},
                    "U_equiv": {"U": os.path.join(run_path, "linear_U_equiv.npy")},
                    "g_model": {"theta": os.path.join(run_path, "linear_g_theta_model.npy"),
                                "phi": os.path.join(run_path, "linear_g_phi_model.npy")},
                    "g_equiv": {"theta": os.path.join(run_path, "linear_g_theta_equiv.npy"),
                                "phi": os.path.join(run_path, "linear_g_phi_equiv.npy")}}
    linear_results = {}  # final dictionary for JSON output

    for label, paths in linear_paths.items():
        linear_results[label] = {}

        # A — full
        res_A = evaluate_linear_baseline(
            label, paths,
            true_U=A["dU_m2_s2"].to_numpy() if "U" in paths else None,
            true_theta=A["dg_theta_mGal"].to_numpy(),
            true_phi=A["dg_phi_mGal"].to_numpy(),
            idx_subset=idx_A
        )
        if res_A is not None:
            linear_results[label]["A"] = res_A

        # F — high acceleration
        res_F = evaluate_linear_baseline(
            label, paths,
            true_U=F["dU_m2_s2"].to_numpy() if "U" in paths else None,
            true_theta=F["dg_theta_mGal"].to_numpy(),
            true_phi=F["dg_phi_mGal"].to_numpy(),
            idx_subset=idx_F
        )
        if res_F is not None:
            linear_results[label]["F"] = res_F

        # C — complement
        res_C = evaluate_linear_baseline(
            label, paths,
            true_U=C["dU_m2_s2"].to_numpy() if "U" in paths else None,
            true_theta=C["dg_theta_mGal"].to_numpy(),
            true_phi=C["dg_phi_mGal"].to_numpy(),
            idx_subset=idx_C
        )
        if res_C is not None:
            linear_results[label]["C"] = res_C

    meta = {
        "mode": mode,

        # Number of samples in each dataset
        "samples": {
            "A": len(A),
            "F": len(F),
            "C": len(C),
        },

        "nn": {
            "A": res_A,
            "F": res_F,
            "C": res_C,
        },

        "linear": linear_results,

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "stats": {
            "nn_pred": {
                "A": pred_A,
                "F": pred_F,
                "C": pred_C,
            },

            # NN true stats
            "nn_true": {
                "A": true_A,
                "F": true_F,
                "C": true_C,
            },

            # Linear predicted stats
            "linear_pred": {
                "U_model": {
                    "A": linear_results["U_model"]["A"]["stats"],
                    "F": linear_results["U_model"]["F"]["stats"],
                    "C": linear_results["U_model"]["C"]["stats"],
                },
                "U_equiv": {
                    "A": linear_results["U_equiv"]["A"]["stats"],
                    "F": linear_results["U_equiv"]["F"]["stats"],
                    "C": linear_results["U_equiv"]["C"]["stats"],
                },
                "g_model": {
                    "A": linear_results["g_model"]["A"]["stats"],
                    "F": linear_results["g_model"]["F"]["stats"],
                    "C": linear_results["g_model"]["C"]["stats"],
                },
                "g_equiv": {
                    "A": linear_results["g_equiv"]["A"]["stats"],
                    "F": linear_results["g_equiv"]["F"]["stats"],
                    "C": linear_results["g_equiv"]["C"]["stats"],
                }
            }
        }
    }

    meta_file = os.path.join(latest_run, "test_results_report.json")

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"🧩 Metadata saved to {meta_file}")
    print("✅ Done.")
    return model, test_df


if __name__ == "__main__":
    mp.freeze_support()
    main()
