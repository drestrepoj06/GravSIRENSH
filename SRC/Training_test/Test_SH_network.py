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

    print(f"📈 Loaded {len(test_df):,} test samples")

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

    lon = torch.tensor(test_df["lon"].values, dtype=torch.float32, device=device)
    lat = torch.tensor(test_df["lat"].values, dtype=torch.float32, device=device)

    true_U = test_df["dU_m2_s2"].values
    true_theta = test_df["dg_theta_mGal"].values
    true_phi = test_df["dg_phi_mGal"].values
    true_mag = test_df["dg_total_mGal"].values

    print("⚙️ Running model inference...")

    start = time.time()

    def to_np(t):
        """Detach and convert tensor to NumPy."""
        return t.detach().cpu().numpy()

    if mode == "U":
        # Predict potential and optionally gradients
        lon = lon.clone().detach().requires_grad_(True)
        lat = lat.clone().detach().requires_grad_(True)
        U_scaled, grads = model(lon, lat, return_gradients = True)
        U_pred = scaler.unscale_potential(U_scaled).detach()
        g_scaled = torch.stack(grads, dim=1)
        g_phys = scaler.unscale_gravity(g_scaled)

        # gravity components
        g_theta = g_phys[:, 0]
        g_phi = g_phys[:, 1]

        g_theta = g_theta.detach()
        g_phi = g_phi.detach()

        g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

        mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)
        mse_g = np.mean((to_np(g_theta) - true_theta) ** 2) + \
                np.mean((to_np(g_phi) - true_phi) ** 2)

    elif mode == "g_direct":
        preds = model(lon, lat)
        g_pred = (scaler.unscale_gravity(preds)).detach()
        g_theta = g_pred[:, 0]
        g_phi = g_pred[:, 1]
        g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

        mse_U = None
        mse_g = np.mean((to_np(g_theta) - true_theta) ** 2) + np.mean(
            (to_np(g_phi) - true_phi) ** 2
        )

    elif mode == "g_indirect":
        # Recompute U_pred with autograd
        lon = lon.clone().detach().requires_grad_(True)
        lat = lat.clone().detach().requires_grad_(True)

        U_scaled,  grads = model(lon, lat)
        U_pred = scaler.unscale_potential(U_scaled).detach()

        g_scaled = torch.stack(grads, dim=1)
        g_phys = scaler.unscale_gravity(g_scaled)

        # gravity components
        g_theta = g_phys[:, 0]
        g_phi = g_phys[:, 1]

        g_theta = g_theta.detach()
        g_phi = g_phi.detach()

        g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

        mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)
        mse_g = np.mean((to_np(g_theta) - true_theta) ** 2) + \
                np.mean((to_np(g_phi) - true_phi) ** 2)

    elif mode == "g_hybrid":
        out = model(lon, lat)  # out is a dict
        U_pred_scaled = out["U_pred"]  # (N,1)
        g_pred_scaled = out["g_pred"]  # (N,2)
        g_from_gradU_scaled = out["g_from_gradU"]  # (N,2)
        U_pred = scaler.unscale_potential(U_pred_scaled).detach()
        g_pred = scaler.unscale_gravity(g_pred_scaled).detach()
        g_from_gradU = scaler.unscale_gravity(g_from_gradU_scaled).detach()

        g_theta, g_phi = g_pred[:, 0], g_pred[:, 1]
        g_theta_grad, g_phi_grad = g_from_gradU[:, 0], g_from_gradU[:, 1]

        g_mag_grad = torch.sqrt(g_theta_grad ** 2 + g_phi_grad ** 2)

        g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

        mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)

        mse_g = (
                np.mean((to_np(g_theta) - true_theta) ** 2) +
                np.mean((to_np(g_phi) - true_phi) ** 2)
        )

        mse_grad = (
                np.mean((to_np(g_theta_grad) - true_theta) ** 2) +
                np.mean((to_np(g_phi_grad) - true_phi) ** 2)
        )

        mse_consistency = (
                np.mean((to_np(g_theta_grad) - to_np(g_theta)) ** 2) +
                np.mean((to_np(g_phi_grad) - to_np(g_phi)) ** 2)
        )

    elif mode == "U_g_direct":
        U_pred, g_pred = model(lon, lat)
        U_pred = scaler.unscale_potential(U_pred).detach()
        g_pred = (scaler.unscale_gravity(g_pred)).detach()
        g_theta = g_pred[:, 0]
        g_phi = g_pred[:, 1]
        g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

        mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)
        mse_g = np.mean((to_np(g_theta) - true_theta) ** 2) + np.mean(
            (to_np(g_phi) - true_phi) ** 2
        )

    # elif mode == "U_g_indirect":
    #
    #     U_scaled, grads = model(lon, lat, return_gradients=True)
    #     U_pred = scaler.unscale_potential(U_scaled).detach()
    #
    #     g_theta, g_phi, _ = scaler.unscale_acceleration_from_potential(
    #         grads,
    #         lat=lat,
    #         r=None
    #     )
    #     g_theta = (g_theta * 1e5).detach()
    #     g_phi = (g_phi * 1e5).detach()
    #     g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)
    #
    #     mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)
    #     mse_g = np.mean((to_np(g_theta) - true_theta) ** 2) + \
    #             np.mean((to_np(g_phi) - true_phi) ** 2)
    #
    # elif mode == "U_g_hybrid":
    #
    #     # Unpack outputs
    #     U_pred_scaled, g_pred_scaled, (gtheta_fromU_scaled, gphi_fromU_scaled) = model(lon, lat)
    #
    #     # ---- Unscale everything ----
    #     U_pred = scaler.unscale_potential(U_pred_scaled).detach()
    #
    #     g_pred = scaler.unscale_gravity(g_pred_scaled).detach()
    #     g_theta = g_pred[:, 0]
    #     g_phi = g_pred[:, 1]
    #
    #     # Unscale gradient-based gravity
    #     g_from_gradU = scaler.unscale_gravity(
    #         torch.stack([gtheta_fromU_scaled, gphi_fromU_scaled], dim=1)
    #     ).detach()
    #
    #     g_theta_grad = g_from_gradU[:, 0]
    #     g_phi_grad = g_from_gradU[:, 1]
    #
    #     # Magnitudes
    #     g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)
    #     g_mag_grad = torch.sqrt(g_theta_grad ** 2 + g_phi_grad ** 2)
    #
    #     mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)
    #
    #     mse_g = (
    #             np.mean((to_np(g_theta) - true_theta) ** 2) +
    #             np.mean((to_np(g_phi) - true_phi) ** 2)
    #     )
    #
    #     mse_grad = (
    #             np.mean((to_np(g_theta_grad) - true_theta) ** 2) +
    #             np.mean((to_np(g_phi_grad) - true_phi) ** 2)
    #     )
    #
    #     mse_consistency = (
    #             np.mean((to_np(g_theta_grad) - to_np(g_theta)) ** 2) +
    #             np.mean((to_np(g_phi_grad) - to_np(g_phi)) ** 2)
    #     )

    else:
        raise ValueError(f"Unsupported mode '{mode}'")
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"⏱️ Inference done in {time.time() - start:.1f}s")

    if mse_U is not None:
        U_pred_np = U_pred.detach().cpu().numpy().ravel()
    else:
        U_pred_np = None

    g_theta_np = g_theta.detach().cpu().numpy()
    g_phi_np = g_phi.detach().cpu().numpy()
    g_mag_np = g_mag.detach().cpu().numpy()

    # Handle gradient-based g only if available
    if mse_grad is not None:
        g_theta_grad_np = g_theta_grad.detach().cpu().numpy()
        g_phi_grad_np = g_phi_grad.detach().cpu().numpy()
        g_mag_grad_np = g_mag_grad.detach().cpu().numpy()
    else:
        g_theta_grad_np = None
        g_phi_grad_np = None
        g_mag_grad_np = None

    prefix = os.path.join(latest_run, "test_results")

    if U_pred_np is not None:
        np.save(f"{prefix}_U.npy", U_pred_np)

    # Save gradient-based gravity only when it exists
    if g_theta_grad_np is not None:
        np.save(f"{prefix}_g_theta_grad.npy", g_theta_grad_np)
        np.save(f"{prefix}_g_phi_grad.npy", g_phi_grad_np)
        np.save(f"{prefix}_g_mag_grad.npy", g_mag_grad_np)

    # Save direct predicted gravity
    np.save(f"{prefix}_g_theta.npy", g_theta_np)
    np.save(f"{prefix}_g_phi.npy", g_phi_np)
    np.save(f"{prefix}_g_mag.npy", g_mag_np)

    print(f"💾 Predictions saved in {latest_run}")

    # === Metadata summary ===
    def stats(arr):
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    pred_stats = {
        "g_theta": stats(g_theta_np),
        "g_phi": stats(g_phi_np),
        "g_mag": stats(g_mag_np),
    }

    if U_pred_np is not None:
        pred_stats["U"] = stats(U_pred_np)

    # Add stats of gradient-based gravity where available
    if g_theta_grad_np is not None:
        pred_stats["g_theta_grad"] = stats(g_theta_grad_np)
        pred_stats["g_phi_grad"] = stats(g_phi_grad_np)
        pred_stats["g_mag_grad"] = stats(g_mag_grad_np)

    # === True stats ===
    true_stats = {
        "g_theta": stats(test_df["dg_theta_mGal"].to_numpy()),
        "g_phi": stats(test_df["dg_phi_mGal"].to_numpy()),
        "g_mag": stats(test_df["dg_total_mGal"].to_numpy()),
    }

    if "dU_m2_s2" in test_df.columns:
        true_stats["U"] = stats(test_df["dU_m2_s2"].to_numpy())

    # === Linear baseline stats ===
    linear_paths = {
        "U_model": {
            "U": os.path.join(run_path, "linear_U_model.npy")
        },
        "U_equiv": {
            "U": os.path.join(run_path, "linear_U_equiv.npy")
        },
        "g_model": {
            "theta": os.path.join(run_path, "linear_g_theta_model.npy"),
            "phi": os.path.join(run_path, "linear_g_phi_model.npy")
        },
        "g_equiv": {
            "theta": os.path.join(run_path, "linear_g_theta_equiv.npy"),
            "phi": os.path.join(run_path, "linear_g_phi_equiv.npy")
        }
    }

    linear_results = {}
    mse_linear = {}

    for label, paths in linear_paths.items():

        if "U" in paths:
            U_path = paths["U"]

            if not os.path.exists(U_path):
                print(f"⚠ Missing potential baseline for {label}")
                continue

            lin_U = np.load(U_path)

            if len(lin_U) != len(true_U):
                print(f"⚠ Length mismatch in potential baseline {label}")
                continue

            mse = float(np.mean((lin_U - true_U) ** 2))
            mse_linear[label] = mse

            linear_results[label] = {
                "U": stats(lin_U)
            }

            print(f"✔ Loaded linear potential: {label} → MSE={mse:.3e}")
            continue

        theta_path = paths["theta"]
        phi_path = paths["phi"]

        if not (os.path.exists(theta_path) and os.path.exists(phi_path)):
            print(f"⚠ Missing gravity component files for {label}")
            continue

        lin_theta = np.load(theta_path)
        lin_phi = np.load(phi_path)

        if len(lin_theta) != len(true_theta):
            print(f"⚠ Length mismatch in gravity baseline {label}")
            continue

        mse_theta = np.mean((lin_theta - true_theta) ** 2)
        mse_phi = np.mean((lin_phi - true_phi) ** 2)

        mse_total = float(mse_theta + mse_phi)
        mse_linear[label] = mse_total

        linear_results[label] = {
            "theta": stats(lin_theta),
            "phi": stats(lin_phi)
        }

        print(f"✔ Loaded linear gravity: {label} → MSE={mse_total:.3e}")

    meta = {
        "mode": mode,
        "samples": len(test_df),

        # NN MSEs
        "mse_U": float(mse_U) if mse_U is not None else None,
        "mse_g": float(mse_g),
        "mse_grad": float(mse_grad) if mse_grad is not None else None,
        "mse_consistency": float(mse_consistency) if mse_consistency is not None else None,

        # Linear baselines: dictionary with 4 entries
        "mse_linear": mse_linear,  # { "U_model": ..., "U_equiv": ..., "g_model": ..., "g_equiv": ... }

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "stats": {
            "pred": pred_stats,
            "true": true_stats,

            # These match linear_results dict produced earlier
            "linear_U_model": linear_results.get("U_model"),
            "linear_U_equiv": linear_results.get("U_equiv"),
            "linear_g_model": linear_results.get("g_model"),
            "linear_g_equiv": linear_results.get("g_equiv"),
        },
    }

    meta_file = f"{prefix}_report.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"🧩 Metadata saved to {meta_file}")
    print("✅ Done.")
    return model, test_df


if __name__ == "__main__":
    mp.freeze_support()
    main()
