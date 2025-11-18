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
from SRC.Location_encoder.SH_siren import SH_SIREN, SHSirenScaler


def main(run_path=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    runs_dir = os.path.join(base_dir, "Outputs", "Runs")
    data_dir = os.path.join(base_dir, "Data")

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
    scaler = SHSirenScaler(mode=mode)
    scaler.U_mean = scaler_data.get("U_mean")
    scaler.U_std = scaler_data.get("U_std")

    g_mean = scaler_data.get("g_mean")
    g_std = scaler_data.get("g_std")

    if g_mean is not None:
        scaler.g_mean = torch.tensor(g_mean, dtype=torch.float32, device=device)
    if g_std is not None:
        scaler.g_std = torch.tensor(g_std, dtype=torch.float32, device=device)

    # === Initialize model ===
    cache_path = os.path.join(base_dir, "Data", "cache_test.npy")
    model = SH_SIREN(
        lmax=config["lmax"],
        hidden_features=config["hidden_features"],
        hidden_layers=config["hidden_layers"],
        first_omega_0=config["first_omega_0"],
        hidden_omega_0=config["hidden_omega_0"],
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

    # === Prepare test inputs ===
    lon = torch.tensor(test_df["lon"].values, dtype=torch.float32, device=device)
    lat = torch.tensor(test_df["lat"].values, dtype=torch.float32, device=device)

    true_U = test_df["dU_m2_s2"].values
    true_theta = test_df["dg_theta_mGal"].values
    true_phi = test_df["dg_phi_mGal"].values
    true_mag = np.sqrt(true_theta**2 + true_phi**2)

    print("⚙️ Running model inference...")

    start = time.time()

    def to_np(t):
        """Detach and convert tensor to NumPy."""
        return t.detach().cpu().numpy()

    if mode == "U":
        # Predict potential and optionally gradients
        U_scaled, grads_phys = model(lon, lat, return_gradients=True)
        U_pred = scaler.unscale_potential(U_scaled).detach()
        dU_dlon, dU_dlat, _ = grads_phys
        g_theta = (-dU_dlat * 1e5).detach()
        g_phi = (-dU_dlon * 1e5).detach()
        g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

        mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)
        mse_g = np.mean((to_np(g_mag) - true_mag) ** 2)

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
        U_pred, (g_theta, g_phi) = model(lon, lat)
        U_pred = scaler.unscale_potential(U_pred).detach()
        g_theta = g_theta.detach()
        g_phi = g_phi.detach()
        g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

        mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)
        mse_g = np.mean((to_np(g_theta) - true_theta) ** 2) + np.mean(
            (to_np(g_phi) - true_phi) ** 2
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

    elif mode == "U_g_indirect":
        U_pred, (g_theta, g_phi) = model(lon, lat)
        U_pred = scaler.unscale_potential(U_pred).detach()
        g_theta = g_theta.detach()
        g_phi = g_phi.detach()
        g_mag = torch.sqrt(g_theta ** 2 + g_phi ** 2)

        mse_U = np.mean((to_np(U_pred).ravel() - true_U) ** 2)
        mse_g = np.mean((to_np(g_theta) - true_theta) ** 2) + np.mean(
            (to_np(g_phi) - true_phi) ** 2
        )

    else:
        raise ValueError(f"Unsupported mode '{mode}'")
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"⏱️ Inference done in {time.time() - start:.1f}s")

    # === Convert to NumPy ===
    if mse_U is not None:
        U_pred_np = U_pred.detach().cpu().numpy().ravel()
    else:
        U_pred_np = None
    g_theta_np = g_theta.detach().cpu().numpy()
    g_phi_np = g_phi.detach().cpu().numpy()
    g_mag_np = g_mag.detach().cpu().numpy()

    # === Compute summary statistics ===
    print("📊 Test metrics:")
    if mse_U is not None:
        print(f"   MSE(U) = {mse_U:.3e}")
    print(f"   MSE(g) = {mse_g:.3e}")

    # === Save predictions inside the same run folder ===
    prefix = os.path.join(latest_run, "test_results")

    if U_pred_np is not None:
        np.save(f"{prefix}_U.npy", U_pred_np)
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

    true_stats = {
        "g_theta": stats(test_df["dg_theta_mGal"].to_numpy()),
        "g_phi": stats(test_df["dg_phi_mGal"].to_numpy()),
        "g_mag": stats(test_df["dg_total_mGal"].to_numpy()),
    }

    if "dU_m2_s2" in test_df.columns:
        true_stats["U"] = stats(test_df["dU_m2_s2"].to_numpy())
    # === Metadata summary ===
    meta = {
        "mode": mode,
        "samples": len(test_df),
        "mse_U": float(mse_U) if mse_U is not None else None,
        "mse_g": float(mse_g),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "pred": pred_stats,
            "true": true_stats
        }
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
