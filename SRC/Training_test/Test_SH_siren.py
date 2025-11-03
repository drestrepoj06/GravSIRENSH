import os
import json
import torch
import pandas as pd
import numpy as np
import sys
import multiprocessing as mp
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_siren import SH_SIREN, SHSirenScaler, SH_LINEAR

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    models_dir = os.path.join(base_dir, 'Outputs', 'Models')
    data_dir = os.path.join(base_dir, 'Data')

    # ───────────────────────────────
    # Load test data
    # ───────────────────────────────
    test_path = os.path.join(data_dir, "Samples_2190-2_250k_r0_test.parquet")
    test_df = pd.read_parquet(test_path)
    print(f"📈 Loaded {len(test_df):,} test samples")

    # ───────────────────────────────
    # Load latest model + config
    # ───────────────────────────────
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    config_files = [f for f in os.listdir(models_dir) if f.endswith('_model_config.json')]
    if not model_files or not config_files:
        raise FileNotFoundError("❌ No model/config files found in Outputs/Models")

    latest_model_path = max([os.path.join(models_dir, f) for f in model_files], key=os.path.getmtime)
    latest_config_path = max([os.path.join(models_dir, f) for f in config_files], key=os.path.getmtime)

    print(f"📂 Model file:  {os.path.basename(latest_model_path)}")
    print(f"📂 Config file: {os.path.basename(latest_config_path)}")

    with open(latest_config_path, "r") as f:
        config = json.load(f)

    print(f"🧩 Loaded config: Lmax={config['lmax']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(latest_model_path, map_location=device)
    scaler_data = checkpoint["scaler"]

    scaler = SHSirenScaler(
        U_min=scaler_data["U_min"],
        U_max=scaler_data["U_max"],
    )

    cache_path = os.path.join(base_dir, "Data", "cache_test")

    if config["model_type"].lower() == "siren":
        model = SH_SIREN(
            lmax=config["lmax"],
            hidden_features=config["hidden_features"],
            hidden_layers=config["hidden_layers"],
            out_features=config["out_features"],
            first_omega_0=config["first_omega_0"],
            hidden_omega_0=config["hidden_omega_0"],
            device=device,
            scaler=scaler,
            cache_path=cache_path
        )
        print("🌀 Loaded SH-SIREN model for testing")

    elif config["model_type"].lower() == "linear":
        model = SH_LINEAR(
            lmax=config["lmax"],
            out_features=config["out_features"],
            device=device,
            scaler=scaler,
            cache_path=cache_path
        )
        print("📈 Loaded SH-LINEAR model for testing")

    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print("✅ Model and scaler loaded successfully.")

    # True values (for comparison)
    true_theta = test_df["dg_theta_mGal"].values
    true_phi = test_df["dg_phi_mGal"].values
    true_mag = test_df["dg_total_mGal"].values


    print("⚙️ Computing potential and accelerations via autograd...")

    lon = torch.tensor(test_df["lon"].values, dtype=torch.float32, device=device, requires_grad=True)
    lat = torch.tensor(test_df["lat"].values, dtype=torch.float32, device=device, requires_grad=True)

    start = time.time()
    U_scaled, grads_phys = model(lon, lat, return_gradients=True)
    U_pred = scaler.unscale_potential(U_scaled)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"⏱️ Forward + gradients done in {time.time() - start:.1f}s")

    # Unpack gradients
    g_lon, g_lat, g_r = grads_phys
    g_mag = torch.sqrt(g_lon**2 + g_lat**2)

    # Convert to NumPy
    U_pred = U_pred.detach().cpu().numpy().ravel()
    pred_g_lon = g_lon.detach().cpu().numpy()
    pred_g_lat = g_lat.detach().cpu().numpy()
    pred_g_mag = g_mag.detach().cpu().numpy()

    # ───────────────────────────────
    # Compute statistics and errors
    # ───────────────────────────────
    mse_theta = np.mean((pred_g_lat - true_theta) ** 2)
    mse_phi = np.mean((pred_g_lon - true_phi) ** 2)
    mse_mag = np.mean((pred_g_mag - true_mag) ** 2)

    print(f"📊 MSEθ={mse_theta:.3e}  MSEφ={mse_phi:.3e}  MSE|g|={mse_mag:.3e}")

    # Global stats
    stats = {
        "true": {
            "θ_min": float(true_theta.min()), "θ_max": float(true_theta.max()), "θ_std": float(true_theta.std()),
            "φ_min": float(true_phi.min()), "φ_max": float(true_phi.max()), "φ_std": float(true_phi.std()),
            "|g|_min": float(true_mag.min()), "|g|_max": float(true_mag.max()), "|g|_std": float(true_mag.std())
        },
        "pred": {
            "θ_min": float(pred_g_lat.min()), "θ_max": float(pred_g_lat.max()), "θ_std": float(pred_g_lat.std()),
            "φ_min": float(pred_g_lon.min()), "φ_max": float(pred_g_lon.max()), "φ_std": float(pred_g_lon.std()),
            "|g|_min": float(pred_g_mag.min()), "|g|_max": float(pred_g_mag.max()), "|g|_std": float(pred_g_mag.std())
        },
    }

    outputs_dir = os.path.join(base_dir, "Outputs", "Predictions")
    os.makedirs(outputs_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(latest_model_path))[0]
    prefix = os.path.join(outputs_dir, model_name)

    np.save(f"{prefix}_U.npy", U_pred)
    np.save(f"{prefix}_g_theta.npy", pred_g_lat)
    np.save(f"{prefix}_g_phi.npy", pred_g_lon)
    np.save(f"{prefix}_g_mag.npy", pred_g_mag)
    print(f"💾 Predictions saved to {outputs_dir}")

    # Metadata
    meta = {
        "model_file": os.path.basename(latest_model_path),
        "config_file": os.path.basename(latest_config_path),
        "samples": len(test_df),
        "mse_theta_mgal2": float(mse_theta),
        "mse_phi_mgal2": float(mse_phi),
        "mse_mag_mgal2": float(mse_mag),
        "stats": stats,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_file = f"{prefix}_autograd_test_report.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"🧩 Metadata saved to {meta_file}")
    print("✅ Done.")
    return model, test_df


if __name__ == '__main__':
    mp.freeze_support()
    main()