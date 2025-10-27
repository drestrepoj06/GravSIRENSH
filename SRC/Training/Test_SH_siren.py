"""Test of the SIRENSH network
jhonr"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
from sklearn.metrics import mean_squared_error

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_siren import SH_SIREN, SHSirenScaler


def main():
    # === Paths and configuration ===
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    outputs_dir = os.path.join(base_dir, "Outputs", "Models")

    # External test dataset (new distribution)
    test_data_path = os.path.join(base_dir, "Data", "Samples_2190-2_250k_r0_test.parquet")

    # === Find latest trained model/config ===
    model_files = [f for f in os.listdir(outputs_dir) if f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError("❌ No trained model (.pth) found in Outputs/Models.")
    latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)))
    model_path = os.path.join(outputs_dir, latest_model)

    config_files = [f for f in os.listdir(outputs_dir) if f.endswith("_model_config.json")]
    latest_config = max(config_files, key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)))
    config_path = os.path.join(outputs_dir, latest_config)

    print(f"📂 Model file:  {latest_model}")
    print(f"📂 Config file: {latest_config}")

    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model and scaler ===
    checkpoint = torch.load(model_path, map_location=device)

    scaler = SHSirenScaler(r_scale=checkpoint["scaler"]["r_scale"])
    scaler.U_min = checkpoint["scaler"]["U_min"]
    scaler.U_max = checkpoint["scaler"]["U_max"]

    model = SH_SIREN(
        lmax=config["lmax"],
        hidden_features=config["hidden_features"],
        hidden_layers=config["hidden_layers"],
        out_features=config["out_features"],
        first_omega_0=config["first_omega_0"],
        hidden_omega_0=config["hidden_omega_0"],
        device=device,
        scaler=scaler
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Model and scaler loaded successfully on {device}.")

    # === Load and scale new test data ===
    test_df = pd.read_parquet(test_data_path)
    print(f"📈 Loaded {len(test_df):,} test samples from external dataset")

    # Scale potential using the *training* scaler limits (do NOT refit)
    test_df["dV_m2_s2_scaled"] = scaler.scale_potential(test_df["dV_m2_s2"].values)

    # Scale spatial coordinates
    lon_t = torch.tensor(test_df["lon"].values, dtype=torch.float32)
    lat_t = torch.tensor(test_df["lat"].values, dtype=torch.float32)
    r_t = torch.tensor(test_df["radius_m"].values, dtype=torch.float32)

    lon_scaled, lat_scaled, r_scaled = scaler.scale_inputs(lon_t, lat_t, r_t)
    test_df["radius_bar"] = r_scaled.numpy()

    lon = lon_t.to(device)
    lat = lat_t.to(device)
    r = r_t.to(device)

    # === Generate predictions ===
    with torch.no_grad():
        preds_scaled = model(lon, lat, r).cpu().numpy().ravel()

    preds_unscaled = scaler.unscale_potential(preds_scaled)
    test_df["predicted_dV_m2_s2"] = preds_unscaled

    true_vals = test_df["dV_m2_s2"].values
    mse = mean_squared_error(true_vals, preds_unscaled)

    print(f"📊 RMSE: {mse:.6e}")


    # === Save predictions ===
    preds_dir = os.path.join(base_dir, "Outputs", "Predictions")
    os.makedirs(preds_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lmax = config["lmax"]

    npy_path = os.path.join(preds_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_preds.npy")

    np.save(npy_path, preds_unscaled)
    test_df.to_parquet(parquet_path, index=False)

    print(f"💾 Predictions saved:")
    print(f"   • {npy_path}")
    print(f"   • {parquet_path}")

    # === Save evaluation report ===
    report = {
        "timestamp": timestamp,
        "device": str(device),
        "lmax": lmax,
        "n_samples": len(test_df),
        "mse": float(mse),
        "model_file": model_path,
        "config_file": config_path,
        "preds_file": npy_path
    }
    report_path = os.path.join(preds_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"🧾 Test report saved to {report_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()