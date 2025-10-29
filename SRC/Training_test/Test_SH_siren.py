"""Test of the SIRENSH network
jhonr"""
import os
import sys
import torch
import numpy as np
import pandas as pd
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
    test_data_path = os.path.join(base_dir, "Data", "Samples_10-2_50_r0_test.parquet")

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

    scaler = SHSirenScaler()
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

    lon = lon_t.to(device)
    lat = lat_t.to(device)
    r = r_t.to(device)

    # --- model grads in scaled space (per DEG for lon/lat) ---
    U_phys, grads_phys = model(lon, lat, r, return_gradients=True, physical_units=True)

    # grads_phys = (dU/dlon, dU/dlat, dU/dr) but dU/dr≈0 now
    a_theta = -grads_phys[0]
    a_phi = -grads_phys[1]

    # 6) Convert to mGal and magnitude
    a_theta_mGal = a_theta.detach().cpu().numpy().ravel() * 1e5
    a_phi_mGal = a_phi.detach().cpu().numpy().ravel() * 1e5
    g_mag = torch.sqrt(a_theta ** 2 + a_phi ** 2)
    a_mag_mGal = g_mag.detach().cpu().numpy().ravel() * 1e5
    U_pred = U_phys.detach().cpu().numpy().ravel()

    true_u = test_df["dV_m2_s2"].values
    true_theta = test_df["dg_theta_mGal"].values
    true_phi = test_df["dg_phi_mGal"].values
    true_mag = test_df["dg_total_mGal"].values

    # === Compute statistics ===
    true_stats = {
        "true_U_min": float(true_u.min()), "true_U_max": float(true_u.max()),
        "true_U_mean": float(true_u.mean()), "true_U_std": float(true_u.std()),
        "true_a_theta_min": float(true_theta.min()), "true_a_theta_max": float(true_theta.max()),
        "true_a_theta_mean": float(true_theta.mean()), "true_a_theta_std": float(true_theta.std()),
        "true_a_phi_min": float(true_phi.min()), "true_a_phi_max": float(true_phi.max()),
        "true_a_phi_mean": float(true_phi.mean()), "true_a_phi_std": float(true_phi.std()),
        "true_a_mag_min": float(true_mag.min()), "true_a_mag_max": float(true_mag.max()),
        "true_a_mag_mean": float(true_mag.mean()), "true_a_mag_std": float(true_mag.std()),
    }

    pred_stats = {
        "pred_U_min": float(U_pred.min()), "pred_U_max": float(U_pred.max()),
        "pred_U_mean": float(U_pred.mean()), "pred_U_std": float(U_pred.std()),
        "pred_a_theta_min": float(a_theta_mGal.min()), "pred_a_theta_max": float(a_theta_mGal.max()),
        "pred_a_theta_mean": float(a_theta_mGal.mean()), "pred_a_theta_std": float(a_theta_mGal.std()),
        "pred_a_phi_min": float(a_phi_mGal.min()), "pred_a_phi_max": float(a_phi_mGal.max()),
        "pred_a_phi_mean": float(a_phi_mGal.mean()), "pred_a_phi_std": float(a_phi_mGal.std()),
        "pred_a_mag_min": float(a_mag_mGal.min()), "pred_a_mag_max": float(a_mag_mGal.max()),
        "pred_a_mag_mean": float(a_mag_mGal.mean()), "pred_a_mag_std": float(a_mag_mGal.std()),
    }

    # === Compute MSE per component ===
    mse_u= mean_squared_error(true_u, U_pred)
    mse_theta = mean_squared_error(true_theta, a_theta_mGal)
    mse_phi = mean_squared_error(true_phi, a_phi_mGal)
    mse_mag = mean_squared_error(true_mag, a_mag_mGal)
    print(f"📊 MSE_U: {mse_u:.6e}")
    print(f"📊 MSE_theta: {mse_theta:.6e}")
    print(f"📊 MSE_phi:   {mse_phi:.6e}")
    print(f"📊 MSE_mag:   {mse_mag:.6e}")

    # === Save acceleration magnitude predictions ===
    preds_dir = os.path.join(base_dir, "Outputs", "Predictions")
    os.makedirs(preds_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_file = os.path.join(preds_dir, f"{model_name}_preds.npy".replace(",", ""))

    np.save(output_file, a_mag_mGal)
    print(f"💾 Acceleration magnitude predictions saved to {output_file}")

    # === Save metadata ===
    meta = {
        "model_file": os.path.basename(model_path),
        "config_file": os.path.basename(config_path),
        "device": str(device),
        "lmax": config["lmax"],
        "hidden_layers": config.get("hidden_layers", None),
        "samples": len(test_df),
        "mse_u_m2/s22": float(mse_u),
        "mse_theta_mgal2": float(mse_theta),
        "mse_phi_mgal2": float(mse_phi),
        "mse_mag_mgal2": float(mse_mag),
        **true_stats,
        **pred_stats
    }

    report_path = output_file.replace(".npy", "_test_report.json")
    with open(report_path, "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()