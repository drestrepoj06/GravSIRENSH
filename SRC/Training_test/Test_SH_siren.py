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
    # Enable gradient tracking
    deg2rad = np.pi / 180.0

    # Enable gradient tracking
    lon.requires_grad_(True)
    lat.requires_grad_(True)
    r.requires_grad_(True)

    # Get scaled-output gradients (model outputs U_scaled)
    U_scaled, (dUscaled_dlon_deg, dUscaled_dlat_deg, dUscaled_dr) = model(
        lon, lat, r, return_gradients=True, physical_units=False  # important: False
    )

    # Unscale gradients from U_scaled -> U (m^2/s^2) using S = (Umax - Umin)/2
    S = 0.5 * (scaler.U_max - scaler.U_min)
    dU_dlon_deg = dUscaled_dlon_deg * S
    dU_dlat_deg = dUscaled_dlat_deg * S
    dU_dr = dUscaled_dr * S  # already per meter

    # Convert degree-gradients -> radian-gradients
    dU_dlon = dU_dlon_deg * (1.0 / deg2rad)  # ∂U/∂lon [per rad]
    dU_dlat = dU_dlat_deg * (1.0 / deg2rad)  # ∂U/∂lat [per rad]

    # Map to spherical (θ, φ), with θ = colatitude = π/2 - lat
    lat_rad = lat * deg2rad
    theta = 0.5 * np.pi - lat_rad
    sin_theta = torch.sin(theta).clamp_min(1e-12)  # avoid division by zero

    # ∂/∂θ = - ∂/∂lat
    dU_dtheta = -dU_dlat

    # Spherical components (m/s^2): g_r, g_theta (southward), g_phi (eastward)
    g_r = dU_dr
    g_theta = (1.0 / r) * dU_dtheta
    g_phi = (1.0 / (r * sin_theta)) * dU_dlon

    # Magnitude
    g_mag = torch.sqrt(g_r ** 2 + g_theta ** 2 + g_phi ** 2)

    # Move to CPU numpy
    a_r_mps2 = g_r.detach().cpu().numpy().ravel()
    a_theta_mps2 = g_theta.detach().cpu().numpy().ravel()
    a_phi_mps2 = g_phi.detach().cpu().numpy().ravel()
    a_mag_mps2 = g_mag.detach().cpu().numpy().ravel()

    # Convert to mGal for comparison (1 m/s^2 = 1e5 mGal)
    a_r_mGal = a_r_mps2 * 1e5
    a_theta_mGal = a_theta_mps2 * 1e5
    a_phi_mGal = a_phi_mps2 * 1e5
    a_mag_mGal = a_mag_mps2 * 1e5

    test_df["pred_dg_r_mGal"] = a_r_mGal
    test_df["pred_dg_theta_mGal"] = a_theta_mGal
    test_df["pred_dg_phi_mGal"] = a_phi_mGal
    test_df["pred_dg_mag_mGal"] = a_mag_mGal

    # === True (already in mGal)
    dg_r = test_df["dg_r_mGal"].values
    dg_theta = test_df["dg_theta_mGal"].values
    dg_phi = test_df["dg_phi_mGal"].values
    true_mag = np.sqrt(dg_r ** 2 + dg_theta ** 2 + dg_phi ** 2)

    # === Pred (now in mGal)
    pred_r = test_df["pred_dg_r_mGal"].values
    pred_theta = test_df["pred_dg_theta_mGal"].values
    pred_phi = test_df["pred_dg_phi_mGal"].values
    pred_mag = test_df["pred_dg_mag_mGal"].values

    true_stats = {
        "true_a_r_min": float(dg_r.min()), "true_a_r_max": float(dg_r.max()),
        "true_a_r_mean": float(dg_r.mean()), "true_a_r_std": float(dg_r.std()),
        "true_a_theta_min": float(dg_theta.min()), "true_a_theta_max": float(dg_theta.max()),
        "true_a_theta_mean": float(dg_theta.mean()), "true_a_theta_std": float(dg_theta.std()),
        "true_a_phi_min": float(dg_phi.min()), "true_a_phi_max": float(dg_phi.max()),
        "true_a_phi_mean": float(dg_phi.mean()), "true_a_phi_std": float(dg_phi.std()),
        "true_a_mag_min": float(true_mag.min()), "true_a_mag_max": float(true_mag.max()),
        "true_a_mag_mean": float(true_mag.mean()), "true_a_mag_std": float(true_mag.std()),
    }

    pred_stats = {
        "pred_a_r_min": float(pred_r.min()), "pred_a_r_max": float(pred_r.max()),
        "pred_a_r_mean": float(pred_r.mean()), "pred_a_r_std": float(pred_r.std()),
        "pred_a_theta_min": float(pred_theta.min()), "pred_a_theta_max": float(pred_theta.max()),
        "pred_a_theta_mean": float(pred_theta.mean()), "pred_a_theta_std": float(pred_theta.std()),
        "pred_a_phi_min": float(pred_phi.min()), "pred_a_phi_max": float(pred_phi.max()),
        "pred_a_phi_mean": float(pred_phi.mean()), "pred_a_phi_std": float(pred_phi.std()),
        "pred_a_mag_min": float(pred_mag.min()), "pred_a_mag_max": float(pred_mag.max()),
        "pred_a_mag_mean": float(pred_mag.mean()), "pred_a_mag_std": float(pred_mag.std()),
    }

    true_vals = test_df["dg_total_mGal"].values
    mse = mean_squared_error(true_vals, a_mag_mGal)
    print(f"📊 MSE: {mse:.6e}")
    # === Save predictions ===
    preds_dir = os.path.join(base_dir, "Outputs", "Predictions")
    os.makedirs(preds_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lmax = config["lmax"]

    npy_path = os.path.join(preds_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_preds.npy")

    np.save(npy_path, a_mag_mGal)

    print(f"💾 Predictions saved:")
    print(f"   • {npy_path}")


    # === Combine into one report ===
    report = {
        "timestamp": timestamp,
        "device": str(device),
        "lmax": lmax,
        "mse": mse,
        "model_file": model_path,
        "config_file": config_path,
        "preds_file": npy_path,
        "true_stats": true_stats,
        "pred_stats": pred_stats
    }
    report_path = os.path.join(preds_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"🧾 Test report saved to {report_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()