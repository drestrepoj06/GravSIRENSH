import os
import json
import torch
import pandas as pd
import numpy as np
import sys
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_siren import SH_SIREN, SHSirenScaler

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    models_dir = os.path.join(base_dir, 'Outputs', 'Models')
    data_dir = os.path.join(base_dir, 'Data')

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

    print(f"🧩 Loaded config: Lmax={config['lmax']}, hidden_layers={config['hidden_layers']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(latest_model_path, map_location=device)
    scaler_data = checkpoint["scaler"]

    scaler = SHSirenScaler(
        r_scale=scaler_data["r_scale"],
        t_min=scaler_data["t_min"],
        t_max=scaler_data["t_max"],
        target_name=scaler_data.get("target_name", None),
        target_units=scaler_data.get("target_units", None),
    )

    cache_path = os.path.join(base_dir, "Data", "cache_basis_test")

    model = SH_SIREN(
        lmax=config["lmax"],
        hidden_features=config["hidden_features"],
        hidden_layers=config["hidden_layers"],
        out_features=config["out_features"],
        first_omega_0=config["first_omega_0"],
        hidden_omega_0=config["hidden_omega_0"],
        device=device,
        scaler=scaler,
        cache_path=cache_path,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print("✅ Model and scaler loaded successfully.")

    test_path = os.path.join(data_dir, "Samples_2190-2_250k_r0_test.parquet")
    test_df = pd.read_parquet(test_path)
    print(f"📈 Loaded {len(test_df):,} test samples")

    true_values = test_df["dV_m2_s2"].values.astype(np.float32)
    with torch.no_grad():
        preds_scaled = model(test_df).cpu().numpy().ravel()

    preds = scaler.unscale_target(preds_scaled)

    true_min = true_values.min()
    true_max = true_values.max()
    true_mean = true_values.mean()
    true_std = true_values.std()
    pred_min = preds.min()
    pred_max = preds.max()
    pred_mean = preds.mean()
    pred_std = preds.std()

    print(f"📉 Predictions statistics:")
    print(f"   • Min:  {pred_min:.6f}")
    print(f"   • Max:  {pred_max:.6f}")
    print(f"   • Mean: {pred_mean:.6f}")
    print(f"   • Std:  {pred_std:.6f}")

    print(f"📉 True statistics:")
    print(f"   • Min:  {true_min:.6f}")
    print(f"   • Max:  {true_max:.6f}")
    print(f"   • Mean: {true_mean:.6f}")
    print(f"   • Std:  {true_std:.6f}")

    mse = np.mean((preds - true_values) ** 2)
    print(f"📊 MSE: {mse:.6f} (mGal²)")
    outputs_dir = os.path.join(base_dir, "Outputs", "Predictions")
    os.makedirs(outputs_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(latest_model_path))[0]
    output_file = os.path.join(
        outputs_dir,
        f"{model_name}_preds.npy".replace(",", "")
    )

    np.save(output_file, preds)
    print(f"💾 Predictions saved to {output_file}")

    meta = {
        "model_file": os.path.basename(latest_model_path),
        "config_file": os.path.basename(latest_config_path),
        "lmax": config["lmax"],
        "hidden_layers": config["hidden_layers"],
        "samples": len(test_df),
        "mse_mgal2": float(mse),
        "pred_min": float(pred_min),
        "pred_max": float(pred_max),
        "pred_mean": float(pred_mean),
        "pred_std": float(pred_std),
        "true_min": float(true_min),
        "true_max": float(true_max),
        "true_mean": float(true_mean),
        "true_std": float(true_std)
    }

    with open(output_file.replace(".npy", "_test_report.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"🧩 Metadata saved to {output_file.replace('.npy', '_test_report.json')}")

    return model, test_df, preds

if __name__ == '__main__':
    mp.freeze_support()
    main()