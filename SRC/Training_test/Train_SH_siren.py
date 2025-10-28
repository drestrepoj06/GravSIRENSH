import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time
import numpy as np
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_siren import SH_SIREN, SHSirenScaler
# from SRC.Models.SH_embedding import SHEmbedding


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_dir, 'Data', 'Samples_2190-2_5.0M_r0_train.parquet')

    print("📂 Loading dataset...")
    df = pd.read_parquet(data_path)
    df["orig_index"] = np.arange(len(df))

    # Keep 500k samples for validation
    val_df = df.sample(n=500000, random_state=42)
    train_df = df.drop(val_df.index)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    # --- Scaler for ΔV (potential) ---
    scaler = SHSirenScaler(r_scale=6378136.3)
    scaler.fit_target(train_df["dV_m2_s2"].values, target_name="dV", target_units="m²/s²")

    for subdf in [train_df, val_df]:
        subdf["dV_scaled"] = scaler.scale_target(subdf["dV_m2_s2"].values)
        _, _, r_scaled = scaler.scale_inputs(
            torch.tensor(subdf["lon"].values),
            torch.tensor(subdf["lat"].values),
            torch.tensor(subdf["radius_m"].values),
        )
        subdf["radius_bar"] = r_scaled.numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # --- Data to tensors ---
    def df_to_tensors(df):
        lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        lat = torch.tensor(df["lat"].values, dtype=torch.float32)
        r = torch.tensor(df["radius_m"].values, dtype=torch.float32)
        idx = torch.tensor(df["orig_index"].values, dtype=torch.long)
        y = torch.tensor(df["dV_scaled"].values, dtype=torch.float32).unsqueeze(1)
        return lon, lat, r, idx, y

    lon_train, lat_train, r_train, idx_train, y_train = df_to_tensors(train_df)
    lon_val, lat_val, r_val, idx_val, y_val = df_to_tensors(val_df)

    train_loader = DataLoader(
        TensorDataset(lon_train, lat_train, r_train, idx_train, y_train),
        batch_size=10240,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(lon_val, lat_val, r_val, idx_val, y_val),
        batch_size=10240,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Model ---
    lmax = 10
    hidden_features = 8
    hidden_layers = 2
    out_features = 1
    first_omega_0 = 20
    hidden_omega_0 = 1.0

    cache_path = os.path.join(base_dir, "Data", "cache_basis_train.npy")
    model = SH_SIREN(
        lmax=lmax,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
        device=device,
        scaler=scaler,
        cache_path=cache_path,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6)

    epochs = 1
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for lon_b, lat_b, r_b, idx_b, y_b in train_loader:
            optimizer.zero_grad(set_to_none=True)

            batch_df = {"lon": lon_b, "lat": lat_b, "radius_m": r_b, "orig_index": idx_b}
            Y = model.prepare_input(batch_df).detach()

            y_true = y_b.to(device)
            y_pred = model.siren(Y)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lon_b, lat_b, r_b, idx_b, y_b in val_loader:
                batch_df = {"lon": lon_b, "lat": lat_b, "radius_m": r_b, "orig_index": idx_b}
                y_true = y_b.to(device)
                y_pred = model(df=batch_df)
                val_loss += criterion(y_pred, y_true).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        epoch_time = time.time() - epoch_start

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

    # --- Save results ---
    outputs_dir = os.path.join(base_dir, "Outputs", "Models")
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = f"sh_siren_pyshtools_lmax{model.lmax}_{timestamp}"
    save_path = os.path.join(outputs_dir, f"{model_name}.pth")

    np.save(os.path.join(outputs_dir, f"{model_name}_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(outputs_dir, f"{model_name}_val_losses.npy"), np.array(val_losses))

    torch.save({
        "state_dict": model.state_dict(),
        "scaler": {
            "r_scale": scaler.r_scale,
            "t_min": scaler.t_min,
            "t_max": scaler.t_max,
            "target_name": scaler.target_name,
            "target_units": scaler.target_units,
        },
    }, save_path)

    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "lmax": lmax,
        "hidden_features": hidden_features,
        "hidden_layers": hidden_layers,
        "out_features": out_features,
        "first_omega_0": first_omega_0,
        "hidden_omega_0": hidden_omega_0,
        "scaler": {
            "r_scale": float(scaler.r_scale),
            "t_min": float(scaler.t_min),
            "t_max": float(scaler.t_max),
            "target_name": scaler.target_name,
            "target_units": scaler.target_units,
        },
        "training": {
            "epochs": epochs,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
        },
        "paths": {
            "model_file": save_path,
            "train_losses": os.path.join(outputs_dir, f"{model_name}_train_losses.npy"),
            "val_losses": os.path.join(outputs_dir, f"{model_name}_val_losses.npy"),
        },
    }

    config_path = os.path.join(outputs_dir, f"{model_name}_model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"✅ Model configuration saved to: {config_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
