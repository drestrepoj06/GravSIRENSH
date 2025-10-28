"""Training_test of the SIRENSH network
jhonr"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_siren import SH_SIREN, SHSirenScaler


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_dir, 'Data', 'Samples_2190-2_5.0M_r0_train.parquet')

    df = pd.read_parquet(data_path)
    # df = df.sample(n=500000, random_state=42).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    scaler = SHSirenScaler(r_scale=6378136.3)
    scaler.fit_potential(train_df["dV_m2_s2"].values)

    for subdf in [train_df, val_df]:
        subdf["dV_m2_s2_scaled"] = scaler.scale_potential(subdf["dV_m2_s2"].values)
        _, _, r_scaled = scaler.scale_inputs(
            torch.tensor(subdf["lon"].values),
            torch.tensor(subdf["lat"].values),
            torch.tensor(subdf["radius_m"].values)
        )
        subdf["radius_bar"] = r_scaled.numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def df_to_tensors(df):
        lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        lat = torch.tensor(df["lat"].values, dtype=torch.float32)
        r = torch.tensor(df["radius_m"].values, dtype=torch.float32)
        y = torch.tensor(df["dV_m2_s2_scaled"].values, dtype=torch.float32).unsqueeze(1)
        return lon, lat, r, y

    lon_train, lat_train, r_train, y_train = df_to_tensors(train_df)
    lon_val, lat_val, r_val, y_val = df_to_tensors(val_df)

    train_dataset = TensorDataset(lon_train, lat_train, r_train, y_train)
    val_dataset = TensorDataset(lon_val, lat_val, r_val, y_val)

    # num_workers = max(1, os.cpu_count() // 2) uncomment in GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=10240,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        # persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=10240,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        # persistent_workers=True
    )

    lmax = 10
    hidden_features = 8
    hidden_layers = 2
    out_features = 1
    first_omega_0 = 20
    hidden_omega_0 = 1.0

    model = SH_SIREN(
        lmax=lmax,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
        device=device,
        scaler=scaler
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 1
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for lon_b, lat_b, r_b, y_b in train_loader:
            lon_b = lon_b.to(device)
            lat_b = lat_b.to(device)
            r_b = r_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            y_pred = model(lon_b, lat_b, r_b)
            loss = criterion(y_pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lon_b, lat_b, r_b, y_b in val_loader:
                lon_b = lon_b.to(device)
                lat_b = lat_b.to(device)
                r_b = r_b.to(device)
                y_b = y_b.to(device)
                y_pred = model(lon_b, lat_b, r_b)
                val_loss += criterion(y_pred, y_b).item()

        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{epochs}] - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Time: {epoch_time:.1f}s")

    outputs_dir = os.path.join(base_dir, "Outputs")
    save_dir = os.path.join(outputs_dir, "Models")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}.pth")
    np.save(os.path.join(save_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(save_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_val_losses.npy"), np.array(val_losses))

    torch.save({
        "state_dict": model.state_dict(),
        "scaler": {
            "r_scale": scaler.r_scale,
            "U_min": scaler.U_min,
            "U_max": scaler.U_max
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
            "U_min": float(scaler.U_min),
            "U_max": float(scaler.U_max)
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
            "train_losses": os.path.join(save_dir, "train_losses.npy"),
            "val_losses": os.path.join(save_dir, "val_losses.npy"),
        }
    }

    config_path = os.path.join(save_dir, f"sh_siren_torch_lmax{lmax}_{timestamp}_model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    print(f"✅ Model configuration saved to: {config_path}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()