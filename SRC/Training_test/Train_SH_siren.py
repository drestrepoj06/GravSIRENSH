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
    data_path = os.path.join(base_dir, 'Data', 'Samples_2190_5M_r0.parquet')

    print("📂 Loading dataset...")
    df = pd.read_parquet(data_path)
    df["orig_index"] = np.arange(len(df))

    # Keep only validation set of 500k samples (train = rest)
    val_df = df.sample(n=500000, random_state=42)
    train_df = df.drop(val_df.index)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    scaler = SHSirenScaler(r_scale=6378136.3)
    scaler.fit_acceleration(train_df["dg_total_mGal"].values.reshape(-1, 1))

    for subdf in [train_df, val_df]:
        _, _, r_scaled = scaler.scale_inputs(
            torch.tensor(subdf["lon"].values),
            torch.tensor(subdf["lat"].values),
            torch.tensor(subdf["radius_m"].values)
        )
        subdf["radius_bar"] = r_scaled.numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    def df_to_tensors(df):
        lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        lat = torch.tensor(df["lat"].values, dtype=torch.float32)
        r = torch.tensor(df["radius_m"].values, dtype=torch.float32)
        idx = torch.tensor(df["orig_index"].values, dtype=torch.long)

        # Use dg_total_mGal as scalar target
        a_np = df["dg_total_mGal"].values.reshape(-1, 1)

        # Scale to [-1, 1]
        a_scaled = scaler.scale_acceleration(a_np)

        y = torch.tensor(a_scaled, dtype=torch.float32)
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
    lmax = 10
    hidden_features = 8
    hidden_layers = 2
    out_features = 1
    first_omega_0 = 20
    hidden_omega_0 = 1.0

    cache_path = os.path.join(base_dir, "Data", f"cache_basis_train.npy")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6)

    epochs = 50

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for lon_b, lat_b, r_b, idx_b, y_b in train_loader:
            # Always clear gradients first
            optimizer.zero_grad(set_to_none=True)

            # Build batch dictionary (stay on CPU is fine if prepare_input expects it)
            batch_df = {
                "lon": lon_b,
                "lat": lat_b,
                "radius_m": r_b,
                "orig_index": idx_b,
            }

            # Embedding does not require gradients
            Y = model.prepare_input(batch_df).detach()

            # Forward + backward + optimize
            y_true = y_b.to(device)
            y_pred = model.siren(Y)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lon_b, lat_b, r_b, idx_b, y_b in val_loader:
                batch_df = {
                    "lon": lon_b,
                    "lat": lat_b,
                    "radius_m": r_b,
                    "orig_index": idx_b,
                }
                y_true = y_b.to(device)
                y_pred = model(df=batch_df)
                val_loss += criterion(y_pred, y_true).item()


        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        epoch_time = time.time() - epoch_start

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

    outputs_dir = os.path.join(base_dir, "Outputs")
    save_dir = os.path.join(outputs_dir, "Models")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"sh_siren_pyshtools_lmax{model.lmax}_{timestamp}.pth")
    np.save(os.path.join(save_dir, f"sh_siren_pyshtools_lmax{model.lmax}_{timestamp}_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(save_dir, f"sh_siren_pyshtools_lmax{model.lmax}_{timestamp}_val_losses.npy"), np.array(val_losses))

    torch.save({
        "state_dict": model.state_dict(),
        "scaler": {
            "r_scale": scaler.r_scale,
            "a_min": scaler.a_min,
            "a_max": scaler.a_max,
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
            "a_min": float(scaler.a_min),
            "a_max": float(scaler.a_max),
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
            "train_losses": os.path.join(save_dir, f"sh_siren_pyshtools_lmax{model.lmax}_{timestamp}_train_losses.npy"),
            "val_losses": os.path.join(save_dir, f"sh_siren_pyshtools_lmax{model.lmax}_{timestamp}_val_losses.npy"),
        }
    }

    config_path = os.path.join(save_dir, f"sh_siren_pyshtools_lmax{model.lmax}_{timestamp}_model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    print(f"✅ Model configuration saved to: {config_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
