"""Training of the SIRENSH network
jhonr"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import time
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Models.SH_siren import SH_SIREN, SHSirenScaler


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_dir, 'Data', 'Samples_2190_5M_r0_complete.parquet')

    df = pd.read_parquet(data_path)
    df = df.sample(n=500000, random_state=42).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

    scaler = SHSirenScaler(r_scale=6378136.3)
    scaler.fit_potential(train_df["V_full_m2_s2"].values)

    for subdf in [train_df, val_df, test_df]:
        subdf["V_full_scaled"] = scaler.scale_potential(subdf["V_full_m2_s2"].values)
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
        r   = torch.tensor(df["radius_m"].values, dtype=torch.float32)
        y   = torch.tensor(df["V_full_scaled"].values, dtype=torch.float32).unsqueeze(1)
        return lon, lat, r, y

    lon_train, lat_train, r_train, y_train = df_to_tensors(train_df)
    lon_val, lat_val, r_val, y_val         = df_to_tensors(val_df)

    train_dataset = TensorDataset(lon_train, lat_train, r_train, y_train)
    val_dataset   = TensorDataset(lon_val,   lat_val,   r_val,   y_val)

    # num_workers = max(1, os.cpu_count() // 2) uncomment in GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        #persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        #persistent_workers=True
    )

    model = SH_SIREN(
        lmax=10,
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        first_omega_0=20,
        hidden_omega_0=1.0,
        device=device,
        scaler=scaler,
        normalize_input=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 1
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for lon_b, lat_b, r_b, y_b in train_loader:
            lon_b = lon_b.to(device)
            lat_b = lat_b.to(device)
            r_b   = r_b.to(device)
            y_b   = y_b.to(device)

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
                r_b   = r_b.to(device)
                y_b   = y_b.to(device)
                y_pred = model(lon_b, lat_b, r_b)
                val_loss += criterion(y_pred, y_b).item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch+1}/{epochs}] - Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Time: {epoch_time:.1f}s")

    save_dir = os.path.join(base_dir, 'SRC', 'Models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'sh_siren_trained.pth')

    torch.save({
        "state_dict": model.state_dict(),
        "scaler": {
            "r_scale": scaler.r_scale,
            "U_min": scaler.U_min,
            "U_max": scaler.U_max,
            "a_scale": scaler.a_scale,
        },
    }, save_path)

    print(f"✅ Model and scaler saved to: {save_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
