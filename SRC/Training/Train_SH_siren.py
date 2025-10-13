# SRC/Training/train_sh_siren.py
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from SRC.Models.SH_siren import SH_SIREN

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_path = os.path.join(base_dir, 'Data', 'Samples_2190_5M.parquet')


df = pd.read_parquet(data_path)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

lon = torch.tensor(train_df['lon'].values, dtype=torch.float32)
lat = torch.tensor(train_df['lat'].values, dtype=torch.float32)

y = torch.tensor(train_df['dV_m2_s2'].values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(lon, lat, y)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

model = SH_SIREN(lmax=10, hidden_features=128, hidden_layers=4, out_features=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 50

val_lon = torch.tensor(val_df['lon'].values, dtype=torch.float32)
val_lat = torch.tensor(val_df['lat'].values, dtype=torch.float32)
val_y = torch.tensor(val_df['dV_m2_s2'].values, dtype=torch.float32).unsqueeze(1)

val_dataset = TensorDataset(val_lon, val_lat, val_y)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    # ---- Training ----
    for lon_batch, lat_batch, y_batch in dataloader:
        lon_batch = lon_batch.to(device)
        lat_batch = lat_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(lon_batch, lat_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(dataloader)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lon_batch, lat_batch, y_batch in val_loader:
            lon_batch, lat_batch, y_batch = lon_batch.to(device), lat_batch.to(device), y_batch.to(device)
            y_pred = model(lon_batch, lat_batch)
            val_loss += loss_fn(y_pred, y_batch).item()

    avg_val_loss = val_loss / len(val_loader)

    # ---- Log losses ----
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")


save_path = os.path.join(base_dir, 'Models', 'sh_siren_trained.pth')
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to: {save_path}")
