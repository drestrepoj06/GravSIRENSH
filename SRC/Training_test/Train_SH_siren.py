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
from SRC.Location_encoder.SH_siren import SH_SIREN, SHSirenScaler, SH_LINEAR
# from SRC.Models.SH_embedding import SHEmbedding


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_dir, 'Data', 'Samples_2190-2_5.0M_r0_train.parquet')

    print("Loading dataset...")
    df = pd.read_parquet(data_path)
    df["orig_index"] = np.arange(len(df))

    # Keep 500k samples for validation
    val_df = df.sample(n=500000, random_state=42)
    train_df = df.drop(val_df.index)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    # --- Scaler for target variable ---
    scaler = SHSirenScaler()
    scaler.fit_potential(train_df["dV_m2_s2"].values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # --- Data to tensors ---
    def df_to_tensors(df, include_radial=True):
        lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        lat = torch.tensor(df["lat"].values, dtype=torch.float32)

        if include_radial:
            y = torch.tensor(
                df[["dg_r_mGal", "dg_theta_mGal", "dg_phi_mGal"]].values,
                dtype=torch.float32
            )
        else:
            y = torch.tensor(
                df[["dg_theta_mGal", "dg_phi_mGal"]].values,
                dtype=torch.float32
            )

        return lon, lat, y

    lon_train, lat_train, y_train = df_to_tensors(train_df, include_radial=False)
    lon_val, lat_val, y_val = df_to_tensors(val_df, include_radial=False)

    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        TensorDataset(lon_train, lat_train, y_train),
        batch_size=10240,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=2,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        TensorDataset(lon_val, lat_val, y_val),
        batch_size=10240,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=2,
        prefetch_factor=2
    )

    lmax = 10
    hidden_features = 8
    hidden_layers = 2
    out_features = 1
    first_omega_0 = 20
    hidden_omega_0 = 1.0
    cache_path = os.path.join(base_dir, "Data", "cache_train.npy")

    model_type = "siren"

    if model_type.lower() == "siren":
        model = SH_SIREN(
            lmax=lmax,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            device=device,
            scaler=scaler,
            cache_path=cache_path
        )
        print("🌀 Using SH-SIREN model")

    elif model_type.lower() == "linear":
        model = SH_LINEAR(
            lmax=lmax,
            out_features=out_features,
            device=device,
            scaler=scaler,
            cache_path=cache_path
        )
        print("Using SH-LINEAR model (classical expansion)")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-6
    )

    epochs = 50
    train_losses, val_losses = [], []

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 10
    min_delta = 1e-5
    outputs_dir = os.path.join(base_dir, "Outputs", "Models")
    os.makedirs(outputs_dir, exist_ok=True)
    best_model_path = os.path.join(outputs_dir, f"{model_type}_best.pth")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for lon_b, lat_b, a_true_b in train_loader:
            lon_b = lon_b.to(device).requires_grad_(True)
            lat_b = lat_b.to(device).requires_grad_(True)
            a_true_b = a_true_b.to(device)

            optimizer.zero_grad(set_to_none=True)

            U_pred, grads_phys= model(
                lon_b, lat_b,
                return_gradients=True,
                create_graph = True
            )

            g_lon, g_lat, g_r = grads_phys
            a_pred_b = torch.stack([g_lat * 1e5, g_lon * 1e5], dim=1)

            loss = criterion(a_pred_b, a_true_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.set_grad_enabled(True):
            for lon_b, lat_b, a_true_b in val_loader:
                lon_b = lon_b.to(device).requires_grad_(True)
                lat_b = lat_b.to(device).requires_grad_(True)
                a_true_b = a_true_b.to(device)

                U_pred, grads_phys = model(lon_b, lat_b, return_gradients=True)

                g_lon, g_lat, g_r = grads_phys

                a_pred_b = torch.stack([g_lat * 1e5, g_lon * 1e5], dim=1)

                val_loss += criterion(a_pred_b, a_true_b).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        epoch_time = time.time() - epoch_start

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} "
              f"| LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation improved → model saved (ValLoss={avg_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} — validation loss plateaued.")
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if isinstance(model, SH_SIREN):
        model_label = "sh_siren"
    elif isinstance(model, SH_LINEAR):
        model_label = "sh_linear"
    else:
        model_label = "unknown"

    model_name = f"{model_label}_pyshtools_lmax{model.lmax}_{timestamp}"
    final_model_path = os.path.join(outputs_dir, f"{model_name}.pth")

    checkpoint_state_dict = torch.load(os.path.join(outputs_dir, f"{model_type}_best.pth"))
    model.load_state_dict(checkpoint_state_dict)

    torch.save({
        "state_dict": model.state_dict(),
        "scaler": {
            "U_min": float(scaler.U_min),
            "U_max": float(scaler.U_max),
        }
    }, final_model_path)

    # Save losses
    np.save(os.path.join(outputs_dir, f"{model_name}_train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(outputs_dir, f"{model_name}_val_losses.npy"), np.array(val_losses))

    # Build configuration dictionary
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "model_type": model_label,
        "lmax": model.lmax,
        "out_features": int(getattr(model, "out_features", 1)),
        "scaler": {
            "U_min": float(scaler.U_min),
            "U_max": float(scaler.U_max),
        },
        "training": {
            "epochs": epochs,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "best_val_loss": float(best_val_loss),
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
        },
        "paths": {
            "model_file": final_model_path,
            "train_losses": os.path.join(outputs_dir, f"{model_name}_train_losses.npy"),
            "val_losses": os.path.join(outputs_dir, f"{model_name}_val_losses.npy"),
        },
    }

    # Add SIREN-specific parameters if applicable
    if isinstance(model, SH_SIREN):
        config.update({
            "hidden_features": hidden_features,
            "hidden_layers": hidden_layers,
            "first_omega_0": first_omega_0,
            "hidden_omega_0": hidden_omega_0,
        })

    # Save configuration JSON
    config_path = os.path.join(outputs_dir, f"{model_name}_model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nBest model and configuration saved to:\n → {final_model_path}\n → {config_path}")
    temp_checkpoint = os.path.join(outputs_dir, f"{model_type}_best.pth")
    if os.path.exists(temp_checkpoint):
        os.remove(temp_checkpoint)
        print("Temporary checkpoint removed.")

if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
