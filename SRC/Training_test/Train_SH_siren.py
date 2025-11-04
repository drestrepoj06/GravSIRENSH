import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import json
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_siren import SHSirenScaler, Gravity

class GravityDataset(torch.utils.data.Dataset):
    def __init__(self, df, scaler, mode="g_direct", include_radial=False):
        """
        mode:
          - "U"              → learn potential
          - "g_direct"       → learn acceleration directly (scaled)
          - "g_indirect"     → will be derived from U (so target = U)
          - "U_g_direct"     → multitask (U + g_direct)
          - "U_g_indirect"   → multitask (U + gradients(U))
        """
        self.mode = mode
        self.scaler = scaler

        self.lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        self.lat = torch.tensor(df["lat"].values, dtype=torch.float32)

        if mode in ["g_direct", "U_g_direct"]:
            cols = ["dg_theta_mGal", "dg_phi_mGal"]
            if include_radial:
                cols = ["dg_r_mGal"] + cols
            a_phys = df[cols].values
            a_scaled = self.scaler.scale_gravity(a_phys)
            self.y = torch.tensor(a_scaled, dtype=torch.float32)

        elif mode in ["U", "g_indirect", "U_g_indirect"]:
            U_phys = df["dU_m2_s2"].values
            U_scaled = self.scaler.scale_potential(U_phys)
            self.y = torch.tensor(U_scaled, dtype=torch.float32)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.lon)

    def __getitem__(self, idx):
        return self.lon[idx], self.lat[idx], self.y[idx]


class GravityDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, scaler, mode="g_direct", batch_size=10240):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.scaler = scaler
        self.mode = mode
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = GravityDataset(self.train_df, self.scaler, mode=self.mode)
        self.val_dataset = GravityDataset(self.val_df, self.scaler, mode=self.mode)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_dir, 'Data', 'Samples_2190-2_5.0M_r0_train.parquet')

    print("📂 Loading dataset...")
    df = pd.read_parquet(data_path)

    val_df = df.sample(n=500_000, random_state=42)
    train_df = df.drop(val_df.index)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    scaler = SHSirenScaler(mode="U")
    scaler.fit(train_df)
    model_cfg = dict(
        lmax=10,
        hidden_features=8,
        hidden_layers=2,
        out_features=1,
        first_omega_0=20,
        hidden_omega_0=1.0,
        device=device,
        scaler=scaler,
        cache_path=os.path.join(base_dir, "Data", "cache_train.npy"),
        exclude_degrees=None,
        mode = "U"
    )

    datamodule = GravityDataModule(train_df, val_df, batch_size=10240, scaler = scaler)
    module = Gravity(model_cfg, scaler)
    pl.seed_everything(42, workers=True)
    wandb_logger = WandbLogger(
        project="grav_siren",  # name of your W&B project
        name=f"lmax{model_cfg['lmax']}_hidden{model_cfg['hidden_features']}",  # run name
        save_dir=os.path.join(base_dir, "Outputs", "wandb_logs")  # local backup directory
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,
        logger=wandb_logger
    )

    trainer.fit(module, datamodule=datamodule)

    outputs_dir = os.path.join(base_dir, "Outputs", "Models")
    os.makedirs(outputs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"sh_siren_pyshtools_lmax{module.model.lmax}_{timestamp}"
    final_model_path = os.path.join(outputs_dir, f"{model_name}.pth")

    # --- Save model weights and scaler ---
    torch.save({
        "state_dict": module.model.state_dict(),
        "scaler": {
            "U_min": float(module.scaler.U_min),
            "U_max": float(module.scaler.U_max),
        }
    }, final_model_path)

    # --- Build configuration dictionary ---
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "lmax": module.model.lmax,
        "exclude_degrees": model_cfg.get("exclude_degrees"),
        "out_features": int(model_cfg.get("out_features", 1)),
        "scaler": {
            "U_min": float(module.scaler.U_min),
            "U_max": float(module.scaler.U_max),
        },
        "training": {
            "epochs": trainer.max_epochs,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
        },
        "paths": {
            "model_file": final_model_path,
        },
    }

    # Add model hyperparameters
    config.update({
        "hidden_features": model_cfg["hidden_features"],
        "hidden_layers": model_cfg["hidden_layers"],
        "first_omega_0": model_cfg["first_omega_0"],
        "hidden_omega_0": model_cfg["hidden_omega_0"],
    })

    # --- Save configuration JSON ---
    config_path = os.path.join(outputs_dir, f"{model_name}_model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n✅ Model saved at: {final_model_path}")
    print(f"📝 Config saved at: {config_path}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
