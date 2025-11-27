import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import json
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.SH_siren import SHSirenScaler, Gravity
import SRC.Training_test.Test_SH_siren as test_script
from SRC.Visualizations.Geographic_plots import GravityDataPlotter
from SRC.Linear.Linear_equivalent import LinearEquivalentGenerator

class GravityDataset(torch.utils.data.Dataset):
    def __init__(self, df, scaler, mode="g_direct", include_radial=False):
        """
        mode:
          - "U"              : predict potential only
          - "g_direct"       : predict gravity components directly
          - "g_indirect"     : predict potential and derive g = -∇U
          - "U_g_direct"     : predict potential and g directly (multi-output)
          - "U_g_indirect"   : predict potential and derive g = -∇U
          - "g_hybrid"      : predict g directly and force the gradient of U to be equal to gpred
          - "U_g_hybrid"    : predict U and g directly and force the gradient of U to be equal to gpred
        """
        self.mode = mode
        self.scaler = scaler

        self.lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        self.lat = torch.tensor(df["lat"].values, dtype=torch.float32)

        if mode == "U":
            U_phys = df["dU_m2_s2"].values
            U_scaled = self.scaler.scale_potential(U_phys)
            cols = ["dg_theta_mGal", "dg_phi_mGal"]
            g_phys = df[cols].values
            g_scaled = self.scaler.scale_gravity(g_phys)
            self.y = torch.tensor(U_scaled, dtype=torch.float32).unsqueeze(1)

        elif mode in ["g_direct"]:
            cols = ["dg_theta_mGal", "dg_phi_mGal"]
            if include_radial:
                cols = ["dg_r_mGal"] + cols
            g_phys = df[cols].values
            g_scaled = self.scaler.scale_gravity(g_phys)
            self.y = torch.tensor(g_scaled, dtype=torch.float32)

        # --- Indirect gravity (predict U, compare g = ∇U vs target g) ---
        elif mode == "g_indirect":
            cols = ["dg_theta_mGal", "dg_phi_mGal"]
            g_phys = df[cols].values
            g_scaled = self.scaler.scale_gravity(g_phys)
            self.y = torch.tensor(g_scaled, dtype=torch.float32)

        elif mode == "g_hybrid":
            U_phys = df["dU_m2_s2"].values
            U_scaled = self.scaler.scale_potential(U_phys)
            cols = ["dg_theta_mGal", "dg_phi_mGal"]
            if include_radial:
                cols = ["dg_r_mGal"] + cols
            g_phys = df[cols].values
            g_scaled = self.scaler.scale_gravity(g_phys)
            y = np.column_stack([U_scaled, g_scaled])
            self.y = torch.tensor(y, dtype=torch.float32)

        # --- U + g (direct multitask) ---
        elif mode == "U_g_direct":
            U_phys = df["dU_m2_s2"].values
            g_phys = df[["dg_theta_mGal", "dg_phi_mGal"]].values
            U_scaled = self.scaler.scale_potential(U_phys)
            g_scaled = self.scaler.scale_gravity(g_phys)
            y = np.column_stack([U_scaled, g_scaled])
            self.y = torch.tensor(y, dtype=torch.float32)

        # --- U + g (indirect multitask) ---
        # elif mode == "U_g_indirect":
        #     U_phys = df["dU_m2_s2"].values
        #     g_phys = df[["dg_theta_mGal", "dg_phi_mGal"]].values
        #     U_scaled = self.scaler.scale_potential(U_phys)
        #     g_scaled = self.scaler.scale_gravity(g_phys)
        #     y = np.column_stack([U_scaled, g_scaled])
        #     self.y = torch.tensor(y, dtype=torch.float32)

        # elif mode == "U_g_hybrid":
        #     U_phys = df["dU_m2_s2"].values
        #     g_phys = df[["dg_theta_mGal", "dg_phi_mGal"]].values
        #     U_scaled = self.scaler.scale_potential(U_phys)
        #     g_scaled = self.scaler.scale_gravity(g_phys)
        #     y = np.column_stack([U_scaled, g_scaled])
        #     self.y = torch.tensor(y, dtype=torch.float32)
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
                          shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available(), persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available(), persistent_workers=True)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_dir, 'Data', 'Samples_2190-2_5.0M_r0_train.parquet')

    print("📂 Loading dataset...")
    df = pd.read_parquet(data_path)
    val_df = df.sample(n=500_000, random_state=42)
    train_df = df.drop(val_df.index)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "U"
    lr = 5e-3
    batch_size = 262144
    lmax = 0
    hidden_layers = 2
    hidden_features = 4
    first_omega_0 = 20
    hidden_omega_0 = 1.0
    exclude_degrees = None
    epochs = 1
    #warmup_U_epochs = 2000

    run_name = (
        f"sh_siren_LR={lr}_mode={mode}_BS={batch_size}_"
        f"lmax={lmax}_layers={hidden_layers}_neurons={hidden_features}_"
        f"first_omega={first_omega_0}_hidden_omega={hidden_omega_0}_"
        f"exclude_degrees={exclude_degrees}"
    )
    run_dir = os.path.join(base_dir, "Outputs", "Runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"🧩 Run directory: {run_dir}")
    scaler = SHSirenScaler(mode=mode).fit(train_df)
    model_cfg = dict(
        lmax=lmax,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
        device=device,
        scaler=scaler,
        cache_path=os.path.join(base_dir, "Data", "cache_train.npy"),
        exclude_degrees=exclude_degrees,
        mode=mode
        #warmup_U_epochs = warmup_U_epochs
    )

    datamodule = GravityDataModule(train_df, val_df, scaler=scaler, mode=mode, batch_size=batch_size)
    module = Gravity(model_cfg, scaler, lr=lr)
    pl.seed_everything(42, workers=True)

    wandb_logger = WandbLogger(
        project="grav_siren",
        name=run_name,
        log_model=False
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,      # number of epochs with no improvement
        min_delta=1e-4,    # minimum improvement to count
        mode="min",
        verbose=True
    )

    # (optional) best checkpoint saving
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-{epoch:04d}-{val_loss:.4f}"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[early_stop, checkpoint],
        devices=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        logger=wandb_logger
    )

    trainer.fit(module, datamodule=datamodule)
    actual_epochs = trainer.current_epoch
    model_path = os.path.join(run_dir, "model.pth")
    torch.save({
        "state_dict": module.model.state_dict(),
        "scaler": {
            "U_mean": float(module.scaler.U_mean) if module.scaler.U_mean is not None else None,
            "U_std": float(module.scaler.U_std) if module.scaler.U_std is not None else None,
            "g_mean": module.scaler.g_mean.tolist() if module.scaler.g_mean is not None else None,
            "g_std": module.scaler.g_std.tolist() if module.scaler.g_std is not None else None,
        }
    }, model_path)

    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "mode": mode,
        "lr": lr,
        "epochs_requested": epochs,
        "epochs_trained": actual_epochs,
        "batch_size": batch_size,
        "lmax": lmax,
        "hidden_layers": hidden_layers,
        "hidden_features": hidden_features,
        "first_omega_0": first_omega_0,
        "hidden_omega_0": hidden_omega_0,
        "exclude_degrees": exclude_degrees,
        "train_samples": len(train_df),
        "val_samples": len(val_df)
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n✅ Model saved at: {model_path}")
    print(f"📝 Config saved at: {config_path}")

    data_path = os.path.join(base_dir, "Data", "Samples_2190-2_250k_r0_test.parquet")
    if trainer.is_global_zero:
        print("Generating linear equivalent model...")
        gen = LinearEquivalentGenerator(run_dir, data_path)
    output_dir = run_dir
    predictions_dir = run_dir

    test_script.main(run_path=run_dir)

    PLOTS_BY_MODE = {
        "U": ["potential", "acceleration"],
        "U_g_direct": ["potential", "acceleration"],
        "g_indirect": ["potential", "acceleration"],
        "g_direct": ["acceleration"],
        "g_hybrid": ["potential", "acceleration", "gradients"],
        #"U_g_indirect": ["potential", "acceleration"],
        #"U_g_hybrid": ["potential", "acceleration", "gradients"],
    }

    targets_to_plot = PLOTS_BY_MODE.get(mode, None)

    if targets_to_plot is None:
        print(f"⚠️ Mode '{mode}' not recognized for plotting.")
    else:
        for target in targets_to_plot:
            print(f"📡 Plotting {target} maps...")

            plotter = GravityDataPlotter(
                data_path=data_path,
                output_dir=output_dir,
                predictions_dir=predictions_dir,
                linear_dir=predictions_dir,
                target_type=target
            )

            plotter.plot_map()
            plotter.plot_scatter()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()