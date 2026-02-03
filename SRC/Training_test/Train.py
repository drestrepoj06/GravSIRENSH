"""Training of a neural network for the estimation of Earth's
gravity field
jhonr"""
import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import json
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from SRC.Location_encoder.Coordinates_network import MANDS2022Scaler, Scaler, Gravity
import SRC.Training_test.Test as test_script
from SRC.Visualizations.Geographic_plots import GravityDataPlotter


class GravityDataset(torch.utils.data.Dataset):
    def __init__(self, df, scaler, mode="g_direct"):
        self.mode = mode
        self.scaler = scaler

        self.lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        self.lat = torch.tensor(df["lat"].values, dtype=torch.float32)

        def _scale_g(g_phys):
            if hasattr(self.scaler, "scale_gravity"):
                return self.scaler.scale_gravity(g_phys)
            if hasattr(self.scaler, "scale_accel_uniform"):
                return self.scaler.scale_accel_uniform(g_phys)
            raise AttributeError(
                "Scaler must implement either scale_gravity(...) or scale_accel_uniform(...)."
            )

        if mode == "U":
            U_phys = df["dU_m2_s2"].values
            U_scaled = self.scaler.scale_potential(U_phys)
            self.y = torch.tensor(U_scaled, dtype=torch.float32).unsqueeze(1)

        elif mode in ["g_direct"]:
            cols = ["dg_theta_mGal", "dg_phi_mGal", "dg_r_mGal"]
            g_phys = df[cols].values
            g_scaled = _scale_g(g_phys)
            self.y = torch.tensor(g_scaled, dtype=torch.float32)
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

    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)


    df = pd.read_parquet(data_path)
    val_df = df.sample(n=500_000, random_state=42)
    train_df = df.drop(val_df.index)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "U"
    lr = 5e-3
    batch_size = 262144
    lmax = 3
    hidden_layers = 2
    hidden_features = 40
    first_omega_0 = 20
    hidden_omega_0 = 1.0
    exclude_degrees = None
    epochs = 1
    arch = "sirensh"  # "sirensh, linearsh or mands2022"

    if arch == "mands2022":
        scaler = MANDS2022Scaler().fit(train_df)
    else:
        scaler = Scaler(mode=mode).fit(train_df)

    if arch == "mands2022":
        run_name = (
            f"{arch}_LR={lr}_mode={mode}_BS={batch_size}_"
            f"layers={hidden_layers}_neurons={hidden_features}"
        )

    elif arch == "linearsh":
        run_name = (
            f"{arch}_LR={lr}_mode={mode}_BS={batch_size}_"
            f"lmax={lmax}_layers={hidden_layers}_neurons={hidden_features}_"
            f"exclude_degrees={exclude_degrees}"
        )

    elif arch == "sirensh":
        run_name = (
            f"{arch}_LR={lr}_mode={mode}_BS={batch_size}_"
            f"lmax={lmax}_layers={hidden_layers}_neurons={hidden_features}_"
            f"first_omega={first_omega_0}_hidden_omega={hidden_omega_0}_"
            f"exclude_degrees={exclude_degrees}"
        )

    else:
        raise ValueError(f"Unknown architecture '{arch}'")

    run_dir = os.path.join(base_dir, "Outputs", "Runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    if arch == "mands2022":
        model_cfg = dict(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            device=device,
            scaler=scaler,
            mode=mode,
            arch=arch,
        )
    elif arch == "sirensh":
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
            mode=mode,
            arch=arch,
        )
    else:
        model_cfg = dict(
            lmax=lmax,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            device=device,
            scaler=scaler,
            cache_path=os.path.join(base_dir, "Data", "cache_train.npy"),
            exclude_degrees=exclude_degrees,
            mode=mode,
            arch=arch,
        )

    datamodule = GravityDataModule(train_df, val_df, scaler=scaler, mode=mode, batch_size=batch_size)
    module = Gravity(model_cfg, scaler, lr=lr)
    pl.seed_everything(42, workers=True)

    wandb_logger = WandbLogger(
        project="siren",
        name=run_name,
        log_model=False
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-{epoch:04d}-{val_loss:.4f}"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint],
        devices=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=wandb_logger,
        inference_mode=False
    )

    trainer.fit(module, datamodule=datamodule)
    actual_epochs = trainer.current_epoch
    best_path = checkpoint.best_model_path
    ckpt = torch.load(best_path, map_location="cpu")

    inner_state_dict = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }


    model_path = os.path.join(run_dir, "model.pth")

    scaler_payload = {}

    if hasattr(scaler, "U_mean") or hasattr(scaler, "U_std") or hasattr(scaler, "g_mean") or hasattr(scaler, "g_std"):
        scaler_payload.update({
            "U_mean": float(scaler.U_mean) if getattr(scaler, "U_mean", None) is not None else None,
            "U_std": float(scaler.U_std) if getattr(scaler, "U_std", None) is not None else None,
            "g_mean": getattr(scaler, "g_mean", None).tolist() if getattr(scaler, "g_mean", None) is not None else None,
            "g_std": getattr(scaler, "g_std", None).tolist() if getattr(scaler, "g_std", None) is not None else None,
        })

    if hasattr(scaler, "a_scale"):
        a_scale = getattr(scaler, "a_scale", None)
        scaler_payload["a_scale"] = float(a_scale) if a_scale is not None else None

    if hasattr(scaler, "U_scale"):
        U_scale = getattr(scaler, "U_scale", None)
        scaler_payload["U_scale"] = float(U_scale) if U_scale is not None else None

    if hasattr(scaler, "base"):
        base = scaler.base
        scaler_payload["base"] = {
            "U_mean": float(getattr(base, "U_mean", None)) if getattr(base, "U_mean", None) is not None else None,
            "U_std": float(getattr(base, "U_std", None)) if getattr(base, "U_std", None) is not None else None,
            "g_mean": getattr(base, "g_mean", None).tolist() if getattr(base, "g_mean", None) is not None else None,
            "g_std": getattr(base, "g_std", None).tolist() if getattr(base, "g_std", None) is not None else None,
        }

    for k in ["lon_min", "lon_max", "lat_min", "lat_max"]:
        if hasattr(scaler, k):
            v = getattr(scaler, k, None)
            scaler_payload[k] = float(v) if v is not None else None

    if hasattr(scaler, "U_min") and hasattr(scaler, "U_max"):
        scaler_payload["U_min"] = (
            float(scaler.U_min) if scaler.U_min is not None else None
        )
        scaler_payload["U_max"] = (
            float(scaler.U_max) if scaler.U_max is not None else None
        )

    # Acceleration (vector, 3,)
    if hasattr(scaler, "a_min") and hasattr(scaler, "a_max"):
        scaler_payload["a_min"] = (
            scaler.a_min.tolist() if scaler.a_min is not None else None
        )
        scaler_payload["a_max"] = (
            scaler.a_max.tolist() if scaler.a_max is not None else None
        )

    torch.save({
        "state_dict": inner_state_dict,
        "scaler": scaler_payload,
        "architecture": arch,
    }, model_path)

    config = {
        # common
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        #"gpu_name": gpu_name if gpu_name is not None else None,
        "architecture": arch,
        "mode": mode,
        "lr": lr,
        "epochs_requested": epochs,
        "epochs_trained": actual_epochs,
        "batch_size": batch_size,
        "hidden_layers": hidden_layers,
        "hidden_features": hidden_features,
        "exclude_degrees": exclude_degrees,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }

    if arch in ["sirensh", "linearsh"]:
        config["lmax"] = lmax

    if arch == "sirensh":
        config["first_omega_0"] = first_omega_0
        config["hidden_omega_0"] = hidden_omega_0

    if arch == "mands2022":
        if hasattr(scaler, "r_scale"):
            config["r_scale"] = float(getattr(scaler, "r_scale"))
        if hasattr(scaler, "lon_min"):
            config["lon_min"] = float(getattr(scaler, "lon_min"))
            config["lon_max"] = float(getattr(scaler, "lon_max"))
            config["lat_min"] = float(getattr(scaler, "lat_max"))
        if hasattr(scaler, "a_scale"):
            a_scale = getattr(scaler, "a_scale", None)
            config["a_scale"] = float(a_scale) if a_scale is not None and not hasattr(a_scale, "tolist") else (
                a_scale.tolist() if a_scale is not None else None
            )

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nModel saved at: {model_path}")
    print(f"Config saved at: {config_path}")

    data_path = os.path.join(base_dir, "Data", "Samples_2190-2_250k_r0_test.parquet")
    test_script.main(run_path=run_dir)

    PLOTS_BY_MODE = {
        "U": ["potential"],
        "g_direct": ["acceleration"]
    }

    targets_to_plot = PLOTS_BY_MODE.get(mode, None)

    if targets_to_plot is None:
        print(f"Mode '{mode}' not recognized for plotting.")
    else:
        for target in targets_to_plot:
            print(f"Plotting {target} maps...")

            plotter = GravityDataPlotter(
                data_path=data_path,
                output_dir=run_dir,
                predictions_dir=run_dir,
                linear_dir=run_dir,
                target_type=target
            )

            plotter.plot_map()
            plotter.plot_scatter()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()