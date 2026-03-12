"""Training of hybrid or numerical models for gravity field modeling
jhonr"""
import os
import sys
import torch
import pandas as pd
from datetime import datetime
import numpy as np
import json
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Dict, Any
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from numerical.numerical_wrapper import Scaler as numerical_scaler
from numerical.numerical_wrapper import Numerical
from hybrid.SIREN_SH import Hybrid
from hybrid.SIREN_SH import Scaler as hybrid_scaler
from analytical.pyshtools_expansion import Analytical
from train_test.data_modules import SplitDataset
from visualizations.geographic_plots import Plotter
from train_test.runtime import run_runtime_benchmark

# Bsed on the implementation exposed in: https://github.com/MarcCoru/locationencoder/blob/685892ee1f9945d368b2dbfbb153dc1f9813f8bd/train.py
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_data_path = os.path.join(base_dir, 'data', 'Samples_2190-2_5.0M_r0_train.parquet')
    test_data_path = os.path.join(base_dir, "data", "Samples_2190-2_250k_r0_test.parquet")

    #if torch.cuda.is_available():
        #gpu_id = torch.cuda.current_device()
        #gpu_name = torch.cuda.get_device_name(gpu_id)


    df = pd.read_parquet(train_data_path)
    val_df = df.sample(n=500_000, random_state=42)
    train_df = df.drop(val_df.index)
    test_df = pd.read_parquet(test_data_path)

    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")
    hparams_file = os.path.join(base_dir, "hparams.yaml")
    with open(hparams_file, "r") as f:
        hp = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = hp["mode"]
    arch = hp["arch"]

    lr = float(hp["lr"])
    batch_size = int(hp["batch_size"])
    epochs = int(hp["epochs"])

    hidden_layers = int(hp["hidden_layers"])
    hidden_features = int(hp["hidden_features"])

    exclude_degrees = hp.get("exclude_degrees", None)

    if arch == "numerical":
        scaler = numerical_scaler().fit(train_df)
        run_name = (
            f"{arch}_LR={lr}_mode={mode}_BS={batch_size}_"
            f"layers={hidden_layers}_neurons={hidden_features}"
        )
        run_dir = os.path.join(base_dir, "outputs", "runs", run_name)
        os.makedirs(run_dir, exist_ok=True)
        module = Numerical(hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            scaler=scaler,
            mode=mode, lr=lr, run_dir = run_dir)
    else:
        lmax = int(hp["lmax"])
        first_omega_0 = float(hp["first_omega_0"])
        hidden_omega_0 = float(hp["hidden_omega_0"])
        scaler = hybrid_scaler(mode=mode).fit(train_df)
        run_name = (
            f"{arch}_LR={lr}_mode={mode}_BS={batch_size}_"
            f"lmax={lmax}_layers={hidden_layers}_neurons={hidden_features}_"
            f"first_omega={first_omega_0}_hidden_omega={hidden_omega_0}_"
            f"exclude_degrees={exclude_degrees}"
        )
        run_dir = os.path.join(base_dir, "outputs", "runs", run_name)
        os.makedirs(run_dir, exist_ok=True)
        module = Hybrid(lmax=lmax,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            scaler=scaler,
            cache_path=os.path.join(base_dir, "data", "cache_train.npy"),
            exclude_degrees=exclude_degrees,
            mode=mode,lr=lr, run_dir = run_dir)
    datamodule = SplitDataset(train_df, val_df, test_df, scaler=scaler, mode=mode, batch_size=batch_size)
    pl.seed_everything(42, workers=True)
    # wandb_logger = WandbLogger(
    #     project="siren",
    #     name=run_name,
    #     log_model=False
    # )
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
        # logger=wandb_logger, To activate the wandb logger flawlessly, change its waiting time
        inference_mode=False
    )

    trainer.fit(module, datamodule=datamodule)
    actual_epochs = trainer.current_epoch

    best_ckpt_path = checkpoint.best_model_path

    best_module = type(module).load_from_checkpoint(
        best_ckpt_path,
        scaler=scaler,
        mode=mode,
        lr=lr,
        run_dir=run_dir,
        **(
            {
                "hidden_features": hidden_features,
                "hidden_layers": hidden_layers,
            }
            if arch == "numerical"
            else {
                "lmax": int(hp["lmax"]),
                "hidden_features": hidden_features,
                "hidden_layers": hidden_layers,
                "first_omega_0": float(hp["first_omega_0"]),
                "hidden_omega_0": float(hp["hidden_omega_0"]),
                "cache_path": os.path.join(base_dir, "data", "cache_train.npy"),
                "exclude_degrees": exclude_degrees,
            }
        )
    )

    runtime_model_path = os.path.join(run_dir, "model_runtime.pth") # Saves the model for runtime experiment

    runtime_ckpt = {
        "state_dict": best_module.state_dict(),
        "arch": arch,
        "mode": mode,
        "model_hparams": {
            "hidden_features": hidden_features,
            "hidden_layers": hidden_layers,
        },
    }

    if arch == "numerical":
        runtime_ckpt["coord_scaler"] = {
            "lon_min": float(scaler.lon_min),
            "lon_max": float(scaler.lon_max),
            "lat_min": float(scaler.lat_min),
            "lat_max": float(scaler.lat_max),
        }

    if arch != "numerical":
        runtime_ckpt["model_hparams"].update({
            "lmax": int(hp["lmax"]),
            "first_omega_0": float(hp["first_omega_0"]),
            "hidden_omega_0": float(hp["hidden_omega_0"]),
            "exclude_degrees": exclude_degrees,
        })

    torch.save(runtime_ckpt, runtime_model_path)
    n_params = sum(
        p.numel() for p in trainer.lightning_module.parameters() if p.requires_grad
    )

    config: Dict[str, Any] = {
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
        "num_parameters": int(n_params),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }

    if arch == "hybrid":
        config["lmax"] = int(hp["lmax"])
        config["first_omega_0"] = float(hp["first_omega_0"])
        config["hidden_omega_0"] = float(hp["hidden_omega_0"])

    test_results = trainer.test(ckpt_path="best", datamodule=datamodule)

    pred_path = os.path.join(run_dir, "predictions.npz")
    data = np.load(pred_path)

    lon = data["lon"].astype(np.float64)
    lat = data["lat"].astype(np.float64)
    mode = str(data["mode"])

    # Generate the equivalent analytical model
    ana = Analytical(run_dir=run_dir)

    equiv = ana.equiv
    du_lin, ath_lin, aph_lin, ar_lin = ana.evaluate_on_coords(
        lat=lat,
        lon=lon,
        du_grid=equiv["du_grid"],
        lats_grid=equiv["lats"],
        lons_grid=equiv["lons"],
        clm_full=equiv["clm_full"],
        clm_low=equiv["clm_low"],
        r0=equiv["r0"],
        l=equiv["l"]
    )

    if mode == "u":
        y_pred_linear = du_lin.astype(np.float32)
    else:
        y_pred_linear = np.stack([ath_lin, aph_lin, ar_lin], axis=1).astype(np.float32)

    out = {k: data[k] for k in data.files}
    out["y_pred_linear"] = y_pred_linear

    np.savez_compressed(pred_path, **out)

    runtime_info = run_runtime_benchmark(run_dir)

    def stats(a):
        a = np.asarray(a)
        return {
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std()),
        }

    def rmse_u(pred, true):
        pred = np.asarray(pred).reshape(-1)
        true = np.asarray(true).reshape(-1)
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    def rmse_a_l2(pred, true):
        pred = np.asarray(pred)
        true = np.asarray(true)
        mse = np.mean((pred - true) ** 2, axis=0)
        return float(np.sqrt(mse.sum()) / 1e5)

    if os.path.exists(pred_path):
        data = np.load(pred_path)

        y_true = data["y_true"]
        y_pred = data["y_pred"]
        is_dist = data["is_dist"].astype(bool)

        n_test = int(len(y_true))
        n_dist = int(is_dist.sum())

        config["test_eval"] = {
            "samples": {
                "all": n_test,
                "dist": n_dist,
            },
            "model_metrics": test_results[0] if len(test_results) else {},
            "pred_stats": {"all": {}, "dist": {}},
            "true_stats": {"all": {}, "dist": {}},

            "analytical_metrics": {"all": None, "dist": None},
            "analytical_stats": {"all": {}, "dist": {}},
        }

        if y_pred.ndim == 2:
            pred_mag = np.sqrt((y_pred ** 2).sum(axis=1))
            true_mag = np.sqrt((y_true ** 2).sum(axis=1))

            config["test_eval"]["pred_stats"]["all"]["a_theta"] = stats(y_pred[:, 0])
            config["test_eval"]["pred_stats"]["all"]["a_phi"] = stats(y_pred[:, 1])
            config["test_eval"]["pred_stats"]["all"]["a_rad"] = stats(y_pred[:, 2])
            config["test_eval"]["pred_stats"]["all"]["a_mag"] = stats(pred_mag)

            config["test_eval"]["true_stats"]["all"]["a_theta"] = stats(y_true[:, 0])
            config["test_eval"]["true_stats"]["all"]["a_phi"] = stats(y_true[:, 1])
            config["test_eval"]["true_stats"]["all"]["a_rad"] = stats(y_true[:, 2])
            config["test_eval"]["true_stats"]["all"]["a_mag"] = stats(true_mag)

            if is_dist.any():
                config["test_eval"]["pred_stats"]["dist"]["a_theta"] = stats(y_pred[is_dist, 0])
                config["test_eval"]["pred_stats"]["dist"]["a_phi"] = stats(y_pred[is_dist, 1])
                config["test_eval"]["pred_stats"]["dist"]["a_rad"] = stats(y_pred[is_dist, 2])
                config["test_eval"]["pred_stats"]["dist"]["a_mag"] = stats(pred_mag[is_dist])

                config["test_eval"]["true_stats"]["dist"]["a_theta"] = stats(y_true[is_dist, 0])
                config["test_eval"]["true_stats"]["dist"]["a_phi"] = stats(y_true[is_dist, 1])
                config["test_eval"]["true_stats"]["dist"]["a_rad"] = stats(y_true[is_dist, 2])
                config["test_eval"]["true_stats"]["dist"]["a_mag"] = stats(true_mag[is_dist])

        else:
            yt = y_true.reshape(-1)
            yp = y_pred.reshape(-1)

            config["test_eval"]["pred_stats"]["all"]["u"] = stats(yp)
            config["test_eval"]["true_stats"]["all"]["u"] = stats(yt)

            if is_dist.any():
                config["test_eval"]["pred_stats"]["dist"]["u"] = stats(yp[is_dist])
                config["test_eval"]["true_stats"]["dist"]["u"] = stats(yt[is_dist])

        if "y_pred_linear" in data.files:
            y_lin = data["y_pred_linear"]

            if y_lin.ndim == 2:
                lin_mag = np.sqrt((y_lin ** 2).sum(axis=1))

                config["test_eval"]["analytical_metrics"]["all"] = {
                    "rmse": rmse_a_l2(y_lin, y_true)
                }
                if is_dist.any():
                    config["test_eval"]["analytical_metrics"]["dist"] = {
                        "rmse": rmse_a_l2(y_lin[is_dist], y_true[is_dist])
                    }

                # predicted stats
                config["test_eval"]["analytical_stats"]["all"]["a_theta"] = stats(y_lin[:, 0])
                config["test_eval"]["analytical_stats"]["all"]["a_phi"] = stats(y_lin[:, 1])
                config["test_eval"]["analytical_stats"]["all"]["a_rad"] = stats(y_lin[:, 2])
                config["test_eval"]["analytical_stats"]["all"]["a_mag"] = stats(lin_mag)

                if is_dist.any():
                    config["test_eval"]["analytical_stats"]["dist"]["a_theta"] = stats(y_lin[is_dist, 0])
                    config["test_eval"]["analytical_stats"]["dist"]["a_phi"] = stats(y_lin[is_dist, 1])
                    config["test_eval"]["analytical_stats"]["dist"]["a_rad"] = stats(y_lin[is_dist, 2])
                    config["test_eval"]["analytical_stats"]["dist"]["a_mag"] = stats(lin_mag[is_dist])

            else:
                lin_u = y_lin.reshape(-1)
                true_u = y_true.reshape(-1)

                config["test_eval"]["analytical_metrics"]["all"] = {
                    "rmse": rmse_u(lin_u, true_u)
                }
                if is_dist.any():
                    config["test_eval"]["analytical_metrics"]["dist"] = {
                        "rmse": rmse_u(lin_u[is_dist], true_u[is_dist])
                    }

                config["test_eval"]["analytical_stats"]["all"]["u"] = stats(lin_u)
                if is_dist.any():
                    config["test_eval"]["analytical_stats"]["dist"]["u"] = stats(lin_u[is_dist])
        config["runtime_benchmark"] = runtime_info
    config_path = os.path.join(run_dir, "results.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    plots_by_mode = {
        "u": ["potential"],
        "a": ["acceleration"]
    }
    targets_to_plot = plots_by_mode.get(mode, None)
    if targets_to_plot is None:
        print(f"Mode '{mode}' not recognized for plotting.")
    else:
        for target in targets_to_plot:
            print(f"Plotting {target} maps...")

            plotter = Plotter(
                predictions_path=pred_path,
                output_dir=run_dir,
                target_type=target
            )
            plotter.plot_scatter()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()