"""
SH_network: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from hybrid.SH_embedding import SHEmbedding
from hybrid.siren import Siren
import lightning.pytorch as pl
import numpy as np
from pathlib import Path

# Scaler that transforms outputs based on normalization of the mean and std
class Scaler:

    def __init__(self, mode="u"):
        self.mode = mode
        self.u_mean = None
        self.u_std  = None
        self.a_mean = None
        self.a_std  = None

    def fit(self, df):
        if self.mode == "u":
            u = df["dU_m2_s2"].to_numpy()
            self.u_mean = float(u.mean())
            self.u_std  = float(u.std())

        if self.mode == "a":
            a_cols = ["dg_theta_mGal", "dg_phi_mGal", "dg_r_mGal"]
            a_vals = df[a_cols].to_numpy()
            self.a_mean = a_vals.mean(axis=0)
            self.a_std  = a_vals.std(axis=0)

        return self

    def _as_torch(self, x, like: torch.Tensor):
        if torch.is_tensor(x):
            return x.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(x, device=like.device, dtype=like.dtype)

    def scale_potential(self, u):
        if self.u_mean is None:
            raise ValueError("Scaler not fitted for potential.")
        return (u - self.u_mean) / self.u_std

    def unscale_potential(self, u_scaled):
        mean = self._as_torch(self.u_mean, u_scaled)
        std = self._as_torch(self.u_std, u_scaled)

        return u_scaled * std + mean

    def scale_acceleration(self, a):
        if self.a_mean is None:
            raise ValueError("Scaler not fitted for acceleration.")
        return (a - self.a_mean) / self.a_std

    def unscale_acceleration(self, a_scaled):
        mean = self._as_torch(self.a_mean, a_scaled)
        std = self._as_torch(self.a_std, a_scaled)
        return a_scaled * std + mean

#SIREN(SH) hybrid model, proposed by Russwurm et al., (2024), modified for the problem of gravity field
# modeling following a modified experimental setup: https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/locationencoder.py

class Hybrid(pl.LightningModule):
    def __init__(
        self,
        *,
        lmax=10,
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        first_omega_0=30.0,
        hidden_omega_0=1.0,
        normalization="4pi",
        exclude_degrees=None,
        cache_path=None,
        scaler=None,
        mode="u",
        lr=1e-4,
        run_dir
    ):
        super().__init__()
        self.scaler = scaler
        self.mode = mode
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.scaler = scaler
        self.mode = mode
        self.lmax = lmax
        self.normalization = normalization
        self.out_features = int(out_features)
        self.cache_path = cache_path
        self.run_dir = run_dir

        self.test_sse = None
        self.test_n = 0
        self.test_sse_dist = None
        self.test_n_dist = 0

        self.embedding = SHEmbedding(
            lmax=lmax,
            normalization=normalization,
            cache_path=cache_path,
            use_theta_lut=True,
            n_theta=18001,
            exclude_degrees=exclude_degrees
        )

        if lmax > 0 and self.embedding.use_theta_lut:
            self.embedding.build_theta_lut()

        if lmax == 0:
            in_features = 2
        else:
            dummy_lon = torch.tensor([0.0])
            dummy_lat = torch.tensor([0.0])
            y_dummy = self.embedding(dummy_lon, dummy_lat)
            in_features = y_dummy.shape[1]

        out_features = 1 if mode == "u" else 3
        self.siren = Siren(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

    def on_test_epoch_start(self):
        self.test_sse = [0.0, 0.0, 0.0] if self.mode != "u" else 0.0
        self.test_sse_dist = [0.0, 0.0, 0.0] if self.mode != "u" else 0.0
        self.test_n = 0
        self.test_n_dist = 0

        self._pred_lon = []
        self._pred_lat = []
        self._pred_true = []
        self._pred_pred = []
        self._pred_is_dist = []

    def forward(self, lon, lat):
        net_device = next(self.siren.parameters()).device
        lon = lon.to(net_device)
        lat = lat.to(net_device)
        y = self.embedding(lon, lat)
        y = y.to(net_device)
        return self.siren(y)

    def _compute_loss(self, y_pred, y_true):
        return self.criterion(y_pred.view(-1), y_true.view(-1))

    def training_step(self, batch, batch_idx):
        lon_b, lat_b, y_true_b = batch
        y_pred_b = self(lon_b, lat_b)
        loss = self._compute_loss(y_pred_b, y_true_b)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lon_b, lat_b, y_true_b = batch
        y_pred_b = self(lon_b, lat_b)
        loss = self._compute_loss(y_pred_b, y_true_b)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        lon, lat, y_true_scaled, is_dist = batch

        # forward once
        y_pred_scaled = self(lon, lat)
        y_true_scaled = y_true_scaled.to(y_pred_scaled.device)

        # mask as torch bool on same device
        m = is_dist.to(y_pred_scaled.device).bool()

        if self.mode == "u":
            # y_*_scaled: (B,1) or (B,)
            yp_s = y_pred_scaled.detach().view(-1)
            yt_s = y_true_scaled.detach().view(-1)

            yp = self.scaler.unscale_potential(yp_s)
            yt = self.scaler.unscale_potential(yt_s)

            # accumulate SSE in physical units
            e2 = (yp - yt) ** 2
            self.test_sse += e2.sum().item()
            self.test_n += e2.numel()

            if m.any():
                e2d = e2[m.view(-1)]
                self.test_sse_dist += e2d.sum().item()
                self.test_n_dist += e2d.numel()

            # buffers for saving (numpy)
            self._pred_lon.append(lon.detach().cpu().numpy())
            self._pred_lat.append(lat.detach().cpu().numpy())
            self._pred_true.append(yt.detach().cpu().numpy())  # (B,)
            self._pred_pred.append(yp.detach().cpu().numpy())  # (B,)
            self._pred_is_dist.append(m.detach().cpu().numpy())  # (B,)

        else:
            yp_s = y_pred_scaled.detach()
            yt_s = y_true_scaled.detach()

            yp = self.scaler.unscale_acceleration(yp_s)
            yt = self.scaler.unscale_acceleration(yt_s)

            e2 = (yp - yt) ** 2

            self.test_sse[0] += e2[:, 0].sum().item()
            self.test_sse[1] += e2[:, 1].sum().item()
            self.test_sse[2] += e2[:, 2].sum().item()
            self.test_n += e2.shape[0]

            if m.any():
                e2d = e2[m]
                self.test_sse_dist[0] += e2d[:, 0].sum().item()
                self.test_sse_dist[1] += e2d[:, 1].sum().item()
                self.test_sse_dist[2] += e2d[:, 2].sum().item()
                self.test_n_dist += e2d.shape[0]

            self._pred_lon.append(lon.detach().cpu().numpy())
            self._pred_lat.append(lat.detach().cpu().numpy())
            self._pred_true.append(yt.detach().cpu().numpy())
            self._pred_pred.append(yp.detach().cpu().numpy())
            self._pred_is_dist.append(m.detach().cpu().numpy())

    def on_test_epoch_end(self):
        if self.mode == "u":
            rmse_all = (self.test_sse / self.test_n) ** 0.5
            self.log("RMSE/test_all", rmse_all)

            if self.test_n_dist > 0:
                rmse_dist = (self.test_sse_dist / self.test_n_dist) ** 0.5
                self.log("RMSE/test_dist", rmse_dist)

        else:
            mse_theta = self.test_sse[0] / self.test_n
            mse_phi = self.test_sse[1] / self.test_n
            mse_rad = self.test_sse[2] / self.test_n

            rmse_all = (mse_theta + mse_phi + mse_rad)**0.5 / 1e5
            self.log("RMSE/test_all", rmse_all)

            if self.test_n_dist > 0:
                mse_theta_d = self.test_sse_dist[0] / self.test_n_dist
                mse_phi_d = self.test_sse_dist[1] / self.test_n_dist
                mse_rad_d = self.test_sse_dist[2] / self.test_n_dist

                rmse_dist = (mse_theta_d + mse_phi_d + mse_rad_d)**0.5 / 1e5
                self.log("RMSE/test_dist", rmse_dist)

        if self.trainer.is_global_zero:
            lon = np.concatenate(self._pred_lon)
            lat = np.concatenate(self._pred_lat)
            y_true = np.concatenate(self._pred_true)
            y_pred = np.concatenate(self._pred_pred)
            is_dist = np.concatenate(self._pred_is_dist)

            save_dir = Path(self.run_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            out_path = save_dir / "predictions.npz"

            np.savez_compressed(
                out_path,
                lon=lon,
                lat=lat,
                y_true=y_true,
                y_pred=y_pred,
                is_dist=is_dist,
                mode=self.mode,
            )
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }