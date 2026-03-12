import torch
import torch.nn as nn
from numerical.gelu import Gelu
import numpy as np
import lightning.pytorch as pl
from pathlib import Path

# Using https://github.com/MartinAstro/GravNN/blob/5debb42013097944c0398fe5b570d7cd9ebd43bd/GravNN/Preprocessors/UniformScaler.py#L3
# Scale of raw coordinates and output using minmax, according to Martin & Schaub (2022) implementation
# Important!: Instead of cartesian vectors, lat and lon values are used, for comparison with SIREN(SH) with L_max = 0
class Scaler:
    def __init__(self):

        self.lon_min = None
        self.lon_max = None
        self.lat_min = None
        self.lat_max = None
        self.u_min = None
        self.u_max = None
        self.a_min = None
        self.a_max = None
        self.eps = 1e-12
        # For numerical stability

    def fit(self, df):
        self.lon_min = float(df["lon"].min())
        self.lon_max = float(df["lon"].max())
        self.lat_min = float(df["lat"].min())
        self.lat_max = float(df["lat"].max())

        if "dU_m2_s2" in df.columns:
            u = df["dU_m2_s2"].to_numpy(dtype=float)
            self.u_min = float(u.min())
            self.u_max = float(u.max())

            # protect against degenerate range
            if not np.isfinite(self.u_min) or not np.isfinite(self.u_max) or abs(self.u_max - self.u_min) < self.eps:
                # fallback so scaling doesn't blow up
                self.u_min = 0.0
                self.u_max = 1.0

        a_cols = ["dg_theta_mGal", "dg_phi_mGal", "dg_r_mGal"]
        a = df[a_cols].to_numpy(dtype=float)

        self.a_min = a.min(axis=0).astype(float)
        self.a_max = a.max(axis=0).astype(float)

        # protect per-component degeneracy / NaNs
        rng = self.a_max - self.a_min
        bad = (~np.isfinite(self.a_min)) | (~np.isfinite(self.a_max)) | (np.abs(rng) < self.eps)
        if np.any(bad):
            # set degenerate components to a safe range
            self.a_min = np.where(bad, 0.0, self.a_min)
            self.a_max = np.where(bad, 1.0, self.a_max)

        return self

    def _as_torch(self, x, like: torch.Tensor):
        if torch.is_tensor(x):
            return x.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(x, device=like.device, dtype=like.dtype)

    def scale_potential(self, u):
        if self.u_min is None or self.u_max is None:
            raise ValueError("Scaler not fitted for potential (U_min/U_max missing).")

        denom = (self.u_max - self.u_min)
        if torch.is_tensor(u):
            u_min = self._as_torch(self.u_min, u)
            denom_t = self._as_torch(denom if abs(denom) >= self.eps else 1.0, u)
            return 2.0 * (u - u_min) / denom_t - 1.0

        denom = denom if abs(denom) >= self.eps else 1.0
        return 2.0 * (u - self.u_min) / denom - 1.0

    def unscale_potential(self, u_scaled):
        if self.u_min is None or self.u_max is None:
            raise ValueError("Scaler not fitted for potential (U_min/U_max missing).")

        denom = (self.u_max - self.u_min)
        if torch.is_tensor(u_scaled):
            u_min = self._as_torch(self.u_min, u_scaled)
            denom_t = self._as_torch(denom if abs(denom) >= self.eps else 1.0, u_scaled)
            return (u_scaled + 1.0) * 0.5 * denom_t + u_min

        denom = denom if abs(denom) >= self.eps else 1.0
        return (u_scaled + 1.0) * 0.5 * denom + self.u_min

    def scale_acceleration(self, a):
        if self.a_min is None or self.a_max is None:
            raise ValueError("Scaler not fitted for accel (a_min/a_max missing).")

        rng = (self.a_max - self.a_min)
        rng = np.where(np.abs(rng) < self.eps, 1.0, rng)

        if torch.is_tensor(a):
            a_min = self._as_torch(self.a_min, a)
            rng_t = self._as_torch(rng, a)
            return 2.0 * (a - a_min) / rng_t - 1.0

        return 2.0 * (a - self.a_min) / rng - 1.0

    def unscale_acceleration(self, a_scaled):
        if self.a_min is None or self.a_max is None:
            raise ValueError("Scaler not fitted for accel (a_min/a_max missing).")

        rng = (self.a_max - self.a_min)
        rng = np.where(np.abs(rng) < self.eps, 1.0, rng)

        if torch.is_tensor(a_scaled):
            a_min = self._as_torch(self.a_min, a_scaled)
            rng_t = self._as_torch(rng, a_scaled)
            return (a_scaled + 1.0) * 0.5 * rng_t + a_min

        return (a_scaled + 1.0) * 0.5 * rng + self.a_min

    def scale_coords(self, lonlat_deg):
        lon = lonlat_deg[..., 0]
        lat = lonlat_deg[..., 1]
        lon_s = 2.0 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1.0
        lat_s = 2.0 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1.0
        return np.stack([lon_s, lat_s], axis=-1)

    def unscale_coords(self, lonlat_scaled):
        lon_s = lonlat_scaled[..., 0]
        lat_s = lonlat_scaled[..., 1]
        lon = (lon_s + 1.0) * 0.5 * (self.lon_max - self.lon_min) + self.lon_min
        lat = (lat_s + 1.0) * 0.5 * (self.lat_max - self.lat_min) + self.lat_min
        return np.stack([lon, lat], axis=-1)

# Class to replicate Martin & Schaub's (2022) architecture scaling inputs and generating outputs

class Numerical(pl.LightningModule):
    def __init__(
        self,
        *,
        hidden_features=128,
        hidden_layers=4,
        mode="u",
        scaler=None,
        lr=1e-4,
        run_dir=None,
    ):
        super().__init__()
        self.scaler = scaler
        self.mode = mode
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.run_dir = run_dir

        self.test_sse = None
        self.test_n = 0
        self.test_sse_dist = None
        self.test_n_dist = 0

        out_features = 1 if mode == "u" else 3
        self.net = Gelu(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
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

    def _scale_lonlat(self, lon: torch.Tensor, lat: torch.Tensor):
        lon = lon.to(self.device)
        lat = lat.to(self.device)

        lonlat = torch.stack([lon, lat], dim=-1)
        lonlat_np = lonlat.detach().cpu().numpy()
        lonlat_scaled_np = self.scaler.scale_coords(lonlat_np)

        lonlat_scaled = torch.as_tensor(
            lonlat_scaled_np, dtype=lon.dtype, device=self.device
        )
        return lonlat_scaled[..., 0], lonlat_scaled[..., 1]

    def forward(self, lon, lat):
        lon_s, lat_s = self._scale_lonlat(lon, lat)
        x = torch.stack([lon_s, lat_s], dim=-1)
        return self.net(x)

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

        y_pred_scaled = self(lon, lat)
        y_true_scaled = y_true_scaled.to(y_pred_scaled.device)

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
            self._pred_true.append(yt.detach().cpu().numpy())
            self._pred_pred.append(yp.detach().cpu().numpy())
            self._pred_is_dist.append(m.detach().cpu().numpy())

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

            # buffers for saving (numpy)
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

            rmse_all = (mse_theta + mse_phi + mse_rad) ** 0.5 / 1e5
            self.log("RMSE/test_all", rmse_all)

            if self.test_n_dist > 0:
                mse_theta_d = self.test_sse_dist[0] / self.test_n_dist
                mse_phi_d = self.test_sse_dist[1] / self.test_n_dist
                mse_rad_d = self.test_sse_dist[2] / self.test_n_dist

                rmse_dist = (mse_theta_d + mse_phi_d + mse_rad_d) ** 0.5 / 1e5
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