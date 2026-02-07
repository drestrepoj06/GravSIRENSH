"""
SH_network: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.location_encoder.SH_embedding import SHEmbedding
from SRC.location_encoder.networks import SIRENNet, LINEARNet, GELUNet
import lightning.pytorch as pl
import numpy as np

# Scaling target outputs in the range [-1, 1],
# based on the scaling preferred for SIRENNETS: https://github.com/vsitzmann/siren
# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/explore_siren.ipynb
class Scaler:

    def __init__(self, mode="U"):
        self.mode = mode
        self.U_mean = None
        self.U_std  = None
        self.g_mean = None
        self.g_std  = None

    def fit(self, df):
        if self.mode in ["U"]:
            U = df["dU_m2_s2"].to_numpy()
            self.U_mean = float(U.mean())
            self.U_std  = float(U.std())

        if self.mode in ["U", "g_direct"]:
            g_cols = ["dg_theta_mGal", "dg_phi_mGal", "dg_r_mGal"]
            g_vals = df[g_cols].to_numpy()
            self.g_mean = g_vals.mean(axis=0)
            self.g_std  = g_vals.std(axis=0)

        return self

    def _as_torch(self, x, like: torch.Tensor):
        if torch.is_tensor(x):
            return x.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(x, device=like.device, dtype=like.dtype)

    def scale_potential(self, U):
        if self.U_mean is None:
            raise ValueError("Scaler not fitted for potential.")
        return (U - self.U_mean) / self.U_std

    def unscale_potential(self, U_scaled):
        mean = self._as_torch(self.U_mean, U_scaled)
        std = self._as_torch(self.U_std, U_scaled)

        return U_scaled * std + mean

    def scale_gravity(self, g):
        if self.g_mean is None:
            raise ValueError("Scaler not fitted for gravity.")
        return (g - self.g_mean) / self.g_std

    def unscale_gravity(self, g_scaled):
        mean = self._as_torch(self.g_mean, g_scaled)
        std = self._as_torch(self.g_std, g_scaled)
        return g_scaled * std + mean

# Using https://github.com/MartinAstro/GravNN/blob/5debb42013097944c0398fe5b570d7cd9ebd43bd/GravNN/Preprocessors/UniformScaler.py#L3
# Scale of raw coordinates too using minmax scaling and output using minmax
class MANDS2022Scaler:
    def __init__(self):

        self.lon_min = None
        self.lon_max = None
        self.lat_min = None
        self.lat_max = None

        # U is scalar
        self.U_min = None
        self.U_max = None

        # g vector (3,) min/max
        self.a_min = None
        self.a_max = None

        self.eps = 1e-12

    def fit(self, df):
        self.lon_min = float(df["lon"].min())
        self.lon_max = float(df["lon"].max())
        self.lat_min = float(df["lat"].min())
        self.lat_max = float(df["lat"].max())

        if "dU_m2_s2" in df.columns:
            U = df["dU_m2_s2"].to_numpy(dtype=float)
            self.U_min = float(U.min())
            self.U_max = float(U.max())

            # protect against degenerate range
            if not np.isfinite(self.U_min) or not np.isfinite(self.U_max) or abs(self.U_max - self.U_min) < self.eps:
                # fallback so scaling doesn't blow up
                self.U_min = 0.0
                self.U_max = 1.0

        g_cols = ["dg_theta_mGal", "dg_phi_mGal", "dg_r_mGal"]
        g = df[g_cols].to_numpy(dtype=float)  # (N,3)

        self.a_min = g.min(axis=0).astype(float)
        self.a_max = g.max(axis=0).astype(float)

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

    def scale_potential(self, U):
        if self.U_min is None or self.U_max is None:
            raise ValueError("MANDS2022Scaler not fitted for potential (U_min/U_max missing).")

        denom = (self.U_max - self.U_min)
        if torch.is_tensor(U):
            U_min = self._as_torch(self.U_min, U)
            denom_t = self._as_torch(denom if abs(denom) >= self.eps else 1.0, U)
            return 2.0 * (U - U_min) / denom_t - 1.0

        denom = denom if abs(denom) >= self.eps else 1.0
        return 2.0 * (U - self.U_min) / denom - 1.0

    def unscale_potential(self, U_scaled):
        if self.U_min is None or self.U_max is None:
            raise ValueError("MANDS2022Scaler not fitted for potential (U_min/U_max missing).")

        denom = (self.U_max - self.U_min)
        if torch.is_tensor(U_scaled):
            U_min = self._as_torch(self.U_min, U_scaled)
            denom_t = self._as_torch(denom if abs(denom) >= self.eps else 1.0, U_scaled)
            return (U_scaled + 1.0) * 0.5 * denom_t + U_min

        denom = denom if abs(denom) >= self.eps else 1.0
        return (U_scaled + 1.0) * 0.5 * denom + self.U_min

    def scale_accel_uniform(self, a):
        if self.a_min is None or self.a_max is None:
            raise ValueError("MANDS2022Scaler not fitted for accel (a_min/a_max missing).")

        rng = (self.a_max - self.a_min)
        rng = np.where(np.abs(rng) < self.eps, 1.0, rng)

        if torch.is_tensor(a):
            a_min = self._as_torch(self.a_min, a)
            rng_t = self._as_torch(rng, a)
            return 2.0 * (a - a_min) / rng_t - 1.0

        return 2.0 * (a - self.a_min) / rng - 1.0

    def unscale_accel_uniform(self, a_scaled):
        if self.a_min is None or self.a_max is None:
            raise ValueError("MANDS2022Scaler not fitted for accel (a_min/a_max missing).")

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

# Based on the code https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/locationencoder.py
# But only for the encoder SH + Siren network and the autograd of Martin & Schaub (2022)

class SH_SIREN(nn.Module):
    def __init__(
        self,
        lmax=10,
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        first_omega_0=30.0,
        hidden_omega_0=1.0,
        device='cuda',
        normalization="4pi",
        exclude_degrees=None,
        cache_path=None,
        scaler=None,
        mode="U"
    ):
        """
        mode:
          - "U"              : predict potential only
          - "g_direct"       : predict gravity components directly
        """
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.mode = mode
        self.lmax = lmax
        self.normalization = normalization
        self.out_features = int(out_features)
        self.cache_path = cache_path

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
            Y_dummy = self.embedding(dummy_lon, dummy_lat)
            in_features = Y_dummy.shape[1]

        if mode in ["U"]:
            out_features = 1
        elif mode in ["g_direct"]:
            out_features = 3
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self.siren = SIRENNet(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

    def forward(self, lon, lat):

        net_device = next(self.siren.parameters()).device
        lon = lon.to(net_device)
        lat = lat.to(net_device)
        Y = self.embedding(lon, lat)
        Y = Y.to(net_device)
        return self.siren(Y)


class SH_LINEAR(nn.Module):
    def __init__(
        self,
        lmax=10,
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        device='cuda',
        normalization="4pi",
        exclude_degrees=None,
        cache_path=None,
        scaler=None,
        mode="U"
    ):
        super().__init__()

        self.device = device
        self.scaler = scaler
        self.mode = mode
        self.lmax = lmax
        self.normalization = normalization
        self.out_features = int(out_features)
        self.cache_path = cache_path

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
            Y_dummy = self.embedding(torch.tensor([0.0]), torch.tensor([0.0]))
            in_features = Y_dummy.shape[1]

        if mode in ["U"]:
            out_dim = 1
        elif mode == "g_direct":
            out_dim = 3
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self.net = LINEARNet(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_dim
        )

    def forward(self, lon, lat):

        if self.mode == "U":
            Y = self.embedding(lon, lat).to(self.device)
            return self.net(Y)

        elif self.mode == "g_direct":
            Y = self.embedding(lon, lat).to(self.device)
            return self.net(Y)

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")

# Class to replicate Martin & Schaub's (2022) architecture, but with a traditional loss

class MANDS2022(nn.Module):
    def __init__(
        self,
        hidden_features=128,
        hidden_layers=4,
        device="cuda",
        scaler=None,
        mode="U",
    ):
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.mode = mode

        if self.scaler is None or not hasattr(self.scaler, "scale_coords"):
            raise ValueError("MANDS2022 requires a scaler with a .scale_coords(...) method.")

        out_features = 1 if mode == "U" else 3
        self.net = GELUNet(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
        )

        self.to(self.device)

    def _scale_lonlat(self, lon, lat):
        lon = lon.to(self.device)
        lat = lat.to(self.device)

        lonlat = torch.stack([lon, lat], dim=-1)              # (N,2)
        lonlat_np = lonlat.detach().cpu().numpy()             # scaler is numpy-based
        lonlat_scaled_np = self.scaler.scale_coords(lonlat_np)

        lonlat_scaled = torch.as_tensor(
            lonlat_scaled_np, dtype=lon.dtype, device=self.device
        )
        return lonlat_scaled[..., 0], lonlat_scaled[..., 1]

    def forward(self, lon, lat):
        lon_s, lat_s = self._scale_lonlat(lon, lat)

        if self.mode == "U":
            x = torch.stack([lon_s, lat_s], dim=-1)
            U_pred = self.net(x)
            return U_pred
        elif self.mode == "g_direct":
            x = torch.stack([lon_s, lat_s], dim=-1)
            g_pred = self.net(x)
            return g_pred


class Gravity(pl.LightningModule):
    def __init__(self, model_cfg, scaler, lr=1e-4):
        super().__init__()

        arch = model_cfg.pop("arch", "sirensh").lower()

        if arch == "sirensh":
            self.model = SH_SIREN(**model_cfg)
            self.mode = model_cfg.get("mode", "U")

        elif arch == "linearsh":
            self.model = SH_LINEAR(**model_cfg)
            self.mode = model_cfg.get("mode", "U")

        elif arch == "mands2022":
            model_cfg = dict(model_cfg)
            model_cfg["scaler"] = scaler
            self.model = MANDS2022(**model_cfg)
            self.mode = model_cfg.get("mode", "U")

        else:
            raise ValueError(
                f"Unknown architecture '{arch}'. Expected 'sirensh', 'linearsh', or 'mands2022'."
            )

        self.scaler = scaler
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, lon, lat):
        return self.model(lon, lat)

    def _compute_loss(self, y_pred, y_true, return_components=False):

        if self.mode == "U":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            return (loss, None, None, None, loss) if return_components else loss

        elif self.mode == "g_direct":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            return (None, loss, None, None, loss) if return_components else loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


    def training_step(self, batch, batch_idx=None):

        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)

        loss_components = self._compute_loss(y_pred_b, y_true_b, return_components=True)

        total_loss = loss_components[-1]
        if isinstance(total_loss, torch.Tensor) and total_loss.ndim > 0:
            total_loss = total_loss.mean()

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx=None):
        with torch.enable_grad():
            lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
            y_pred_b = self.model(lon_b, lat_b)

            loss_components = self._compute_loss(y_pred_b, y_true_b, return_components=True)

            total_loss = loss_components[-1]
            if isinstance(total_loss, torch.Tensor) and total_loss.ndim > 0:
                total_loss = total_loss.mean()

            self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
            return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }