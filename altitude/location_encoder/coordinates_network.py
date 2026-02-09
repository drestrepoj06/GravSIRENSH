"""
SH_network: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from altitude.location_encoder.SH_embedding import SHPlusR, SHEmbedding
from altitude.location_encoder.networks import SIRENNet, LINEARNet, GELUNet
import lightning.pytorch as pl
import numpy as np

# Scaling target outputs in the range [-1, 1],
# based on the scaling preferred for SIRENNETS: https://github.com/vsitzmann/siren
# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/explore_siren.ipynb
class Scaler:

    def __init__(self, mode="U", r_scale=6378136.6):
        self.mode = mode
        self.r_scale = r_scale
        self.U_mean = None
        self.U_std  = None
        self.g_mean = None
        self.g_std  = None

    def fit(self, df):
        if self.mode in ["U", "g_indirect"]:
            U = df["dU_m2_s2"].to_numpy()
            self.U_mean = float(U.mean())
            self.U_std  = float(U.std())

        if self.mode in ["U", "g_direct", "g_indirect"]:
            g_cols = ["dg_theta_mGal", "dg_phi_mGal", "dg_r_mGal"]
            g_vals = df[g_cols].to_numpy()
            self.g_mean = g_vals.mean(axis=0)
            self.g_std  = g_vals.std(axis=0)

        return self

    def scale_potential(self, U):
        if self.U_mean is None:
            raise ValueError("Scaler not fitted for potential.")
        return (U - self.U_mean) / self.U_std

    def unscale_potential(self, U_scaled):
        return U_scaled * self.U_std + self.U_mean

    def scale_gravity(self, g):
        if self.g_mean is None:
            raise ValueError("Scaler not fitted for gravity.")
        return (g - self.g_mean) / self.g_std

    def unscale_gravity(self, g_scaled):
        return g_scaled * self.g_std + self.g_mean

# Using https://github.com/MartinAstro/GravNN/blob/5debb42013097944c0398fe5b570d7cd9ebd43bd/GravNN/Preprocessors/UniformScaler.py#L3
# Scale of raw coordinates too
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
          - "g_indirect"     : predict potential and derive g = -∇U
        """
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.mode = mode
        self.lmax = lmax
        self.normalization = normalization
        self.out_features = int(out_features)
        self.cache_path = cache_path
        self.r_scale = 6378136.6

        base_emb = SHEmbedding(
            lmax=lmax,
            normalization=normalization,
            cache_path=cache_path,
            use_theta_lut=True,
            n_theta=18001,
            exclude_degrees=exclude_degrees
        )
        if lmax > 0 and base_emb.use_theta_lut:
            base_emb.build_theta_lut()

        # NEW: embedding that concatenates rho = r/R
        self.embedding = SHPlusR(base_emb, R_ref=self.r_scale)

        if lmax == 0:
            in_features = 3
        else:
            dummy_lon = torch.tensor([0.0])
            dummy_lat = torch.tensor([0.0])
            dummy_r = torch.tensor([self.r_scale])  # r = R
            X_dummy = self.embedding(dummy_lon, dummy_lat, dummy_r)
            in_features = X_dummy.shape[1]

        if mode in ["U", "g_indirect"]:
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

    def forward(self, lon, lat, r=None, return_gradients=False):
        """
        lon, lat in degrees (as before)
        r in meters (radius). If you have altitude h, pass r = R + h.
        return_radial:
            if True, include dU/dr in returned gradients (when return_gradients=True)
        """
        net_device = next(self.siren.parameters()).device

        if r is None:
            # if you still run on reference sphere
            r = torch.full_like(lon, fill_value=self.r_scale)

        if self.mode == "U":
            if return_gradients:
                with torch.set_grad_enabled(True):
                    lon = lon.to(net_device).requires_grad_(True)
                    lat = lat.to(net_device).requires_grad_(True)
                    r = r.to(net_device).requires_grad_(True)

                    X = self.embedding(lon, lat, r)  # NEW
                    outputs = self.siren(X)

                    inputs = [lon, lat, r]
                    dU_dlon, dU_dlat, dU_dr = torch.autograd.grad(
                        outputs=outputs,
                        inputs=[lon, lat, r],
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=self.training,
                        retain_graph=self.training,  # set True only if you need another grad call on same graph
                        only_inputs=True
                    )

                    eps = 1e-12
                    r_safe = torch.clamp(r, min=eps)
                    cos_lat = torch.clamp(torch.cos(lat), min=eps)  # avoid division blow-up near poles
                    g_r = -dU_dr
                    g_phi = -(1.0 / r_safe) * dU_dlat
                    g_lam = -(1.0 / (r_safe * cos_lat)) * dU_dlon

                    return outputs, (g_phi, g_lam, g_r)

            lon = lon.to(net_device)
            lat = lat.to(net_device)
            r = r.to(net_device)
            X = self.embedding(lon, lat, r)
            return self.siren(X)

        elif self.mode == "g_direct":
            lon = lon.to(net_device)
            lat = lat.to(net_device)
            r = r.to(net_device)
            X = self.embedding(lon, lat, r)
            return self.siren(X)


        elif self.mode == "g_indirect":
            lon = lon.to(net_device).requires_grad_(True)  # λ (rad)
            lat = lat.to(net_device).requires_grad_(True)  # φ (rad)
            r = r.to(net_device).requires_grad_(True)  # radius
            X = self.embedding(lon, lat, r)
            U = self.siren(X)  # shape [N, 1] (or [N])
            dU_dlon, dU_dlat, dU_dr = torch.autograd.grad(
                outputs=U,
                inputs=[lon, lat, r],
                grad_outputs=torch.ones_like(U),
                create_graph=self.training,
                retain_graph=self.training,  # set True only if you need another grad call on same graph
                only_inputs=True
            )

            eps = 1e-12
            r_safe = torch.clamp(r, min=eps)
            cos_lat = torch.clamp(torch.cos(lat), min=eps)  # avoid division blow-up near poles
            g_r = -dU_dr
            g_phi = -(1.0 / r_safe) * dU_dlat
            g_lam = -(1.0 / (r_safe * cos_lat)) * dU_dlon

            return U, (g_phi, g_lam, g_r)

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")


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
        mode="U",
        r_scale=6378136.6,
        **kwargs
    ):
        super().__init__()

        self.device = device
        self.scaler = scaler
        self.mode = mode
        self.lmax = lmax
        self.normalization = normalization
        self.out_features = int(out_features)
        self.cache_path = cache_path
        self.r_scale = float(r_scale)

        base_emb = SHEmbedding(
            lmax=lmax,
            normalization=normalization,
            cache_path=cache_path,
            use_theta_lut=True,
            n_theta=18001,
            exclude_degrees=exclude_degrees
        )

        if lmax > 0 and base_emb.use_theta_lut:
            base_emb.build_theta_lut()

        # NEW: append rho=r/R
        self.embedding = SHPlusR(base_emb, R_ref=self.r_scale)

        # infer in_features
        if lmax == 0:
            in_features = 3  # [lon_norm, lat_norm, rho]
        else:
            dummy_lon = torch.tensor([0.0])
            dummy_lat = torch.tensor([0.0])
            dummy_r   = torch.tensor([self.r_scale])
            X_dummy = self.embedding(dummy_lon, dummy_lat, dummy_r)
            in_features = X_dummy.shape[1]

        if mode in ["U", "g_indirect"]:
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

    def forward(self, lon, lat, r=None, return_gradients=False):
        """
        lon, lat in degrees
        r in meters (radius). If you have altitude h, pass r = R + h.

        return_radial:
            if True and return_gradients=True, also return dU/dr (or -dU/dr depending on sign below).
        """
        if r is None:
            r = torch.full_like(lon, fill_value=self.r_scale)

        if self.mode == "U":

            if return_gradients:
                with torch.set_grad_enabled(True):
                    lon = lon.to(self.device).requires_grad_(True)
                    lat = lat.to(self.device).requires_grad_(True)
                    r   = r.to(self.device).requires_grad_(True)

                    X = self.embedding(lon, lat, r).to(self.device)
                    outputs = self.net(X)

                    inputs = [lon, lat, r]
                    dU_dlon, dU_dlat, dU_dr = torch.autograd.grad(
                        outputs=outputs,
                        inputs=inputs,
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True
                    )

                    eps = 1e-12
                    r_safe = torch.clamp(r, min=eps)
                    cos_lat = torch.clamp(torch.cos(lat), min=eps)  # avoid division blow-up near poles
                    g_r = -dU_dr
                    g_phi = -(1.0 / r_safe) * dU_dlat
                    g_lam = -(1.0 / (r_safe * cos_lat)) * dU_dlon

                    return outputs, (g_phi, g_lam, g_r)

            lon = lon.to(self.device)
            lat = lat.to(self.device)
            r   = r.to(self.device)

            X = self.embedding(lon, lat, r).to(self.device)
            return self.net(X)

        elif self.mode == "g_direct":
            lon = lon.to(self.device)
            lat = lat.to(self.device)
            r   = r.to(self.device)

            X = self.embedding(lon, lat, r).to(self.device)
            return self.net(X)


        elif self.mode == "g_indirect":

            lon = lon.to(self.device).requires_grad_(True)  # λ (rad)
            lat = lat.to(self.device).requires_grad_(True)  # φ (rad)
            r = r.to(self.device).requires_grad_(True)  # radius
            X = self.embedding(lon, lat, r)
            U = self.net(X)  # shape [N, 1] (or [N])
            dU_dlon, dU_dlat, dU_dr = torch.autograd.grad(
                outputs=U,
                inputs=[lon, lat, r],
                grad_outputs=torch.ones_like(U),
                create_graph=self.training,
                retain_graph=self.training,  # set True only if you need another grad call on same graph
                only_inputs=True
            )
            eps = 1e-12
            r_safe = torch.clamp(r, min=eps)
            cos_lat = torch.clamp(torch.cos(lat), min=eps)  # avoid division blow-up near poles
            g_r = -dU_dr
            g_phi = -(1.0 / r_safe) * dU_dlat
            g_lam = -(1.0 / (r_safe * cos_lat)) * dU_dlon
            return U, (g_phi, g_lam, g_r)

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")

# Class to replicate Martin & Schaub's (2022) architecture, with some modifications

class MANDS2022(nn.Module):
    """
    MANDS2022 that takes (lon, lat, r) as inputs.
    Scaling is done in torch so autograd can give derivatives w.r.t lon/lat/r.
    Conventions:
      - Returns U_neg = -U_raw
      - Always returns derivatives wrt (lon, lat, r)
      - g ordering: (g_theta (lat), g_phi (lon), g_r)
    """

    def __init__(
        self,
        hidden_features=128,
        hidden_layers=4,
        device="cuda",
        scaler=None,     # MANDS2022Scaler (for output scaling if you want; input scaling done here)
        mode="g_indirect",
        r_scale=6378136.6,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.mode = mode
        self.r_scale = float(r_scale)

        if mode not in ["U", "g_direct", "g_indirect"]:
            raise ValueError(f"Unsupported MANDS2022S mode: {mode}")

        out_features = 1 if mode in ["U", "g_indirect"] else 3  # if g_direct now includes gr
        # If you still want g_direct only (g_theta,g_phi), set out_features=2 and ignore gr.

        self.net = GELUNet(
            in_features=3,  # lon_s, lat_s, rho
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
        )

        self.to(self.device)

    def _scale_inputs(self, lon_deg, lat_deg, r_m):
        """
        Differentiable scaling in torch.
        lon_deg, lat_deg in degrees
        r_m in meters (radius)

        Returns:
          lon_s, lat_s, rho
        """
        lon_deg = lon_deg.to(self.device).float()
        lat_deg = lat_deg.to(self.device).float()
        r_m     = r_m.to(self.device).float()

        deg2rad = torch.pi / 180.0
        lon_rad = lon_deg * deg2rad
        lat_rad = lat_deg * deg2rad

        # uniform angular scale (matches your MANDS2022SScaler idea)
        lon_s = lon_rad / torch.pi
        lat_s = lat_rad / torch.pi

        rho = r_m / self.r_scale

        return lon_s, lat_s, rho

    def forward(self, lon, lat, r, return_gradients=False):
        """
        Always returns:
          - U_neg (Nx1) for modes U/g_indirect, or g_pred (Nx3) for g_direct
          - derivatives wrt raw lon/lat/r: (dU_dlon, dU_dlat, dU_dr)
          - and for g_indirect also (g_theta, g_phi, g_r) in requested order

        Note: derivatives are w.r.t raw lon/lat in DEGREES and r in METERS.
        """
        # scale inputs (but keep raw vars for grads)
        lon = lon.to(self.device).float().requires_grad_(True)
        lat = lat.to(self.device).float().requires_grad_(True)
        r   = r.to(self.device).float().requires_grad_(True)

        lon_s, lat_s, rho = self._scale_inputs(lon, lat, r)
        if self.mode == "U":
            if return_gradients:
                with torch.set_grad_enabled(True):
                    x = torch.stack([lon_s, lat_s, rho], dim=-1)
                    U_raw = self.net(x)
                    inputs = [lon, lat, r]
                    dU_dlon, dU_dlat, dU_dr = torch.autograd.grad(
                        outputs=U_raw,
                        inputs=inputs,
                        grad_outputs=torch.ones_like(U_raw),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True
                        )

                    eps = 1e-12
                    r_safe = torch.clamp(r, min=eps)
                    cos_lat = torch.clamp(torch.cos(lat), min=eps)  # avoid division blow-up near poles
                    g_r = -dU_dr
                    g_phi = -(1.0 / r_safe) * dU_dlat
                    g_lam = -(1.0 / (r_safe * cos_lat)) * dU_dlon

                    return U_raw, (g_phi, g_lam, g_r)
            else:
                x = torch.stack([lon_s, lat_s, rho], dim=-1)
                U_raw = self.net(x)
                return U_raw

        elif self.mode == "g_direct":
            x = torch.stack([lon_s, lat_s, rho], dim=-1)
            g_pred = self.net(x)
            return g_pred

        elif self.mode == "g_indirect":
            x = torch.stack([lon_s, lat_s, rho], dim=-1)
            U_raw = self.net(x)          # Nx1

            dU_dlon, dU_dlat, dU_dr = torch.autograd.grad(
                outputs=U_raw,
                inputs=[lon, lat, r],
                grad_outputs=torch.ones_like(U_raw),
                create_graph=self.training,
                retain_graph=self.training,  # set True only if you need another grad call on same graph
                only_inputs=True
            )

            eps = 1e-12
            r_safe = torch.clamp(r, min=eps)
            cos_lat = torch.clamp(torch.cos(lat), min=eps)  # avoid division blow-up near poles

            g_r = -dU_dr
            g_phi = -(1.0 / r_safe) * dU_dlat
            g_lam = -(1.0 / (r_safe * cos_lat)) * dU_dlon

            return U_raw, (g_phi, g_lam, g_r)

class Gravity(pl.LightningModule):
    def __init__(self, model_cfg, scaler, lr=1e-4):
        super().__init__()

        arch = model_cfg.pop("arch", "sirensh").lower()

        self.mode = model_cfg.get("mode", "U")

        if arch == "sirensh":
            self.model = SH_SIREN(**model_cfg)
            self.model.mode = self.mode

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
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, lon, lat, r):
        return self.model(lon, lat, r = r)

    def _compute_loss(self, y_pred, y_true, return_components=False):

        if self.mode in ("U", "g_direct"):
            if isinstance(y_pred, (tuple, list)):
                y_pred = y_pred[0]
            if isinstance(y_true, (tuple, list)):
                y_true = y_true[0]

        if self.mode == "U":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            return (loss, None, None, None, loss) if return_components else loss


        elif self.mode == "g_direct":
            loss = self.criterion(y_pred, y_true)
            return (None, loss, None, None, loss) if return_components else loss

        elif self.mode == "g_indirect":
            if isinstance(y_pred, (tuple, list)):
                if len(y_pred) == 2:
                    U_pred, g_tuple = y_pred
                elif len(y_pred) in (3, 4):
                    U_pred = y_pred[0]
                    g_tuple = y_pred[1:]
                else:
                    raise ValueError(f"Unexpected y_pred length: {len(y_pred)}")
            else:
                raise ValueError("g_indirect expects tuple output")
            g_pred = torch.stack(list(g_tuple), dim=1)
            if isinstance(y_true, (tuple, list)):
                y_true = y_true[0]
            loss = self.criterion(g_pred, y_true)
            return (None, loss, None, None, loss) if return_components else loss
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


    def training_step(self, batch, batch_idx=None):

        lon_b, lat_b, r_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b, r_b)
        loss_components = self._compute_loss(y_pred_b, y_true_b, return_components=True)

        total_loss = loss_components[-1]
        if isinstance(total_loss, torch.Tensor) and total_loss.ndim > 0:
            total_loss = total_loss.mean()

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx=None):
        lon_b, lat_b, r_b, y_true_b = [b.to(self.device) for b in batch]
        lon_b = lon_b.detach().requires_grad_(True)
        lat_b = lat_b.detach().requires_grad_(True)
        r_b = r_b.detach().requires_grad_(True)

        with torch.enable_grad():
            y_pred_b = self.model(lon_b, lat_b, r_b)

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