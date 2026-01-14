"""
SH_network: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Location_encoder.SH_embedding import SHPlusR, SHEmbedding
from SRC.Location_encoder.Networks import SIRENNet, LINEARNet, GELUNet
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
class PINNScaler:
    def __init__(self, base_scaler: Scaler, a_scale=None, U_scale=None):
        self.base = base_scaler
        self.a_scale = None if a_scale is None else float(a_scale)
        self.U_scale = None if U_scale is None else float(U_scale)
        self.Rang = np.pi

    def fit(self, df):
        if self.U_scale is None:
            if "dU_m2_s2" in df.columns:
                U = df["dU_m2_s2"].to_numpy(dtype=float)
                rmsU = float(np.sqrt(np.mean(U ** 2)))
                self.U_scale = rmsU if rmsU > 0 else 1.0
            else:
                self.U_scale = 1.0

        if self.a_scale is None:
            g_cols = ["dg_theta_mGal", "dg_phi_mGal"]
            g = df[g_cols].to_numpy(dtype=float)
            rms = float(np.sqrt(np.mean(g ** 2)))
            self.a_scale = rms if rms > 0 else 1.0

        return self

    def scale_coords(self, lonlatr):
        """
        lonlatr: array-like (..., 3) = [lon_deg, lat_deg, r_m]
        returns: (..., 3) = [lon_scaled, lat_scaled, rho]
        """
        lon_deg = lonlatr[..., 0]
        lat_deg = lonlatr[..., 1]
        r_m = lonlatr[..., 2]

        lon_rad = lon_deg * (np.pi / 180.0)
        lat_rad = lat_deg * (np.pi / 180.0)

        lon_scaled = lon_rad / self.Rang  # [-1, 1] if lon in [-pi, pi]
        lat_scaled = lat_rad / self.Rang  # [-0.5, 0.5] if lat in [-pi/2, pi/2]

        rho = r_m / self.r_scale  # ~ 1 + h/R

        return np.stack([lon_scaled, lat_scaled, rho], axis=-1)

    def unscale_coords(self, coords_scaled):
        lon_scaled = coords_scaled[..., 0]
        lat_scaled = coords_scaled[..., 1]
        rho = coords_scaled[..., 2]

        lon_rad = lon_scaled * self.Rang
        lat_rad = lat_scaled * self.Rang

        lon_deg = lon_rad * (180.0 / np.pi)
        lat_deg = lat_rad * (180.0 / np.pi)

        r_m = rho * self.r_scale
        return np.stack([lon_deg, lat_deg, r_m], axis=-1)

    def scale_potential(self, U):
        return U / self.U_scale

    def unscale_potential(self, U_scaled):
        return U_scaled * self.U_scale

    def scale_accel_uniform(self, a):
        return a / self.a_scale

    def unscale_accel_uniform(self, a_scaled):
        return a_scaled * self.a_scale

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
                    grads = torch.autograd.grad(
                        outputs=outputs,
                        inputs=inputs,
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True
                    )

                    g_theta = -grads[1]
                    g_phi = -grads[0]
                    g_r = -grads[2]

                    return outputs, (g_theta, g_phi, g_r)

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
            with torch.set_grad_enabled(True):
                lon = lon.to(net_device).requires_grad_(True)
                lat = lat.to(net_device).requires_grad_(True)
                r = r.to(net_device).requires_grad_(True)

                X = self.embedding(lon, lat, r)
                U = self.siren(X)  # potential

                inputs = [lon, lat, r]
                grads = torch.autograd.grad(
                    outputs=U,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(U),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True
                )

                # what you called g_theta/g_phi before (still same mapping)
                g_theta = -grads[1]
                g_phi = -grads[0]
                g_r = -grads[2]

                return U, (g_theta, g_phi, g_r)

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
                    grads = torch.autograd.grad(
                        outputs=outputs,
                        inputs=inputs,
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True
                    )

                    g_theta = -grads[1]
                    g_phi = -grads[0]
                    g_r = -grads[2]

                    return outputs, (g_theta, g_phi, g_r)

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
            with torch.set_grad_enabled(True):
                lon = lon.to(self.device).requires_grad_(True)
                lat = lat.to(self.device).requires_grad_(True)
                r   = r.to(self.device).requires_grad_(True)

                X = self.embedding(lon, lat, r).to(self.device)
                U_pred = self.net(X)

                inputs = [lon, lat, r]
                grads = torch.autograd.grad(
                    outputs=U_pred,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(U_pred),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True,
                )

                g_theta = -grads[1]
                g_phi   = -grads[0]
                g_r = -grads[2]

                return U_pred, (g_theta, g_phi, g_r)

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")

# Class to replicate Martin & Schaub's (2022) architecture, with some modifications

class PINN(nn.Module):
    """
    PINN that takes (lon, lat, r) as inputs.
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
        scaler=None,     # PINNScaler (for output scaling if you want; input scaling done here)
        mode="g_indirect",
        r_scale=6378136.6,
    ):
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.mode = mode
        self.r_scale = float(r_scale)

        if mode not in ["U", "g_direct", "g_indirect"]:
            raise ValueError(f"Unsupported PINN mode: {mode}")

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

        # uniform angular scale (matches your PINNScaler idea)
        lon_s = lon_rad / torch.pi
        lat_s = lat_rad / torch.pi

        rho = r_m / self.r_scale

        return lon_s, lat_s, rho

    def forward(self, lon, lat, r):
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

        if self.mode == "g_direct":
            x = torch.stack([lon_s, lat_s, rho], dim=-1)
            g_pred = self.net(x)  # Nx3: (g_theta, g_phi, g_r) in whatever convention you train
            # If you train only 2 components, change out_features and return Nx2.
            return g_pred

        # U and g_indirect: compute U and derivatives
        x = torch.stack([lon_s, lat_s, rho], dim=-1)
        U_raw = self.net(x)          # Nx1

        grads = torch.autograd.grad(
            outputs=U_raw,
            inputs=[lon, lat, r],
            grad_outputs=torch.ones_like(U_raw),
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
        )
        dU_dlon, dU_dlat, dU_dr = grads  # derivatives w.r.t (deg, deg, m)

        g_theta = -dU_dlat
        g_phi   = -dU_dlon
        g_r     = -dU_dr

        return U_raw, (g_theta, g_phi, g_r)

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


        elif arch == "pinn":
            model_cfg = dict(model_cfg)
            model_cfg["scaler"] = scaler
            self.model = PINN(**model_cfg)
            self.mode = model_cfg.get("mode", "U")

        else:
            raise ValueError(
                f"Unknown architecture '{arch}'. Expected 'sirensh', 'linearsh', or 'pinn'."
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

        component_names = ["U", "g", "grad", "consistency"]
        for name, comp in zip(component_names, loss_components[:-1]):
            if comp is not None:
                if isinstance(comp, torch.Tensor) and comp.ndim > 0:
                    comp = comp.mean()
                self.log(f"train_{name}_loss", comp, on_step=False, on_epoch=True)

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

        component_names = ["U", "g", "grad", "consistency"]

        for name, comp in zip(component_names, loss_components[:-1]):
            if comp is not None:
                if isinstance(comp, torch.Tensor) and comp.ndim > 0:
                    comp = comp.mean()
                self.log(f"val_{name}_loss", comp, on_epoch=True)


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