"""
SH_network: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Location_encoder.SH_embedding import SHEmbedding
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
            g_cols = ["dg_theta_mGal", "dg_phi_mGal"]
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

    def scale_coords(self, lonlat_deg):
        return lonlat_deg * (np.pi / 180.0)

    def unscale_coords(self, lonlat_rad):
        return lonlat_rad * (180.0 / np.pi)

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

        if mode in ["U", "g_indirect"]:
            out_features = 1
        elif mode in ["g_direct"]:
            out_features = 2
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

    def forward(self, lon, lat, return_gradients=False, r=None):

        net_device = next(self.siren.parameters()).device

        if self.mode == "U":
            if return_gradients:
                with torch.set_grad_enabled(True):
                    lon = lon.to(net_device).requires_grad_(True)
                    lat = lat.to(net_device).requires_grad_(True)

                    Y = self.embedding(lon, lat)
                    Y = Y.to(net_device)

                    outputs = self.siren(Y)

                    grads = torch.autograd.grad(
                        outputs=outputs,
                        inputs=[lon, lat],
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True
                    )
                    return outputs, grads
            else:
                lon = lon.to(net_device)
                lat = lat.to(net_device)

                Y = self.embedding(lon, lat)
                Y = Y.to(net_device)

                return self.siren(Y)

        elif self.mode == "g_direct":
            lon = lon.to(net_device)
            lat = lat.to(net_device)

            Y = self.embedding(lon, lat)
            Y = Y.to(net_device)

            return self.siren(Y)

        elif self.mode == "g_indirect":
            with torch.set_grad_enabled(True):
                lon = lon.to(net_device).requires_grad_(True)
                lat = lat.to(net_device).requires_grad_(True)

                Y = self.embedding(lon, lat)
                Y = Y.to(net_device)

                outputs = self.siren(Y)

                grads = torch.autograd.grad(
                    outputs=outputs,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(outputs),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True
                )

                g_theta = -grads[1]
                g_phi = -grads[0]

                return outputs, (g_theta, g_phi)
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
        first_omega_0=None,
        hidden_omega_0=None,
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

        if mode in ["U", "g_indirect"]:
            out_dim = 1
        elif mode == "g_direct":
            out_dim = 2
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self.net = LINEARNet(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_dim
        )

    def forward(self, lon, lat, return_gradients=False, r=None):

        if self.mode == "U":

            if return_gradients:
                with torch.set_grad_enabled(True):
                    lon = lon.to(self.device).requires_grad_(True)
                    lat = lat.to(self.device).requires_grad_(True)

                    Y = self.embedding(lon, lat).to(self.device)
                    outputs = self.net(Y)

                    grads = torch.autograd.grad(
                        outputs=outputs,
                        inputs=[lon, lat],
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True
                    )
                    return outputs, grads

            else:
                Y = self.embedding(lon, lat).to(self.device)
                return self.net(Y)

        elif self.mode == "g_direct":
            Y = self.embedding(lon, lat).to(self.device)
            return self.net(Y)

        elif self.mode == "g_indirect":
            with torch.set_grad_enabled(True):
                lon = lon.to(self.device).requires_grad_(True)
                lat = lat.to(self.device).requires_grad_(True)

                Y = self.embedding(lon, lat).to(self.device)
                U_pred = self.net(Y)

                grads = torch.autograd.grad(
                    outputs=U_pred,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(U_pred),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True,
                )
                g_theta = -grads[1]
                g_phi = -grads[0]

                return U_pred, (g_theta, g_phi)


        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")

# Class to replicate Martin & Schaub's (2022) architecture

class PINN(nn.Module):
    """
    PINN with raw lon/lat input (degrees), scaled inside using PINNScaler.scale_coords.
    Supports modes:
      - "U":         predict potential only
      - "g_direct":  predict (g_theta, g_phi) directly
      - "g_indirect":predict U and derive g = -∇U (w.r.t. scaled coords)
    """

    def __init__(
        self,
        hidden_features=128,
        hidden_layers=4,
        device="cuda",
        scaler=None,     # PINNScaler
        mode="g_indirect",
    ):
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.mode = mode

        if self.scaler is None or not hasattr(self.scaler, "scale_coords"):
            raise ValueError("PINN requires a scaler with a .scale_coords(...) method.")

        if mode not in ["U", "g_direct", "g_indirect"]:
            raise ValueError(f"Unsupported PINN mode: {mode}")

        out_features = 1 if mode in ["U", "g_indirect"] else 2

        self.net = GELUNet(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
        )

        self.to(self.device)

    def _scale_lonlat(self, lon, lat):
        """
        lon, lat: raw degrees (Tensor).
        returns scaled lon/lat (Tensor), e.g. radians.
        """
        lon = lon.to(self.device)
        lat = lat.to(self.device)

        lonlat = torch.stack([lon, lat], dim=-1)              # (N,2)
        lonlat_np = lonlat.detach().cpu().numpy()             # scaler is numpy-based
        lonlat_scaled_np = self.scaler.scale_coords(lonlat_np)

        lonlat_scaled = torch.as_tensor(
            lonlat_scaled_np, dtype=lon.dtype, device=self.device
        )
        return lonlat_scaled[..., 0], lonlat_scaled[..., 1]

    def forward(self, lon, lat, return_gradients=False):
        """
        Returns:
          - mode "U":
                if return_gradients: (U, grads) where grads=(dU/dlon_s, dU/dlat_s)
                else: U
          - mode "g_direct":
                g = (g_theta, g_phi) directly (shape Nx2)
          - mode "g_indirect":
                (U, (g_theta, g_phi)) where g = -∇U in scaled coord space
        """
        lon_s, lat_s = self._scale_lonlat(lon, lat)

        if self.mode == "g_direct":
            x = torch.stack([lon_s, lat_s], dim=-1)
            g_pred = self.net(x)  # Nx2 (scaled accel domain)
            return g_pred

        # For U and g_indirect we may need grads
        with torch.set_grad_enabled(True):
            lon_s = lon_s.requires_grad_(True)
            lat_s = lat_s.requires_grad_(True)

            x = torch.stack([lon_s, lat_s], dim=-1)
            U_pred = self.net(x)  # Nx1

            grads = torch.autograd.grad(
                outputs=U_pred,
                inputs=[lon_s, lat_s],
                grad_outputs=torch.ones_like(U_pred),
                create_graph=self.training,
                retain_graph=self.training,
                only_inputs=True,
            )
            dU_dlon, dU_dlat = grads[0], grads[1]

            if self.mode == "U":
                if return_gradients:
                    return U_pred, (dU_dlon, dU_dlat)
                return U_pred

            # g_indirect: g = -∇U
            g_theta = -dU_dlat
            g_phi   = -dU_dlon
            return U_pred, (g_theta, g_phi)

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


        elif arch == "pinn":
            model_cfg = dict(model_cfg)
            model_cfg["scaler"] = scaler
            self.model = PINN(**model_cfg)
            self.mode = model_cfg.get("mode", "g_indirect")

        else:
            raise ValueError(
                f"Unknown architecture '{arch}'. Expected 'sirensh', 'linearsh', or 'pinn'."
            )

        self.scaler = scaler
        self.lr = lr

        self.lambda_consistency = 1e-1

        self.scaler = scaler
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, lon, lat):
        return self.model(lon, lat)

    def _compute_loss(self, y_pred, y_true, return_components=False):

        if self.mode == "U":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            return (loss, None, None, None, loss) if return_components else loss

        elif self.mode == "g_direct":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            return (None, loss, None, None, loss) if return_components else loss

        elif self.mode == "g_indirect":
            U_pred, (g_theta, g_phi) = y_pred
            g_pred = torch.stack([g_theta, g_phi], dim=1)
            loss = self.criterion(g_pred.view(-1), y_true.view(-1))
            loss = loss
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
        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)

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