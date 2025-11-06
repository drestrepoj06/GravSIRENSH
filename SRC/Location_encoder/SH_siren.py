"""
SH-SIREN: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Location_encoder.SH_embedding import SHEmbedding
from SRC.Location_encoder.Siren import SIRENNet
import numpy as np
import lightning.pytorch as pl

# Scaling target outputs in the range [-1, 1],
# based on the scaling preferred for SIRENNETS: https://github.com/vsitzmann/siren
# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/explore_siren.ipynb
class SHSirenScaler:
    """
    Handles scaling of potential U and/or gravity g,
    depending on experiment mode.
    """
    def __init__(self, mode="U", r_scale=6.371e6):
        self.mode = mode
        self.r_scale = r_scale
        self.U_min = self.U_max = None
        self.g_min = self.g_max = None
        self.u_scale = None

    # === FITTING ===
    def fit(self, df):
        # Fit potential whenever U is used (directly or for indirect g)
        if self.mode in ["U", "g_indirect", "U_g_indirect", "U_g_direct"]:
            self.U_min = float(df["dU_m2_s2"].min())
            self.U_max = float(df["dU_m2_s2"].max())
            self.u_scale = 0.5 * (self.U_max - self.U_min)

        # Fit gravity whenever g is a target (direct) or part of the loss
        if self.mode in ["g_direct", "g_indirect", "U_g_direct", "U_g_indirect"]:
            g_cols = ["dg_theta_mGal", "dg_phi_mGal"]
            g_values = df[g_cols].to_numpy()
            self.g_min = float(g_values.min())
            self.g_max = float(g_values.max())

        return self

    # === POTENTIAL ===
    def scale_potential(self, U):
        if self.U_min is None or self.U_max is None:
            raise ValueError("Scaler not fitted for potential.")
        return 2 * (U - self.U_min) / (self.U_max - self.U_min) - 1

    def unscale_potential(self, U_scaled):
        if self.U_min is None or self.U_max is None:
            raise ValueError("Scaler not fitted for potential.")
        return (U_scaled + 1) * 0.5 * (self.U_max - self.U_min) + self.U_min

    # === ACCELERATION ===
    def scale_gravity(self, g):
        """Scale g directly using its min/max."""
        if self.g_min is None or self.g_max is None:
            raise ValueError("Scaler not fitted for gravity.")
        return 2 * (g - self.g_min) / (self.g_max - self.g_min) - 1

    def unscale_gravity(self, g_scaled):
        """Inverse of scale_gravity."""
        return (g_scaled + 1) * 0.5 * (self.g_max - self.g_min) + self.g_min

    def unscale_acceleration_from_potential(self, grads, lat, r=None):
        """
        grads: (dU_dlon, dU_dlat, [dU_dr]) from model space.
        Converts to physical accelerations using potential scaling.
        """
        if self.U_min is None or self.U_max is None:
            raise ValueError("Scaler not fitted for potential (required for indirect g).")

        S = 0.5 * (self.U_max - self.U_min)
        deg2rad = np.pi / 180.0
        dU_dlon, dU_dlat, *rest = grads

        if r is None:
            r_phys = torch.tensor(self.r_scale, dtype=dU_dlon.dtype, device=dU_dlon.device)
        else:
            r_phys = r

        lat_rad = lat * deg2rad
        dU_dlon_phys = S * (deg2rad / (r_phys * torch.cos(lat_rad))) * dU_dlon
        dU_dlat_phys = S * (deg2rad / r_phys) * dU_dlat

        if rest:
            dU_dr = rest[0]
            dU_dr_phys = S / self.r_scale * dU_dr if dU_dr is not None else torch.zeros_like(dU_dlon)
        else:
            dU_dr_phys = None

        return dU_dlon_phys, dU_dlat_phys, dU_dr_phys

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
          - "U_g_direct"     : predict potential and g directly (multi-output)
          - "U_g_indirect"   : predict potential and derive g = -∇U
        """
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.mode = mode
        self.lmax = lmax
        self.normalization = normalization
        self.out_features = int(out_features)
        self.cache_path = cache_path

        # --- Embedding setup ---
        self.embedding = SHEmbedding(
            lmax=lmax,
            normalization=normalization,
            cache_path=cache_path,
            use_theta_lut=True,
            n_theta=18001,
            exclude_degrees=exclude_degrees
        )

        self.embedding.build_theta_lut()
        dummy_phi = torch.tensor([0.0])
        dummy_lat = torch.tensor([0.0])
        Y_dummy = self.embedding(dummy_phi, dummy_lat)
        in_features = Y_dummy.shape[1]

        # --- Determine network output size ---
        if mode in ["U", "g_indirect"]:
            out_features = 1
        elif mode in ["g_direct"]:
            out_features = 2  # (g_theta, g_phi)
        elif mode in ["U_g_direct", "U_g_indirect"]:
            out_features = 3  # (U, g_theta, g_phi)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        # --- Build SIREN ---
        self.siren = SIRENNet(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

    def forward(self, lon, lat, return_gradients=False, r=None, create_graph=False):
        """
        Forward pass supporting all 5 experimental configurations.
        """
        Y = self.embedding(lon, lat).to(self.device)
        outputs = self.siren(Y)

        # --- MODE 1: Potential only ---
        if self.mode == "U":
            if return_gradients:
                with torch.set_grad_enabled(True):
                    lon = lon.clone().detach().requires_grad_(True)
                    lat = lat.clone().detach().requires_grad_(True)
                    Y = self.embedding(lon, lat).to(self.device)
                    outputs = self.siren(Y)

                    grads = torch.autograd.grad(
                        outputs=outputs,
                        inputs=[lon, lat],
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True,
                        allow_unused=True
                    )
                    grads_phys = self.scaler.unscale_acceleration_from_potential(
                        grads, lat=lat, r=r
                    )
                    return outputs, grads_phys
            else:
                return outputs

        # --- MODE 2: Gravity direct ---
        elif self.mode == "g_direct":
            return outputs  # (g_theta, g_phi)

        # --- MODE 3: Gravity indirect (from potential) ---
        elif self.mode == "g_indirect":
            with torch.set_grad_enabled(True):
                lon = lon.clone().detach().requires_grad_(True)
                lat = lat.clone().detach().requires_grad_(True)

                Y = self.embedding(lon, lat).to(self.device)
                outputs = self.siren(Y)

                grads = torch.autograd.grad(
                    outputs=outputs,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(outputs),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True,
                    allow_unused=True
                )
                grads_phys = self.scaler.unscale_acceleration_from_potential(grads, lat=lat, r=r)
                g_theta = -grads_phys[1]
                g_phi = -grads_phys[0]
            return outputs, (g_theta, g_phi)

        # --- MODE 4: U + g (direct) ---
        elif self.mode == "U_g_direct":
            U_pred = outputs[:, 0:1]
            g_pred = outputs[:, 1:]
            return U_pred, g_pred

        # --- MODE 5: U + g (indirect) ---
        elif self.mode == "U_g_indirect":
            with torch.set_grad_enabled(True):
                lon = lon.clone().detach().requires_grad_(True)
                lat = lat.clone().detach().requires_grad_(True)

                Y = self.embedding(lon, lat).to(self.device)
                outputs = self.siren(Y)

                U_pred = outputs[:, 0:1]
                grads = torch.autograd.grad(
                    outputs=U_pred,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(U_pred),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True,
                    allow_unused=True
                )
                grads_phys = self.scaler.unscale_acceleration_from_potential(grads, lat=lat, r=r)
                g_theta = -grads_phys[1]
                g_phi = -grads_phys[0]
            return U_pred, (g_theta, g_phi)

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")

class Gravity(pl.LightningModule):
    def __init__(self, model_cfg, scaler, lr=1e-4):
        super().__init__()
        self.model = SH_SIREN(**model_cfg)
        self.scaler = scaler
        self.mode = model_cfg.get("mode", "U")
        self.criterion = nn.MSELoss()
        self.lr = lr  # 🔹 store user-defined learning rate

    def forward(self, lon, lat):
        return self.model(lon, lat)

    def _compute_loss(self, y_pred, y_true):
        if self.mode == "U":
            return self.criterion(y_pred.view(-1), y_true.view(-1))

        elif self.mode == "g_direct":
            return self.criterion(y_pred.view(-1), y_true.view(-1))

        elif self.mode == "g_indirect":
            U_pred, (g_theta, g_phi) = y_pred
            g_pred = torch.stack([g_theta, g_phi], dim=1)
            return self.criterion(g_pred.view(-1), y_true.view(-1))

        elif self.mode == "U_g_direct":
            U_pred, g_pred = y_pred
            U_true = y_true[:, 0:1]
            g_true = y_true[:, 1:]
            loss_U = self.criterion(U_pred.view(-1), U_true.view(-1))
            loss_g = self.criterion(g_pred.reshape(-1), g_true.reshape(-1))
            return loss_U + loss_g

        elif self.mode == "U_g_indirect":
            U_pred, (g_theta, g_phi) = y_pred
            U_true = y_true[:, 0:1]
            g_true = y_true[:, 1:]
            g_pred = torch.stack([g_theta, g_phi], dim=1)
            loss_U = self.criterion(U_pred.view(-1), U_true.view(-1))
            loss_g = self.criterion(g_pred.reshape(-1), g_true.reshape(-1))
            return loss_U + loss_g

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def training_step(self, batch):
        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)
        loss = self._compute_loss(y_pred_b, y_true_b)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)
        val_loss = self._compute_loss(y_pred_b, y_true_b)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        """Use externally provided learning rate."""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
