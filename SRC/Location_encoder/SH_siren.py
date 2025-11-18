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
    Handles scaling of potential U and/or gravity g using MEAN/STD scaling.
    This avoids the catastrophic gradient amplification created by min–max scaling.
    """

    def __init__(self, mode="U", r_scale=6.371e6):
        self.mode = mode
        self.r_scale = r_scale

        # Potential stats
        self.U_mean = None
        self.U_std  = None

        # Gravity stats (if direct g training)
        self.g_mean = None
        self.g_std  = None

    # === FITTING ===
    def fit(self, df):
        # Potential always needs standardization if used at all
        if self.mode in ["U", "g_indirect", "U_g_indirect", "U_g_direct"]:
            U = df["dU_m2_s2"].to_numpy()   # in m²/s²
            self.U_mean = float(U.mean())
            self.U_std  = float(U.std())

        if self.mode in ["g_direct", "U_g_direct"]:
            g_cols = ["dg_theta_mGal", "dg_phi_mGal"]
            g_vals = df[g_cols].to_numpy()
            self.g_mean = g_vals.mean(axis=0)   # shape (2,)
            self.g_std  = g_vals.std(axis=0)    # shape (2,)

        return self

    # === POTENTIAL ===
    def scale_potential(self, U):
        """Standardize U."""
        if self.U_mean is None:
            raise ValueError("Scaler not fitted for potential.")
        return (U - self.U_mean) / self.U_std

    def unscale_potential(self, U_scaled):
        """Inverse of standardization."""
        return U_scaled * self.U_std + self.U_mean

    # === DIRECT GRAVITY ===
    def scale_gravity(self, g):
        """Standardize g (theta, phi)."""
        if self.g_mean is None:
            raise ValueError("Scaler not fitted for gravity.")
        return (g - self.g_mean) / self.g_std

    def unscale_gravity(self, g_scaled):
        """Inverse standardization."""
        return g_scaled * self.g_std + self.g_mean

    # === INDIRECT GRAVITY (FROM POTENTIAL GRADS) ===
    def unscale_acceleration_from_potential(self, grads, lat, r=None):
        """
        grads = (dU_dlon_scaled, dU_dlat_scaled, [dU_dr_scaled])
        These gradients come from the NN output in STANDARDIZED POTENTIAL space.

        Converts them into PHYSICAL accelerations in m/s².
        """

        if self.U_mean is None or self.U_std is None:
            raise ValueError("Scaler not fitted for potential (required for indirect g).")

        # recover the scaling factor
        S = self.U_std   # <<<<< CRITICAL FIX: THIS IS THE ONLY SCALING YOU APPLY

        dU_dlon_scaled, dU_dlat_scaled, *rest = grads
        deg2rad = np.pi / 180.0

        # radius
        if r is None:
            r_phys = torch.tensor(self.r_scale, dtype=dU_dlon_scaled.dtype,
                                  device=dU_dlon_scaled.device)
        else:
            r_phys = r

        # convert lat to radians
        lat_rad = lat * deg2rad

        # === Convert angular derivatives to physical gradients ===
        # dU/dλ = (std_U) * (1/(r cos φ)) * dU_scaled/dlon
        dU_dlon_phys = S * (deg2rad / (r_phys * torch.cos(lat_rad))) * dU_dlon_scaled

        # dU/dφ = (std_U) * (1/r) * dU_scaled/dlat
        dU_dlat_phys = S * (deg2rad / r_phys) * dU_dlat_scaled

        # Radial derivative (if present)
        if rest:
            dU_dr_scaled = rest[0]
            dU_dr_phys = S * (1.0 / self.r_scale) * dU_dr_scaled
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
                    lon = lon.to(self.device).requires_grad_(True)
                    lat = lat.to(self.device).requires_grad_(True)
                    Y = self.embedding(lon, lat)
                    outputs = self.siren(Y)

                    grads = torch.autograd.grad(
                        outputs=outputs,
                        inputs=[lon, lat],
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=self.training,
                        retain_graph=self.training,
                        only_inputs=True
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
                lon = lon.to(self.device).requires_grad_(True)
                lat = lat.to(self.device).requires_grad_(True)

                Y = self.embedding(lon, lat)
                outputs = self.siren(Y)

                grads = torch.autograd.grad(
                    outputs=outputs,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(outputs),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True
                )
                grads_phys = self.scaler.unscale_acceleration_from_potential(grads, lat=lat, r=r)
                g_theta = -grads_phys[1]*1e5
                g_phi = -grads_phys[0]*1e5
            return outputs, (g_theta, g_phi)

        # --- MODE 4: U + g (direct) ---
        elif self.mode == "U_g_direct":
            U_pred = outputs[:, 0:1]
            g_pred = outputs[:, 1:]
            return U_pred, g_pred

        # --- MODE 5: U + g (indirect) ---
        elif self.mode == "U_g_indirect":
            with torch.set_grad_enabled(True):
                lon = lon.to(self.device).requires_grad_(True)
                lat = lat.to(self.device).requires_grad_(True)

                Y = self.embedding(lon, lat)
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
                g_theta = -grads_phys[1]*1e5
                g_phi = -grads_phys[0]*1e5
            return U_pred, (g_theta, g_phi)

        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")

class Gravity(pl.LightningModule):
    def __init__(self, model_cfg, scaler, lr=1e-4):
        super().__init__()
        self.alpha_U = model_cfg.pop("alpha_U", 0.9)
        self.model = SH_SIREN(**model_cfg)
        self.scaler = scaler
        self.mode = model_cfg.get("mode", "U")
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, lon, lat):
        return self.model(lon, lat)

    def _compute_loss(self, y_pred, y_true, return_components=False):
        if self.mode == "U":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            return (loss, None, loss) if return_components else loss

        elif self.mode == "g_direct":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            return (None, loss, loss) if return_components else loss

        elif self.mode == "g_indirect":
            U_pred, (g_theta, g_phi) = y_pred
            g_pred = torch.stack([g_theta, g_phi], dim=1)
            loss = self.criterion(g_pred.view(-1), y_true.view(-1))
            return (None, loss, loss) if return_components else loss

        elif self.mode in ["U_g_direct", "U_g_indirect"]:
            # Extract predictions
            if self.mode == "U_g_direct":
                U_pred, g_pred = y_pred
            else:
                # indirect: unpack gradients and stack
                U_pred, (g_theta, g_phi) = y_pred
                g_pred = torch.stack([g_theta, g_phi], dim=1)

            # Extract true values
            U_true = y_true[:, 0:1]
            g_true = y_true[:, 1:]

            # Compute component losses
            loss_U = self.criterion(U_pred.view(-1), U_true.view(-1))
            loss_g = self.criterion(g_pred.reshape(-1), g_true.reshape(-1))

            # Combined loss
            loss = self.alpha_U * loss_U + (1 - self.alpha_U) * loss_g

            return (loss_U, loss_g, loss) if return_components else loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def training_step(self, batch):
        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)

        # get separated losses
        loss_U, loss_g, total_loss = self._compute_loss(y_pred_b, y_true_b, return_components=True)

        # log total loss
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # log components only if they exist (mode includes U and g)
        if loss_U is not None:
            self.log("train_U_loss", loss_U, on_step=False, on_epoch=True)
        if loss_g is not None:
            self.log("train_g_loss", loss_g, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch):
        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)

        loss_U, loss_g, total_loss = self._compute_loss(y_pred_b, y_true_b, return_components=True)

        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)

        if loss_U is not None:
            self.log("val_U_loss", loss_U, on_epoch=True)
        if loss_g is not None:
            self.log("val_g_loss", loss_g, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        """Use externally provided learning rate."""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
