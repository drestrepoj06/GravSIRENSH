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
        if self.mode in ["U", "g_indirect", "U_g_indirect", "U_g_direct", "g_hybrid", "U_g_hybrid"]:
            U = df["dU_m2_s2"].to_numpy()   # in m²/s²
            self.U_mean = float(U.mean())
            self.U_std  = float(U.std())

        if self.mode in ["g_direct", "g_indirect", "U_g_direct", "g_hybrid", "U_g_hybrid", "U_g_indirect"]:
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
        S = self.U_std

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
          - "g_hybrid"      : predict g directly and force the gradient of U to be equal to gpred
          - "U_g_hybrid"    : predict U and g directly and force the gradient of U to be equal to gpred
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
        elif mode in ["U_g_direct", "U_g_indirect", "g_hybrid", "U_g_hybrid"]:
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

    def forward(self, lon, lat, return_gradients=False, r=None):
        """
        Forward pass supporting all 5 experimental configurations.
        """
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
                    return outputs, grads
            else:
                Y = self.embedding(lon, lat).to(self.device)
                outputs = self.siren(Y)
                return outputs

        # --- MODE 2: Gravity direct ---
        elif self.mode == "g_direct":
            Y = self.embedding(lon, lat).to(self.device)
            outputs = self.siren(Y)
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
                g_theta = grads[1]
                g_phi = grads[0]
            return outputs, (g_theta, g_phi)

        elif self.mode == "g_hybrid":
            # compute ∇U_pred using autograd
            with torch.set_grad_enabled(True):
                lon = lon.to(self.device).requires_grad_(True)
                lat = lat.to(self.device).requires_grad_(True)

                # recompute U_pred with grad-enabled coords
                Y = self.embedding(lon, lat).to(self.device)
                outputs = self.siren(Y)
                U_pred = outputs[:, 0:1]
                g_pred = outputs[:, 1:]

                grads = torch.autograd.grad(
                    outputs=U_pred,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(U_pred),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True,
                    allow_unused=True
                )

                grads_phys = self.scaler.unscale_acceleration_from_potential(
                    grads, lat=lat, r=r
                )
                g_theta_from_U = -grads_phys[1] * 1e5
                g_phi_from_U = -grads_phys[0] * 1e5

            # return predicted g, and computed g from ∇U
            return {
                "U_pred": U_pred,
                "g_pred": g_pred,
                "g_from_gradU": torch.stack([g_theta_from_U, g_phi_from_U], dim=-1)
            }

        # --- MODE 5: U + g (direct) ---
        elif self.mode == "U_g_direct":
            Y = self.embedding(lon, lat).to(self.device)
            outputs = self.siren(Y)
            U_pred = outputs[:, 0:1]
            g_pred = outputs[:, 1:]
            return U_pred, g_pred

        # --- MODE 6: U + g (indirect) ---
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

        # --- MODE 7: U + g (hybrid) ---
        elif self.mode == "U_g_hybrid": # shape (N,2)

            with torch.set_grad_enabled(True):
                lon = lon.to(self.device).requires_grad_(True)
                lat = lat.to(self.device).requires_grad_(True)

                Y = self.embedding(lon, lat).to(self.device)
                outputs = self.siren(Y)
                U_pred = outputs[:, 0:1]
                g_pred = outputs[:, 1:]

                grads = torch.autograd.grad(
                    outputs=U_pred,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(U_pred),
                    create_graph=self.training,
                    retain_graph=self.training,
                    only_inputs=True,
                    allow_unused=True,
                )

                grads_phys = self.scaler.unscale_acceleration_from_potential(
                    grads, lat=lat, r=r
                )
                g_theta_from_U = -grads_phys[1] * 1e5
                g_phi_from_U = -grads_phys[0] * 1e5

            return U_pred, g_pred, (g_theta_from_U, g_phi_from_U)
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")

class Gravity(pl.LightningModule):
    def __init__(self, model_cfg, scaler, lr=1e-4):
        super().__init__()
        self.model = SH_SIREN(**model_cfg)
        self.mode = model_cfg.get("mode", "U")
        if self.mode in ["U_g_direct", "U_g_indirect"]:
            self.log_sigma_U = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_g = nn.Parameter(torch.tensor(0.0))

        if self.mode == "g_hybrid":
            # Only g_hybrid needs sigma_g and sigma_consistency
            self.log_sigma_grad = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_g = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_consistency = nn.Parameter(torch.tensor(0.0))

        if self.mode == "U_g_hybrid":
            # U_g_hybrid needs all three
            self.log_sigma_U = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_grad = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_g = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_consistency = nn.Parameter(torch.tensor(0.0))
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

        elif self.mode == "g_hybrid":
            U_pred = y_pred["U_pred"]  # (N,1)
            g_pred = y_pred["g_pred"]  # (N,2)
            Ugrad_pred = y_pred["g_from_gradU"]

            g_true = y_true[:, 1:]

            loss_g = self.criterion(g_pred.reshape(-1),
                                    g_true.reshape(-1))

            loss_grad = self.criterion(Ugrad_pred.reshape(-1),
                                       g_true.reshape(-1))

            loss_consistency = self.criterion(
                Ugrad_pred.reshape(-1),
                g_pred.reshape(-1)
            )

            sigma_g = torch.exp(self.log_sigma_g)
            sigma_grad = torch.exp(self.log_sigma_grad)

            λ = 0.1  # recommended range: 0.01–0.5

            loss = (
                    loss_g / (2 * sigma_g ** 2) +
                    loss_grad / (2 * sigma_grad ** 2) +
                    λ * loss_consistency +
                    torch.log(sigma_g * sigma_grad)
            )

            if return_components:
                return (None, loss_g, loss_grad, loss_consistency, loss)

            return loss

        elif self.mode in ["U_g_direct", "U_g_indirect"]:

            if self.mode == "U_g_direct":
                U_pred, g_pred = y_pred
            else:
                U_pred, (g_theta, g_phi) = y_pred
                g_pred = torch.stack([g_theta, g_phi], dim=1)

            U_true = y_true[:, :1]
            g_true = y_true[:, 1:]

            loss_U = self.criterion(U_pred, U_true)
            loss_g = self.criterion(g_pred, g_true)
            w_U = torch.sigmoid(self.log_sigma_U)
            w_g = torch.sigmoid(self.log_sigma_g)

            W_sum = w_U + w_g
            w_U = w_U / W_sum
            w_g = w_g / W_sum

            loss = w_U * loss_U + w_g * loss_g

            return (loss_U, loss_g, None, None, loss) if return_components else loss

        elif self.mode == "U_g_hybrid":
            U_pred, g_pred, (gtheta_from_U, gphi_from_U) = y_pred
            Ugrad_pred = torch.stack([gtheta_from_U, gphi_from_U], dim=1)
            U_true = y_true[:, 0:1]
            g_true = y_true[:, 1:]
            loss_U = self.criterion(U_pred.reshape(-1), U_true.reshape(-1))
            loss_g = self.criterion(g_pred.reshape(-1), g_true.reshape(-1))
            loss_grad = self.criterion(Ugrad_pred.reshape(-1), g_true.reshape(-1))
            loss_consistency = self.criterion(
                Ugrad_pred.reshape(-1),
                g_pred.reshape(-1)
            )

            sigma_U = torch.exp(self.log_sigma_U)
            sigma_g = torch.clamp(torch.exp(self.log_sigma_g), 1e-4, 0.5)
            sigma_grad = torch.clamp(torch.exp(self.log_sigma_grad), 1e-4, 0.5)
            sigma_consist = torch.clamp(torch.exp(self.log_sigma_consistency), 1e-1, 0.5)

            loss = (
                    loss_U / (2 * sigma_U ** 2)
                    + loss_g / (2 * sigma_g ** 2)
                    + loss_grad / (2 * sigma_grad ** 2)
                    + loss_consistency / (2 * sigma_consist ** 2)
                    + torch.log(sigma_U * sigma_g * sigma_grad * sigma_consist)
            )

            if return_components:
                return (loss_U, loss_g, loss_grad, loss_consistency, loss)
            return loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _log_sigmas(self, prefix):

        if hasattr(self, "log_sigma_U"):
            w_U = torch.sigmoid(self.log_sigma_U)
            self.log(f"{prefix}_w_U", w_U, on_epoch=True)

        if hasattr(self, "log_sigma_g"):
            w_g = torch.sigmoid(self.log_sigma_g)
            self.log(f"{prefix}_w_g", w_g, on_epoch=True)

        if hasattr(self, "log_sigma_grad"):
            w_grad = torch.sigmoid(self.log_sigma_grad)
            self.log(f"{prefix}_w_grad", w_grad, on_epoch=True)

        if hasattr(self, "log_sigma_consistency"):
            w_con = torch.sigmoid(self.log_sigma_consistency)
            self.log(f"{prefix}_w_consistency", w_con, on_epoch=True)

    def training_step(self, batch, batch_idx=None):

        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)

        loss_components = self._compute_loss(y_pred_b, y_true_b, return_components=True)

        total_loss = loss_components[-1]
        if isinstance(total_loss, torch.Tensor) and total_loss.ndim > 0:
            total_loss = total_loss.mean()

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Component names MUST match the returned structure:
        # (loss_U, loss_g, loss_grad, loss_consistency, total_loss)
        component_names = ["U", "g", "grad", "consistency"]

        # Iterate through U, g, grad, consistency (skip total_loss at the end)
        for name, comp in zip(component_names, loss_components[:-1]):
            if comp is not None:
                if isinstance(comp, torch.Tensor) and comp.ndim > 0:
                    comp = comp.mean()
                self.log(f"train_{name}_loss", comp, on_step=False, on_epoch=True)

        # Log all sigmas
        self._log_sigmas("train")
        # Log current learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx=None):

        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)

        # Compute all loss components
        loss_components = self._compute_loss(y_pred_b, y_true_b, return_components=True)

        # Last component is the total loss
        total_loss = loss_components[-1]
        if isinstance(total_loss, torch.Tensor) and total_loss.ndim > 0:
            total_loss = total_loss.mean()

        # Log total validation loss
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)

        # Component names MUST match the returned structure
        component_names = ["U", "g", "grad", "consistency"]

        # Log U, g, grad, consistency
        for name, comp in zip(component_names, loss_components[:-1]):
            if comp is not None:
                if isinstance(comp, torch.Tensor) and comp.ndim > 0:
                    comp = comp.mean()
                self.log(f"val_{name}_loss", comp, on_epoch=True)

        # Log learned sigmas
        self._log_sigmas("val")

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
