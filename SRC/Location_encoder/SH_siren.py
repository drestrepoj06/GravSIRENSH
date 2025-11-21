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

        if self.mode in ["g_direct", "U_g_direct", "g_hybrid", "U_g_hybrid", "U_g_indirect"]:
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

        # --- MODE 4: g (hybrid) ---#

        elif self.mode == "g_hybrid":
            # model outputs: [U_pred, g_theta_pred, g_phi_pred]
            U_pred = outputs[:, 0:1]
            g_pred = outputs[:, 1:]  # shape: (N, 2)

            # compute ∇U_pred using autograd
            with torch.set_grad_enabled(True):
                lon = lon.to(self.device).requires_grad_(True)
                lat = lat.to(self.device).requires_grad_(True)

                # recompute U_pred with grad-enabled coords
                Y = self.embedding(lon, lat)
                out2 = self.siren(Y)
                U2 = out2[:, 0:1]

                grads = torch.autograd.grad(
                    outputs=U2,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(U2),
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
        elif self.mode == "U_g_hybrid":

            U_pred = outputs[:, 0:1]
            g_pred = outputs[:, 1:]  # shape (N,2)

            with torch.set_grad_enabled(True):
                lon = lon.to(self.device).requires_grad_(True)
                lat = lat.to(self.device).requires_grad_(True)

                Y = self.embedding(lon, lat)
                out2 = self.siren(Y)
                U2 = out2[:, 0:1]

                grads = torch.autograd.grad(
                    outputs=U2,
                    inputs=[lon, lat],
                    grad_outputs=torch.ones_like(U2),
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
        self.log_sigma_U = torch.nn.Parameter(torch.tensor(0.0))
        self.log_sigma_g = torch.nn.Parameter(torch.tensor(0.0))
        self.log_sigma_consistency = torch.nn.Parameter(torch.tensor(0.0))
        self.scaler = scaler
        self.mode = model_cfg.get("mode", "U")
        if self.mode in ["g_hybrid", "U_g_direct", "U_g_indirect", "U_g_hybrid"]:
            self.register_buffer("U_std_buf", torch.as_tensor(scaler.U_std, dtype=torch.float32))
            self.register_buffer("g_std_buf", torch.as_tensor(scaler.g_std, dtype=torch.float32))
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, lon, lat):
        return self.model(lon, lat)

    def _compute_loss(self, y_pred, y_true, return_components=False):

        if self.mode == "U":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            loss = loss.mean()
            return (loss, None, loss) if return_components else loss

        elif self.mode == "g_direct":
            loss = self.criterion(y_pred.view(-1), y_true.view(-1))
            loss = loss.mean()
            return (None, loss, loss) if return_components else loss

        elif self.mode == "g_indirect":
            U_pred, (g_theta, g_phi) = y_pred
            g_pred = torch.stack([g_theta, g_phi], dim=1)
            loss = self.criterion(g_pred.view(-1), y_true.view(-1))
            loss = loss.mean()
            return (None, loss, loss) if return_components else loss

        elif self.mode == "g_hybrid":
            U_pred = y_pred["U_pred"]  # (N,1)
            g_pred = y_pred["g_pred"]  # (N,2)
            Ugrad_pred = y_pred["g_from_gradU"]  # (N,2)

            g_true = y_true[:, 1:]  # shape (N,2)

            loss_g = self.criterion(
                g_pred.reshape(-1),
                g_true.reshape(-1)
            ).mean()

            loss_consistency = self.criterion(
                Ugrad_pred.reshape(-1),
                g_pred.reshape(-1)
            ).mean()
            sigma_g = torch.exp(self.log_sigma_g)
            sigma_consist = torch.exp(self.log_sigma_consistency)

            loss = (loss_g / (2 * sigma_g ** 2)) \
                   + (loss_consistency / (2 * sigma_consist ** 2)) \
                   + torch.log(sigma_g * sigma_consist)

            if return_components:
                return (loss_g, loss_consistency, loss)

            return loss

        elif self.mode in ["U_g_direct", "U_g_indirect"]:

            # Extract predictions
            if self.mode == "U_g_direct":
                U_pred, g_pred = y_pred
                normalize = False
            else:
                U_pred, (g_theta, g_phi) = y_pred
                g_pred = torch.stack([g_theta, g_phi], dim=1)
                normalize = True

            U_true = y_true[:, 0:1]
            g_true = y_true[:, 1:]

            loss_U = self.criterion(U_pred.view(-1), U_true.view(-1))
            loss_g = self.criterion(g_pred.reshape(-1), g_true.reshape(-1))

            loss_U = loss_U.mean()
            loss_g = loss_g.mean()

            # optional normalization
            if normalize:
                loss_U = loss_U / (self.U_std_buf ** 2)
                loss_g = loss_g / (self.g_std_buf ** 2)

            # learned weighting
            sigma_U = torch.exp(self.log_sigma_U)
            sigma_g = torch.exp(self.log_sigma_g)

            loss = (loss_U / (2 * sigma_U ** 2)) + (loss_g / (2 * sigma_g ** 2))
            loss += torch.log(sigma_U * sigma_g)
            return (loss_U, loss_g, loss) if return_components else loss

        elif self.mode == "U_g_hybrid":

            U_pred, g_pred, (gtheta_from_U, gphi_from_U) = y_pred
            Ugrad_pred = torch.stack([gtheta_from_U, gphi_from_U], dim=1)

            U_true = y_true[:, 0:1]
            g_true = y_true[:, 1:]

            # (1) U_pred vs U_true
            loss_U = self.criterion(U_pred.reshape(-1), U_true.reshape(-1)).mean()

            # (2) g_pred vs g_true
            loss_g = self.criterion(g_pred.reshape(-1), g_true.reshape(-1)).mean()

            # (3) ∇U_pred vs g_pred
            loss_consistency = self.criterion(
                Ugrad_pred.reshape(-1),
                g_pred.reshape(-1)
            ).mean()

            sigma_U = torch.exp(self.log_sigma_U)
            sigma_g = torch.exp(self.log_sigma_g)
            sigma_consist = torch.exp(self.log_sigma_consistency)

            loss = (loss_U / (2 * sigma_U ** 2)) \
                   + (loss_g / (2 * sigma_g ** 2)) \
                   + (loss_consistency / (2 * sigma_consist ** 2)) \
                   + torch.log(sigma_U * sigma_g * sigma_consist)

            if return_components:
                return (loss_U, loss_g, loss_consistency, loss)
            return loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _log_sigmas(self, prefix):

        # sigma_U
        if hasattr(self, "log_sigma_U"):
            sigma_U = torch.exp(self.log_sigma_U)
            self.log(f"{prefix}_sigma_U", sigma_U, on_epoch=True)

        # sigma_g
        if hasattr(self, "log_sigma_g"):
            sigma_g = torch.exp(self.log_sigma_g)
            self.log(f"{prefix}_sigma_g", sigma_g, on_epoch=True)

        # sigma_consistency (only in hybrid modes)
        if hasattr(self, "log_sigma_consistency"):
            sigma_consistency = torch.exp(self.log_sigma_consistency)
            self.log(f"{prefix}_sigma_consistency", sigma_consistency, on_epoch=True)

    def training_step(self, batch, batch_idx=None):

        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)

        loss_components = self._compute_loss(y_pred_b, y_true_b, return_components=True)

        # ---- total loss ----
        total_loss = loss_components[-1]
        if isinstance(total_loss, torch.Tensor) and total_loss.ndim > 0:
            total_loss = total_loss.mean()

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # ---- component losses ----
        component_names = ["U", "g", "consistency"]

        for name, comp in zip(component_names, loss_components[:-1]):
            if comp is not None:
                if isinstance(comp, torch.Tensor) and comp.ndim > 0:
                    comp = comp.mean()  # component fix
                self.log(f"train_{name}_loss", comp, on_step=False, on_epoch=True)

        # ---- sigmas ----
        self._log_sigmas("train")

        return total_loss

    def validation_step(self, batch, batch_idx=None):

        lon_b, lat_b, y_true_b = [b.to(self.device) for b in batch]
        y_pred_b = self.model(lon_b, lat_b)

        loss_components = self._compute_loss(y_pred_b, y_true_b, return_components=True)

        total_loss = loss_components[-1]
        if isinstance(total_loss, torch.Tensor) and total_loss.ndim > 0:
            total_loss = total_loss.mean()  # REQUIRED FIX

        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)

        component_names = ["U", "g", "consistency"]

        for name, comp in zip(component_names, loss_components[:-1]):
            if comp is not None:
                if isinstance(comp, torch.Tensor) and comp.ndim > 0:
                    comp = comp.mean()  # component fix
                self.log(f"val_{name}_loss", comp, on_epoch=True)

        self._log_sigmas("val")

        return total_loss

    def configure_optimizers(self):
        """Use externally provided learning rate."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
