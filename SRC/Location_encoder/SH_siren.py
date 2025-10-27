"""
SH-SIREN: Combines differentiable Spherical Harmonics embedding with a SIREN network.
jhonr
"""

import torch
import torch.nn as nn
from SRC.Location_encoder.SH_embedding import SHEmbedding
from SRC.Location_encoder.Siren import SIRENNet
import numpy as np
import os
import pandas as pd
import glob
import re

# Scaling acceleration outputs in the range [-1, 1],
# based on the scaling preferred for SIRENNETS: https://github.com/vsitzmann/siren
# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/explore_siren.ipynb
class SHSirenScaler:

    def __init__(self, r_scale=None, a_min=None, a_max=None):
        self.r_scale = r_scale
        self.a_min = a_min
        self.a_max = a_max

    def scale_inputs(self, lon, lat, r):
        if self.r_scale:
            r = r / self.r_scale
        return lon, lat, r

    def fit_acceleration(self, a_components):
        a_all = np.concatenate([np.ravel(a) for a in np.atleast_2d(a_components).T], axis=0)
        self.a_min = np.min(a_all)
        self.a_max = np.max(a_all)
        return self

    def scale_acceleration(self, a_components):
        if self.a_min is None or self.a_max is None:
            raise ValueError("Call fit_acceleration() before scaling.")
        return 2 * (a_components - self.a_min) / (self.a_max - self.a_min) - 1

    def unscale_acceleration(self, a_scaled):
        if self.a_min is None or self.a_max is None:
            raise ValueError("Scaler not fitted.")
        return (a_scaled + 1) * 0.5 * (self.a_max - self.a_min) + self.a_min


# Based on the code https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/locationencoder.py
# But only for the encoder SH + Siren network

class SH_SIREN(nn.Module):

    def __init__(self, lmax=10, hidden_features=128, hidden_layers=4, out_features=1,
                 first_omega_0=30.0, hidden_omega_0=1.0, device='cuda',
                 normalization="4pi", cache_path=None, scaler=None):
        super().__init__()
        self.device = device
        self.scaler = scaler
        self.lmax = lmax
        self.normalization = normalization
        self.cache_path = cache_path

        self.embedding = SHEmbedding(lmax=lmax, normalization=normalization, cache_path=cache_path)

        n_basis = (lmax + 1) ** 2
        self.siren = SIRENNet(
            in_features=n_basis,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        ).to(device)

        # ----------------------------------------------------------------------
        # Handle cached SH basis
        # ----------------------------------------------------------------------
        self.Y_cache = None

        if self.cache_path is not None:
            base, ext = os.path.splitext(self.cache_path or "cache_basis")
            if not ext:
                ext = ".npy"

            target_file = f"{base}_lmax{self.lmax}{ext}"
            search_pattern = f"{base}_lmax*.npy"

            cache_files = glob.glob(search_pattern)
            selected_file = None
            lmax_found = None

            if os.path.exists(target_file):
                selected_file = target_file
                lmax_found = self.lmax
                print(f"⚡ Found exact cache for lmax={self.lmax}")

            elif cache_files:
                matches = []
                for f in cache_files:
                    m = re.search(r"_lmax(\d+)", f)
                    if m:
                        matches.append((int(m.group(1)), f))
                matches.sort(key=lambda x: x[0])
                larger = [m for m in matches if m[0] > self.lmax]
                if larger:
                    lmax_found, selected_file = larger[0]
                    print(f"⚡ Found larger cache lmax={lmax_found}. Will slice down to lmax={self.lmax}")

            if selected_file is None:
                print(f"⚠️ No cache available for base '{base}'. "
                      f"It will be generated automatically from the "
                      f"dataset the first time `prepare_input()` is called.")
                self.Y_cache = None
                self.cache_path = target_file  # remember where to save later
            else:
                # Load existing cache (full or sliced)
                mmap_obj = np.load(selected_file, mmap_mode="r")
                n_cols_needed = (self.lmax + 1) ** 2
                n_cols_available = mmap_obj.shape[1]
                if n_cols_needed > n_cols_available:
                    raise ValueError(f"Requested lmax={self.lmax} exceeds columns in cache (lmax={lmax_found})")
                Y = np.array(mmap_obj[:, :n_cols_needed], copy=True).astype(np.float32, copy=False)
                del mmap_obj

                if lmax_found != self.lmax:
                    print(f"💾 Saving sliced cache to: {target_file}")
                    np.save(target_file, Y)

                self.Y_cache = Y
                print(f"✅ Loaded cache: shape={self.Y_cache.shape}")

        else:
            print("⚠️ No cache_path provided. Embeddings will be computed on the fly.")
            self.Y_cache = None

    def prepare_input(self, df, lon_col='lon', lat_col='lat', use_cache=True):
        if isinstance(df, dict):
            if lon_col in df and lat_col in df:
                lon = df[lon_col]
                lat = df[lat_col]
                idx = np.arange(len(lon))
            elif "X" in df:
                X = df["X"]
                lon, lat = X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy()
                idx = np.arange(len(lon))
            elif "orig_index" in df:
                idx = df["orig_index"].cpu().numpy() if torch.is_tensor(df["orig_index"]) else df["orig_index"]
            else:
                raise ValueError("Unrecognized dict format passed to prepare_input()")
        else:
            lon, lat = df[lon_col].values, df[lat_col].values
            idx = df["orig_index"].values if "orig_index" in df.columns else df.index.values

        base, ext = os.path.splitext(self.embedding.cache_path or "cache_basis")
        if not ext:
            ext = ".npy"
        cache_file = f"{base}_lmax{self.lmax}{ext}"

        # ------------------------------------------------------------------
        # Try to use existing cache, otherwise recompute automatically
        # ------------------------------------------------------------------
        if use_cache and os.path.exists(cache_file):
            print(f"📂 Loading cached SH basis from {cache_file}")
            Y = np.load(cache_file, mmap_mode="r")[idx]
        else:
            print(f"⚙️ Cache not found for lmax={self.lmax}. Recomputing SH basis...")
            Y = self.embedding.from_dataframe(
                pd.DataFrame({lon_col: lon, lat_col: lat}),
                lon_col=lon_col,
                lat_col=lat_col,
                use_cache=False
            )

        Y_torch = torch.tensor(Y, dtype=torch.float32, device=self.device)
        return Y_torch

    def forward(self, df=None, Y=None, return_gradients=False):
        if Y is None:
            if df is None:
                raise ValueError("Provide either DataFrame (df) or precomputed SH basis (Y).")
            Y = self.prepare_input(df, use_cache=True)

        if return_gradients:
            raise RuntimeError("Gradients unavailable: SH basis is precomputed and non-differentiable.")
        a_scaled = self.siren(Y)
        return a_scaled



