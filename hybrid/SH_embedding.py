"""Creation of spherical harmonic basis functions with pyshtools and store them in cache for training/test data
jhonr"""

import numpy as np
import pyshtools as pysh
import multiprocessing as mp
import os
import torch

# Derivation of SH basis functions using the pyshtools library
# Creation of LUT tables for the datasets based on real SH expansions
# If no cache path is specified, the basis functions are calculated on the fly, being slower
# Inspired by the tutorials exposed in https://shtools.github.io/SHTOOLS/python-examples.html
def _amp_worker(th_deg, lmax, normalization):
    ylm = pysh.expand.spharm(lmax, th_deg, 0.0, normalization=normalization, kind="real")
    a_list = []
    for l in range(lmax + 1):
        a_l = [ylm[0, l, m] for m in range(l + 1)]
        a_list.extend(a_l)
    return np.array(a_list, dtype=np.float32)

class SHEmbedding:
    def __init__(self, lmax=10, normalization="4pi", cache_path=None,
                 use_theta_lut=True, n_theta=18001, exclude_degrees=None):
        self.lmax = lmax
        self.normalization = normalization
        self.cache_path = cache_path
        self.use_theta_lut = use_theta_lut
        if self.lmax == 0:
            self.use_theta_lut = False
        if cache_path is None:
            self.use_theta_lut = False
        self.n_theta = n_theta
        self.exclude_degrees = exclude_degrees or []

        self.a_lut = None
        self.theta_grid = None

    def _lut_paths(self):
        base, ext = os.path.splitext(self.cache_path or "cache_amp")
        if not ext:
            ext = ".npy"
        lut_file = f"{base}_ampsLUT_lmax{self.lmax}_n{self.n_theta}{ext}"
        grid_file = f"{base}_theta_grid_n{self.n_theta}{ext}"
        return lut_file, grid_file

    def build_theta_lut(self, force=False):
        """Build a θ-LUT once: a_lut[n_theta, S_amp], theta_grid[n_theta]."""
        lut_file, grid_file = self._lut_paths()
        if (not force) and os.path.exists(lut_file) and os.path.exists(grid_file):
            self.a_lut = torch.from_numpy(np.load(lut_file)).float()
            self.theta_grid = torch.from_numpy(np.load(grid_file)).float()
            return

        theta_grid = torch.linspace(0.0, torch.pi, self.n_theta)  # [0, π]
        theta_deg = (theta_grid * 180.0 / np.pi).cpu().numpy()

        n_proc = min(os.cpu_count(), 8)
        with mp.Pool(processes=n_proc) as pool:
            results = pool.starmap(
                _amp_worker,
                [(thd, self.lmax, self.normalization) for thd in theta_deg]
            )
        a_lut = np.vstack(results).astype(np.float32)
        np.save(lut_file, a_lut)
        np.save(grid_file, theta_grid.cpu().numpy())

        self.a_lut = torch.from_numpy(a_lut).float()
        self.theta_grid = theta_grid.float()

    @torch.no_grad()
    def _prepare_lut(self, device):
        if self.a_lut is None or self.theta_grid is None:
            self.build_theta_lut()
        self._a_lut_dev = self.a_lut.to(device, non_blocking=True)
        self._theta_grid_dev = self.theta_grid.to(device, non_blocking=True)

    def _interp_a_theta(self, theta_rad):
        a_lut = self._a_lut_dev
        grid = self._theta_grid_dev

        th = torch.clamp(theta_rad, 0.0, torch.pi - 1e-12)

        idx_hi = torch.searchsorted(grid, th, right=False)
        idx_hi = torch.clamp(idx_hi, 1, grid.numel() - 1)
        idx_lo = idx_hi - 1

        th0 = grid[idx_lo]
        th1 = grid[idx_hi]
        denom = (th1 - th0)

        denom = torch.where(denom > 0, denom, torch.ones_like(denom))

        t = (th - th0) / denom

        a0 = a_lut[idx_lo, :]
        a1 = a_lut[idx_hi, :]

        a = a0 + t.unsqueeze(1) * (a1 - a0)
        return a

    def forward(self, lon, lat):

        lon = lon.float()
        lat = lat.float()

        if self.lmax == 0:
            lon_rad = torch.deg2rad(lon)
            lat_rad = torch.deg2rad(lat)

            lon_norm = lon_rad / torch.pi  # [-1,1]
            lat_norm = lat_rad / (0.5 * torch.pi)  # [-1,1]

            return torch.stack([lon_norm, lat_norm], dim=-1)

        device = lon.device

        phi = torch.deg2rad(lon)
        theta = torch.deg2rad(90.0 - lat)

        if self.use_theta_lut:
            self._prepare_lut(device)
            a_pack = self._interp_a_theta(theta)
        else:
            a_pack = self._amps_onthefly(theta, device=device)

        y = self._assemble_torch(phi, a_pack, self.lmax)
        return y

    def _amps_onthefly(self, theta_rad: torch.Tensor, device=None, chunk=4096):
        theta_deg = (theta_rad.detach().cpu().numpy() * 180.0 / np.pi).astype(np.float32)
        n = int(theta_deg.shape[0])
        s = (self.lmax + 1) * (self.lmax + 2) // 2

        out = np.empty((n, s), dtype=np.float32)

        for i in range(0, n, chunk):
            th = theta_deg[i:i + chunk]
            n = int(th.shape[0])

            ylm = pysh.expand.spharm(
                self.lmax, th, 0.0,
                normalization=self.normalization, kind="real"
            )

            blocks = []
            for l in range(self.lmax + 1):
                a = ylm[0, l, 0:l + 1]  # unknown shape depending on pysh

                a = np.asarray(a, dtype=np.float32)

                # Normalize to shape (n, l+1)
                if a.ndim == 1:
                    # could be (l+1,) when n==1
                    a = a.reshape(1, -1)
                elif a.ndim == 2:
                    # could be (n, l+1) or (l+1, n)
                    if a.shape[0] == (l + 1) and a.shape[1] == n:
                        a = a.T
                    elif a.shape[0] == n and a.shape[1] == (l + 1):
                        pass
                    else:
                        raise ValueError(
                            f"Unexpected slice shape for l={l}: {a.shape}, n={n}"
                        )
                else:
                    raise ValueError(
                        f"Unexpected ndim for l={l}: {a.ndim}, shape={a.shape}"
                    )

                blocks.append(a)

            a = np.concatenate(blocks, axis=1).astype(np.float32)  # (n, S)
            out[i:i + n] = a

        a_t = torch.from_numpy(out).float()
        if device is not None:
            a_t = a_t.to(device, non_blocking=True)
        return a_t

    def _assemble_torch(self, phi, a_pack, lmax):
        """
        Vectorized assembly of real spherical harmonics.
        """
        device = phi.device
        n = a_pack.shape[0]

        if self.exclude_degrees:
            y = torch.empty((n, (lmax + 1) ** 2), dtype=torch.float32, device=device)

            m_grid = torch.arange(lmax + 1, dtype=torch.float32, device=device)
            sin_mphi = torch.sin(phi[:, None] * m_grid[None, :])
            cos_mphi = torch.cos(phi[:, None] * m_grid[None, :])

            col = 0
            amp_col = 0
            for l in range(lmax + 1):
                if l in self.exclude_degrees:
                    amp_col += (l + 1)
                    continue

                a_l = a_pack[:, amp_col:amp_col + (l + 1)]
                amp_col += (l + 1)

                for m in range(l, 0, -1):
                    y[:, col] = a_l[:, m] * sin_mphi[:, int(m)]
                    col += 1
                y[:, col] = a_l[:, 0]
                col += 1
                for m in range(1, l + 1):
                    y[:, col] = a_l[:, m] * cos_mphi[:, int(m)]
                    col += 1

            y = y[:, :col]
            return y

        y = torch.empty((n, (lmax + 1) ** 2), dtype=torch.float32, device=device)

        m_grid = torch.arange(lmax + 1, dtype=torch.float32, device=device)
        sin_mphi = torch.sin(phi[:, None] * m_grid[None, :])
        cos_mphi = torch.cos(phi[:, None] * m_grid[None, :])

        amp_col = 0

        for l in range(lmax + 1):

            a_l = a_pack[:, amp_col:amp_col + (l + 1)]
            amp_col += (l + 1)

            start = l * l

            if l > 0:
                sin_amp = torch.flip(a_l[:, 1:], dims=[1])
                sin_trig = torch.flip(sin_mphi[:, 1:l + 1], dims=[1])
                y[:, start:start + l] = sin_amp * sin_trig

            y[:, start + l] = a_l[:, 0]

            if l > 0:
                cos_amp = a_l[:, 1:]
                cos_trig = cos_mphi[:, 1:l + 1]
                y[:, start + l + 1:start + l + 1 + l] = cos_amp * cos_trig

        return y

    __call__ = forward