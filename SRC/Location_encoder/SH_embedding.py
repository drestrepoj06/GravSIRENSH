"""Creation of spherical harmonic basis functions with pyshtools and store them in cache
jhonr"""

import numpy as np
import pyshtools as pysh
import multiprocessing as mp
import os
import torch


def _amp_worker(th_deg, lmax, normalization):
    # A_{l,m}(theta) = N_{l,m} P_l^m(cos theta), evaluated at phi=0 so cos-branch holds amplitudes
    ylm = pysh.expand.spharm(lmax, th_deg, 0.0, normalization=normalization, kind="real")
    # Pack [A_{0,0}, A_{1,0}, A_{1,1}, ..., A_{L,0},...,A_{L,L}]
    A_list = []
    for l in range(lmax + 1):
        A_l = [ylm[0, l, m] for m in range(l + 1)]  # cos-branch m=0..l
        A_list.extend(A_l)
    return np.array(A_list, dtype=np.float32)


class SHEmbedding:
    def __init__(self, lmax=10, normalization="4pi", cache_path=None,
                 use_theta_lut=True, n_theta=18001):
        self.lmax = lmax
        self.normalization = normalization
        self.cache_path = cache_path
        self.use_theta_lut = use_theta_lut
        self.n_theta = n_theta

        self.A_lut = None
        self.theta_grid = None

    def _lut_paths(self):
        base, ext = os.path.splitext(self.cache_path or "cache_amp")
        if not ext:
            ext = ".npy"
        lut_file = f"{base}_ampsLUT_lmax{self.lmax}_n{self.n_theta}{ext}"
        grid_file = f"{base}_theta_grid_n{self.n_theta}{ext}"
        return lut_file, grid_file

    def build_theta_lut(self, force=False):
        """Build a θ-LUT once: A_lut[n_theta, S_amp], theta_grid[n_theta]."""
        lut_file, grid_file = self._lut_paths()
        if (not force) and os.path.exists(lut_file) and os.path.exists(grid_file):
            self.A_lut = torch.from_numpy(np.load(lut_file)).float()
            self.theta_grid = torch.from_numpy(np.load(grid_file)).float()
            return

        print(f"⚙️ Building θ-LUT for lmax={self.lmax} with n_theta={self.n_theta} ...")
        theta_grid = torch.linspace(0.0, torch.pi, self.n_theta)  # [0, π]
        theta_deg = (theta_grid * 180.0 / np.pi).cpu().numpy()

        n_proc = min(os.cpu_count(), 8)
        with mp.Pool(processes=n_proc) as pool:
            results = pool.starmap(
                _amp_worker,
                [(thd, self.lmax, self.normalization) for thd in theta_deg]
            )
        A_lut = np.vstack(results).astype(np.float32)
        np.save(lut_file, A_lut)
        np.save(grid_file, theta_grid.cpu().numpy())
        print(f"💾 Saved LUT: {lut_file} and grid: {grid_file}")

        self.A_lut = torch.from_numpy(A_lut).float()
        self.theta_grid = theta_grid.float()

    @torch.no_grad()
    def _prepare_lut(self, device):
        if self.A_lut is None or self.theta_grid is None:
            self.build_theta_lut()
        self._A_lut_dev = self.A_lut.to(device, non_blocking=True)
        self._theta_grid_dev = self.theta_grid.to(device, non_blocking=True)

    def _interp_A_theta(self, theta_rad):
        """
        Differentiable linear interpolation of A(theta) from θ-LUT.
        theta_rad: (N,) radians in [0, π]
        Returns A_batch: (N, S_amp)
        """
        A_lut = self._A_lut_dev
        grid = self._theta_grid_dev

        # Clamp theta to grid
        th = torch.clamp(theta_rad, 0.0, torch.pi - 1e-12)

        # searchsorted upper index
        # Note: torch.searchsorted requires grid 1D ascending
        idx_hi = torch.searchsorted(grid, th, right=False)
        idx_hi = torch.clamp(idx_hi, 1, grid.numel() - 1)
        idx_lo = idx_hi - 1

        th0 = grid[idx_lo]     # (N,)
        th1 = grid[idx_hi]     # (N,)
        denom = (th1 - th0)
        # safe to avoid /0 if two grid points equal (should not happen with linspace)
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))

        t = (th - th0) / denom  # (N,)

        # Gather rows
        A0 = A_lut[idx_lo, :]   # (N, S_amp)
        A1 = A_lut[idx_hi, :]   # (N, S_amp)

        # Linear interpolation (differentiable wrt th via t)
        A = A0 + t.unsqueeze(1) * (A1 - A0)
        return A

    def forward(self, lon, lat):
        """
        If use_theta_lut=True -> differentiable wrt lat via LUT interpolation.
        Else, fall back to per-sample cache (needs idx), not differentiable wrt lat.
        """
        lon = lon.to(torch.float32)
        lat = lat.to(torch.float32)
        device = lon.device

        phi = torch.deg2rad(lon)                   # (N,)
        theta = torch.deg2rad(90.0 - lat)          # (N,)

        if self.use_theta_lut:
            self._prepare_lut(device)
            A_pack = self._interp_A_theta(theta)   # (N, S_amp) differentiable

        Y = self._assemble_torch(phi, A_pack, self.lmax)
        return Y

    def _assemble_torch(self, phi, A_pack, lmax):
        N = A_pack.shape[0]
        Y = torch.empty((N, (lmax + 1) ** 2), dtype=torch.float32, device=phi.device)
        m_grid = torch.arange(lmax + 1, dtype=torch.float32, device=phi.device)  # 0..L
        sin_mphi = torch.sin(phi[:, None] * m_grid[None, :])
        cos_mphi = torch.cos(phi[:, None] * m_grid[None, :])

        col = 0
        amp_col = 0
        for l in range(lmax + 1):
            A_l = A_pack[:, amp_col:amp_col + (l + 1)]  # (N, l+1)
            amp_col += (l + 1)
            for m in range(l, 0, -1):
                Y[:, col] = A_l[:, m] * sin_mphi[:, int(m)]
                col += 1
            Y[:, col] = A_l[:, 0]; col += 1
            for m in range(1, l + 1):
                Y[:, col] = A_l[:, m] * cos_mphi[:, int(m)]
                col += 1
        return Y

    __call__ = forward