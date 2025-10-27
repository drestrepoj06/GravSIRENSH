"""Creation of spherical harmonic basis functions with pyshtools and store them in cache
jhonr"""

import numpy as np
import pandas as pd
import pyshtools as pysh
import multiprocessing as mp
import os


def _sh_worker(th, ph, lmax, normalization):
    """Worker for parallel spherical harmonic computation (single point)."""
    ylm = pysh.expand.spharm(lmax, th, ph, normalization=normalization, kind="real")
    y_list = []
    for l in range(lmax + 1):
        for m in range(l, 0, -1):
            y_list.append(ylm[1, l, m])
        y_list.append(ylm[0, l, 0])
        for m in range(1, l + 1):
            y_list.append(ylm[0, l, m])
    return np.array(y_list, dtype=np.float32)


class SHEmbedding:
    def __init__(self, lmax: int = 10, normalization: str = "4pi", cache_path: str = None):
        self.lmax = lmax
        self.normalization = normalization
        self.cache_path = cache_path

    # -------------------------------------------------------------------------
    # 1. Merge chunks safely
    # -------------------------------------------------------------------------
    def merge_chunks(self, chunk_files, final_path=None, delete_after=False):
        """Safely merge partial .npy chunks into one valid file (memory-mapped)."""
        if final_path is None:
            base, ext = os.path.splitext(self.cache_path or "cache_basis")
            if not ext:
                ext = ".npy"
            final_path = f"{base}_lmax{self.lmax}{ext}"

        print(f"🧩 Merging {len(chunk_files)} chunks into {final_path} ...")

        # inspect first chunk
        first = np.load(chunk_files[0], mmap_mode="r")
        n_cols = first.shape[1]
        dtype = first.dtype
        total_rows = sum(np.load(f, mmap_mode="r").shape[0] for f in chunk_files)

        # allocate output memmap
        Y = np.lib.format.open_memmap(final_path, mode="w+", dtype=dtype, shape=(total_rows, n_cols))

        # sequential copy
        offset = 0
        for i, f in enumerate(chunk_files):
            arr = np.load(f, mmap_mode="r")
            rows = arr.shape[0]
            Y[offset:offset + rows] = arr
            offset += rows
            print(f"   ✅ Chunk {i + 1}/{len(chunk_files)} merged ({rows} rows)")
        del Y  # flush to disk

        # optionally delete temporary chunks
        if delete_after:
            for f in chunk_files:
                try:
                    os.remove(f)
                except OSError:
                    pass

        print(f"💾 Merged cache saved to {final_path}")
        return final_path

    def from_dataframe(self, df: pd.DataFrame, lon_col="lon", lat_col="lat",
                       use_cache=True, parallel=True, vectorized=False,
                       chunked=False, chunk_size=500_000, delete_after=True):
        """Compute or load SH basis from a DataFrame of lon/lat points."""
        lon = df[lon_col].values
        lat = df[lat_col].values

        base, ext = os.path.splitext(self.cache_path or "cache_basis")
        if not ext:
            ext = ".npy"
        cache_file = f"{base}_lmax{self.lmax}{ext}"

        # load cache if available
        if use_cache:
            # Look for exact match first
            if os.path.exists(cache_file):
                print(f"📂 Loading cached SH basis from {cache_file}")
                return np.load(cache_file, mmap_mode="r")

            # Look for higher-degree cache to slice from
            base, ext = os.path.splitext(self.cache_path or "cache_basis")
            for l in range(self.lmax + 1, 2000):  # scan for possible larger files
                candidate = f"{base}_lmax{l}{ext}"
                if os.path.exists(candidate):
                    print(f"🔎 Found higher-degree cache ({candidate}). Slicing down to lmax={self.lmax}.")
                    Y_full = np.load(candidate, mmap_mode="r")
                    n_basis = (self.lmax + 1) ** 2
                    return Y_full[:, :n_basis]

        # --- chunked mode ---
        if chunked:
            n = len(df)
            chunks = (n + chunk_size - 1) // chunk_size
            print(f"🧩 Chunked computation mode: {chunks} chunks (~{chunk_size:,} each)")

            chunk_files = []
            for i in range(chunks):
                print(f"🧩 Processing chunk {i + 1}/{chunks}")
                df_chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
                cache_i = f"{base}_part{i:02d}_lmax{self.lmax}{ext}"
                Y = self._compute_basis(df_chunk[lon_col].values,
                                        df_chunk[lat_col].values,
                                        parallel=parallel,
                                        vectorized=vectorized)
                np.save(cache_i, Y.astype(np.float32))
                chunk_files.append(cache_i)
                del Y

            merged_file = self.merge_chunks(chunk_files, final_path=cache_file, delete_after=delete_after)
            print(f"✅ Final merged cache: {merged_file}")
            return np.load(cache_file, mmap_mode="r")

        # --- single-shot mode ---
        print(f"⚙️ Computing SH basis for {len(df):,} samples (lmax={self.lmax}) ...")
        Y = self._compute_basis(lon, lat, parallel=parallel, vectorized=vectorized)
        np.save(cache_file, Y.astype(np.float32))
        print(f"💾 Cached SH basis saved to {cache_file}")
        return Y

    # -------------------------------------------------------------------------
    # 3. Computation backends
    # -------------------------------------------------------------------------
    def _compute_basis(self, lon, lat, parallel=True, vectorized=False):
        if vectorized:
            return self._compute_basis_vectorized(lon, lat)
        elif parallel:
            return self._compute_basis_parallel(lon, lat)
        else:
            return self._compute_basis_serial(lon, lat)

    def _compute_basis_serial(self, lon, lat):
        n_points = len(lat)
        n_basis = (self.lmax + 1) ** 2
        Y = np.zeros((n_points, n_basis), dtype=np.float32)
        theta = 90.0 - lat
        for i, (th, ph) in enumerate(zip(theta, lon)):
            ylm = pysh.expand.spharm(self.lmax, th, ph,
                                     normalization=self.normalization,
                                     kind="real")
            y_list = []
            for l in range(self.lmax + 1):
                for m in range(l, 0, -1):
                    y_list.append(ylm[1, l, m])
                y_list.append(ylm[0, l, 0])
                for m in range(1, l + 1):
                    y_list.append(ylm[0, l, m])
            Y[i, :] = np.array(y_list, dtype=np.float32)
        return Y

    def _compute_basis_parallel(self, lon, lat):
        theta = 90.0 - lat
        n_proc = min(4, os.cpu_count())
        with mp.Pool(processes=n_proc) as pool:
            results = pool.starmap(
                _sh_worker,
                [(th, ph, self.lmax, self.normalization) for th, ph in zip(theta, lon)]
            )
        return np.vstack(results).astype(np.float32)

    def _compute_basis_vectorized(self, lon, lat):
        """Vectorized evaluation for regular grids."""
        unique_lats = np.unique(lat)
        unique_lons = np.unique(lon)
        n_lat, n_lon = len(unique_lats), len(unique_lons)
        if n_lat * n_lon != len(lat):
            print("⚠️ Lat/Lon not on a regular grid — falling back to parallel mode.")
            return self._compute_basis_parallel(lon, lat)

        Y_grid = []
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                Ylm = pysh.expand.spharm(
                    l, 90 - unique_lats[:, None], unique_lons[None, :],
                    normalization=self.normalization, kind="real"
                )
                Y_grid.append(Ylm.flatten())
        return np.stack(Y_grid, axis=1).astype(np.float32)

