import os
import sys
import time
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytical.pyshtools_expansion import Analytical
from numerical.numerical_wrapper import Numerical
from hybrid.SIREN_SH import Hybrid


class RuntimeCoordScaler:
    def __init__(self, lon_min, lon_max, lat_min, lat_max):
        self.lon_min = float(lon_min)
        self.lon_max = float(lon_max)
        self.lat_min = float(lat_min)
        self.lat_max = float(lat_max)

    def scale_coords(self, lonlat_np):
        lon = lonlat_np[..., 0]
        lat = lonlat_np[..., 1]

        lon_scaled = 2.0 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1.0
        lat_scaled = 2.0 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1.0

        return np.stack([lon_scaled, lat_scaled], axis=-1)


def load_runtime_model(run_path, device):
    model_path = os.path.join(run_path, "model_runtime.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing runtime checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")

    arch = ckpt["arch"]
    mode = ckpt["mode"]
    hp = ckpt["model_hparams"]

    if arch == "numerical":

        coord_scaler = ckpt["coord_scaler"]

        scaler = RuntimeCoordScaler(
            lon_min=coord_scaler["lon_min"],
            lon_max=coord_scaler["lon_max"],
            lat_min=coord_scaler["lat_min"],
            lat_max=coord_scaler["lat_max"],
        )

        model = Numerical(
            hidden_features=hp["hidden_features"],
            hidden_layers=hp["hidden_layers"],
            scaler=scaler,
            mode=mode,
        )

    elif arch == "hybrid":

        model = Hybrid(
            lmax=hp["lmax"],
            hidden_features=hp["hidden_features"],
            hidden_layers=hp["hidden_layers"],
            first_omega_0=hp["first_omega_0"],
            hidden_omega_0=hp["hidden_omega_0"],
            scaler=None,
            cache_path=None,
            exclude_degrees=hp.get("exclude_degrees"),
            mode=mode,
            run_dir=run_path
        )

    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    return model, arch, mode, hp

def load_runtime_dataset(test_path, n_samples=10_000, seed=42):
    df = pd.read_parquet(test_path)

    if len(df) < n_samples:
        raise ValueError(
            f"Dataset has only {len(df)} rows, cannot sample {n_samples}."
        )

    df_sample = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    return df_sample


def build_lonlat_tensors(df, device):
    lon = torch.tensor(df["lon"].values, dtype=torch.float32, device=device)
    lat = torch.tensor(df["lat"].values, dtype=torch.float32, device=device)
    return lon, lat

def benchmark_model(model, lon, lat, device, warmup_runs=1, timed_runs=3):
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(lon, lat)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []

    with torch.no_grad():
        for _ in range(timed_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(lon, lat)

            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times, dtype=float)

    return {
        "samples": int(lon.shape[0]),
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "times_s": times.tolist(),
        "mean_s": float(times.mean()),
        "std_s": float(times.std(ddof=0)),
        "min_s": float(times.min()),
        "max_s": float(times.max()),
        "seconds_per_sample_mean": float(times.mean() / lon.shape[0]),
        "samples_per_second_mean": float(lon.shape[0] / times.mean()),
        "device": str(device),
    }

def benchmark_analytical(
    analytical_model,
    df_runtime,
    warmup_runs=1,
    timed_runs=3,
):
    lat = df_runtime["lat"].to_numpy(dtype=np.float64)
    lon = df_runtime["lon"].to_numpy(dtype=np.float64)

    equiv = analytical_model.equiv

    for _ in range(warmup_runs):
        _ = analytical_model.evaluate_on_coords(
            lat=lat,
            lon=lon,
            du_grid=equiv["du_grid"],
            lats_grid=equiv["lats"],
            lons_grid=equiv["lons"],
            clm_full=equiv["clm_full"],
            clm_low=equiv["clm_low"],
            r0=equiv["r0"],
            l=equiv["l"],
            return_timing=False,
        )

    times = []

    for _ in range(timed_runs):
        t0 = time.perf_counter()

        _ = analytical_model.evaluate_on_coords(
            lat=lat,
            lon=lon,
            du_grid=equiv["du_grid"],
            lats_grid=equiv["lats"],
            lons_grid=equiv["lons"],
            clm_full=equiv["clm_full"],
            clm_low=equiv["clm_low"],
            r0=equiv["r0"],
            l=equiv["l"],
            return_timing=False,
        )

        times.append(time.perf_counter() - t0)

    times = np.array(times, dtype=float)

    return {
        "samples": int(len(lat)),
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "times_s": times.tolist(),
        "mean_s": float(times.mean()),
        "std_s": float(times.std(ddof=0)),
        "min_s": float(times.min()),
        "max_s": float(times.max()),
        "seconds_per_sample_mean": float(times.mean() / len(lat)),
        "samples_per_second_mean": float(len(lat) / times.mean()),
        "device": "cpu",
        "l_equiv": int(equiv["l"]),
    }


def run_runtime_benchmark(run_path):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")

    test_path = os.path.join(data_dir, "Samples_2190-2_250k_r0_test.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_runtime = load_runtime_dataset(test_path, n_samples=10_000, seed=42)
    lon, lat = build_lonlat_tensors(df_runtime, device)

    model, arch, mode, hp = load_runtime_model(
        run_path=run_path,
        device=device,
    )

    bench_model = benchmark_model(
        model=model,
        lon=lon,
        lat=lat,
        device=device,
        warmup_runs=1,
        timed_runs=5,
    )

    analytical_model = Analytical(run_path)

    bench_analytical = benchmark_analytical(
        analytical_model=analytical_model,
        df_runtime=df_runtime,
        warmup_runs=1,
        timed_runs=5,
    )

    result = {
        "benchmark_model": bench_model,
        "benchmark_analytical": bench_analytical,
        "sh_cache_used": False,
    }


    return result

