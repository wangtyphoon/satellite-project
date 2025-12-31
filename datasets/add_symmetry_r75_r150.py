#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd


CSV_PATH = Path(__file__).parent / "gpm_passes_swath_true.csv"
NPY_DIR = Path(__file__).parent / "zFactorFinal"

VAR_NAME = "zFactorFinal"
AGG = "max"
NPY_PREFIX = "1_max_"

GRID_KM = 1.0
RADII_KM = (25.0, 75.0, 150.0)
N_SAMPLES = 360
MAX_WAVENUMBER = 6
THICK_RING_HALF_WIDTH_KM = 5.0
THICK_RING_RADII_COUNT = 11
THICK_RING_STAT = "mean"
THICK_RING_WEIGHT = "uniform"
THICK_RING_SIGMA_KM = 2.5


def _radius_label(radius_km: float) -> str:
    if float(radius_km).is_integer():
        return str(int(radius_km))
    return f"{radius_km:g}"


def bilinear_sample(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = data.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    if not np.any(valid):
        return np.array([])

    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]
    xv = x[valid]
    yv = y[valid]

    wa = (x1v - xv) * (y1v - yv)
    wb = (xv - x0v) * (y1v - yv)
    wc = (x1v - xv) * (yv - y0v)
    wd = (xv - x0v) * (yv - y0v)

    a = data[y0v, x0v]
    b = data[y0v, x1v]
    c = data[y1v, x0v]
    d = data[y1v, x1v]

    return wa * a + wb * b + wc * c + wd * d


def bilinear_sample_full(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = data.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    values = np.full_like(x, np.nan, dtype=float)
    if not np.any(valid):
        return values

    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]
    xv = x[valid]
    yv = y[valid]

    wa = (x1v - xv) * (y1v - yv)
    wb = (xv - x0v) * (y1v - yv)
    wc = (x1v - xv) * (yv - y0v)
    wd = (xv - x0v) * (yv - y0v)

    a = data[y0v, x0v]
    b = data[y0v, x1v]
    c = data[y1v, x0v]
    d = data[y1v, x1v]

    values[valid] = wa * a + wb * b + wc * c + wd * d
    return values


def sample_ring_full(
    data: np.ndarray, radius_km: float, n_samples: int, grid_km: float
) -> tuple[np.ndarray, np.ndarray]:
    center_x = (data.shape[1] - 1) / 2.0
    center_y = (data.shape[0] - 1) / 2.0
    r = radius_km / grid_km
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    values = bilinear_sample_full(data, x, y)
    return theta, values


def sample_ring(
    data: np.ndarray, radius_km: float, n_samples: int, grid_km: float
) -> tuple[np.ndarray, np.ndarray]:
    theta, values = sample_ring_full(data, radius_km, n_samples, grid_km)
    if values.size == 0:
        return theta, values
    valid = np.isfinite(values)
    return theta[valid], values[valid]


def sample_thick_ring(
    data: np.ndarray,
    r_min_km: float,
    r_max_km: float,
    n_radii: int,
    n_samples: int,
    grid_km: float,
    radial_stat: str = "mean",
    radial_weight: str = "uniform",
    radial_sigma_km: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.linspace(r_min_km, r_max_km, n_radii)
    values_stack = []
    theta = None
    for r_km in radii:
        theta, values = sample_ring_full(data, r_km, n_samples, grid_km)
        values_stack.append(values)
    values_stack = np.vstack(values_stack)

    if radial_weight == "gaussian":
        center_km = 0.5 * (r_min_km + r_max_km)
        sigma_km = radial_sigma_km or (r_max_km - r_min_km) / 6.0
        weights = np.exp(-0.5 * ((radii - center_km) / sigma_km) ** 2)
        valid = np.isfinite(values_stack)
        weighted = np.where(valid, values_stack, 0.0) * weights[:, None]
        weight_sum = np.sum(weights[:, None] * valid, axis=0)
        values = np.where(
            weight_sum > 0.0, np.sum(weighted, axis=0) / weight_sum, np.nan
        )
    else:
        if radial_stat == "median":
            values = np.nanmedian(values_stack, axis=0)
        else:
            values = np.nanmean(values_stack, axis=0)

    return theta, values


def lsha_coeffs(
    theta: np.ndarray, values: np.ndarray, max_wavenumber: int
) -> tuple[float, np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.nan, np.array([]), np.array([])
    valid = np.isfinite(values) & np.isfinite(theta)
    if not np.any(valid):
        return np.nan, np.array([]), np.array([])
    theta = theta[valid]
    values = values[valid]
    design = [np.ones_like(theta)]
    for n in range(1, max_wavenumber + 1):
        design.append(np.cos(n * theta))
        design.append(np.sin(n * theta))
    matrix = np.column_stack(design)
    coeffs, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    a0 = float(coeffs[0])
    an = coeffs[1::2]
    bn = coeffs[2::2]
    return a0, an, bn


def symmetry_index(a0: float, an: np.ndarray, bn: np.ndarray) -> float:
    if not np.isfinite(a0):
        return np.nan
    energy0 = a0**2
    if len(an) == 0:
        return 1.0
    energy_waves = 0.5 * np.sum(an**2 + bn**2)
    total = energy0 + energy_waves
    return float(energy0 / total) if total > 0 else np.nan


def compute_symmetry(data: np.ndarray) -> dict[float, float]:
    result = {}
    for radius_km in RADII_KM:
        r_min = max(0.0, radius_km - THICK_RING_HALF_WIDTH_KM)
        r_max = radius_km + THICK_RING_HALF_WIDTH_KM
        theta, values = sample_thick_ring(
            data,
            r_min,
            r_max,
            THICK_RING_RADII_COUNT,
            N_SAMPLES,
            GRID_KM,
            radial_stat=THICK_RING_STAT,
            radial_weight=THICK_RING_WEIGHT,
            radial_sigma_km=THICK_RING_SIGMA_KM,
        )
        a0, an, bn = lsha_coeffs(theta, values, MAX_WAVENUMBER)
        result[radius_km] = symmetry_index(a0, an, bn)
    return result


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "granule_file" not in df.columns:
        raise ValueError("CSV missing granule_file column.")

    radius_cols = {}
    for radius_km in RADII_KM:
        label = _radius_label(radius_km)
        col = f"{VAR_NAME}_{AGG}_r{label}_symmetry"
        radius_cols[radius_km] = col
        if col not in df.columns:
            df[col] = np.nan

    cache = {}
    missing = 0
    processed = 0
    for row_idx, row in df.iterrows():
        granule_file = row.get("granule_file", None)
        if pd.isna(granule_file):
            continue
        stem = Path(str(granule_file)).stem
        if stem not in cache:
            npy_path = NPY_DIR / f"{NPY_PREFIX}{stem}.npy"
            if not npy_path.exists():
                cache[stem] = None
            else:
                data = np.load(npy_path)
                if data.ndim != 2:
                    raise ValueError(f"Expected 2D array, got {data.shape} in {npy_path}")
                cache[stem] = compute_symmetry(data)
        sym_map = cache.get(stem)
        if sym_map is None:
            missing += 1
            continue
        for radius_km, col in radius_cols.items():
            df.at[row_idx, col] = sym_map.get(radius_km, np.nan)
        processed += 1

    df.to_csv(CSV_PATH, index=False)
    print(f"Updated rows: {processed}")
    print(f"Missing npy: {missing}")
    print(f"Wrote CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
