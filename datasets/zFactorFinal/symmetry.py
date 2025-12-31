#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRID_KM = 1.0
RADII_KM = (25.0, 75.0, 150.0)
N_SAMPLES = 360
MAX_WAVENUMBER = 6
OUTPUT_PNG = "symmetry_first_npy.png"
THICK_RING_HALF_WIDTH_KM = 5.0
THICK_RING_RADII_COUNT = 11
THICK_RING_STAT = "mean"
THICK_RING_WEIGHT = "uniform"
THICK_RING_SIGMA_KM = 2.5


def bilinear_sample(data, x, y):
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

    values = wa * a + wb * b + wc * c + wd * d
    return values


def bilinear_sample_full(data, x, y):
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


def sample_ring_full(data, radius_km, n_samples, grid_km):
    center_x = (data.shape[1] - 1) / 2.0
    center_y = (data.shape[0] - 1) / 2.0
    r = radius_km / grid_km
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    values = bilinear_sample_full(data, x, y)
    return theta, values


def sample_ring(data, radius_km, n_samples, grid_km):
    theta, values = sample_ring_full(data, radius_km, n_samples, grid_km)
    valid = np.isfinite(values)
    return theta[valid], values[valid]


def sample_thick_ring(
    data,
    r_min_km,
    r_max_km,
    n_radii,
    n_samples,
    grid_km,
    radial_stat="mean",
    radial_weight="uniform",
    radial_sigma_km=None,
):
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
        values = np.where(weight_sum > 0.0, np.sum(weighted, axis=0) / weight_sum, np.nan)
    else:
        if radial_stat == "median":
            values = np.nanmedian(values_stack, axis=0)
        else:
            values = np.nanmean(values_stack, axis=0)

    return theta, values


def lsha_coeffs(theta, values, max_wavenumber):
    if values.size == 0:
        return np.nan, [], []
    valid = np.isfinite(values) & np.isfinite(theta)
    if not np.any(valid):
        return np.nan, [], []
    theta = theta[valid]
    values = values[valid]
    X = [np.ones_like(theta)]
    for n in range(1, max_wavenumber + 1):
        X.append(np.cos(n * theta))
        X.append(np.sin(n * theta))
    A = np.column_stack(X)
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    a0 = coeffs[0]
    an = coeffs[1::2]
    bn = coeffs[2::2]
    return a0, an, bn


def symmetry_index(a0, an, bn):
    if not np.isfinite(a0):
        return np.nan
    energy0 = a0 ** 2
    if len(an) == 0:
        return 1.0
    energy_waves = 0.5 * np.sum(an ** 2 + bn ** 2)
    total = energy0 + energy_waves
    return energy0 / total if total > 0 else np.nan


def main():
    npy_files = sorted(Path(__file__).parent.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError("No npy files found in datasets/zFactorFinal.")

    npy_path = npy_files[2]
    data = np.load(npy_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.shape} in {npy_path}")

    # --- distance-aware imshow extent (km) ---
    h, w = data.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # pixel centers mapped to km, centered at (0,0)
    # left edge is at x=-cx*GRID_KM, right edge at x=(w-1-cx)*GRID_KM, etc.
    extent = (
        (-cx) * GRID_KM,            # xmin (km)
        (w - 1 - cx) * GRID_KM,     # xmax (km)
        (-cy) * GRID_KM,            # ymin (km)
        (h - 1 - cy) * GRID_KM,     # ymax (km)
    )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(
        data,
        origin="lower",
        cmap="jet",
        extent=extent,
        interpolation="nearest",
    )

    # --- draw radius circles (km) ---
    for r_km in RADII_KM:
        circ = plt.Circle((0.0, 0.0), r_km, fill=False, linewidth=1.5)
        ax.add_patch(circ)
        ax.text(
            r_km / np.sqrt(2),
            r_km / np.sqrt(2),
            f"{r_km:g} km",
            ha="left",
            va="bottom",
            fontsize=9,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x distance from center (km)")
    ax.set_ylabel("y distance from center (km)")
    ax.set_title(f"{npy_path.name}")
    fig.colorbar(im, ax=ax, label="zFactorFinal")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / OUTPUT_PNG, bbox_inches="tight")

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
        sym = symmetry_index(a0, an, bn)
        radius_label = f"{radius_km:g}km (ring {r_min:g}-{r_max:g}km)"
        print(
            f" radius={radius_label}: "
            f"symmetry={sym:.4f}, a0={a0:.4f}, n={values.size}"
        )


if __name__ == "__main__":
    main()
