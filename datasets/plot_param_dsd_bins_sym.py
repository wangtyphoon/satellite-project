#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize paramDSD NW/DM as axis-symmetric averages (radius vs height bins).

Input: .npy from save_param_dsd.py or save_param_dsd_V2y.py
Expected shapes:
  - (2, n_bins, y, x) or
  - (2, y, x) for no-bin data
  - (2, n_bins, n_rad) or
  - (2, n_rad) for radial-only outputs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import regrid_zfactorfinal as rzf


# ===============================================
# ======
# User config
# =====================================================
NPY_PATH = None  # set to a specific .npy to bypass CSV lookup
IN_CSV = "gpm_passes_swath_true.csv"
ROW_INDEX = 5
NPY_DIR = "paramDSD"
NPY_TAG = "paramDSD_radial150km.npy"
SWATH_COL = "swath"
GRANULE_COL = "granule_file"
OUT_DIR = "plots_param_dsd_bins"

BIN_START = None  # set int for inclusive start bin index (0-based)
BIN_END = None    # set int for exclusive end bin index (0-based)

GRID_KM = 1.0        # grid spacing used when saving npy
RADIAL_BIN_KM = 5.0  # radial bin size (km)
MAX_RADIUS_KM = 150.0  # set float to cap radius (default: half grid extent)

CLIP_PERCENTILE = (2, 98)  # set None to use full min/max
FIG_DPI = 140


# =====================================================
# Helpers
# =====================================================
def _limits(data: np.ndarray, pct: Optional[Tuple[float, float]]):
    if pct is None:
        return np.nanmin(data), np.nanmax(data)
    lo, hi = np.nanpercentile(data, pct)
    return lo, hi


def _prepare_bins(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    if arr.ndim == 2:
        return arr[:, np.newaxis, ...], 1
    if arr.ndim == 3:
        if arr.shape[1] == arr.shape[2]:
            return arr[:, np.newaxis, ...], 1
        return arr, arr.shape[1]
    if arr.ndim == 4:
        return arr, arr.shape[1]
    raise ValueError(
        f"Unsupported array shape {arr.shape}; expected (2, n_bins, y, x) "
        "or (2, n_bins, n_rad)."
    )


def _grid_centers(step_km: float, size: int) -> np.ndarray:
    start = -size * step_km / 2.0 + step_km / 2.0
    return start + step_km * np.arange(size)


def _radial_profile(field: np.ndarray, r_km: np.ndarray, edges: np.ndarray) -> np.ndarray:
    flat_r = r_km.ravel()
    flat_v = field.ravel()
    good = np.isfinite(flat_v)
    if not np.any(good):
        return np.full(len(edges) - 1, np.nan, dtype=float)
    idx = np.digitize(flat_r[good], edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    if not np.any(valid):
        return np.full(len(edges) - 1, np.nan, dtype=float)
    sums = np.bincount(idx[valid], weights=flat_v[good][valid], minlength=len(edges) - 1)
    counts = np.bincount(idx[valid], minlength=len(edges) - 1)
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    nonzero = counts > 0
    out[nonzero] = sums[nonzero] / counts[nonzero]
    return out


def _resolve_npy_path_from_csv() -> str:
    csv_path = Path(IN_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if len(df) == 0:
        raise ValueError(f"{IN_CSV} is empty.")
    if ROW_INDEX < 0 or ROW_INDEX >= len(df):
        raise IndexError(f"ROW_INDEX {ROW_INDEX} out of range 0..{len(df) - 1}")

    row = df.iloc[ROW_INDEX]
    granule_file = row.get(GRANULE_COL, None)
    if granule_file is None or str(granule_file).strip() == "":
        raise ValueError(f"Row {ROW_INDEX} missing {GRANULE_COL}")
    stem = Path(str(granule_file)).stem
    swath = rzf.normalize_swath_name(row.get(SWATH_COL, None))
    npy_dir = Path(NPY_DIR)
    if swath:
        npy_path = npy_dir / f"{stem}_{swath}_{NPY_TAG}"
        if npy_path.exists():
            return str(npy_path)
    matches = sorted(npy_dir.glob(f"{stem}_*_{NPY_TAG}"))
    if len(matches) == 1:
        return str(matches[0])
    raise FileNotFoundError(f"No unique npy found for row {ROW_INDEX} ({stem}).")


# =====================================================
# Main
# =====================================================
def main() -> None:
    npy_path = NPY_PATH if NPY_PATH else _resolve_npy_path_from_csv()
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing input: {npy_path}")

    data = np.load(npy_path)
    data, n_bins = _prepare_bins(data)
    if data.shape[0] != 2:
        raise ValueError(f"Expected first dim=2 (NW/DM), got {data.shape}")

    b0 = BIN_START if BIN_START is not None else 0
    b1 = BIN_END if BIN_END is not None else n_bins
    b0 = max(0, b0)
    b1 = min(n_bins, b1)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if data.ndim == 3:
        r_centers = (np.arange(data.shape[2]) * RADIAL_BIN_KM) + RADIAL_BIN_KM / 2.0
        if MAX_RADIUS_KM is not None:
            keep = r_centers <= (MAX_RADIUS_KM + 1e-6)
            r_centers = r_centers[keep]
            data = data[:, :, keep]
        nw_stack = data[0, b0:b1].astype(np.float32, copy=False)
        dm_stack = data[1, b0:b1].astype(np.float32, copy=False)
    else:
        size = data.shape[-1]
        centers = _grid_centers(GRID_KM, size)
        xx, yy = np.meshgrid(centers, centers)
        r_km = np.sqrt(xx**2 + yy**2)
        max_r = MAX_RADIUS_KM
        if max_r is None:
            max_r = GRID_KM * size / 2.0
        edges = np.arange(0.0, max_r + RADIAL_BIN_KM, RADIAL_BIN_KM)
        r_centers = 0.5 * (edges[:-1] + edges[1:])

        nbins = b1 - b0
        nw_stack = np.full((nbins, len(r_centers)), np.nan, dtype=float)
        dm_stack = np.full((nbins, len(r_centers)), np.nan, dtype=float)

        for i, b in enumerate(range(b0, b1)):
            nw = data[0, b].astype(np.float32, copy=False)
            dm = data[1, b].astype(np.float32, copy=False)
            nw_stack[i] = _radial_profile(nw, r_km, edges)
            dm_stack[i] = _radial_profile(dm, r_km, edges)

    nw_lim = _limits(nw_stack, CLIP_PERCENTILE)
    dm_lim = _limits(dm_stack, CLIP_PERCENTILE)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    extent = [r_centers[0], r_centers[-1], b1 - 1, b0]

    ax = axes[0]
    im = ax.imshow(
        nw_stack,
        origin="upper",
        aspect="auto",
        extent=extent,
        vmin=nw_lim[0],
        vmax=nw_lim[1],
        cmap="viridis",
    )
    ax.set_title("NW axis-sym (radius vs height bins)")
    ax.set_ylabel("Bin index (top -> bottom)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(
        dm_stack,
        origin="upper",
        aspect="auto",
        extent=extent,
        vmin=dm_lim[0],
        vmax=dm_lim[1],
        cmap="magma",
    )
    ax.set_title("DM axis-sym (radius vs height bins)")
    ax.set_xlabel("Radius from TC center (km)")
    ax.set_ylabel("Bin index (top -> bottom)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    out_path = out_dir / f"paramDSD_radial_height_row{ROW_INDEX}.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
