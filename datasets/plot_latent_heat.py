#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 2LSLH latent heating as axis-symmetric averages (radius vs height bins).

Input: .npy from save_2LSLH_latent_heating.py
Expected shapes:
  - (n_bins, y, x) or
  - (y, x) for no-bin data
  - (n_bins, n_rad) or
  - (n_rad) for radial-only outputs
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
ROW_INDEX = 0
NPY_DIR = "2LSLH"
NPY_TAG = "2LSLH_radial150km.npy"
SWATH_COL = "swath"
GRANULE_COL = "granule_file"
OUT_DIR = "plots_latent_heat"

BIN_START = None  # set int for inclusive start bin index (0-based)
BIN_END = None    # set int for exclusive end bin index (0-based)

GRID_KM = 1.0        # grid spacing used when saving npy
RADIAL_BIN_KM = 5.0  # radial bin size (km)
MAX_RADIUS_KM = 150.0  # set float to cap radius (default: half grid extent)
HEIGHT_BIN_KM = 0.25   # fixed height layer thickness (km)
HEIGHT_START_KM = 0.0  # layer base (km)

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


def _prepare_bins(arr: np.ndarray) -> Tuple[np.ndarray, int, bool]:
    if arr.ndim == 1:
        return arr[np.newaxis, :], 1, True
    if arr.ndim == 2:
        if arr.shape[0] == arr.shape[1]:
            return arr[np.newaxis, ...], 1, False
        return arr, arr.shape[0], True
    if arr.ndim == 3:
        return arr, arr.shape[0], False
    raise ValueError(
        f"Unsupported array shape {arr.shape}; expected (n_bins, y, x) "
        "or (n_bins, n_rad)."
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


def _fmt_array(arr: np.ndarray) -> str:
    return np.array2string(
        arr,
        precision=3,
        floatmode="fixed",
        separator=", ",
        suppress_small=False,
    )


def _print_profiles(
    r_centers: np.ndarray,
    heights: np.ndarray,
    lh_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h_r = np.nansum(lh_pos, axis=0) * HEIGHT_BIN_KM
    h_z = np.nansum(lh_pos, axis=1) * RADIAL_BIN_KM
    print("H_r(r)=∫LH+(r,z)dz")
    print("r_km:", _fmt_array(r_centers))
    print("H_r:", _fmt_array(h_r))
    print("H_z(z)=∫LH+(r,z)dr")
    print("z_km:", _fmt_array(heights))
    print("H_z:", _fmt_array(h_z))
    return h_r, h_z


def _print_features(
    r_centers: np.ndarray,
    heights: np.ndarray,
    lh_pos: np.ndarray,
    h_r: np.ndarray,
) -> None:
    eps = 1e-9
    max_r = MAX_RADIUS_KM if MAX_RADIUS_KM is not None else float(r_centers.max())
    inner_mask = r_centers <= 50.0
    outer_mask = (r_centers > 50.0) & (r_centers <= max_r + 1e-6)

    h_z_core = np.nansum(lh_pos[:, inner_mask], axis=1) * RADIAL_BIN_KM
    if np.all(~np.isfinite(h_z_core)) or np.nanmax(h_z_core) <= 0:
        z_max = float("nan")
        z_c = float("nan")
    else:
        z_max = float(heights[int(np.nanargmax(h_z_core))])
        z_c = float(np.nansum(heights * h_z_core) / (np.nansum(h_z_core) + eps))

    mask_8_12 = (heights >= 8.0) & (heights <= 12.0)
    total_core = np.nansum(h_z_core) * HEIGHT_BIN_KM
    f_8_12 = float(
        (np.nansum(h_z_core[mask_8_12]) * HEIGHT_BIN_KM) / (total_core + eps)
    )
    mask_hi = heights >= 6.0
    mask_lo = heights < 6.0
    ti = float(
        (np.nansum(h_z_core[mask_hi]) * HEIGHT_BIN_KM)
        / (np.nansum(h_z_core[mask_lo]) * HEIGHT_BIN_KM + eps)
    )

    e_core = float(
        np.nansum(lh_pos[:, inner_mask]) * HEIGHT_BIN_KM * RADIAL_BIN_KM
    )
    e_out = float(
        np.nansum(lh_pos[:, outer_mask]) * HEIGHT_BIN_KM * RADIAL_BIN_KM
    )
    f_core = float(e_core / (e_core + e_out + eps))

    if np.all(~np.isfinite(h_r)) or np.nanmax(h_r) <= 0:
        r_max = float("nan")
        r_c = float("nan")
    else:
        r_max = float(r_centers[int(np.nanargmax(h_r))])
        r_c = float(np.nansum(r_centers * h_r) / (np.nansum(h_r) + eps))

    print("Feature set (inner-core r<=50 km):")
    print(f"height_of_max_heating_km={z_max:.3f}")
    print(f"heating_centroid_height_km={z_c:.3f}")
    print(f"upper_level_heating_fraction_8_12km={f_8_12:.3f}")
    print(f"top_heavy_index_6km={ti:.3f}")
    print("Core/outer heating contrast:")
    print(f"total_heating_core={e_core:.3f}")
    print(f"total_heating_outer={e_out:.3f}")
    print(f"core_heating_fraction={f_core:.3f}")
    print(f"radius_of_max_heating_km={r_max:.3f}")
    print(f"radial_heating_centroid_km={r_c:.3f}")


# =====================================================
# Main
# =====================================================
def main() -> None:
    npy_path = NPY_PATH if NPY_PATH else _resolve_npy_path_from_csv()
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing input: {npy_path}")

    data = np.load(npy_path)
    print(f"Loaded {npy_path} with shape {data.shape}")
    data, n_bins, is_radial = _prepare_bins(data)

    b0 = BIN_START if BIN_START is not None else 0
    b1 = BIN_END if BIN_END is not None else n_bins
    b0 = max(0, b0)
    b1 = min(n_bins, b1)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_radial:
        r_centers = (np.arange(data.shape[-1]) * RADIAL_BIN_KM) + RADIAL_BIN_KM / 2.0
        if MAX_RADIUS_KM is not None:
            keep = r_centers <= (MAX_RADIUS_KM + 1e-6)
            r_centers = r_centers[keep]
            data = data[:, keep]
        lh_stack = data[b0:b1].astype(np.float32, copy=False)
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
        lh_stack = np.full((nbins, len(r_centers)), np.nan, dtype=float)

        for i, b in enumerate(range(b0, b1)):
            lh = data[b].astype(np.float32, copy=False)
            lh_stack[i] = _radial_profile(lh, r_km, edges)

    lh_lim = _limits(lh_stack, CLIP_PERCENTILE)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
    heights = HEIGHT_START_KM + (np.arange(n_bins) + 0.5) * HEIGHT_BIN_KM
    heights = heights[b0:b1]

    lh_pos = np.nan_to_num(lh_stack, nan=0.0)
    lh_pos = np.where(lh_pos > 0.0, lh_pos, 0.0)
    h_r, _ = _print_profiles(r_centers, heights, lh_pos)
    _print_features(r_centers, heights, lh_pos, h_r)
    extent = [r_centers[0], r_centers[-1], heights[0], heights[-1]]

    im = ax.imshow(
        lh_stack,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=lh_lim[0],
        vmax=lh_lim[1],
        cmap="viridis",
    )
    ax.set_title("LH axis-sym (radius vs height bins)")
    ax.set_xlabel("Radius from TC center (km)")
    ax.set_ylabel("Height (km)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    out_path = out_dir / f"latent_heat_radial_height_row{ROW_INDEX}.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
