#!/usr/bin/env python3
"""
Add latent-heating radial features to gpm_passes_swath_true.csv.
Features match the prints in plot_latent_heat.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import regrid_zfactorfinal as rzf


IN_CSV = Path("gpm_passes_swath_true.csv")
OUT_CSV = Path("gpm_passes_swath_true.csv")
NPY_DIR = Path("2LSLH")
NPY_TAG = "2LSLH_radial150km.npy"

RADIAL_BIN_KM = 5.0
MAX_RADIUS_KM = 150.0
HEIGHT_BIN_KM = 0.25
HEIGHT_START_KM = 0.0
BIN_START = None
BIN_END = None

SWATH_COL = "swath"
GRANULE_COL = "granule_file"

OUT_COLS = [
    "lh_height_of_max_heating_km",
    "lh_heating_centroid_height_km",
    "lh_upper_level_heating_fraction_8_12km",
    "lh_top_heavy_index_6km",
    "lh_total_heating_core",
    "lh_total_heating_outer",
    "lh_core_heating_fraction",
    "lh_radius_of_max_heating_km",
    "lh_radial_heating_centroid_km",
]


def _resolve_npy_path(row: pd.Series) -> Path | None:
    granule_file = row.get(GRANULE_COL, None)
    if granule_file is None or str(granule_file).strip() == "":
        return None
    stem = Path(str(granule_file)).stem
    swath = rzf.normalize_swath_name(row.get(SWATH_COL, None))
    if swath:
        npy_path = NPY_DIR / f"{stem}_{swath}_{NPY_TAG}"
        if npy_path.exists():
            return npy_path
    matches = sorted(NPY_DIR.glob(f"{stem}_*_{NPY_TAG}"))
    if len(matches) == 1:
        return matches[0]
    return None


def _prepare_bins(arr: np.ndarray) -> tuple[np.ndarray, int, bool]:
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


def _compute_features(
    r_centers: np.ndarray,
    heights: np.ndarray,
    lh_pos: np.ndarray,
) -> dict[str, float]:
    eps = 1e-9
    if r_centers.size == 0 or heights.size == 0:
        return {col: float("nan") for col in OUT_COLS}

    max_r = MAX_RADIUS_KM if MAX_RADIUS_KM is not None else float(r_centers.max())
    inner_mask = r_centers <= 50.0
    outer_mask = (r_centers > 50.0) & (r_centers <= max_r + 1e-6)
    if not np.any(inner_mask):
        return {col: float("nan") for col in OUT_COLS}

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

    e_core = float(np.nansum(lh_pos[:, inner_mask]) * HEIGHT_BIN_KM * RADIAL_BIN_KM)
    e_out = float(np.nansum(lh_pos[:, outer_mask]) * HEIGHT_BIN_KM * RADIAL_BIN_KM)
    f_core = float(e_core / (e_core + e_out + eps))

    h_r = np.nansum(lh_pos, axis=0) * HEIGHT_BIN_KM
    if np.all(~np.isfinite(h_r)) or np.nanmax(h_r) <= 0:
        r_max = float("nan")
        r_c = float("nan")
    else:
        r_max = float(r_centers[int(np.nanargmax(h_r))])
        r_c = float(np.nansum(r_centers * h_r) / (np.nansum(h_r) + eps))

    return {
        "lh_height_of_max_heating_km": z_max,
        "lh_heating_centroid_height_km": z_c,
        "lh_upper_level_heating_fraction_8_12km": f_8_12,
        "lh_top_heavy_index_6km": ti,
        "lh_total_heating_core": e_core,
        "lh_total_heating_outer": e_out,
        "lh_core_heating_fraction": f_core,
        "lh_radius_of_max_heating_km": r_max,
        "lh_radial_heating_centroid_km": r_c,
    }


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV, low_memory=False)
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    for idx, row in df.iterrows():
        npy_path = _resolve_npy_path(row)
        if npy_path is None:
            continue
        data = np.load(npy_path)
        data, n_bins, is_radial = _prepare_bins(data)

        if n_bins <= 1:
            print(f"Row {idx} skip: no vertical bins in {npy_path.name}")
            continue

        b0 = BIN_START if BIN_START is not None else 0
        b1 = BIN_END if BIN_END is not None else n_bins
        b0 = max(0, b0)
        b1 = min(n_bins, b1)

        heights = HEIGHT_START_KM + (np.arange(n_bins) + 0.5) * HEIGHT_BIN_KM
        heights = heights[b0:b1]
        if heights.size == 0:
            print(f"Row {idx} skip: empty height range")
            continue

        if is_radial:
            r_centers = (np.arange(data.shape[-1]) * RADIAL_BIN_KM) + RADIAL_BIN_KM / 2.0
            if MAX_RADIUS_KM is not None:
                keep = r_centers <= (MAX_RADIUS_KM + 1e-6)
                r_centers = r_centers[keep]
                data = data[:, keep]
            lh_stack = data[b0:b1].astype(np.float32, copy=False)
        else:
            size = data.shape[-1]
            centers = _grid_centers(1.0, size)
            xx, yy = np.meshgrid(centers, centers)
            r_km = np.sqrt(xx**2 + yy**2)
            max_r = MAX_RADIUS_KM
            if max_r is None:
                max_r = 1.0 * size / 2.0
            edges = np.arange(0.0, max_r + RADIAL_BIN_KM, RADIAL_BIN_KM)
            r_centers = 0.5 * (edges[:-1] + edges[1:])

            nbins = b1 - b0
            lh_stack = np.full((nbins, len(r_centers)), np.nan, dtype=float)
            for i, b in enumerate(range(b0, b1)):
                lh = data[b].astype(np.float32, copy=False)
                lh_stack[i] = _radial_profile(lh, r_km, edges)

        lh_pos = np.nan_to_num(lh_stack, nan=0.0)
        lh_pos = np.where(lh_pos > 0.0, lh_pos, 0.0)
        feats = _compute_features(r_centers, heights, lh_pos)
        for col, val in feats.items():
            df.at[idx, col] = val

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
