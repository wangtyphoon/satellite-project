#!/usr/bin/env python3
"""
Add flagShallowRain pct_zero features to gpm_passes_swath_true_with_stormtop.csv.
Computes pct_zero within 50/100/150 km and upshear half-plane within 250 km.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import regrid_zfactorfinal as rzf


IN_CSV = Path("gpm_passes_swath_true.csv")
OUT_CSV = Path("gpm_passes_swath_true.csv")
NPY_DIR = Path("light_rain_npy")
UPSHEAR_RADIUS_KM = 100.0

SWATH_COL = "swath"
GRANULE_COL = "granule_file"
SHEAR_DIR_COL = "era5_shear_dir_2p5_8p5_deg"

OUT_COLS = [
    "flagShallowRain_pct_zero_r50",
    "flagShallowRain_pct_zero_r100",
    "flagShallowRain_pct_zero_r150",
    "flagShallowRain_pct_zero_upshear_r150",
]


def _resolve_npy_path(row: pd.Series) -> Path | None:
    granule_file = row.get(GRANULE_COL, None)
    if granule_file is None or str(granule_file).strip() == "":
        return None
    stem = Path(str(granule_file)).stem
    swath = rzf.normalize_swath_name(row.get(SWATH_COL, None))
    if swath:
        npy_path = NPY_DIR / f"{stem}_{swath}_light_rain.npy"
        return npy_path if npy_path.exists() else None
    matches = sorted(NPY_DIR.glob(f"{stem}_*_light_rain.npy"))
    if len(matches) == 1:
        return matches[0]
    return None


def _pct_zero(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    vals = values[mask]
    return float(np.mean(vals == 0) * 100.0)


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV, low_memory=False)
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    half = rzf.GRID_EXTENT_KM
    step = rzf.GRID_KM
    centers = rzf._grid_centers(step, rzf.GRID_SIZE, half)
    grid_x, grid_y = np.meshgrid(centers, centers)
    dist_km = np.hypot(grid_x, grid_y)
    angle = (np.degrees(np.arctan2(grid_x, grid_y)) + 360.0) % 360.0

    for idx, row in df.iterrows():
        npy_path = _resolve_npy_path(row)
        if npy_path is None:
            continue
        grid_flag = np.load(npy_path)
        if grid_flag.shape != dist_km.shape:
            print(f"Row {idx} skip: shape mismatch {grid_flag.shape} vs {dist_km.shape}")
            continue

        finite = np.isfinite(grid_flag) & np.isfinite(dist_km)
        df.at[idx, "flagShallowRain_pct_zero_r50"] = _pct_zero(
            grid_flag, finite & (dist_km <= 50.0)
        )
        df.at[idx, "flagShallowRain_pct_zero_r100"] = _pct_zero(
            grid_flag, finite & (dist_km <= 100.0)
        )
        df.at[idx, "flagShallowRain_pct_zero_r150"] = _pct_zero(
            grid_flag, finite & (dist_km <= 150.0)
        )

        shear_dir = row.get(SHEAR_DIR_COL, np.nan)
        if np.isfinite(shear_dir):
            angle_rel = (angle - float(shear_dir) + 360.0) % 360.0
            upshear = (angle_rel >= 90.0) & (angle_rel < 270.0)
            df.at[idx, "flagShallowRain_pct_zero_upshear_r100"] = _pct_zero(
                grid_flag, finite & (dist_km <= UPSHEAR_RADIUS_KM) & upshear
            )

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
