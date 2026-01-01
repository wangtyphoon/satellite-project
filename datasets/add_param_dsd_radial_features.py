#!/usr/bin/env python3
"""
Add NW/DM radial features to gpm_passes_swath_true.csv.
Uses save_param_dsd_V2y.py outputs (axis-symmetric radial means).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import regrid_zfactorfinal as rzf


IN_CSV = Path("gpm_passes_swath_true.csv")
OUT_CSV = Path("gpm_passes_swath_true.csv")
NPY_DIR = Path("paramDSD")

RADIAL_BIN_KM = 5.0
RADIUS_INNER_KM = 25.0
RADIUS_OUTER_KM = 75.0
BIN_START = 131
BIN_END = 179

SWATH_COL = "swath"
GRANULE_COL = "granule_file"

OUT_COLS = [
    "nw_mean_r0_75_b131_179",
    "nw_std_r0_75_b131_179",
    "nw_max_r0_75_b131_179",
    "dm_mean_r0_75_b131_179",
    "dm_std_r0_75_b131_179",
    "dm_max_r0_75_b131_179",
    "nw_mean_r25_75_b131_179",
    "nw_std_r25_75_b131_179",
    "nw_max_r25_75_b131_179",
    "dm_mean_r25_75_b131_179",
    "dm_std_r25_75_b131_179",
    "dm_max_r25_75_b131_179",
]


def _resolve_npy_path(row: pd.Series) -> Path | None:
    granule_file = row.get(GRANULE_COL, None)
    if granule_file is None or str(granule_file).strip() == "":
        return None
    stem = Path(str(granule_file)).stem
    swath = rzf.normalize_swath_name(row.get(SWATH_COL, None))
    if swath:
        npy_path = NPY_DIR / f"{stem}_{swath}_paramDSD_radial150km.npy"
        return npy_path if npy_path.exists() else None
    matches = sorted(NPY_DIR.glob(f"{stem}_*_paramDSD_radial150km.npy"))
    if len(matches) == 1:
        return matches[0]
    return None


def _nan_stats(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.nanmean(values)),
        float(np.nanstd(values)),
        float(np.nanmax(values)),
    )


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

        if data.ndim == 2:
            print(f"Row {idx} skip: no vertical bins in {npy_path.name}")
            continue
        if data.ndim != 3:
            print(f"Row {idx} skip: unexpected shape {data.shape}")
            continue

        r_centers = (np.arange(data.shape[2]) * RADIAL_BIN_KM) + RADIAL_BIN_KM / 2.0
        mask_0_75 = r_centers <= RADIUS_OUTER_KM
        mask_25_75 = (r_centers > RADIUS_INNER_KM) & (r_centers <= RADIUS_OUTER_KM)
        if not np.any(mask_0_75):
            print(f"Row {idx} skip: no radial bins within {RADIUS_OUTER_KM} km")
            continue

        n_bins = data.shape[1]
        b0 = max(0, BIN_START)
        b1 = min(n_bins - 1, BIN_END)
        if b1 < b0:
            print(f"Row {idx} skip: bin range {BIN_START}-{BIN_END} outside {n_bins}")
            continue

        nw = data[0, b0 : b1 + 1, :]
        dm = data[1, b0 : b1 + 1, :]

        nw_vals_0_75 = nw[:, mask_0_75].ravel()
        dm_vals_0_75 = dm[:, mask_0_75].ravel()
        nw_vals_25_75 = nw[:, mask_25_75].ravel()
        dm_vals_25_75 = dm[:, mask_25_75].ravel()

        nw_mean, nw_std, nw_max = _nan_stats(nw_vals_0_75)
        dm_mean, dm_std, dm_max = _nan_stats(dm_vals_0_75)
        df.at[idx, "nw_mean_r0_75_b131_179"] = nw_mean
        df.at[idx, "nw_std_r0_75_b131_179"] = nw_std
        df.at[idx, "nw_max_r0_75_b131_179"] = nw_max
        df.at[idx, "dm_mean_r0_75_b131_179"] = dm_mean
        df.at[idx, "dm_std_r0_75_b131_179"] = dm_std
        df.at[idx, "dm_max_r0_75_b131_179"] = dm_max

        nw_mean, nw_std, nw_max = _nan_stats(nw_vals_25_75)
        dm_mean, dm_std, dm_max = _nan_stats(dm_vals_25_75)
        df.at[idx, "nw_mean_r25_75_b131_179"] = nw_mean
        df.at[idx, "nw_std_r25_75_b131_179"] = nw_std
        df.at[idx, "nw_max_r25_75_b131_179"] = nw_max
        df.at[idx, "dm_mean_r25_75_b131_179"] = dm_mean
        df.at[idx, "dm_std_r25_75_b131_179"] = dm_std
        df.at[idx, "dm_max_r25_75_b131_179"] = dm_max

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
