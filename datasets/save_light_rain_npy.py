#!/usr/bin/env python3
"""
Save light_rain (flagShallowRain) arrays as storm-centered regridded .npy.
Pipeline mirrors regrid_zfactorfinal.py, with only dataset/variable differences.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import regrid_zfactorfinal as rzf

IN_CSV = "gpm_passes_swath_true.csv"
DOWNLOAD_DIR_TEMPLATE = "data_gpm_2adpr_{year}"
IBTRACS_CSV_TEMPLATE = "ibtracs_WP_{year}.csv"

SWATH_COL = "swath"
SID_COL = "SID"
PASS_TIME_COL = "pass_time_utc"
PASS_START_COL = "pass_start_utc"
PASS_END_COL = "pass_end_utc"
SOURCE_COL = "source"
GRANULE_COL = "granule_file"

DATASET_PATH = "CSF/flagShallowRain"
OUT_DIR = Path("light_rain_npy")

def _mask_fill(arr: np.ndarray, fill_value: float | int | None) -> np.ndarray:
    out = arr.astype(float, copy=True)
    if fill_value is not None:
        out[out == float(fill_value)] = np.nan
    return out


def main() -> None:
    # =========================
    # Config (edit as needed)
    # =========================
    csv_path = Path(IN_CSV)
    file_path = None
    swath = None
    subset = (slice(None), slice(None))
    max_rows = None
    overwrite = False

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if file_path is not None:
        hdf_path = Path(file_path)
        if not h5py.is_hdf5(hdf_path):
            raise OSError(f"Not a valid HDF5 file: {hdf_path}")
        storm_lat = None
        storm_lon = None
        sid = "unknown"
        year = None
        pass_time = None
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty:
            raise ValueError(f"{csv_path} is empty.")

        row_slice, col_slice = subset
        root = Path(rzf._project_root())
        indices = df.index
        if max_rows is not None:
            indices = indices[:max_rows]

        processed = 0
        skipped = 0
        missing = 0

        for row_idx in indices:
            row = df.loc[row_idx]
            try:
                year = rzf._infer_year_from_row(row)
                granule_file = row[GRANULE_COL]
                swath_row = rzf.normalize_swath_name(row.get(SWATH_COL, None))
                sid = row[SID_COL]
                pass_time = rzf._resolve_pass_time(row)
            except Exception:
                missing += 1
                continue

            download_dir = root / DOWNLOAD_DIR_TEMPLATE.format(year=year)
            hdf_path = download_dir / granule_file
            if not hdf_path.exists() or not h5py.is_hdf5(hdf_path):
                missing += 1
                continue

            ibtracs_path = root / IBTRACS_CSV_TEMPLATE.format(year=year)
            if not ibtracs_path.exists():
                missing += 1
                continue
            try:
                track_df = rzf.load_track_for_sid(ibtracs_path, sid)
                storm_lat, storm_lon = rzf.interpolate_track_position(track_df, pass_time)
            except Exception:
                missing += 1
                continue
            if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
                missing += 1
                continue

            with h5py.File(hdf_path, "r") as f:
                group = rzf.resolve_swath(f, swath_row or swath)
                base_name = f"{hdf_path.stem}_{group}_light_rain"
                out_path = OUT_DIR / f"{base_name}.npy"
                if out_path.exists() and not overwrite:
                    skipped += 1
                    continue

                ds_flag = f[f"{group}/{DATASET_PATH}"]
                ds_lat = f[f"{group}/Latitude"]
                ds_lon = f[f"{group}/Longitude"]

                flag_arr = _mask_fill(ds_flag[:], ds_flag.attrs.get("_FillValue"))
                lat_arr = _mask_fill(ds_lat[:], ds_lat.attrs.get("_FillValue"))
                lon_arr = _mask_fill(ds_lon[:], ds_lon.attrs.get("_FillValue"))

                attrs = {k: ds_flag.attrs[k] for k in ds_flag.attrs.keys()}
                scale = attrs.get("scale_factor", None)
                offset = attrs.get("add_offset", None)
                if scale is not None or offset is not None:
                    scale = float(scale) if scale is not None else 1.0
                    offset = float(offset) if offset is not None else 0.0
                    flag_arr = flag_arr * scale + offset
                flag_arr = rzf._apply_valid_range(flag_arr, attrs)

                flag_arr = flag_arr[row_slice, col_slice]
                lat_arr = lat_arr[row_slice, col_slice]
                lon_arr = lon_arr[row_slice, col_slice]

                lat_arr[(lat_arr < -90.0) | (lat_arr > 90.0)] = np.nan
                lon_arr[(lon_arr < -180.0) | (lon_arr > 180.0)] = np.nan
                x_km, y_km = rzf._latlon_to_local_km(lat_arr, lon_arr, storm_lat, storm_lon)
                valid_xy = np.isfinite(x_km) & np.isfinite(y_km)
                valid = valid_xy & np.isfinite(flag_arr)

                half = rzf.GRID_EXTENT_KM
                step = rzf.GRID_KM
                swath_mask = rzf._grid_swath_mask(
                    x_km[valid_xy], y_km[valid_xy], step, rzf.GRID_SIZE, half
                )
                grid_flag = rzf._regrid_to_grid(
                    x_km[valid],
                    y_km[valid],
                    flag_arr[valid],
                    rzf.INTERP_METHOD,
                    step,
                    rzf.GRID_SIZE,
                    half,
                )
                grid_flag = np.where(np.isnan(grid_flag) & swath_mask, 0.0, grid_flag)

                np.save(out_path, grid_flag)
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} rows...")

        print(f"Saved: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Missing: {missing}")
        return

    row_slice, col_slice = subset

    with h5py.File(hdf_path, "r") as f:
        group = rzf.resolve_swath(f, swath)
        ds_flag = f[f"{group}/{DATASET_PATH}"]
        ds_lat = f[f"{group}/Latitude"]
        ds_lon = f[f"{group}/Longitude"]

        flag_arr = _mask_fill(ds_flag[:], ds_flag.attrs.get("_FillValue"))
        lat_arr = _mask_fill(ds_lat[:], ds_lat.attrs.get("_FillValue"))
        lon_arr = _mask_fill(ds_lon[:], ds_lon.attrs.get("_FillValue"))

        attrs = {k: ds_flag.attrs[k] for k in ds_flag.attrs.keys()}
        scale = attrs.get("scale_factor", None)
        offset = attrs.get("add_offset", None)
        if scale is not None or offset is not None:
            scale = float(scale) if scale is not None else 1.0
            offset = float(offset) if offset is not None else 0.0
            flag_arr = flag_arr * scale + offset
        flag_arr = rzf._apply_valid_range(flag_arr, attrs)

        flag_arr = flag_arr[row_slice, col_slice]
        lat_arr = lat_arr[row_slice, col_slice]
        lon_arr = lon_arr[row_slice, col_slice]

        if storm_lat is not None and storm_lon is not None:
            lat_arr[(lat_arr < -90.0) | (lat_arr > 90.0)] = np.nan
            lon_arr[(lon_arr < -180.0) | (lon_arr > 180.0)] = np.nan
            x_km, y_km = rzf._latlon_to_local_km(lat_arr, lon_arr, storm_lat, storm_lon)
            valid_xy = np.isfinite(x_km) & np.isfinite(y_km)
            valid = valid_xy & np.isfinite(flag_arr)

            half = rzf.GRID_EXTENT_KM
            step = rzf.GRID_KM
            swath_mask = rzf._grid_swath_mask(
                x_km[valid_xy], y_km[valid_xy], step, rzf.GRID_SIZE, half
            )
            grid_flag = rzf._regrid_to_grid(
                x_km[valid],
                y_km[valid],
                flag_arr[valid],
                rzf.INTERP_METHOD,
                step,
                rzf.GRID_SIZE,
                half,
            )
            grid_flag = np.where(np.isnan(grid_flag) & swath_mask, 0.0, grid_flag)
        else:
            grid_flag = flag_arr

        stem = hdf_path.stem
        suffix = f"{group}"
        base_name = f"{stem}_{suffix}_light_rain"
        out_path = OUT_DIR / f"{base_name}.npy"
        if out_path.exists() and not overwrite:
            print(f"Skipped existing: {out_path}")
            return
        np.save(out_path, grid_flag)
        print(f"Saved: {out_path}")
        if year is not None:
            print(f"Year: {year}, SID: {sid}, pass_time: {pass_time}")


if __name__ == "__main__":
    main()
