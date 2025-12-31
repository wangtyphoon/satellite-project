#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute quadrant reflectivity differences (clockwise) within a storm-centered radius
using ERA5 shear direction, then add the feature to the pass CSV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

IN_CSV = Path(__file__).resolve().parent / "gpm_passes_swath_true.csv"
OUT_CSV = IN_CSV
IBTRACS_TEMPLATE = Path(__file__).resolve().parent.parent / "ibtracs_WP_{year}.csv"

NPY_DIR = Path(__file__).resolve().parent / "zFactorFinal"
NPY_PREFIX = "1_mean_"

PASS_TIME_COL = "pass_time_utc"
PASS_START_COL = "pass_start_utc"
PASS_END_COL = "pass_end_utc"
SOURCE_COL = "source"
SID_COL = "SID"
GRANULE_COL = "granule_file"

ERA5_DIR_TEMPLATE = Path(__file__).resolve().parent / "data_era5_shear_{year}"
ERA5_FILE_RADIUS_DEG = 10.0

SHEAR_LEVELS = (200, 850)
SHEAR_INNER_DEG = 2.5
SHEAR_OUTER_DEG = 8.5

GRID_KM = 1.0
RADIUS_KM = 50.0
AGG = "mean"
KEEP_ZERO = True

MAX_ROWS = None


def _wrap_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _to_utc_datetime(value) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _round_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.Timedelta(minutes=30)).floor("H")


def _infer_year_from_row(row: pd.Series) -> int:
    if PASS_TIME_COL in row and pd.notna(row[PASS_TIME_COL]):
        return pd.to_datetime(row[PASS_TIME_COL], utc=True).year
    if SOURCE_COL in row and pd.notna(row[SOURCE_COL]):
        m = re.search(r"(\\d{4})", str(row[SOURCE_COL]))
        if m:
            return int(m.group(1))
    raise ValueError("Could not infer year from row (pass_time_utc/source missing).")


def _resolve_pass_time(row: pd.Series) -> pd.Timestamp:
    if PASS_TIME_COL in row and pd.notna(row[PASS_TIME_COL]):
        return pd.to_datetime(row[PASS_TIME_COL], utc=True)
    if PASS_START_COL in row and PASS_END_COL in row:
        start = pd.to_datetime(row[PASS_START_COL], utc=True)
        end = pd.to_datetime(row[PASS_END_COL], utc=True)
        return start + (end - start) / 2
    raise ValueError("No pass_time_utc or pass_start_utc/pass_end_utc available.")


@dataclass
class TrackCache:
    by_year: Dict[int, pd.DataFrame]
    by_sid: Dict[str, pd.DataFrame]


def load_track_for_sid(cache: TrackCache, sid: str, year: int) -> pd.DataFrame:
    if sid in cache.by_sid:
        return cache.by_sid[sid]

    if year not in cache.by_year:
        ib_path = IBTRACS_TEMPLATE.with_name(IBTRACS_TEMPLATE.name.format(year=year))
        if not ib_path.exists():
            raise FileNotFoundError(f"Missing IBTrACS CSV: {ib_path}")
        cache.by_year[year] = pd.read_csv(ib_path, low_memory=False)

    df = cache.by_year[year]
    time_col = "ISO_TIME" if "ISO_TIME" in df.columns else None
    lat_col = "USA_LAT" if "USA_LAT" in df.columns else ("LAT" if "LAT" in df.columns else None)
    lon_col = "USA_LON" if "USA_LON" in df.columns else ("LON" if "LON" in df.columns else None)
    sid_col = "SID" if "SID" in df.columns else None
    if not (time_col and lat_col and lon_col and sid_col):
        raise ValueError("IBTrACS CSV missing required columns (SID/time/lat/lon).")

    sub = df[df[sid_col] == sid].copy()
    if sub.empty:
        raise ValueError(f"SID {sid} not found in IBTrACS CSV.")

    sub["time_utc"] = pd.to_datetime(sub[time_col], errors="coerce", utc=True)
    sub["lat"] = pd.to_numeric(sub[lat_col], errors="coerce")
    sub["lon"] = pd.to_numeric(sub[lon_col], errors="coerce")
    sub = sub.dropna(subset=["time_utc", "lat", "lon"]).sort_values("time_utc")

    cache.by_sid[sid] = sub
    return sub


def interpolate_track_position(track_df: pd.DataFrame, target_time: pd.Timestamp) -> Tuple[float, float]:
    t0 = pd.Timestamp("1970-01-01", tz="UTC")
    tt = (track_df["time_utc"] - t0).dt.total_seconds().to_numpy()
    lat = track_df["lat"].astype(float).to_numpy()
    lon = track_df["lon"].astype(float).to_numpy()

    m = np.isfinite(tt) & np.isfinite(lat) & np.isfinite(lon)
    tt = tt[m]
    lat = lat[m]
    lon = lon[m]
    if len(tt) < 2:
        return np.nan, np.nan

    lon_u = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    q = (target_time - t0).total_seconds()
    lat_i = np.interp(q, tt, lat, left=np.nan, right=np.nan)
    lon_i = np.interp(q, tt, lon_u, left=np.nan, right=np.nan)
    lon_i = _wrap_lon(lon_i)
    return float(lat_i), float(lon_i)


def _select_level(da: xr.DataArray, level: int) -> xr.DataArray:
    for dim in ("level", "pressure_level"):
        if dim in da.dims:
            coord = da.coords.get(dim, None)
            if coord is not None:
                try:
                    return da.sel({dim: level})
                except KeyError:
                    return da.sel({dim: level}, method="nearest")
            return da.sel({dim: level})
    raise KeyError("Missing level/pressure_level dimension in ERA5 file.")


def _squeeze_time(da: xr.DataArray) -> xr.DataArray:
    for dim in ("valid_time", "time"):
        if dim in da.dims and da.sizes.get(dim, 0) == 1:
            da = da.isel({dim: 0})
    return da


def _get_coord(ds: xr.Dataset, names: Tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in ds.coords:
            return ds.coords[name].values
        if name in ds:
            return ds[name].values
    raise KeyError(f"Missing coordinate: {names}")


def angular_distance_deg(lat, lon, lat0, lon0) -> np.ndarray:
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    lat0_r = np.deg2rad(lat0)
    lon0_r = np.deg2rad(lon0)
    dlat = lat_r - lat0_r
    dlon = lon_r - lon0_r
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r) * np.cos(lat0_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return np.rad2deg(c)


def _build_era5_path(sid: str, era5_time: pd.Timestamp, radius_deg: float) -> Path:
    out_dir = ERA5_DIR_TEMPLATE.with_name(ERA5_DIR_TEMPLATE.name.format(year=era5_time.year))
    fname = f"era5_{sid}_{era5_time.strftime('%Y%m%d_%H%M')}_r{radius_deg:g}.nc"
    return out_dir / fname


def load_era5_uv(
    nc_path: Path,
    levels: Iterable[int],
    cache: Dict[Path, Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]],
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    if nc_path in cache:
        return cache[nc_path]

    ds = xr.open_dataset(nc_path)
    try:
        u = ds["u"]
        v = ds["v"]
    except KeyError as exc:
        ds.close()
        raise KeyError("Missing u/v wind components in ERA5 file.") from exc

    lat = _get_coord(ds, ("latitude", "lat"))
    lon = _get_coord(ds, ("longitude", "lon"))
    uv_by_level: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for level in levels:
        u_level = _squeeze_time(_select_level(u, level)).values
        v_level = _squeeze_time(_select_level(v, level)).values
        uv_by_level[level] = (u_level, v_level)
    ds.close()
    cache[nc_path] = (uv_by_level, lat, lon)
    return cache[nc_path]


def mean_uv_in_annulus_deg(
    u: np.ndarray,
    v: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    center_lat: float,
    center_lon: float,
    inner_deg: float,
    outer_deg: float,
) -> Tuple[float, float]:
    dist = angular_distance_deg(lat2d, lon2d, center_lat, center_lon)
    mask = (dist >= inner_deg) & (dist <= outer_deg) & np.isfinite(u) & np.isfinite(v)
    if not np.any(mask):
        return np.nan, np.nan
    return float(np.mean(u[mask])), float(np.mean(v[mask]))


def shear_direction_deg(u_shear: float, v_shear: float) -> float:
    return float((np.degrees(np.arctan2(u_shear, v_shear)) + 360.0) % 360.0)


def load_zfactor_grid(stem: str) -> np.ndarray:
    npy_path = NPY_DIR / f"{NPY_PREFIX}{stem}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing zFactorFinal npy: {npy_path}")
    data = np.load(npy_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.shape} in {npy_path}")
    return data


def grid_xy_km(shape: Tuple[int, int], grid_km: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = shape
    center_x = (w - 1) / 2.0
    center_y = (h - 1) / 2.0
    x = (np.arange(w) - center_x) * grid_km
    y = (np.arange(h) - center_y) * grid_km
    x2d, y2d = np.meshgrid(x, y)
    return x2d, y2d


def quadrant_indices(angle_rel: np.ndarray) -> np.ndarray:
    q = np.zeros(angle_rel.shape, dtype=np.int8)
    q[(angle_rel >= 0.0) & (angle_rel < 90.0)] = 1   # DR
    q[(angle_rel >= 90.0) & (angle_rel < 180.0)] = 2  # UR
    q[(angle_rel >= 180.0) & (angle_rel < 270.0)] = 3 # UL
    q[(angle_rel >= 270.0) & (angle_rel < 360.0)] = 4 # DL
    return q


def _apply_agg(values: np.ndarray, agg: str) -> float:
    if values.size == 0:
        return np.nan
    if agg == "mean":
        return float(np.nanmean(values))
    if agg == "max":
        return float(np.nanmax(values))
    if agg == "median":
        return float(np.nanmedian(values))
    if agg.startswith("p"):
        pct = float(agg[1:])
        return float(np.nanpercentile(values, pct))
    raise ValueError(f"Unsupported agg: {agg}")


def compute_quadrant_diffs(
    z_grid: np.ndarray,
    shear_dir: float,
    radius_km: float,
    grid_km: float,
    agg: str,
) -> Dict[str, float]:
    if not KEEP_ZERO:
        z_grid = z_grid.astype(float)
        z_grid[z_grid == 0.0] = np.nan

    x_km, y_km = grid_xy_km(z_grid.shape, grid_km)
    r_km = np.hypot(x_km, y_km)
    angle = (np.degrees(np.arctan2(x_km, y_km)) + 360.0) % 360.0
    angle_rel = (angle - shear_dir + 360.0) % 360.0

    within = r_km <= radius_km
    quadrants = quadrant_indices(angle_rel)
    mask = within & np.isfinite(z_grid)

    dr_vals = z_grid[(quadrants == 1) & mask]
    ur_vals = z_grid[(quadrants == 2) & mask]
    ul_vals = z_grid[(quadrants == 3) & mask]
    dl_vals = z_grid[(quadrants == 4) & mask]

    dr_val = _apply_agg(dr_vals, agg)
    ur_val = _apply_agg(ur_vals, agg)
    ul_val = _apply_agg(ul_vals, agg)
    dl_val = _apply_agg(dl_vals, agg)

    diffs = {
        "DL_minus_DR": np.nan,
        "UL_minus_DL": np.nan,
        "UR_minus_UL": np.nan,
        "DR_minus_UR": np.nan,
    }
    if np.isfinite(dl_val) and np.isfinite(dr_val):
        diffs["DL_minus_DR"] = float(dl_val - dr_val)
    if np.isfinite(ul_val) and np.isfinite(dl_val):
        diffs["UL_minus_DL"] = float(ul_val - dl_val)
    if np.isfinite(ur_val) and np.isfinite(ul_val):
        diffs["UR_minus_UL"] = float(ur_val - ul_val)
    if np.isfinite(dr_val) and np.isfinite(ur_val):
        diffs["DR_minus_UR"] = float(dr_val - ur_val)
    return diffs


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    if df.empty:
        raise ValueError(f"{IN_CSV} is empty.")
    if SID_COL not in df.columns or GRANULE_COL not in df.columns:
        raise ValueError("CSV missing SID or granule_file column.")

    radius_label = int(RADIUS_KM) if float(RADIUS_KM).is_integer() else f"{RADIUS_KM:g}"
    out_cols = [
        f"zFactorFinal_{AGG}_r{radius_label}_DL_minus_DR",
        f"zFactorFinal_{AGG}_r{radius_label}_UL_minus_DL",
        f"zFactorFinal_{AGG}_r{radius_label}_UR_minus_UL",
        f"zFactorFinal_{AGG}_r{radius_label}_DR_minus_UR",
    ]
    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan

    track_cache = TrackCache(by_year={}, by_sid={})
    era5_cache: Dict[
        Path, Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]
    ] = {}
    z_cache: Dict[str, Optional[np.ndarray]] = {}

    processed = 0
    missing = 0
    for row_idx, row in df.iterrows():
        if MAX_ROWS is not None and processed >= MAX_ROWS:
            break

        sid = str(row[SID_COL])
        granule_file = row.get(GRANULE_COL, None)
        if pd.isna(granule_file):
            continue
        stem = Path(str(granule_file)).stem

        if stem not in z_cache:
            npy_path = NPY_DIR / f"{NPY_PREFIX}{stem}.npy"
            if not npy_path.exists():
                z_cache[stem] = None
            else:
                z_cache[stem] = load_zfactor_grid(stem)
        z_grid = z_cache.get(stem)
        if z_grid is None:
            missing += 1
            continue

        pass_time = _resolve_pass_time(row)
        year = _infer_year_from_row(row)
        track_df = load_track_for_sid(track_cache, sid, year)
        storm_lat, storm_lon = interpolate_track_position(track_df, pass_time)
        if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
            missing += 1
            continue

        era5_time = _round_to_hour(pass_time)
        era5_path = _build_era5_path(sid, era5_time, ERA5_FILE_RADIUS_DEG)
        if not era5_path.exists():
            missing += 1
            continue

        uv_by_level, lat, lon = load_era5_uv(era5_path, SHEAR_LEVELS, era5_cache)
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        shear_hi, shear_lo = SHEAR_LEVELS
        u_hi, v_hi = uv_by_level[shear_hi]
        u_lo, v_lo = uv_by_level[shear_lo]
        u_hi_mean, v_hi_mean = mean_uv_in_annulus_deg(
            u_hi, v_hi, lat2d, lon2d, storm_lat, storm_lon, SHEAR_INNER_DEG, SHEAR_OUTER_DEG
        )
        u_lo_mean, v_lo_mean = mean_uv_in_annulus_deg(
            u_lo, v_lo, lat2d, lon2d, storm_lat, storm_lon, SHEAR_INNER_DEG, SHEAR_OUTER_DEG
        )
        if not (np.isfinite(u_hi_mean) and np.isfinite(u_lo_mean)):
            missing += 1
            continue

        shear_dir = shear_direction_deg(u_hi_mean - u_lo_mean, v_hi_mean - v_lo_mean)
        diffs = compute_quadrant_diffs(z_grid, shear_dir, RADIUS_KM, GRID_KM, AGG)
        df.at[row_idx, out_cols[0]] = diffs["DL_minus_DR"]
        df.at[row_idx, out_cols[1]] = diffs["UL_minus_DL"]
        df.at[row_idx, out_cols[2]] = diffs["UR_minus_UL"]
        df.at[row_idx, out_cols[3]] = diffs["DR_minus_UR"]
        processed += 1

    df.to_csv(OUT_CSV, index=False)
    print(f"Updated rows: {processed}")
    print(f"Missing rows: {missing}")
    print(f"Wrote CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
