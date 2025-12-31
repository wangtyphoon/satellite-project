#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download ERA5 pressure-level winds from CSV passes and compute 200-850 hPa shear.
The download area is a radius in degrees around the storm center at pass_time_utc.

Requirements:
  pip install cdsapi xarray netCDF4 pandas numpy
  Configure CDS API credentials in ~/.cdsapirc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import cdsapi
import xarray as xr


# =====================================================
# User config (edit here)
# =====================================================
IN_CSV = Path(__file__).resolve().parent / "gpm_passes_swath_true.csv"
IBTRACS_TEMPLATE = Path(__file__).resolve().parent.parent / "ibtracs_WP_{year}.csv"

SID_COL = "SID"
PASS_TIME_COL = "pass_time_utc"  # fallback handled if missing

RADIUS_DEG = 10.0
LEVELS_HPA = [200, 850]
DATASET = "reanalysis-era5-pressure-levels"

DOWNLOAD_DIR_TEMPLATE = "data_era5_shear_{year}"
SUMMARY_CSV = Path(__file__).resolve().parent / "era5_shear_summary.csv"

MAX_ROWS = None  # set to an integer for quick tests


# =====================================================
# Helpers
# =====================================================


def _wrap_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _to_utc_datetime(value) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _round_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.Timedelta(minutes=30)).floor("H")


def _pick_time_column(df: pd.DataFrame) -> str:
    for col in (
        PASS_TIME_COL,
        "pass_mid_inside_effective_swath_nearest_scan_utc",
        "pass_start_utc",
    ):
        if col in df.columns:
            return col
    raise ValueError("No usable pass time column found in CSV.")


@dataclass
class TrackCache:
    by_year: Dict[int, pd.DataFrame]
    by_sid: Dict[str, pd.DataFrame]


def load_track_for_sid(cache: TrackCache, sid: str, year: int) -> pd.DataFrame:
    if sid in cache.by_sid:
        return cache.by_sid[sid]

    if year not in cache.by_year:
        ib_path = Path(str(IBTRACS_TEMPLATE).format(year=year))
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


def build_area(lat: float, lon: float, radius_deg: float) -> Tuple[float, float, float, float]:
    lat_min = max(-90.0, lat - radius_deg)
    lat_max = min(90.0, lat + radius_deg)
    lon_min = _wrap_lon(lon - radius_deg)
    lon_max = _wrap_lon(lon + radius_deg)
    if lon_min > lon_max:
        # Dateline crossing; fall back to global lon coverage to avoid split request.
        lon_min, lon_max = -180.0, 180.0
    return lat_max, lon_min, lat_min, lon_max


def download_era5(path: Path, dt: pd.Timestamp, area: Tuple[float, float, float, float]) -> None:
    client = cdsapi.Client()
    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(level) for level in LEVELS_HPA],
        "year": f"{dt.year:04d}",
        "month": f"{dt.month:02d}",
        "day": f"{dt.day:02d}",
        "time": f"{dt.hour:02d}:00",
        "area": [float(v) for v in area],
    }
    client.retrieve(DATASET, request, str(path))


def _select_level(da: xr.DataArray, level: int) -> xr.DataArray:
    for dim in ("level", "pressure_level"):
        if dim in da.dims:
            return da.sel({dim: level})
    raise KeyError("Missing level/pressure_level dimension in ERA5 file.")


def compute_shear(nc_path: Path) -> Tuple[float, float, Path]:
    ds = xr.open_dataset(nc_path)
    try:
        u = ds["u"]
        v = ds["v"]
    except KeyError as exc:
        ds.close()
        raise KeyError("Missing u/v wind components in ERA5 file.") from exc

    u200 = _select_level(u, LEVELS_HPA[0])
    v200 = _select_level(v, LEVELS_HPA[0])
    u850 = _select_level(u, LEVELS_HPA[1])
    v850 = _select_level(v, LEVELS_HPA[1])

    shear = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
    shear.name = f"shear_{LEVELS_HPA[0]}_{LEVELS_HPA[1]}"
    shear.attrs["long_name"] = "Deep-layer wind shear magnitude"
    shear.attrs["units"] = u.attrs.get("units", "m s-1")

    shear_mean = float(shear.mean().values)
    shear_max = float(shear.max().values)

    shear_path = nc_path.with_name(nc_path.stem + "_shear.nc")
    shear.to_dataset().to_netcdf(shear_path)
    ds.close()
    return shear_mean, shear_max, shear_path


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {IN_CSV}")

    df = pd.read_csv(IN_CSV, low_memory=False)
    if SID_COL not in df.columns:
        raise ValueError(f"Missing {SID_COL} in CSV.")

    time_col = _pick_time_column(df)
    df["pass_time_utc"] = df[time_col].map(_to_utc_datetime)
    df = df.dropna(subset=["pass_time_utc"])

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    cache = TrackCache(by_year={}, by_sid={})
    summary_rows = []

    for idx, row in df.iterrows():
        sid = str(row[SID_COL])
        pass_time = row["pass_time_utc"]
        if pass_time is None or pd.isna(pass_time):
            continue

        year = int(sid[:4])
        track = load_track_for_sid(cache, sid, year)
        center_lat, center_lon = interpolate_track_position(track, pass_time)
        if not np.isfinite(center_lat) or not np.isfinite(center_lon):
            continue

        era5_time = _round_to_hour(pass_time)
        area = build_area(center_lat, center_lon, RADIUS_DEG)

        out_dir = Path(DOWNLOAD_DIR_TEMPLATE.format(year=era5_time.year))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"era5_{sid}_{era5_time.strftime('%Y%m%d_%H%M')}_r{RADIUS_DEG:g}.nc"
        out_path = out_dir / out_name

        if not out_path.exists():
            download_era5(out_path, era5_time, area)

        shear_mean, shear_max, shear_path = compute_shear(out_path)
        summary_rows.append(
            {
                "sid": sid,
                "pass_time_utc": pass_time.isoformat(),
                "era5_time_utc": era5_time.isoformat(),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "area_n": area[0],
                "area_w": area[1],
                "area_s": area[2],
                "area_e": area[3],
                "shear_mean": shear_mean,
                "shear_max": shear_max,
                "era5_file": str(out_path),
                "shear_file": str(shear_path),
            }
        )

        print(f"[{idx}] {sid} {era5_time} shear_mean={shear_mean:.2f} shear_max={shear_max:.2f}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved summary: {SUMMARY_CSV}")
    else:
        print("No rows processed; check input CSV and time columns.")


if __name__ == "__main__":
    main()
