#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count zFactorFinal data points within 250 km of storm center for the first row in
gpm_passes_swath_true.csv.
"""

from __future__ import annotations

import os
import re

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

DATASET_CANDIDATES = [
    "SLV/zFactorFinal",
    "SLV/zFactorFinalNearSurface",
    "SLV/zFactorFinalESurface",
]
CHANNEL = 0
VERTICAL_AGG = "mean"
DIST_THRESHOLD_KM = 250.0
PLOT_OUTPUT = "zfactorfinal_250km_points.png"


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _project_root() -> str:
    return os.path.abspath(os.path.join(_script_dir(), ".."))


def normalize_swath_name(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s.lstrip("/")


def resolve_swath(h5, preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["FS", "NS", "MS", "HS"])
    seen = set()
    for s in candidates:
        s = normalize_swath_name(s)
        if not s or s in seen:
            continue
        seen.add(s)
        if f"{s}/Latitude" in h5 and f"{s}/Longitude" in h5 and f"{s}/ScanTime/Year" in h5:
            return s
    raise ValueError("No matching swath group found in granule.")


def find_dataset_path(h5, swath_prefix, candidates):
    for ds in candidates:
        path = f"{swath_prefix}/{ds}"
        if path in h5:
            return path
    return None


def squeeze_field(data, channel):
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data
    if data.ndim == 4:
        if channel < 0 or channel >= data.shape[-1]:
            raise IndexError(f"CHANNEL {channel} out of range for data shape {data.shape}.")
        return data[..., channel]
    raise ValueError(f"Unsupported data shape {data.shape} for plan view.")


def reduce_vertical(data, agg):
    if data.ndim != 3:
        return data
    if agg is None:
        raise ValueError("VERTICAL_AGG is None but data has vertical bins.")
    if not np.isfinite(data).any():
        return np.full(data.shape[:2], np.nan, dtype=data.dtype)
    if agg == "max":
        return np.nanmax(data, axis=2)
    if agg == "mean":
        return np.nanmean(data, axis=2)
    raise ValueError(f"Unsupported VERTICAL_AGG {agg}.")


def _to_utc_datetime(s):
    return pd.to_datetime(s, errors="coerce", utc=True)


def load_track_for_sid(csv_path, sid):
    df = pd.read_csv(csv_path, low_memory=False)
    time_col = "ISO_TIME" if "ISO_TIME" in df.columns else None
    lat_col = "LAT" if "LAT" in df.columns else ("USA_LAT" if "USA_LAT" in df.columns else None)
    lon_col = "LON" if "LON" in df.columns else ("USA_LON" if "USA_LON" in df.columns else None)
    sid_col = "SID" if "SID" in df.columns else None
    if not (time_col and lat_col and lon_col and sid_col):
        raise ValueError("IBTRACS CSV missing required columns (SID/time/lat/lon).")

    df = df[df[sid_col] == sid].copy()
    if len(df) == 0:
        raise ValueError(f"SID {sid} not found in IBTRACS CSV.")
    df["time_utc"] = _to_utc_datetime(df[time_col])
    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["time_utc", "lat", "lon"])
    df = df.sort_values("time_utc").reset_index(drop=True)
    return df


def interpolate_track_position(track_df, target_time):
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
    lon_i = ((lon_i + 180.0) % 360.0) - 180.0
    return lat_i, lon_i


def _wrap_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def haversine_km(lat, lon, lat0, lon0):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(_wrap_lon(lon))
    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(_wrap_lon(lon0))
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _infer_year_from_row(row) -> int:
    if PASS_TIME_COL in row and pd.notna(row[PASS_TIME_COL]):
        return pd.to_datetime(row[PASS_TIME_COL], utc=True).year
    if SOURCE_COL in row and pd.notna(row[SOURCE_COL]):
        m = re.search(r"(\\d{4})", str(row[SOURCE_COL]))
        if m:
            return int(m.group(1))
    raise ValueError("Could not infer year from row (pass_time_utc/source missing).")


def _resolve_pass_time(row) -> pd.Timestamp:
    if PASS_TIME_COL in row and pd.notna(row[PASS_TIME_COL]):
        return pd.to_datetime(row[PASS_TIME_COL], utc=True)
    if PASS_START_COL in row and PASS_END_COL in row:
        start = pd.to_datetime(row[PASS_START_COL], utc=True)
        end = pd.to_datetime(row[PASS_END_COL], utc=True)
        return start + (end - start) / 2
    raise ValueError("No pass_time_utc or pass_start_utc/pass_end_utc available.")


def main() -> None:
    csv_path = os.path.join(_script_dir(), IN_CSV)
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"{IN_CSV} is empty.")

    row = df.iloc[0]
    year = _infer_year_from_row(row)
    granule_file = row[GRANULE_COL]
    swath_pref = normalize_swath_name(row.get(SWATH_COL, None))
    sid = row[SID_COL]
    pass_time = _resolve_pass_time(row)

    root = _project_root()
    download_dir = os.path.join(root, DOWNLOAD_DIR_TEMPLATE.format(year=year))
    granule_path = os.path.join(download_dir, granule_file)
    if not os.path.exists(granule_path):
        raise FileNotFoundError(f"Granule not found: {granule_path}")

    ibtracs_path = os.path.join(root, IBTRACS_CSV_TEMPLATE.format(year=year))
    track_df = load_track_for_sid(ibtracs_path, sid)
    storm_lat, storm_lon = interpolate_track_position(track_df, pass_time)
    if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
        raise ValueError("Interpolated storm center is not finite.")

    with h5py.File(granule_path, "r") as h5:
        swath = resolve_swath(h5, swath_pref)
        lat = h5[f"{swath}/Latitude"][...].astype(np.float32)
        lon = h5[f"{swath}/Longitude"][...].astype(np.float32)

        data_path = find_dataset_path(h5, swath, DATASET_CANDIDATES)
        if data_path is None:
            raise ValueError(f"No dataset found under {swath} for {DATASET_CANDIDATES}.")
        ds = h5[data_path]
        data = ds[...]
        attrs = {k: ds.attrs[k] for k in ds.attrs.keys()}

    data = squeeze_field(data, CHANNEL).astype(np.float32)
    data = reduce_vertical(data, VERTICAL_AGG)

    fill = attrs.get("_FillValue", None)
    if fill is not None:
        try:
            data[data == float(fill)] = np.nan
        except Exception:
            pass

    if data.shape != lat.shape:
        raise ValueError(f"Shape mismatch: data {data.shape} vs lat {lat.shape}.")

    valid_xy = np.isfinite(lat) & np.isfinite(lon)
    data = np.where(valid_xy & ~np.isfinite(data), 0.0, data)
    valid = valid_xy & np.isfinite(data)
    dist = haversine_km(lat, lon, storm_lat, storm_lon)
    within = valid & (dist <= DIST_THRESHOLD_KM)
    count = int(np.count_nonzero(within))

    print(f"First granule: {granule_file}")
    print(f"SID: {sid}")
    print(f"Pass time (UTC): {pass_time}")
    print(f"Storm center: lat={storm_lat:.3f}, lon={storm_lon:.3f}")
    print(f"Dataset: {data_path}")
    print(f"Points within {DIST_THRESHOLD_KM:.0f} km: {count}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        lon[valid & ~within],
        lat[valid & ~within],
        s=4,
        c="#9e9e9e",
        alpha=0.5,
        label="Outside 250 km",
    )
    ax.scatter(
        lon[within],
        lat[within],
        s=6,
        c="#d62728",
        alpha=0.8,
        label="Within 250 km",
    )
    ax.scatter(
        [storm_lon],
        [storm_lat],
        s=50,
        c="#1f77b4",
        marker="x",
        linewidths=2,
        label="Storm center",
    )
    ax.set_xlim(storm_lon - 3, storm_lon + 3)
    ax.set_ylim(storm_lat - 3, storm_lat + 3)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("zFactorFinal points within 250 km")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = os.path.join(_script_dir(), PLOT_OUTPUT)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
