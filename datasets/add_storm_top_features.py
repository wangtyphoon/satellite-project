#!/usr/bin/env python3
"""
Add storm-top statistics (max height and pixel fractions) within 50/100 km
to gpm_passes_swath_true.csv using 2A DPR heightStormTop and IBTRACS centers.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# =========================
# Config (edit as needed)
# =========================
IN_CSV = Path("gpm_passes_swath_true.csv")
OUT_CSV = Path("gpm_passes_swath_true_with_stormtop.csv")
DOWNLOAD_DIR_TEMPLATE = "data_gpm_2adpr_{year}"
IBTRACS_CSV_TEMPLATE = "ibtracs_WP_{year}.csv"
VERTICAL_AGG = "max"  # "max", "mean", or None

SWATH_COL = "swath"
SID_COL = "SID"
PASS_TIME_COL = "pass_time_utc"
PASS_START_COL = "pass_start_utc"
PASS_END_COL = "pass_end_utc"
SOURCE_COL = "source"
GRANULE_COL = "granule_file"

OUT_COLS = [
    "stormtop_max_km_r50",
    "stormtop_pct_gt10km_r50",
    "stormtop_pct_gt14km_r50",
    "stormtop_max_km_r100",
    "stormtop_pct_gt10km_r100",
    "stormtop_pct_gt14km_r100",
]


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _project_root() -> Path:
    return _script_dir().parent


def _to_utc_datetime(s):
    return pd.to_datetime(s, errors="coerce", utc=True)


def _infer_year_from_row(row) -> int:
    if PASS_TIME_COL in row and pd.notna(row[PASS_TIME_COL]):
        return pd.to_datetime(row[PASS_TIME_COL], utc=True).year
    if SOURCE_COL in row and pd.notna(row[SOURCE_COL]):
        text = str(row[SOURCE_COL])
        for token in text.split("_"):
            if token.isdigit() and len(token) == 4:
                return int(token)
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 4:
            return int(digits[:4])
    raise ValueError("Could not infer year from row (pass_time_utc/source missing).")


def _resolve_pass_time(row) -> pd.Timestamp:
    if PASS_TIME_COL in row and pd.notna(row[PASS_TIME_COL]):
        return pd.to_datetime(row[PASS_TIME_COL], utc=True)
    if PASS_START_COL in row and PASS_END_COL in row:
        start = pd.to_datetime(row[PASS_START_COL], utc=True)
        end = pd.to_datetime(row[PASS_END_COL], utc=True)
        return start + (end - start) / 2
    raise ValueError("No pass_time_utc or pass_start_utc/pass_end_utc available.")


def _normalize_swath_name(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s.lstrip("/")


def _resolve_swath(h5: h5py.File, preferred: str | None = None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["FS", "NS", "MS", "HS"])
    seen = set()
    for s in candidates:
        s = _normalize_swath_name(s)
        if not s or s in seen:
            continue
        seen.add(s)
        if f"{s}/Latitude" in h5 and f"{s}/Longitude" in h5:
            return s
    raise ValueError("No matching swath group found in granule.")


def _mask_fill(arr: np.ndarray, fill_value: float | int | None) -> np.ndarray:
    if fill_value is None:
        return arr.astype(float, copy=False)
    out = arr.astype(float, copy=True)
    out[out == float(fill_value)] = np.nan
    return out


def _apply_scale_offset(arr: np.ndarray, attrs: dict) -> np.ndarray:
    scale = attrs.get("scale_factor", None)
    offset = attrs.get("add_offset", None)
    if scale is None and offset is None:
        return arr
    scale = float(scale) if scale is not None else 1.0
    offset = float(offset) if offset is not None else 0.0
    return arr * scale + offset


def _reduce_vertical(data: np.ndarray, agg: str | None) -> np.ndarray:
    if data.ndim <= 2:
        return data
    if agg is None:
        raise ValueError("VERTICAL_AGG is None but data has vertical bins.")
    if agg == "max":
        return np.nanmax(data, axis=-1)
    if agg == "mean":
        return np.nanmean(data, axis=-1)
    raise ValueError(f"Unsupported VERTICAL_AGG: {agg}")


def _wrap_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def _latlon_to_local_km(lat, lon, lat0, lon0, radius_km=6371.0):
    dlon = _wrap_lon(lon - lon0)
    x = np.deg2rad(dlon) * radius_km * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * radius_km
    return x, y


def _radial_stats(height_m: np.ndarray, dist_km: np.ndarray, radius_km: float) -> dict:
    mask = (
        np.isfinite(height_m)
        & np.isfinite(dist_km)
        & (dist_km <= radius_km)
        & (height_m > 0)
    )
    if not mask.any():
        return {
            "count": 0,
            "max_km": float("nan"),
            "pct_gt_10km": float("nan"),
            "pct_gt_14km": float("nan"),
        }
    vals_km = height_m[mask] / 1000.0
    count = int(mask.sum())
    max_km = float(np.nanmax(vals_km))
    pct_gt_10km = float(np.mean(vals_km > 10.0) * 100.0)
    pct_gt_14km = float(np.mean(vals_km > 14.0) * 100.0)
    return {
        "count": count,
        "max_km": max_km,
        "pct_gt_10km": pct_gt_10km,
        "pct_gt_14km": pct_gt_14km,
    }


def _load_ibtracs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    time_col = "ISO_TIME" if "ISO_TIME" in df.columns else None
    lat_col = "LAT" if "LAT" in df.columns else ("USA_LAT" if "USA_LAT" in df.columns else None)
    lon_col = "LON" if "LON" in df.columns else ("USA_LON" if "USA_LON" in df.columns else None)
    sid_col = "SID" if "SID" in df.columns else None
    if not (time_col and lat_col and lon_col and sid_col):
        raise ValueError("IBTRACS CSV missing required columns (SID/time/lat/lon).")
    out = pd.DataFrame(
        {
            "sid": df[sid_col],
            "time_utc": _to_utc_datetime(df[time_col]),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["sid", "time_utc", "lat", "lon"])
    return out


def _track_for_sid(df: pd.DataFrame, sid: str) -> pd.DataFrame:
    track = df[df["sid"] == sid].copy()
    if len(track) == 0:
        raise ValueError(f"SID {sid} not found in IBTRACS CSV.")
    track = track.sort_values("time_utc").reset_index(drop=True)
    return track


def _interpolate_track_position(track_df: pd.DataFrame, target_time: pd.Timestamp) -> tuple[float, float]:
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
    return float(lat_i), float(lon_i)


def _compute_features(row, ibtracs_cache: dict[int, pd.DataFrame]) -> dict:
    year = _infer_year_from_row(row)
    granule_file = row[GRANULE_COL]
    swath_pref = _normalize_swath_name(row.get(SWATH_COL, None))
    sid = row[SID_COL]
    pass_time = _resolve_pass_time(row)

    root = _project_root()
    granule_path = root / DOWNLOAD_DIR_TEMPLATE.format(year=year) / granule_file
    if not granule_path.exists():
        raise FileNotFoundError(f"Granule not found: {granule_path}")
    if not h5py.is_hdf5(granule_path):
        raise OSError(f"Not a valid HDF5 file: {granule_path}")

    if year not in ibtracs_cache:
        ibtracs_path = root / IBTRACS_CSV_TEMPLATE.format(year=year)
        if not ibtracs_path.exists():
            raise FileNotFoundError(f"IBTRACS CSV not found: {ibtracs_path}")
        ibtracs_cache[year] = _load_ibtracs(ibtracs_path)
    track_df = _track_for_sid(ibtracs_cache[year], sid)
    storm_lat, storm_lon = _interpolate_track_position(track_df, pass_time)
    if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
        raise ValueError("Interpolated storm center is not finite.")

    with h5py.File(granule_path, "r") as h5:
        swath = _resolve_swath(h5, swath_pref)
        ds_height = h5[f"{swath}/PRE/heightStormTop"]
        ds_lat = h5[f"{swath}/Latitude"]
        ds_lon = h5[f"{swath}/Longitude"]

        height = _mask_fill(ds_height[...], ds_height.attrs.get("_FillValue"))
        height = _apply_scale_offset(height, dict(ds_height.attrs))
        height = _reduce_vertical(height, VERTICAL_AGG)
        lat = _mask_fill(ds_lat[...], ds_lat.attrs.get("_FillValue"))
        lon = _mask_fill(ds_lon[...], ds_lon.attrs.get("_FillValue"))

    lat[(lat < -90.0) | (lat > 90.0)] = np.nan
    lon[(lon < -180.0) | (lon > 180.0)] = np.nan
    x_km, y_km = _latlon_to_local_km(lat, lon, storm_lat, storm_lon)
    dist_km = np.hypot(x_km, y_km)

    stats_50 = _radial_stats(height, dist_km, 50.0)
    stats_100 = _radial_stats(height, dist_km, 100.0)

    return {
        "stormtop_max_km_r50": stats_50["max_km"],
        "stormtop_pct_gt10km_r50": stats_50["pct_gt_10km"],
        "stormtop_pct_gt14km_r50": stats_50["pct_gt_14km"],
        "stormtop_max_km_r100": stats_100["max_km"],
        "stormtop_pct_gt10km_r100": stats_100["pct_gt_10km"],
        "stormtop_pct_gt14km_r100": stats_100["pct_gt_14km"],
    }


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV, low_memory=False)
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    ibtracs_cache: dict[int, pd.DataFrame] = {}
    for idx, row in df.iterrows():
        try:
            feats = _compute_features(row, ibtracs_cache)
            for k, v in feats.items():
                df.at[idx, k] = v
        except Exception as exc:
            print(f"Row {idx} failed: {exc}")
            continue

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
