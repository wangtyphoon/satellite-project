#!/usr/bin/env python3
"""
Plot PRE/heightStormTop from GPM DPR 2A files and report shapes.
"""

from __future__ import annotations

from pathlib import Path

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


def _mask_fill(arr: np.ndarray, fill_value: float | int | None) -> np.ndarray:
    if fill_value is None:
        return arr.astype(float, copy=False)
    out = arr.astype(float, copy=True)
    out[out == float(fill_value)] = 0
    return out


def _describe(name: str, arr: np.ndarray, units: str | None) -> None:
    finite = np.isfinite(arr)
    if finite.any():
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
    else:
        vmin = float("nan")
        vmax = float("nan")
    unit_text = f" {units}" if units else ""
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={vmin:.3f}, max={vmax:.3f}{unit_text}")


def _reduce_vertical(data: np.ndarray, agg: str | None) -> np.ndarray:
    if data.ndim <= 2:
        return data
    if agg is None:
        raise ValueError("vertical_agg is None but data has vertical bins.")
    if agg == "max":
        return np.nanmax(data, axis=-1)
    if agg == "mean":
        return np.nanmean(data, axis=-1)
    raise ValueError(f"Unsupported vertical_agg: {agg}")


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


def load_track_for_sid(csv_path: Path, sid: str) -> pd.DataFrame:
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


def interpolate_track_position(track_df: pd.DataFrame, target_time: pd.Timestamp) -> tuple[float, float]:
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


def main() -> None:
    # =========================
    # Config (edit as needed)
    # =========================
    csv_path = Path(IN_CSV)
    row_index = 0  # which row in gpm_passes_swath_true.csv
    file_path = None  # set to Path("...") to override the CSV selection
    swath = None  # set to "FS"/"HS" or keep None to use CSV swath
    subset = (slice(None), slice(None))  # (row_slice, col_slice)
    vertical_agg = "max"  # "max", "mean", or None
    crop_radius_km = 150.0  # crop around storm center
    outdir = Path("plots_pre_bin_storm_top")
    show_plot = False

    if file_path is not None:
        hdf_path = file_path
        if not h5py.is_hdf5(hdf_path):
            raise OSError(f"Not a valid HDF5 file: {hdf_path}")
        storm_lat = None
        storm_lon = None
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"row_index {row_index} out of range (0..{len(df)-1}).")
        row = df.iloc[row_index]
        year = _infer_year_from_row(row)
        granule_file = row[GRANULE_COL]
        swath = _normalize_swath_name(row.get(SWATH_COL, None))
        sid = row[SID_COL]
        pass_time = _resolve_pass_time(row)

        root = _project_root()
        download_dir = root / DOWNLOAD_DIR_TEMPLATE.format(year=year)
        hdf_path = download_dir / granule_file
        if not hdf_path.exists():
            raise FileNotFoundError(f"Granule not found: {hdf_path}")
        if not h5py.is_hdf5(hdf_path):
            raise OSError(f"Not a valid HDF5 file: {hdf_path}")

        ibtracs_path = root / IBTRACS_CSV_TEMPLATE.format(year=year)
        if not ibtracs_path.exists():
            raise FileNotFoundError(f"IBTRACS CSV not found: {ibtracs_path}")
        track_df = load_track_for_sid(ibtracs_path, sid)
        storm_lat, storm_lon = interpolate_track_position(track_df, pass_time)
        if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
            raise ValueError("Interpolated storm center is not finite.")

    row_slice, col_slice = subset
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf_path, "r") as f:
        group = _resolve_swath(f, swath)
        ds_height = f[f"{group}/PRE/heightStormTop"]
        ds_lat = f[f"{group}/Latitude"]
        ds_lon = f[f"{group}/Longitude"]

        height_arr = _mask_fill(ds_height[:], ds_height.attrs.get("_FillValue"))
        height_arr = _reduce_vertical(height_arr, vertical_agg)
        lat_arr = _mask_fill(ds_lat[:], ds_lat.attrs.get("_FillValue"))
        lon_arr = _mask_fill(ds_lon[:], ds_lon.attrs.get("_FillValue"))

        height_arr = height_arr[row_slice, col_slice]
        lat_arr = lat_arr[row_slice, col_slice]
        lon_arr = lon_arr[row_slice, col_slice]

        print(f"File: {hdf_path}")
        _describe("heightStormTop", height_arr, ds_height.attrs.get("units") or ds_height.attrs.get("Units"))
        _describe("Latitude", lat_arr, "deg")
        _describe("Longitude", lon_arr, "deg")

        dist_km = None
        if storm_lat is not None and storm_lon is not None:
            x_km, y_km = _latlon_to_local_km(lat_arr, lon_arr, storm_lat, storm_lon)
            dist_km = np.hypot(x_km, y_km)
            stats_50 = _radial_stats(height_arr, dist_km, 50.0)
            stats_100 = _radial_stats(height_arr, dist_km, 100.0)
            print("Stats within 50 km:")
            print(
                f"  max_height_km={stats_50['max_km']:.2f}, "
                f"pct_gt_10km={stats_50['pct_gt_10km']:.2f}%, "
                f"pct_gt_14km={stats_50['pct_gt_14km']:.2f}% "
                f"(n={stats_50['count']})"
            )
            print("Stats within 100 km:")
            print(
                f"  max_height_km={stats_100['max_km']:.2f}, "
                f"pct_gt_10km={stats_100['pct_gt_10km']:.2f}%, "
                f"pct_gt_14km={stats_100['pct_gt_14km']:.2f}% "
                f"(n={stats_100['count']})"
            )
            height_arr = np.where(dist_km <= crop_radius_km, height_arr, np.nan)

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6), dpi=120)
        im = ax.pcolormesh(lon_arr, lat_arr, height_arr, shading="auto")
        ax.set_title(f"{group}/PRE/heightStormTop")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="m")
        if storm_lat is not None and storm_lon is not None:
            ax.scatter([storm_lon], [storm_lat], s=60, c="white", edgecolors="black", marker="x", linewidths=2)
            dlon = crop_radius_km / (111.0 * np.cos(np.deg2rad(storm_lat)))
            dlat = crop_radius_km / 111.0
            ax.set_xlim(storm_lon - dlon, storm_lon + dlon)
            ax.set_ylim(storm_lat - dlat, storm_lat + dlat)

        fig.suptitle(f"{hdf_path.name}")
        fig.tight_layout()

        if show_plot:
            plt.show()
        else:
            out_path = outdir / f"{hdf_path.stem}_{group}_pre_height_storm_top_map.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
