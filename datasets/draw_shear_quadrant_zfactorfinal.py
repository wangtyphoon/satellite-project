#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute shear direction from ERA5 winds, split zFactorFinal into four quadrants,
and aggregate reflectivity within a 50 km radius (default).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Wedge


IN_CSV = Path(__file__).resolve().parent / "gpm_passes_swath_true.csv"
IBTRACS_TEMPLATE = Path(__file__).resolve().parent.parent / "ibtracs_WP_{year}.csv"
ERA5_DIR_TEMPLATE = Path(__file__).resolve().parent / "data_era5_shear_{year}"
ERA5_FILE_RADIUS_DEG = 10.0

NPY_DIR = Path(__file__).resolve().parent / "zFactorFinal"
GRID_KM_DEFAULT = 1.0

PASS_TIME_COL = "pass_time_utc"
PASS_START_COL = "pass_start_utc"
PASS_END_COL = "pass_end_utc"
SOURCE_COL = "source"
SID_COL = "SID"
GRANULE_COL = "granule_file"

DEFAULT_LEVELS = (200, 850)
DEFAULT_SHEAR_LEVELS = (200, 850)

PLOT_RADIUS_KM = 150.0

ROW_INDEX = 7
RADIUS_KM = 100.0
LEVELS = DEFAULT_LEVELS
SHEAR_LEVELS = DEFAULT_SHEAR_LEVELS
SHEAR_INNER_DEG = 2.5
SHEAR_OUTER_DEG = 8.5
AGGS = ("mean",)
DIFF_AGG = "mean"
GRID_KM = GRID_KM_DEFAULT
KEEP_ZERO = True
OUT_DIR = Path("shear_quadrant_outputs")


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


def load_track_for_sid(csv_path: Path, sid: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
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


def haversine_km(lat, lon, lat0, lon0) -> np.ndarray:
    lat = np.deg2rad(lat)
    lon = np.deg2rad(_wrap_lon(lon))
    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(_wrap_lon(lon0))
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


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


def load_era5_uv(nc_path: Path, levels: Iterable[int]) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
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
    return uv_by_level, lat, lon


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


def load_zfactor_grid(stem: str, prefix: str) -> np.ndarray:
    npy_path = NPY_DIR / f"{prefix}{stem}.npy"
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


def aggregate_quadrants(
    values: np.ndarray,
    quadrants: np.ndarray,
    aggs: Iterable[str],
) -> pd.DataFrame:
    results = []
    for q in (1, 2, 3, 4):
        q_vals = values[quadrants == q]
        q_vals = q_vals[np.isfinite(q_vals)]
        row = {"quadrant": q, "count": int(q_vals.size)}
        for agg in aggs:
            if q_vals.size == 0:
                row[agg] = np.nan
                continue
            if agg == "mean":
                row[agg] = float(np.nanmean(q_vals))
            elif agg == "max":
                row[agg] = float(np.nanmax(q_vals))
            elif agg == "median":
                row[agg] = float(np.nanmedian(q_vals))
            elif agg.startswith("p"):
                pct = float(agg[1:])
                row[agg] = float(np.nanpercentile(q_vals, pct))
            else:
                raise ValueError(f"Unsupported agg: {agg}")
        results.append(row)
    return pd.DataFrame(results)


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


def compute_dl_minus_dr(
    z_grid: np.ndarray,
    quadrants: np.ndarray,
    agg: str,
) -> float:
    dr_vals = z_grid[quadrants == 1]
    dl_vals = z_grid[quadrants == 4]
    dr_vals = dr_vals[np.isfinite(dr_vals)]
    dl_vals = dl_vals[np.isfinite(dl_vals)]

    dr_val = _apply_agg(dr_vals, agg)
    dl_val = _apply_agg(dl_vals, agg)
    if not (np.isfinite(dr_val) and np.isfinite(dl_val)):
        return np.nan
    return float(dl_val - dr_val)


def infer_npy_agg(df: pd.DataFrame, preferred: str) -> str:
    candidates = []
    for agg in ("mean", "max", "median", "p90"):
        if any(col.startswith(f"zFactorFinal_{agg}_") for col in df.columns):
            candidates.append(agg)
    if preferred in candidates:
        return preferred
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    return preferred


def _circle_lonlat(lat0: float, lon0: float, radius_km: float, n: int = 361) -> Tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0.0, 2.0 * np.pi, n)
    dlat = (radius_km / 111.0) * np.cos(ang)
    dlon = (radius_km / (111.0 * np.cos(np.deg2rad(lat0)))) * np.sin(ang)
    return lat0 + dlat, _wrap_lon(lon0 + dlon)


def plot_winds(
    out_path: Path,
    uv_by_level: Dict[int, Tuple[np.ndarray, np.ndarray]],
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> None:
    levels = list(uv_by_level.keys())
    n_levels = len(levels)
    fig, axes = plt.subplots(n_levels, 2, figsize=(10, 3.5 * n_levels), constrained_layout=True)
    if n_levels == 1:
        axes = np.array([axes])

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    circle_lat, circle_lon = _circle_lonlat(center_lat, center_lon, radius_km)

    for i, level in enumerate(levels):
        u, v = uv_by_level[level]
        ax_u = axes[i, 0]
        ax_v = axes[i, 1]

        im_u = ax_u.pcolormesh(lon2d, lat2d, u, shading="auto", cmap="coolwarm")
        im_v = ax_v.pcolormesh(lon2d, lat2d, v, shading="auto", cmap="coolwarm")

        ax_u.plot(circle_lon, circle_lat, color="k", linewidth=1.0, alpha=0.6)
        ax_v.plot(circle_lon, circle_lat, color="k", linewidth=1.0, alpha=0.6)
        ax_u.scatter([center_lon], [center_lat], c="k", s=20, marker="x")
        ax_v.scatter([center_lon], [center_lat], c="k", s=20, marker="x")

        ax_u.set_title(f"ERA5 u wind (level {level} hPa)")
        ax_v.set_title(f"ERA5 v wind (level {level} hPa)")
        ax_u.set_xlabel("Longitude")
        ax_u.set_ylabel("Latitude")
        ax_v.set_xlabel("Longitude")
        ax_v.set_ylabel("Latitude")

        fig.colorbar(im_u, ax=ax_u, label="m s-1")
        fig.colorbar(im_v, ax=ax_v, label="m s-1")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reflectivity(
    out_path: Path,
    data: np.ndarray,
    grid_km: float,
    shear_dir: float,
    radius_km: float,
) -> None:
    h, w = data.shape
    half_extent = grid_km * (w - 1) / 2.0
    extent = [-half_extent, half_extent, -half_extent, half_extent]
    plot_radius = max(PLOT_RADIUS_KM, radius_km * 1.2)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=0.0,
        vmax=np.nanpercentile(data[np.isfinite(data)], 95) if np.isfinite(data).any() else 40.0,
    )
    circle = plt.Circle((0.0, 0.0), radius_km, color="k", fill=False, linewidth=1.0, alpha=0.7)
    ax.add_patch(circle)
    ax.scatter([0.0], [0.0], c="white", s=40, marker="x", linewidths=2)

    upshear_center = 90.0 - (shear_dir + 180.0)
    upshear_wedge = Wedge(
        (0.0, 0.0),
        radius_km,
        upshear_center - 90.0,
        upshear_center + 90.0,
        facecolor="#ffd166",
        alpha=0.18,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(upshear_wedge)

    angle_rad = math.radians(shear_dir)
    ax.arrow(
        0.0,
        0.0,
        radius_km * 0.7 * math.sin(angle_rad),
        radius_km * 0.7 * math.cos(angle_rad),
        width=1.2,
        head_width=6.0,
        head_length=8.0,
        color="white",
        length_includes_head=True,
        alpha=0.9,
    )
    upshear_label_rad = math.radians((shear_dir + 180.0) % 360.0)
    ax.text(
        radius_km * 0.75 * math.sin(upshear_label_rad),
        radius_km * 0.75 * math.cos(upshear_label_rad),
        "UPSHEAR",
        color="black",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffd166", alpha=0.7, edgecolor="none"),
        zorder=4,
    )

    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.set_xlabel("X (km, east)")
    ax.set_ylabel("Y (km, north)")
    ax.set_title("zFactorFinal (storm-centered grid)")
    fig.colorbar(im, ax=ax, label="zFactorFinal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_quadrants(
    out_path: Path,
    data: np.ndarray,
    quadrants: np.ndarray,
    grid_km: float,
    shear_dir: float,
    radius_km: float,
) -> None:
    h, w = data.shape
    half_extent = grid_km * (w - 1) / 2.0
    extent = [-half_extent, half_extent, -half_extent, half_extent]
    plot_radius = max(PLOT_RADIUS_KM, radius_km * 1.2)

    quad_colors = ["#00000000", "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"]
    quad_cmap = ListedColormap(quad_colors)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=0.0,
        vmax=np.nanpercentile(data[np.isfinite(data)], 95) if np.isfinite(data).any() else 40.0,
    )
    ax.imshow(
        quadrants,
        origin="lower",
        extent=extent,
        cmap=quad_cmap,
        alpha=0.18,
        interpolation="nearest",
    )

    upshear_center = 90.0 - (shear_dir + 180.0)
    upshear_wedge = Wedge(
        (0.0, 0.0),
        radius_km,
        upshear_center - 90.0,
        upshear_center + 90.0,
        facecolor="#ffd166",
        alpha=0.18,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(upshear_wedge)

    boundaries = [
        math.radians((shear_dir + 0.0) % 360.0),
        math.radians((shear_dir + 90.0) % 360.0),
        math.radians((shear_dir + 180.0) % 360.0),
        math.radians((shear_dir + 270.0) % 360.0),
    ]
    for angle in boundaries:
        ax.plot(
            [0.0, radius_km * math.sin(angle)],
            [0.0, radius_km * math.cos(angle)],
            color="white",
            linewidth=2.0,
        )

    circle = plt.Circle((0.0, 0.0), radius_km, color="white", fill=False, linewidth=1.0, alpha=0.8)
    ax.add_patch(circle)
    ax.scatter([0.0], [0.0], c="white", s=40, marker="x", linewidths=2)

    angle_rad = math.radians(shear_dir)
    ax.arrow(
        0.0,
        0.0,
        radius_km * 0.7 * math.sin(angle_rad),
        radius_km * 0.7 * math.cos(angle_rad),
        width=1.2,
        head_width=6.0,
        head_length=8.0,
        color="white",
        length_includes_head=True,
        alpha=0.9,
        zorder=4,
    )
    upshear_label_rad = math.radians((shear_dir + 180.0) % 360.0)
    ax.text(
        radius_km * 0.75 * math.sin(upshear_label_rad),
        radius_km * 0.75 * math.cos(upshear_label_rad),
        "UPSHEAR",
        color="black",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffd166", alpha=0.7, edgecolor="none"),
        zorder=5,
    )

    label_r = radius_km * 0.6
    labels = [
        ("DR", (shear_dir + 45.0) % 360.0),
        ("UR", (shear_dir + 135.0) % 360.0),
        ("UL", (shear_dir + 225.0) % 360.0),
        ("DL", (shear_dir + 315.0) % 360.0),
    ]
    for text, angle in labels:
        ang = math.radians(angle)
        x = label_r * math.sin(ang)
        y = label_r * math.cos(ang)
        ax.text(
            x,
            y,
            text,
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"),
        )

    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.set_xlabel("X (km, east)")
    ax.set_ylabel("Y (km, north)")
    ax.set_title("Quadrants by shear direction")
    fig.colorbar(im, ax=ax, label="zFactorFinal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    if df.empty:
        raise ValueError(f"{IN_CSV} is empty.")
    if ROW_INDEX < 0 or ROW_INDEX >= len(df):
        raise IndexError(f"ROW_INDEX {ROW_INDEX} out of range (0-{len(df) - 1}).")

    npy_agg = infer_npy_agg(df, AGGS[0])
    npy_prefix = f"1_{npy_agg}_"

    row = df.iloc[ROW_INDEX]
    sid = str(row[SID_COL])
    granule_file = str(row[GRANULE_COL])
    stem = Path(granule_file).stem
    pass_time = _resolve_pass_time(row)
    year = _infer_year_from_row(row)

    ib_path = IBTRACS_TEMPLATE.with_name(IBTRACS_TEMPLATE.name.format(year=year))
    track_df = load_track_for_sid(ib_path, sid)
    storm_lat, storm_lon = interpolate_track_position(track_df, pass_time)
    if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
        raise ValueError("Interpolated storm center is not finite.")

    era5_time = _round_to_hour(pass_time)
    era5_path = _build_era5_path(sid, era5_time, ERA5_FILE_RADIUS_DEG)
    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {era5_path}")

    uv_by_level, lat, lon = load_era5_uv(era5_path, LEVELS)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    shear_hi, shear_lo = SHEAR_LEVELS
    if shear_hi not in uv_by_level or shear_lo not in uv_by_level:
        raise ValueError("Shear levels not present in loaded ERA5 levels.")

    u_hi, v_hi = uv_by_level[shear_hi]
    u_lo, v_lo = uv_by_level[shear_lo]
    u_hi_mean, v_hi_mean = mean_uv_in_annulus_deg(
        u_hi, v_hi, lat2d, lon2d, storm_lat, storm_lon, SHEAR_INNER_DEG, SHEAR_OUTER_DEG
    )
    u_lo_mean, v_lo_mean = mean_uv_in_annulus_deg(
        u_lo, v_lo, lat2d, lon2d, storm_lat, storm_lon, SHEAR_INNER_DEG, SHEAR_OUTER_DEG
    )
    u_shear = u_hi_mean - u_lo_mean
    v_shear = v_hi_mean - v_lo_mean
    shear_mag = float(np.hypot(u_shear, v_shear))
    shear_dir = shear_direction_deg(u_shear, v_shear)

    z_grid = load_zfactor_grid(stem, npy_prefix).astype(float)
    if not KEEP_ZERO:
        z_grid[z_grid == 0.0] = np.nan

    x_km, y_km = grid_xy_km(z_grid.shape, GRID_KM)
    r_km = np.hypot(x_km, y_km)
    angle = (np.degrees(np.arctan2(x_km, y_km)) + 360.0) % 360.0
    angle_rel = (angle - shear_dir + 360.0) % 360.0

    within = r_km <= RADIUS_KM
    quadrants = quadrant_indices(angle_rel)
    quadrants = np.where(within & np.isfinite(z_grid), quadrants, 0)

    stats = aggregate_quadrants(z_grid[within], quadrants[within], AGGS)
    quad_names = {1: "DR", 2: "UR", 3: "UL", 4: "DL"}
    stats["name"] = stats["quadrant"].map(quad_names)
    stats = stats[["quadrant", "name", "count", *AGGS]]

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"quadrant_stats_row{ROW_INDEX}_{stem}.csv"
    stats.to_csv(stats_path, index=False)

    wind_path = out_dir / f"winds_levels_row{ROW_INDEX}_{stem}.png"
    refl_path = out_dir / f"zfactorfinal_row{ROW_INDEX}_{stem}.png"
    quad_path = out_dir / f"quadrants_row{ROW_INDEX}_{stem}.png"

    plot_winds(wind_path, uv_by_level, lat, lon, storm_lat, storm_lon, RADIUS_KM)
    plot_reflectivity(refl_path, z_grid, GRID_KM, shear_dir, RADIUS_KM)
    plot_quadrants(quad_path, z_grid, quadrants, GRID_KM, shear_dir, RADIUS_KM)

    dl_minus_dr = compute_dl_minus_dr(z_grid, quadrants, DIFF_AGG)
    mean_by_quad = dict(zip(stats["name"], stats["mean"]))
    print(f"Row: {ROW_INDEX}")
    print(f"SID: {sid}")
    print(f"Granule: {granule_file}")
    print(f"Storm center: lat={storm_lat:.3f}, lon={storm_lon:.3f}")
    print(f"ERA5 file: {era5_path}")
    print(f"Shear levels: {shear_hi}-{shear_lo} hPa")
    print(f"Shear mean (u,v): ({u_shear:.2f}, {v_shear:.2f}) m s-1")
    print(f"Shear magnitude: {shear_mag:.2f} m s-1")
    print(f"Shear direction: {shear_dir:.1f} deg (toward, clockwise from north)")
    print(f"Quadrant mean (DR/UR/UL/DL): {mean_by_quad}")
    print(f"DL-DR ({DIFF_AGG}) within {RADIUS_KM:g} km: {dl_minus_dr:.3f}")
    print(f"Wrote stats: {stats_path}")
    print(f"Wrote plots: {wind_path}, {refl_path}, {quad_path}")


if __name__ == "__main__":
    main()
