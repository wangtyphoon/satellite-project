#!/usr/bin/env python3
"""
Plot FS/CSF/flagShallowRain from GPM DPR 2A files and report shapes.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

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
NPY_DIR = Path("light_rain_npy")
LIGHT_RAIN_MIN = 1.0

ERA5_DIR_TEMPLATE = Path(__file__).resolve().parent / "data_era5_shear_{year}"
ERA5_FILE_RADIUS_DEG = 10.0
SHEAR_LEVELS = (200, 850)
SHEAR_INNER_DEG = 2.5
SHEAR_OUTER_DEG = 8.5
UP_SHEAR_HALF_ANGLE_DEG = 90.0
UP_SHEAR_ARROW_KM = 120.0


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


def _radial_stats(values: np.ndarray, dist_km: np.ndarray, radius_km: float) -> dict:
    mask = np.isfinite(values) & np.isfinite(dist_km) & (dist_km <= radius_km)
    if not mask.any():
        return {"count": 0, "mean": float("nan"), "pct_zero": float("nan")}
    vals = values[mask]
    pct_zero = float(np.mean(vals == 0) * 100.0)
    return {"count": int(mask.sum()), "mean": float(np.nanmean(vals)), "pct_zero": pct_zero}


def _round_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.Timedelta(minutes=30)).floor("h")


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


def _get_coord(ds: xr.Dataset, names: tuple[str, ...]) -> np.ndarray:
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


def load_era5_uv(nc_path: Path, levels: tuple[int, int]) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    try:
        ds = xr.open_dataset(nc_path)
    except ImportError as exc:
        raise ImportError("Missing netCDF4 backend for xarray.") from exc
    try:
        u = ds["u"]
        v = ds["v"]
        lat = _get_coord(ds, ("latitude", "lat"))
        lon = _get_coord(ds, ("longitude", "lon"))
        uv_by_level: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for level in levels:
            u_level = _squeeze_time(_select_level(u, level)).values
            v_level = _squeeze_time(_select_level(v, level)).values
            uv_by_level[level] = (u_level, v_level)
        return uv_by_level, lat, lon
    finally:
        ds.close()


def mean_uv_in_annulus_deg(
    u: np.ndarray,
    v: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    center_lat: float,
    center_lon: float,
    inner_deg: float,
    outer_deg: float,
) -> tuple[float, float]:
    dist = angular_distance_deg(lat2d, lon2d, center_lat, center_lon)
    mask = (dist >= inner_deg) & (dist <= outer_deg) & np.isfinite(u) & np.isfinite(v)
    if not np.any(mask):
        return np.nan, np.nan
    return float(np.mean(u[mask])), float(np.mean(v[mask]))


def shear_direction_deg(u_shear: float, v_shear: float) -> float:
    return float((np.degrees(np.arctan2(u_shear, v_shear)) + 360.0) % 360.0)


def _upshear_light_rain_stats(
    flag_arr: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    shear_dir: float,
    radius_km: float,
    light_rain_min: float,
) -> dict:
    angle = (np.degrees(np.arctan2(x_km, y_km)) + 360.0) % 360.0
    angle_rel = (angle - shear_dir + 360.0) % 360.0
    upshear = (angle_rel >= 90.0) & (angle_rel < 270.0)
    dist_km = np.hypot(x_km, y_km)
    valid = np.isfinite(flag_arr) & np.isfinite(dist_km) & (dist_km <= radius_km)
    mask = valid & upshear
    if not np.any(mask):
        return {"count": 0, "count_zero": 0, "pct_zero": float("nan")}
    light = mask & (flag_arr >= light_rain_min)
    zero = mask & (flag_arr == 0)
    count = int(np.count_nonzero(mask))
    count_light = int(np.count_nonzero(light))
    count_zero = int(np.count_nonzero(zero))
    pct_zero = float(count_zero / count * 100.0)
    return {"count": count, "count_light": count_light, "count_zero": count_zero, "pct_zero": pct_zero}


def _add_shear_overlays(ax: plt.Axes, shear_dir: float, radius_km: float, arrow_km: float) -> None:
    if shear_dir is None or not np.isfinite(shear_dir):
        return
    upshear_center = 90.0 - (shear_dir + 180.0)
    upshear_wedge = Wedge(
        (0.0, 0.0),
        radius_km,
        upshear_center - UP_SHEAR_HALF_ANGLE_DEG,
        upshear_center + UP_SHEAR_HALF_ANGLE_DEG,
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
        arrow_km * math.sin(angle_rad),
        arrow_km * math.cos(angle_rad),
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


def main() -> None:
    # =========================
    # Config (edit as needed)
    # =========================
    csv_path = Path(IN_CSV)
    row_index = 23  # which row in gpm_passes_swath_true.csv
    file_path = None  # set to Path("...") to override the CSV selection (npy)
    swath = None  # set to "FS"/"HS" or keep None to use CSV swath
    crop_radius_km = 150.0  # stats radius around storm center
    outdir = Path("plots_csf_flag_shallow_rain")
    show_plot = False

    if file_path is not None:
        npy_path = Path(file_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"NPY not found: {npy_path}")
        storm_lat = None
        storm_lon = None
        pass_time = None
        sid = None
        granule_file = npy_path.name
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"row_index {row_index} out of range (0..{len(df)-1}).")
        row = df.iloc[row_index]
        year = rzf._infer_year_from_row(row)
        granule_file = row[GRANULE_COL]
        swath = rzf.normalize_swath_name(row.get(SWATH_COL, None))
        sid = row[SID_COL]
        pass_time = rzf._resolve_pass_time(row)

        root = Path(rzf._project_root())
        ibtracs_path = root / IBTRACS_CSV_TEMPLATE.format(year=year)
        if not ibtracs_path.exists():
            raise FileNotFoundError(f"IBTRACS CSV not found: {ibtracs_path}")
        track_df = rzf.load_track_for_sid(ibtracs_path, sid)
        storm_lat, storm_lon = rzf.interpolate_track_position(track_df, pass_time)
        if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
            raise ValueError("Interpolated storm center is not finite.")

        stem = Path(granule_file).stem
        if swath:
            npy_path = NPY_DIR / f"{stem}_{swath}_light_rain.npy"
        else:
            matches = sorted(NPY_DIR.glob(f"{stem}_*_light_rain.npy"))
            if len(matches) == 1:
                npy_path = matches[0]
            else:
                raise FileNotFoundError(f"Missing or ambiguous npy for {stem} in {NPY_DIR}")

    shear_dir = None
    if storm_lat is not None and storm_lon is not None and pass_time is not None and sid is not None:
        era5_time = _round_to_hour(pass_time)
        era5_path = _build_era5_path(sid, era5_time, ERA5_FILE_RADIUS_DEG)
        if era5_path.exists():
            try:
                uv_by_level, lat, lon = load_era5_uv(era5_path, SHEAR_LEVELS)
            except ImportError as exc:
                print(f"ERA5 backend not available, skip shear stats: {exc}")
                uv_by_level = None
            if uv_by_level is not None:
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
                if np.isfinite(u_hi_mean) and np.isfinite(u_lo_mean):
                    shear_dir = shear_direction_deg(u_hi_mean - u_lo_mean, v_hi_mean - v_lo_mean)
        else:
            print(f"ERA5 file not found, skip shear stats: {era5_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    grid_flag = np.load(npy_path)
    print(f"File: {npy_path}")
    _describe("flagShallowRain", grid_flag, None)

    half = rzf.GRID_EXTENT_KM
    step = rzf.GRID_KM
    centers = rzf._grid_centers(step, rzf.GRID_SIZE, half)
    grid_x, grid_y = np.meshgrid(centers, centers)
    dist_km = np.hypot(grid_x, grid_y)

    stats_50 = _radial_stats(grid_flag, dist_km, 50.0)
    stats_100 = _radial_stats(grid_flag, dist_km, 100.0)
    stats_150 = _radial_stats(grid_flag, dist_km, 150.0)
    print("Stats within 50 km:")
    print(
        f"  mean={stats_50['mean']:.3f}, "
        f"pct_zero={stats_50['pct_zero']:.2f}% "
        f"(n={stats_50['count']})"
    )
    print("Stats within 100 km:")
    print(
        f"  mean={stats_100['mean']:.3f}, "
        f"pct_zero={stats_100['pct_zero']:.2f}% "
        f"(n={stats_100['count']})"
    )
    print("Stats within 150 km:")
    print(
        f"  mean={stats_150['mean']:.3f}, "
        f"pct_zero={stats_150['pct_zero']:.2f}% "
        f"(n={stats_150['count']})"
    )
    if shear_dir is not None:
        upshear_stats = _upshear_light_rain_stats(
            grid_flag, grid_x, grid_y, shear_dir, crop_radius_km, LIGHT_RAIN_MIN
        )
        print("Upshear (UR+UL) stats:")
        print(
            f"  pct_zero={upshear_stats['pct_zero']:.2f}% "
            f"(n_zero={upshear_stats['count_zero']}, n={upshear_stats['count']})"
        )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6), dpi=120)
    extent = [-rzf.GRID_EXTENT_KM, rzf.GRID_EXTENT_KM, -rzf.GRID_EXTENT_KM, rzf.GRID_EXTENT_KM]
    im = ax.imshow(grid_flag, origin="lower", extent=extent, cmap="viridis")
    ax.set_title(f"{DATASET_PATH} (storm-centered)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.scatter([0.0], [0.0], s=60, c="white", marker="x", linewidths=2)
    ax.set_xlim(-rzf.GRID_EXTENT_KM, rzf.GRID_EXTENT_KM)
    ax.set_ylim(-rzf.GRID_EXTENT_KM, rzf.GRID_EXTENT_KM)
    if shear_dir is not None:
        overlay_radius = min(crop_radius_km, rzf.GRID_EXTENT_KM)
        overlay_arrow = min(UP_SHEAR_ARROW_KM, rzf.GRID_EXTENT_KM * 0.8)
        _add_shear_overlays(ax, shear_dir, overlay_radius, overlay_arrow)
    fig.colorbar(im, ax=ax, label="flagShallowRain")

    fig.suptitle(f"{Path(granule_file).name}")
    fig.tight_layout()

    if show_plot:
        plt.show()
    else:
        out_path = outdir / f"{Path(granule_file).stem}_csf_flag_shallow_rain_map.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
