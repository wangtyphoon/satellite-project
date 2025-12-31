#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regrid zFactorFinal to a 5 km grid (100x100 over 500x500 km) and visualize it.
"""

from __future__ import annotations

import os
import re
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

DATASET_CANDIDATES = [
    "SLV/zFactorFinal",
    "SLV/zFactorFinalNearSurface",
   #"SLV/zFactorFinalESurface",
]
CHANNEL = 0
VERTICAL_AGG = "max"

GRID_KM = 1
GRID_SIZE = 490  # 100x100 over 500x500 km
GRID_EXTENT_KM = GRID_KM * GRID_SIZE / 2.0  # 250 km
PLOT_OUTPUT = "zfactorfinal_regrid_5km.png"
INTERP_METHOD = "gaussian"  # bin_mean, nearest, bilinear, gaussian, barnes, cressman
WEIGHT_RADIUS_KM = 7.5
GAUSS_SIGMA_KM = 5
BARNES_KAPPA_KM2 = 200.0
INNER_RADIUS_KM = 75.0
OUTER_RADIUS_KM = 150


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


def _latlon_to_local_km(lat, lon, lat0, lon0, radius_km=6371.0):
    dlon = _wrap_lon(lon - lon0)
    x = np.deg2rad(dlon) * radius_km * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * radius_km
    return x, y


def _apply_valid_range(data, attrs):
    valid_range = attrs.get("valid_range", None)
    if valid_range is None:
        vmin = attrs.get("valid_min", None)
        vmax = attrs.get("valid_max", None)
        if vmin is not None and vmax is not None:
            valid_range = (vmin, vmax)
    if valid_range is None:
        return data
    lo = float(valid_range[0])
    hi = float(valid_range[1])
    data[(data < lo) | (data > hi)] = np.nan
    return data


def _grid_centers(step_km, size, half_extent_km):
    start = -half_extent_km + step_km / 2.0
    return start + step_km * np.arange(size)


def _grid_swath_mask(x_km, y_km, step_km, size, half_extent_km):
    edges = np.linspace(-half_extent_km, half_extent_km, size + 1)
    xi = np.digitize(x_km, edges) - 1
    yi = np.digitize(y_km, edges) - 1
    in_bounds = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    xi = xi[in_bounds]
    yi = yi[in_bounds]
    mask = np.zeros((size, size), dtype=bool)
    mask[yi, xi] = True
    return mask


def _regrid_to_grid(x_km, y_km, values, method, step_km, size, half_extent_km):
    centers = _grid_centers(step_km, size, half_extent_km)
    if method == "bin_mean":
        edges = np.linspace(-half_extent_km, half_extent_km, size + 1)
        xi = np.digitize(x_km, edges) - 1
        yi = np.digitize(y_km, edges) - 1
        in_bounds = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
        xi = xi[in_bounds]
        yi = yi[in_bounds]
        vals = values[in_bounds]
        grid_sum = np.zeros((size, size), dtype=np.float64)
        grid_count = np.zeros((size, size), dtype=np.int64)
        np.add.at(grid_sum, (yi, xi), vals)
        np.add.at(grid_count, (yi, xi), 1)
        with np.errstate(invalid="ignore"):
            grid_mean = grid_sum / grid_count
        grid_mean[grid_count == 0] = np.nan
        return grid_mean

    if method == "nearest":
        xi = np.floor((x_km + half_extent_km) / step_km).astype(int)
        yi = np.floor((y_km + half_extent_km) / step_km).astype(int)
        in_bounds = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
        xi = xi[in_bounds]
        yi = yi[in_bounds]
        vals = values[in_bounds]
        cx = centers[xi]
        cy = centers[yi]
        dist2 = (x_km[in_bounds] - cx) ** 2 + (y_km[in_bounds] - cy) ** 2
        grid_dist2 = np.full((size, size), np.inf, dtype=np.float64)
        grid_val = np.full((size, size), np.nan, dtype=np.float32)
        for idx in range(len(vals)):
            i = xi[idx]
            j = yi[idx]
            d2 = dist2[idx]
            if d2 < grid_dist2[j, i]:
                grid_dist2[j, i] = d2
                grid_val[j, i] = vals[idx]
        return grid_val

    if method == "bilinear":
        gx = (x_km - centers[0]) / step_km
        gy = (y_km - centers[0]) / step_km
        i0 = np.floor(gx).astype(int)
        j0 = np.floor(gy).astype(int)
        wx1 = gx - i0
        wy1 = gy - j0
        i1 = i0 + 1
        j1 = j0 + 1
        grid_sum = np.zeros((size, size), dtype=np.float64)
        grid_w = np.zeros((size, size), dtype=np.float64)
        for idx in range(len(values)):
            i0i = i0[idx]
            i1i = i1[idx]
            j0i = j0[idx]
            j1i = j1[idx]
            if i1i < 0 or j1i < 0 or i0i >= size or j0i >= size:
                continue
            v = values[idx]
            wx0 = 1.0 - wx1[idx]
            wy0 = 1.0 - wy1[idx]
            if 0 <= i0i < size and 0 <= j0i < size:
                w = wx0 * wy0
                grid_sum[j0i, i0i] += v * w
                grid_w[j0i, i0i] += w
            if 0 <= i1i < size and 0 <= j0i < size:
                w = wx1[idx] * wy0
                grid_sum[j0i, i1i] += v * w
                grid_w[j0i, i1i] += w
            if 0 <= i0i < size and 0 <= j1i < size:
                w = wx0 * wy1[idx]
                grid_sum[j1i, i0i] += v * w
                grid_w[j1i, i0i] += w
            if 0 <= i1i < size and 0 <= j1i < size:
                w = wx1[idx] * wy1[idx]
                grid_sum[j1i, i1i] += v * w
                grid_w[j1i, i1i] += w
        with np.errstate(invalid="ignore", divide="ignore"):
            out = grid_sum / grid_w
        out[grid_w == 0] = np.nan
        return out

    if method in {"gaussian", "barnes", "cressman"}:
        radius = float(WEIGHT_RADIUS_KM)
        radius2 = radius * radius
        grid_sum = np.zeros((size, size), dtype=np.float64)
        grid_w = np.zeros((size, size), dtype=np.float64)
        for idx in range(len(values)):
            x = x_km[idx]
            y = y_km[idx]
            v = values[idx]
            ix0 = int(np.floor((x - radius - centers[0]) / step_km))
            ix1 = int(np.floor((x + radius - centers[0]) / step_km))
            iy0 = int(np.floor((y - radius - centers[0]) / step_km))
            iy1 = int(np.floor((y + radius - centers[0]) / step_km))
            if ix1 < 0 or iy1 < 0 or ix0 >= size or iy0 >= size:
                continue
            ix0 = max(ix0, 0)
            iy0 = max(iy0, 0)
            ix1 = min(ix1, size - 1)
            iy1 = min(iy1, size - 1)
            xs = centers[ix0 : ix1 + 1]
            ys = centers[iy0 : iy1 + 1]
            dx = xs[None, :] - x
            dy = ys[:, None] - y
            d2 = dx * dx + dy * dy
            if method == "gaussian":
                sigma2 = float(GAUSS_SIGMA_KM) ** 2
                w = np.exp(-0.5 * d2 / sigma2)
            elif method == "barnes":
                w = np.exp(-d2 / float(BARNES_KAPPA_KM2))
            else:
                w = (radius2 - d2) / (radius2 + d2)
            w = np.where(d2 <= radius2, w, 0.0)
            grid_sum[iy0 : iy1 + 1, ix0 : ix1 + 1] += w * v
            grid_w[iy0 : iy1 + 1, ix0 : ix1 + 1] += w
        with np.errstate(invalid="ignore", divide="ignore"):
            out = grid_sum / grid_w
        out[grid_w == 0] = np.nan
        return out

    raise ValueError(f"Unsupported INTERP_METHOD {method}.")


def _radial_means(data, step_km, inner_km, outer_km):
    size_y, size_x = data.shape
    if size_x != size_y:
        raise ValueError(f"Expected square grid, got {data.shape}.")
    center = (size_x - 1) / 2.0
    y, x = np.indices(data.shape, dtype=float)
    x = (x - center) * step_km
    y = (y - center) * step_km
    r = np.sqrt(x**2 + y**2)
    inner_mask = r <= inner_km
    outer_mask = (r > inner_km) & (r <= outer_km)
    inner_mean = np.nanmean(data[inner_mask]) if inner_mask.any() else np.nan
    outer_mean = np.nanmean(data[outer_mask]) if outer_mask.any() else np.nan
    return inner_mean, outer_mean


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


def _output_paths(dataset_name, dataset_idx, granule_file):
    out_dir = os.path.join(_script_dir(), dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(granule_file).stem
    out_npy = os.path.join(out_dir, f"{dataset_idx}_{VERTICAL_AGG}_{stem}.npy")
    plot_base = Path(PLOT_OUTPUT)
    out_plot = os.path.join(
        _script_dir(), f"{plot_base.stem}_{stem}{plot_base.suffix}"
    )
    return out_npy, out_plot


def _process_row(row, row_idx):
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
        dataset_name = data_path.split("/")[-1]
        dataset_idx = None
        for i, candidate in enumerate(DATASET_CANDIDATES, start=1):
            if data_path.endswith(candidate):
                dataset_idx = i
                break
        if dataset_idx is None:
            dataset_idx = 0

    data = squeeze_field(data, CHANNEL).astype(np.float32)
    fill = attrs.get("_FillValue", attrs.get("missing_value", None))
    if fill is not None:
        try:
            data[np.isclose(data, float(fill))] = 0 #np.nan
        except Exception:
            pass
    scale = attrs.get("scale_factor", None)
    offset = attrs.get("add_offset", None)
    if scale is not None or offset is not None:
        scale = float(scale) if scale is not None else 1.0
        offset = float(offset) if offset is not None else 0.0
        data = data * scale + offset
    data = _apply_valid_range(data, attrs)
    data = reduce_vertical(data, VERTICAL_AGG)

    if data.shape != lat.shape:
        raise ValueError(f"Shape mismatch: data {data.shape} vs lat {lat.shape}.")

    lat[(lat < -90.0) | (lat > 90.0)] = np.nan
    lon[(lon < -180.0) | (lon > 180.0)] = np.nan
    x_km, y_km = _latlon_to_local_km(lat, lon, storm_lat, storm_lon)
    valid_xy = np.isfinite(x_km) & np.isfinite(y_km)
    valid = valid_xy & np.isfinite(data)

    half = GRID_EXTENT_KM
    step = GRID_KM
    swath_mask = _grid_swath_mask(x_km[valid_xy], y_km[valid_xy], step, GRID_SIZE, half)
    grid_mean = _regrid_to_grid(
        x_km[valid],
        y_km[valid],
        data[valid],
        INTERP_METHOD,
        step,
        GRID_SIZE,
        half,
    )
    grid_mean = np.where(np.isnan(grid_mean) & swath_mask, 0.0, grid_mean)
    inner_mean, outer_mean = _radial_means(
        grid_mean,
        step_km=GRID_KM,
        inner_km=INNER_RADIUS_KM,
        outer_km=OUTER_RADIUS_KM,
    )
    out_npy, out_path = _output_paths(dataset_name, dataset_idx, granule_file)
    np.save(out_npy, grid_mean)

    # fig, ax = plt.subplots(figsize=(6, 6))
    # extent = [-half, half, -half, half]
    # im = ax.imshow(
    #     grid_mean,
    #     origin="lower",
    #     extent=extent,
    #     cmap="turbo",
    #    # 改成
    #     vmin=-10.0,
    #     vmax=40.0,
    # )
    # ax.scatter(0.0, 0.0, s=60, c="white", marker="x", linewidths=2, label="Storm center")
    # ax.set_xlabel("X (km)")
    # ax.set_ylabel("Y (km)")
    # ax.set_title(f"zFactorFinal regridded to 5 km ({INTERP_METHOD})")
    # ax.grid(alpha=0.2)
    # fig.colorbar(im, ax=ax, label="zFactorFinal")
    # fig.tight_layout()
    # #fig.savefig(out_path, dpi=150)
    # plt.close(fig)
    print(f"[{row_idx}] Wrote npy: {out_npy}")
    print(f"[{row_idx}] Wrote plot: {out_path}")
    print(f"Granule: {granule_file}")
    print(f"SID: {sid}")
    print(f"Dataset: {data_path}")
    return inner_mean, outer_mean


def main() -> None:
    csv_path = os.path.join(_script_dir(), IN_CSV)
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"{IN_CSV} is empty.")

    inner_col = f"zFactorFinal_{VERTICAL_AGG}_r{INNER_RADIUS_KM:g}"
    outer_col = f"zFactorFinal_{VERTICAL_AGG}_r{INNER_RADIUS_KM:g}_{OUTER_RADIUS_KM:g}"
    if inner_col not in df.columns:
        df[inner_col] = np.nan
    if outer_col not in df.columns:
        df[outer_col] = np.nan

    error_count = 0
    for row_idx, row in df.iterrows():
        try:
            inner_mean, outer_mean = _process_row(row, row_idx)
            df.at[row_idx, inner_col] = inner_mean
            df.at[row_idx, outer_col] = outer_mean
        except Exception as exc:
            error_count += 1
            print(f"[{row_idx}] Skipped: {exc}")

    df.to_csv(csv_path, index=False)
    if error_count:
        print(f"Done with {error_count} row(s) skipped due to errors.")


if __name__ == "__main__":
    main()
