#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save paramDSD (NW/DM) as storm-centered axis-symmetric radial averages.
Stores radial means (radius bins) within MAX_RADIUS_KM for each vertical bin.
"""

# Directly reused from regrid_zfactorfinal.py:
# - _script_dir, _project_root, normalize_swath_name, resolve_swath, find_dataset_path
# - _to_utc_datetime, load_track_for_sid, interpolate_track_position
# - _wrap_lon, _latlon_to_local_km, _apply_valid_range
# - _grid_centers, _grid_swath_mask, _regrid_to_grid
# - _infer_year_from_row, _resolve_pass_time
#
# Differences vs regrid_zfactorfinal.py:
# - Dataset path uses SLV/paramDSD.
# - Split paramDSD into NW/DM, regrid separately, stack to (2, GRID_SIZE, GRID_SIZE).
# - No vertical aggregation; keep bins when NW/DM has a vertical dimension.
# - Debug prints for data.shape and attribute keys after reading paramDSD.

from __future__ import annotations
import sys
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


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

DATASET_CANDIDATES = ["SLV/paramDSD"]

GRID_KM = 1
GRID_SIZE = 490  # 100x100 over 500x500 km
GRID_EXTENT_KM = GRID_KM * GRID_SIZE / 2.0  # 250 km
INTERP_METHOD = "gaussian"
WEIGHT_RADIUS_KM = 7.5
GAUSS_SIGMA_KM = 5
BARNES_KAPPA_KM2 = 200.0
RADIAL_BIN_KM = 5.0
MAX_RADIUS_KM = 150.0

# Debug/limit controls
ROW_INDEX = None  # set int to run a single row
ROW_MAX = None    # set int to process first N rows
PRINT_EVERY_BIN = 10  # progress print interval when bin count is large
USE_PARALLEL = False
N_WORKERS = 6


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


def _output_path(granule_file, swath) -> str:
    out_dir = os.path.join(_script_dir(), "paramDSD")
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(granule_file).stem
    tag = f"radial{int(MAX_RADIUS_KM)}km"
    return os.path.join(out_dir, f"{stem}_{swath}_paramDSD_{tag}.npy")


def _split_param_dsd(data: np.ndarray):
    shape = data.shape
    dims = [i for i, s in enumerate(shape) if s == 2]
    if len(dims) != 1:
        raise ValueError(
            "Cannot infer NW/DM dimension from shape "
            f"{shape}; expected exactly one dimension of size 2, found {dims}."
        )
    dim = dims[0]
    data = np.moveaxis(data, dim, -1)
    nw = data[..., 0]
    dm = data[..., 1]
    return nw, dm, dim


def _align_bins(field: np.ndarray, lat_shape: tuple) -> np.ndarray:
    if field.ndim == 2:
        if field.shape != lat_shape:
            raise ValueError(f"Shape mismatch: field {field.shape}, lat {lat_shape}.")
        return field
    if field.ndim == 3:
        if field.shape[:2] == lat_shape:
            return field
        if (field.shape[0], field.shape[2]) == lat_shape:
            return np.swapaxes(field, 1, 2)
        if field.shape[1:] == lat_shape:
            return np.moveaxis(field, 0, -1)
    raise ValueError(f"Unsupported field shape {field.shape} for lat {lat_shape}.")


def _regrid_field(
    field: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    valid_xy: np.ndarray,
    swath_mask: np.ndarray,
    step: float,
    size: int,
    half: float,
) -> np.ndarray:
    valid = valid_xy & np.isfinite(field)
    grid = _regrid_to_grid(
        x_km[valid],
        y_km[valid],
        field[valid],
        INTERP_METHOD,
        step,
        size,
        half,
    )
    return np.where(np.isnan(grid) & swath_mask, np.nan, grid)


def _radial_profile(field: np.ndarray, r_km: np.ndarray, edges: np.ndarray) -> np.ndarray:
    flat_r = r_km.ravel()
    flat_v = field.ravel()
    good = np.isfinite(flat_v)
    if not np.any(good):
        return np.full(len(edges) - 1, np.nan, dtype=float)
    idx = np.digitize(flat_r[good], edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    if not np.any(valid):
        return np.full(len(edges) - 1, np.nan, dtype=float)
    sums = np.bincount(idx[valid], weights=flat_v[good][valid], minlength=len(edges) - 1)
    counts = np.bincount(idx[valid], minlength=len(edges) - 1)
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    nonzero = counts > 0
    out[nonzero] = sums[nonzero] / counts[nonzero]
    return out


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

    print(f"[{row_idx}] paramDSD shape: {data.shape}")
    print(f"[{row_idx}] paramDSD attrs: {sorted(attrs.keys())}")

    data = data.astype(np.float32)
    fill = attrs.get("_FillValue", attrs.get("missing_value", None))
    if fill is not None:
        try:
            data[np.isclose(data, float(fill))] = np.nan
        except Exception:
            pass
    scale = attrs.get("scale_factor", None)
    offset = attrs.get("add_offset", None)
    if scale is not None or offset is not None:
        scale = float(scale) if scale is not None else 1.0
        offset = float(offset) if offset is not None else 0.0
        data = data * scale + offset
    data = _apply_valid_range(data, attrs)

    nw, dm, dim = _split_param_dsd(data)
    nw = _align_bins(nw, lat.shape)
    dm = _align_bins(dm, lat.shape)

    lat[(lat < -90.0) | (lat > 90.0)] = np.nan
    lon[(lon < -180.0) | (lon > 180.0)] = np.nan
    x_km, y_km = _latlon_to_local_km(lat, lon, storm_lat, storm_lon)
    valid_xy = np.isfinite(x_km) & np.isfinite(y_km)

    half = GRID_EXTENT_KM
    step = GRID_KM
    swath_mask = _grid_swath_mask(x_km[valid_xy], y_km[valid_xy], step, GRID_SIZE, half)
    max_r = min(float(MAX_RADIUS_KM), float(half))
    edges = np.arange(0.0, max_r + RADIAL_BIN_KM, RADIAL_BIN_KM)
    centers = _grid_centers(step, GRID_SIZE, half)
    xx, yy = np.meshgrid(centers, centers)
    r_km = np.sqrt(xx**2 + yy**2)

    if nw.ndim == 2:
        grid_nw = _regrid_field(nw, x_km, y_km, valid_xy, swath_mask, step, GRID_SIZE, half)
        grid_dm = _regrid_field(dm, x_km, y_km, valid_xy, swath_mask, step, GRID_SIZE, half)
        nw_rad = _radial_profile(grid_nw, r_km, edges)
        dm_rad = _radial_profile(grid_dm, r_km, edges)
        out = np.stack([nw_rad, dm_rad], axis=0)
    else:
        n_bins = nw.shape[2]
        n_rad = len(edges) - 1
        nw_rad = np.full((n_bins, n_rad), np.nan, dtype=np.float32)
        dm_rad = np.full((n_bins, n_rad), np.nan, dtype=np.float32)
        t0 = time.perf_counter()
        def _regrid_bin(b_idx: int):
            nw_b = _regrid_field(
                nw[:, :, b_idx],
                x_km,
                y_km,
                valid_xy,
                swath_mask,
                step,
                GRID_SIZE,
                half,
            )
            dm_b = _regrid_field(
                dm[:, :, b_idx],
                x_km,
                y_km,
                valid_xy,
                swath_mask,
                step,
                GRID_SIZE,
                half,
            )
            return b_idx, nw_b, dm_b

        if USE_PARALLEL and n_bins > 1:
            done = 0
            with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
                futures = {ex.submit(_regrid_bin, b): b for b in range(n_bins)}
                for fut in as_completed(futures):
                    b_idx, nw_b, dm_b = fut.result()
                    nw_rad[b_idx] = _radial_profile(nw_b, r_km, edges)
                    dm_rad[b_idx] = _radial_profile(dm_b, r_km, edges)
                    done += 1
                    if PRINT_EVERY_BIN and done % PRINT_EVERY_BIN == 0:
                        dt = time.perf_counter() - t0
                        print(f"[{row_idx}] bins {done}/{n_bins} done ({dt:.1f}s)")
        else:
            for b in range(n_bins):
                b_idx, nw_b, dm_b = _regrid_bin(b)
                nw_rad[b_idx] = _radial_profile(nw_b, r_km, edges)
                dm_rad[b_idx] = _radial_profile(dm_b, r_km, edges)
                if PRINT_EVERY_BIN and (b + 1) % PRINT_EVERY_BIN == 0:
                    dt = time.perf_counter() - t0
                    print(f"[{row_idx}] bins {b + 1}/{n_bins} done ({dt:.1f}s)")
        # axis 0: NW (index 0), DM (index 1); axis 1: bin; axis 2: radius bins
        out = np.stack([nw_rad, dm_rad], axis=0)

    out = out.astype(np.float32, copy=False)
    out_path = _output_path(granule_file, swath)
    np.save(out_path, out)

    print(f"[{row_idx}] Wrote npy: {out_path}")
    print(f"Granule: {granule_file}")
    print(f"SID: {sid}")
    print(f"Dataset: {data_path}")


def main() -> None:
    if not hasattr(h5py, "File"):
        raise RuntimeError(
            "h5py.File is unavailable. This usually means a shadowed h5py module "
            "or missing h5py install in the current environment."
        )
    csv_path = os.path.join(_script_dir(), IN_CSV)
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"{IN_CSV} is empty.")

    if ROW_INDEX is not None:
        df = df.iloc[[ROW_INDEX]]
    if ROW_MAX is not None:
        df = df.iloc[:ROW_MAX]

    error_count = 0
    for row_idx, row in df.iterrows():
        try:
            _process_row(row, row_idx)
        except Exception as exc:
            error_count += 1
            print(f"[{row_idx}] Skipped: {exc}")

    if error_count:
        print(f"Done with {error_count} row(s) skipped due to errors.")


if __name__ == "__main__":
    main()
