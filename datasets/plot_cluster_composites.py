#!/usr/bin/env python3
"""
Build cluster composites for shallow rain, stormtop, NW/DM, and latent heating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import regrid_zfactorfinal as rzf


CSV_PATHS = [
    Path("gpm_passes_swath_true_hdbscan_bst.csv"),
    Path("gpm_passes_swath_true_hdbscan_delta.csv"),
]
CLUSTER_COL = "cluster_hdbscan"
INCLUDE_NOISE = True

OUT_DIR = Path("plots_cluster_composites")

SHALLOW_DIR = Path("light_rain_npy")
SHALLOW_SUFFIX = "light_rain.npy"

STORMTOP_CACHE_DIR = Path("stormtop_npy")
STORMTOP_SUFFIX = "stormtop.npy"
STORMTOP_DATASET = "PRE/heightStormTop"
STORMTOP_VERTICAL_AGG = "max"

PARAM_DSD_DIR = Path("paramDSD")
PARAM_DSD_SUFFIX = "paramDSD_radial150km.npy"

LATENT_DIR = Path("2LSLH")
LATENT_SUFFIX = "2LSLH_radial150km.npy"

GRID_KM = 1.0
RADIAL_BIN_KM = 5.0
HEIGHT_BIN_KM = 0.25
HEIGHT_START_KM = 0.0
BIN_INTERVAL_M = 125.0
CLIP_PERCENTILE = (2, 98)
FIG_DPI = 140

SWATH_COL = "swath"
GRANULE_COL = "granule_file"
SID_COL = "SID"


def _limits(data: np.ndarray, pct=CLIP_PERCENTILE) -> tuple[float, float]:
    if pct is None:
        return float(np.nanmin(data)), float(np.nanmax(data))
    lo, hi = np.nanpercentile(data, pct)
    return float(lo), float(hi)


def _resolve_npy(row: pd.Series, base_dir: Path, suffix: str) -> Optional[Path]:
    granule = row.get(GRANULE_COL, None)
    if granule is None or str(granule).strip() == "":
        return None
    stem = Path(str(granule)).stem
    swath = rzf.normalize_swath_name(row.get(SWATH_COL, None))
    if swath:
        candidate = base_dir / f"{stem}_{swath}_{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(base_dir.glob(f"{stem}_*_{suffix}"))
    if len(matches) == 1:
        return matches[0]
    return None


def _mask_fill(arr: np.ndarray, fill_value: float | int | None) -> np.ndarray:
    out = arr.astype(float, copy=True)
    if fill_value is not None:
        out[out == float(fill_value)] = np.nan
    return out


def _apply_scale_offset(data: np.ndarray, attrs: dict) -> np.ndarray:
    scale = attrs.get("scale_factor", None)
    offset = attrs.get("add_offset", None)
    if scale is not None or offset is not None:
        scale = float(scale) if scale is not None else 1.0
        offset = float(offset) if offset is not None else 0.0
        data = data * scale + offset
    return data


def _reduce_vertical(data: np.ndarray, agg: str) -> np.ndarray:
    if data.ndim <= 2:
        return data
    if agg == "max":
        return np.nanmax(data, axis=-1)
    if agg == "mean":
        return np.nanmean(data, axis=-1)
    raise ValueError(f"Unsupported vertical_agg: {agg}")


def _get_storm_center(
    row: pd.Series, ibtracs_cache: dict[tuple[int, str], pd.DataFrame]
) -> tuple[float, float]:
    year = rzf._infer_year_from_row(row)
    sid = row.get(SID_COL, None)
    if sid is None or str(sid).strip() == "":
        raise ValueError("Missing SID")
    key = (year, str(sid))
    if key not in ibtracs_cache:
        ibtracs_path = Path(rzf._project_root()) / rzf.IBTRACS_CSV_TEMPLATE.format(year=year)
        if not ibtracs_path.exists():
            raise FileNotFoundError(f"IBTRACS CSV not found: {ibtracs_path}")
        ibtracs_cache[key] = rzf.load_track_for_sid(ibtracs_path, sid)
    track_df = ibtracs_cache[key]
    pass_time = rzf._resolve_pass_time(row)
    storm_lat, storm_lon = rzf.interpolate_track_position(track_df, pass_time)
    return float(storm_lat), float(storm_lon)


def _load_stormtop_grid(
    row: pd.Series, ibtracs_cache: dict[tuple[int, str], pd.DataFrame], cache_dir: Path
) -> Optional[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _resolve_npy(row, cache_dir, STORMTOP_SUFFIX)
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)

    granule = row.get(GRANULE_COL, None)
    if granule is None or str(granule).strip() == "":
        return None
    year = rzf._infer_year_from_row(row)
    granule_path = Path(rzf._project_root()) / rzf.DOWNLOAD_DIR_TEMPLATE.format(year=year) / granule
    if not granule_path.exists():
        return None
    if not h5py.is_hdf5(granule_path):
        return None

    storm_lat, storm_lon = _get_storm_center(row, ibtracs_cache)
    if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
        return None

    swath_pref = rzf.normalize_swath_name(row.get(SWATH_COL, None))
    with h5py.File(granule_path, "r") as h5:
        swath = rzf.resolve_swath(h5, swath_pref)
        ds_height = h5[f"{swath}/{STORMTOP_DATASET}"]
        ds_lat = h5[f"{swath}/Latitude"]
        ds_lon = h5[f"{swath}/Longitude"]

        height = _mask_fill(ds_height[...], ds_height.attrs.get("_FillValue"))
        height = _apply_scale_offset(height, dict(ds_height.attrs))
        height = rzf._apply_valid_range(height, dict(ds_height.attrs))
        height = _reduce_vertical(height, STORMTOP_VERTICAL_AGG)
        lat = _mask_fill(ds_lat[...], ds_lat.attrs.get("_FillValue"))
        lon = _mask_fill(ds_lon[...], ds_lon.attrs.get("_FillValue"))

    lat[(lat < -90.0) | (lat > 90.0)] = np.nan
    lon[(lon < -180.0) | (lon > 180.0)] = np.nan
    x_km, y_km = rzf._latlon_to_local_km(lat, lon, storm_lat, storm_lon)
    valid_xy = np.isfinite(x_km) & np.isfinite(y_km)
    valid = valid_xy & np.isfinite(height)

    half = rzf.GRID_EXTENT_KM
    step = rzf.GRID_KM
    swath_mask = rzf._grid_swath_mask(x_km[valid_xy], y_km[valid_xy], step, rzf.GRID_SIZE, half)
    grid_height = rzf._regrid_to_grid(
        x_km[valid],
        y_km[valid],
        height[valid],
        rzf.INTERP_METHOD,
        step,
        rzf.GRID_SIZE,
        half,
    )
    grid_height[~swath_mask] = np.nan

    cache_path = cache_dir / f"{Path(granule).stem}_{swath}_{STORMTOP_SUFFIX}"
    np.save(cache_path, grid_height)
    return grid_height


def _composite_arrays(arrays: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    sum_data = None
    count_data = None
    for arr in arrays:
        if arr is None:
            continue
        data = np.array(arr, dtype=float, copy=False)
        if sum_data is None:
            sum_data = np.zeros_like(data, dtype=float)
            count_data = np.zeros_like(data, dtype=float)
        if data.shape != sum_data.shape:
            raise ValueError(f"Shape mismatch {data.shape} vs {sum_data.shape}")
        valid = np.isfinite(data)
        sum_data[valid] += data[valid]
        count_data[valid] += 1.0
    if sum_data is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        composite = sum_data / count_data
    composite[count_data == 0] = np.nan
    return composite


def _prepare_param_dsd(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # (2, n_bins, y, x) -> radial bins
        size = arr.shape[-1]
        centers = _grid_centers(GRID_KM, size)
        xx, yy = np.meshgrid(centers, centers)
        r_km = np.sqrt(xx**2 + yy**2)
        max_r = GRID_KM * size / 2.0
        edges = np.arange(0.0, max_r + RADIAL_BIN_KM, RADIAL_BIN_KM)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        n_bins = arr.shape[1]
        out = np.full((2, n_bins, len(r_centers)), np.nan, dtype=float)
        for b in range(n_bins):
            for i in range(2):
                out[i, b] = _radial_profile(arr[i, b], r_km, edges)
        return out
    raise ValueError(f"Unsupported paramDSD shape {arr.shape}")


def _prepare_latent(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        size = arr.shape[-1]
        centers = _grid_centers(GRID_KM, size)
        xx, yy = np.meshgrid(centers, centers)
        r_km = np.sqrt(xx**2 + yy**2)
        max_r = GRID_KM * size / 2.0
        edges = np.arange(0.0, max_r + RADIAL_BIN_KM, RADIAL_BIN_KM)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        n_bins = arr.shape[0]
        out = np.full((n_bins, len(r_centers)), np.nan, dtype=float)
        for b in range(n_bins):
            out[b] = _radial_profile(arr[b], r_km, edges)
        return out
    raise ValueError(f"Unsupported latentHeating shape {arr.shape}")


def _grid_centers(step_km: float, size: int) -> np.ndarray:
    start = -size * step_km / 2.0 + step_km / 2.0
    return start + step_km * np.arange(size)


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


def _plot_grid(data: np.ndarray, title: str, out_path: Path) -> None:
    _plot_grid_imshow(data, title, out_path)


def _plot_grid_imshow(data: np.ndarray, title: str, out_path: Path) -> None:
    size = data.shape[0]
    half = GRID_KM * size / 2.0
    extent = [-half, half, -half, half]
    vmin, vmax = _limits(data)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_grid_contourf(data: np.ndarray, title: str, out_path: Path) -> None:
    size = data.shape[0]
    half = GRID_KM * size / 2.0
    centers = _grid_centers(GRID_KM, size)
    xx, yy = np.meshgrid(centers, centers)
    vmin, vmax = _limits(data)
    levels = np.linspace(vmin, vmax, 21)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)
    im = ax.contourf(xx, yy, data, levels=levels, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_param_dsd(nw: np.ndarray, dm: np.ndarray, title: str, out_path: Path) -> None:
    n_bins, n_rad = nw.shape
    r_centers = (np.arange(n_rad) * RADIAL_BIN_KM) + RADIAL_BIN_KM / 2.0
    height_km = (n_bins - 1 - np.arange(n_bins)) * (BIN_INTERVAL_M / 1000.0)
    extent = [r_centers[0], r_centers[-1], height_km[-1], height_km[0]]
    nw_lim = _limits(nw)
    dm_lim = _limits(dm)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=FIG_DPI)
    ax = axes[0]
    im = ax.imshow(
        nw,
        origin="upper",
        aspect="auto",
        extent=extent,
        vmin=nw_lim[0],
        vmax=nw_lim[1],
        cmap="viridis",
    )
    ax.set_title(f"{title} - NW")
    ax.set_ylabel("Height (km)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(
        dm,
        origin="upper",
        aspect="auto",
        extent=extent,
        vmin=dm_lim[0],
        vmax=dm_lim[1],
        cmap="magma",
    )
    ax.set_title(f"{title} - DM")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Height (km)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_latent(lh: np.ndarray, title: str, out_path: Path) -> None:
    n_bins, n_rad = lh.shape
    r_centers = (np.arange(n_rad) * RADIAL_BIN_KM) + RADIAL_BIN_KM / 2.0
    heights = HEIGHT_START_KM + (np.arange(n_bins) + 0.5) * HEIGHT_BIN_KM
    extent = [r_centers[0], r_centers[-1], heights[0], heights[-1]]
    vmin, vmax = _limits(lh)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    im = ax.imshow(
        lh,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Height (km)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _cluster_ids(series: pd.Series) -> list[float]:
    cluster_ids = sorted(series.dropna().unique().tolist())
    if not INCLUDE_NOISE:
        cluster_ids = [cid for cid in cluster_ids if not np.isclose(cid, -1)]
    return cluster_ids


def _unique_rows(df: pd.DataFrame) -> pd.DataFrame:
    subset_cols = [GRANULE_COL, SWATH_COL]
    subset_cols = [c for c in subset_cols if c in df.columns]
    return df.dropna(subset=[GRANULE_COL]).drop_duplicates(subset=subset_cols)


def _composite_from_rows(
    rows: pd.DataFrame,
    resolve_npy,
    loader,
) -> tuple[Optional[np.ndarray], list[str]]:
    arrays = []
    missing = []
    for _, row in rows.iterrows():
        path = resolve_npy(row)
        if path is None or not path.exists():
            missing.append(str(row.get(GRANULE_COL, "unknown")))
            continue
        arrays.append(loader(path))
    return _composite_arrays(arrays), missing


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ibtracs_cache: dict[tuple[int, str], pd.DataFrame] = {}

    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            print(f"Missing CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        if CLUSTER_COL not in df.columns:
            print(f"Missing cluster column {CLUSTER_COL} in {csv_path}")
            continue

        csv_out_dir = OUT_DIR / csv_path.stem
        csv_out_dir.mkdir(parents=True, exist_ok=True)

        cluster_ids = _cluster_ids(df[CLUSTER_COL])
        for cluster_id in cluster_ids:
            cluster_rows = df[np.isclose(df[CLUSTER_COL], cluster_id)].copy()
            if cluster_rows.empty:
                continue
            unique_rows = _unique_rows(cluster_rows)
            cluster_tag = f"{cluster_id:g}"
            cluster_dir = csv_out_dir / f"cluster_{cluster_tag}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            def _resolve_shallow(row: pd.Series) -> Optional[Path]:
                return _resolve_npy(row, SHALLOW_DIR, SHALLOW_SUFFIX)

            def _resolve_param(row: pd.Series) -> Optional[Path]:
                return _resolve_npy(row, PARAM_DSD_DIR, PARAM_DSD_SUFFIX)

            def _resolve_latent(row: pd.Series) -> Optional[Path]:
                return _resolve_npy(row, LATENT_DIR, LATENT_SUFFIX)

            shallow_comp, shallow_missing = _composite_from_rows(
                unique_rows,
                _resolve_shallow,
                lambda p: np.load(p),
            )
            if shallow_comp is not None:
                out_path = cluster_dir / "composite_shallow_rain.png"
                _plot_grid(shallow_comp, f"Shallow rain composite (cluster {cluster_tag})", out_path)
            if shallow_missing:
                print(f"{csv_path.stem} cluster {cluster_tag} missing shallow rain: {len(shallow_missing)}")

            storm_arrays = []
            storm_missing = 0
            for _, row in unique_rows.iterrows():
                try:
                    grid = _load_stormtop_grid(row, ibtracs_cache, STORMTOP_CACHE_DIR)
                except Exception:
                    grid = None
                if grid is None:
                    storm_missing += 1
                    continue
                storm_arrays.append(grid)
            storm_comp = _composite_arrays(storm_arrays)
            if storm_comp is not None:
                out_path = cluster_dir / "composite_stormtop.png"
                _plot_grid_contourf(
                    storm_comp,
                    f"Stormtop composite (cluster {cluster_tag})",
                    out_path,
                )
            if storm_missing:
                print(f"{csv_path.stem} cluster {cluster_tag} missing stormtop: {storm_missing}")

            param_arrays = []
            param_missing = []
            for _, row in unique_rows.iterrows():
                path = _resolve_param(row)
                if path is None or not path.exists():
                    param_missing.append(str(row.get(GRANULE_COL, "unknown")))
                    continue
                data = np.load(path)
                param_arrays.append(_prepare_param_dsd(data))
            param_comp = _composite_arrays(param_arrays)
            if param_comp is not None:
                out_path = cluster_dir / "composite_param_dsd.png"
                _plot_param_dsd(
                    param_comp[0],
                    param_comp[1],
                    f"paramDSD composite (cluster {cluster_tag})",
                    out_path,
                )
            if param_missing:
                print(f"{csv_path.stem} cluster {cluster_tag} missing paramDSD: {len(param_missing)}")

            latent_arrays = []
            latent_missing = []
            for _, row in unique_rows.iterrows():
                path = _resolve_latent(row)
                if path is None or not path.exists():
                    latent_missing.append(str(row.get(GRANULE_COL, "unknown")))
                    continue
                data = np.load(path)
                latent_arrays.append(_prepare_latent(data))
            latent_comp = _composite_arrays(latent_arrays)
            if latent_comp is not None:
                out_path = cluster_dir / "composite_latent_heat.png"
                _plot_latent(
                    latent_comp,
                    f"Latent heating composite (cluster {cluster_tag})",
                    out_path,
                )
            if latent_missing:
                print(f"{csv_path.stem} cluster {cluster_tag} missing latentHeating: {len(latent_missing)}")


if __name__ == "__main__":
    main()
