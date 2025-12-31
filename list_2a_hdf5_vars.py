#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot selected PRE variables for GPM 2A DPR granules (plan view)."""

import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


PASSES_CSV_TEMPLATE = "gpm_passes_from_ibtracs_{year}.csv"
DOWNLOAD_DIR_TEMPLATE = "data_gpm_2adpr_{year}"

# =====================================================
# User config
# =====================================================
H5_PATH = None  # set to granule path to bypass CSV lookup
YEAR = 2015  # season year for passes CSV
SID = "2015038N08158"  # filter passes CSV by SID
PASS_ROW = 0  # row index after SID filter
SWATH_OVERRIDE = "FS"  # "FS", "NS", "MS", "HS" to override CSV swath
PASS_BUFFER_MINUTES = 1
SAVE_DIR = "CASE_STUDY"
SAVE_SUBDIR = None  # set or leave None to derive from CSV_PREFIX

VARS_CSV = "vars_2a_dpr.csv"  # CSV with per-level columns to drive plotting
CSV_PREFIX = "FS/VER"  # only plot variables under this path prefix
CSV_KIND_COLUMN = "kind"  # optional column name for plot kind

PLOT_ITEMS = [
    {"path": "PRE/zFactorMeasured", "kind": "zfactor"},
    {"path": "PRE/height", "kind": "height"},
    {"path": "PRE/binRealSurface", "kind": "bin"},
    {"path": "PRE/binClutterFreeBottom", "kind": "bin"},
    {"path": "PRE/binStormTop", "kind": "bin"},
    {"path": "PRE/heightStormTop", "kind": "height2d"},
    {"path": "PRE/localZenithAngle", "kind": "angle"},
    {"path": "PRE/elevation", "kind": "angle"},
    {"path": "PRE/snRatioAtRealSurface", "kind": "snr"},
    {"path": "PRE/flagPrecip", "kind": "flag"},
]

ZFACTOR_VMIN = -10.0
ZFACTOR_VMAX = 40.0
ZFACTOR_CMAP = "turbo"
DEFAULT_CMAP = "viridis"
SWATH_CANDIDATES = ["FS", "NS", "MS", "HS"]


def normalize_swath_name(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s.lstrip("/")


def read_scan_times(h5, swath_prefix):
    st_base = f"{swath_prefix}/ScanTime"
    year = h5[f"{st_base}/Year"][...]
    month = h5[f"{st_base}/Month"][...]
    dom = h5[f"{st_base}/DayOfMonth"][...]
    hour = h5[f"{st_base}/Hour"][...]
    minute = h5[f"{st_base}/Minute"][...]
    second = h5[f"{st_base}/Second"][...]
    ms_path = f"{st_base}/MilliSecond"
    if ms_path in h5:
        msec = h5[ms_path][...]
    else:
        msec = np.zeros_like(second)

    dt = pd.to_datetime(
        {
            "year": year.astype(int),
            "month": month.astype(int),
            "day": dom.astype(int),
            "hour": hour.astype(int),
            "minute": minute.astype(int),
            "second": second.astype(int),
        },
        errors="coerce",
        utc=True,
    ) + pd.to_timedelta(msec.astype(int), unit="ms")

    return pd.DatetimeIndex(dt)


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
        lat_path = f"{s}/Latitude"
        lon_path = f"{s}/Longitude"
        st_path = f"{s}/ScanTime/Year"
        if lat_path in h5 and lon_path in h5 and st_path in h5:
            return s
    raise ValueError("No matching swath group found in granule.")


def to_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.size == 1:
        return to_str(value[0])
    return str(value)


def choose_colormap(name):
    if name in plt.colormaps():
        return name
    return "viridis"


def centers_to_edges_2d(values):
    if values.ndim != 2:
        raise ValueError("centers_to_edges_2d expects a 2D array.")
    nrow, ncol = values.shape
    edges = np.empty((nrow + 1, ncol + 1), dtype=values.dtype)

    edges[1:-1, 1:-1] = 0.25 * (
        values[:-1, :-1]
        + values[1:, :-1]
        + values[:-1, 1:]
        + values[1:, 1:]
    )

    edges[0, 1:-1] = values[0, :-1] + (values[0, :-1] - edges[1, 1:-1])
    edges[-1, 1:-1] = values[-1, :-1] + (values[-1, :-1] - edges[-2, 1:-1])
    edges[1:-1, 0] = values[:-1, 0] + (values[:-1, 0] - edges[1:-1, 1])
    edges[1:-1, -1] = values[:-1, -1] + (values[:-1, -1] - edges[1:-1, -2])

    edges[0, 0] = values[0, 0] + (values[0, 0] - edges[1, 1])
    edges[0, -1] = values[0, -1] + (values[0, -1] - edges[1, -2])
    edges[-1, 0] = values[-1, 0] + (values[-1, 0] - edges[-2, 1])
    edges[-1, -1] = values[-1, -1] + (values[-1, -1] - edges[-2, -2])
    return edges


def normalize_lon_360(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return lon % 360.0


def unwrap_longitudes_per_scanline(lon):
    lon = np.array(lon, dtype=np.float64, copy=True)
    if lon.ndim != 2:
        return lon
    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon), axis=1))
    center_idx = lon.shape[1] // 2
    lon0 = lon_unwrapped[:, center_idx]
    lon_wrapped = (lon_unwrapped - lon0[:, None] + 180.0) % 360.0 - 180.0 + lon0[:, None]
    return lon_wrapped.astype(np.float32)


def build_path_from_levels(row, level_cols):
    parts = []
    for col in level_cols:
        val = row.get(col, None)
        if pd.isna(val) or str(val).strip() == "":
            continue
        parts.append(str(val).strip("/"))
    return "/".join(parts)


def load_plot_items_from_csv(csv_path, prefix=None):
    df = pd.read_csv(csv_path)
    level_cols = [c for c in df.columns if c.lower().startswith("level_")]
    items = []
    prefix_norm = None
    if prefix:
        prefix_norm = str(prefix).strip("/")
    for _, row in df.iterrows():
        path = ""
        if level_cols:
            path = build_path_from_levels(row, level_cols)
        if not path and "path" in df.columns:
            path = str(row.get("path", "")).strip("/")
        if not path:
            continue
        if prefix_norm and not path.startswith(prefix_norm):
            continue
        kind = None
        if CSV_KIND_COLUMN and CSV_KIND_COLUMN in df.columns:
            kind_val = row.get(CSV_KIND_COLUMN, None)
            if not pd.isna(kind_val) and str(kind_val).strip() != "":
                kind = str(kind_val).strip()
        items.append({"path": path, "kind": kind})
    return items


def infer_kind(path, data):
    name = path.split("/")[-1].lower()
    if "zfactor" in name:
        return "zfactor"
    if "height" in name and data.ndim >= 2:
        return "height2d" if data.ndim == 2 else "height"
    if "bin" in name:
        return "bin"
    if "flag" in name:
        return "flag"
    if "angle" in name or "zenith" in name:
        return "angle"
    return "auto"


def apply_fill(data, attrs):
    fill = attrs.get("_FillValue", attrs.get("fill_value", None))
    if fill is None:
        return data
    try:
        data = data.astype(np.float32, copy=False)
        data[data == float(fill)] = np.nan
    except Exception:
        pass
    return data


def reduce_vertical(data, agg):
    if data.ndim != 3:
        return data, False
    if not np.isfinite(data).any():
        return np.full(data.shape[:2], np.nan, dtype=data.dtype), True
    if agg == "max":
        return np.nanmax(data, axis=2), True
    if agg == "mean":
        return np.nanmean(data, axis=2), True
    raise ValueError(f"Unsupported vertical aggregation {agg}.")


def height_to_km(values, units):
    u = to_str(units).lower()
    if "km" in u:
        return values
    if "m" in u:
        return values / 1000.0
    return values


def map_bin_to_height(bin_idx, height_3d):
    if height_3d is None or height_3d.ndim != 3:
        return bin_idx
    if bin_idx.ndim == 3 and bin_idx.shape[-1] == 1:
        bin_idx = bin_idx[..., 0]
    elif bin_idx.ndim == 3:
        bin_idx = bin_idx[..., 0]
    idx = bin_idx.astype(np.float32, copy=False)
    if idx.ndim != 2:
        return idx
    valid = np.isfinite(idx)
    idx_int = np.zeros_like(idx, dtype=int)
    if np.any(valid):
        idx_int[valid] = np.rint(idx[valid]).astype(int)
        if np.nanmin(idx_int[valid]) >= 1:
            idx_int[valid] = idx_int[valid] - 1
    idx_int[~valid] = -1
    idx_int = np.clip(idx_int, 0, height_3d.shape[2] - 1)
    height_at = np.take_along_axis(height_3d, idx_int[:, :, None], axis=2)[:, :, 0]
    height_at[~valid] = np.nan
    return height_at


def squeeze_channel(data, channel_idx):
    if data.ndim != 4:
        return data
    if channel_idx < 0 or channel_idx >= data.shape[-1]:
        raise IndexError(f"channel {channel_idx} out of range for shape {data.shape}")
    return data[..., channel_idx]


def plot_field(
    lat,
    lon,
    data,
    title,
    label,
    out_path,
    cmap,
    vmin=None,
    vmax=None,
):
    lon_edges = centers_to_edges_2d(lon)
    lat_edges = centers_to_edges_2d(lat)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(
        lon_edges,
        lat_edges,
        data,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def split_swath_from_path(path):
    path = path.strip("/")
    root = path.split("/")[0] if path else ""
    if root in SWATH_CANDIDATES:
        return root, "/".join(path.split("/")[1:])
    return None, path


def get_swath_context(h5, swath, pass_start, pass_end):
    lat = h5[f"{swath}/Latitude"][...].astype(np.float32)
    lon = h5[f"{swath}/Longitude"][...].astype(np.float32)
    scan_times = read_scan_times(h5, swath)

    mask = None
    if pass_start is not None and pass_end is not None:
        pass_start_plot = pass_start - pd.Timedelta(minutes=PASS_BUFFER_MINUTES)
        pass_end_plot = pass_end + pd.Timedelta(minutes=PASS_BUFFER_MINUTES)
        mask = (scan_times >= pass_start_plot) & (scan_times <= pass_end_plot)
        if not np.any(mask):
            delta = np.abs((scan_times - pass_start).to_numpy()).astype("timedelta64[ns]")
            idx0 = int(np.argmin(delta.view("int64")))
            mask = np.zeros(len(scan_times), dtype=bool)
            mask[idx0] = True
            print("No scanlines in plot window; using nearest scanline only.")
        lat = lat[mask, :]
        lon = lon[mask, :]
        scan_times = scan_times[mask]

    lon = unwrap_longitudes_per_scanline(lon)
    lon = normalize_lon_360(lon)

    height_path = f"{swath}/PRE/height"
    height_data = None
    height_units = ""
    if height_path in h5:
        height_ds = h5[height_path]
        height_data = height_ds[...].astype(np.float32)
        if mask is not None:
            height_data = height_data[mask, :, :]
        height_data = apply_fill(height_data, height_ds.attrs)
        height_units = height_ds.attrs.get("Units", height_ds.attrs.get("units", ""))

    return {
        "lat": lat,
        "lon": lon,
        "scan_times": scan_times,
        "mask": mask,
        "height_data": height_data,
        "height_units": height_units,
    }


def main():
    pass_start = None
    pass_end = None
    swath_from_csv = None
    sid = SID
    granule_file = None

    if H5_PATH:
        granule_path = H5_PATH
    else:
        if YEAR is None:
            raise SystemExit("Set H5_PATH or YEAR to locate the granule.")
        passes_csv = PASSES_CSV_TEMPLATE.format(year=YEAR)
        download_dir = DOWNLOAD_DIR_TEMPLATE.format(year=YEAR)
        df = pd.read_csv(passes_csv)
        if sid is not None:
            df = df[df["SID"] == sid]
        if len(df) == 0:
            raise SystemExit("No rows found for the requested SID.")
        if PASS_ROW < 0 or PASS_ROW >= len(df):
            raise SystemExit(f"PASS_ROW {PASS_ROW} out of range (0..{len(df)-1}).")
        row = df.iloc[PASS_ROW]
        granule_file = row["granule_file"]
        pass_start = pd.to_datetime(row["pass_start_utc"], utc=True)
        pass_end = pd.to_datetime(row["pass_end_utc"], utc=True)
        swath_from_csv = normalize_swath_name(row.get("swath", None))
        granule_path = os.path.join(download_dir, granule_file)

    if not os.path.exists(granule_path):
        raise SystemExit(f"Granule not found: {granule_path}")

    save_dir = SAVE_DIR
    if CSV_PREFIX:
        subdir = SAVE_SUBDIR or CSV_PREFIX.replace("/", "_")
        save_dir = os.path.join(save_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)

    with h5py.File(granule_path, "r") as h5:
        swath_pref = normalize_swath_name(SWATH_OVERRIDE) or swath_from_csv
        default_swath = resolve_swath(h5, swath_pref)
        swath_cache = {}

        plot_items = None
        if VARS_CSV and os.path.exists(VARS_CSV):
            plot_items = load_plot_items_from_csv(VARS_CSV, CSV_PREFIX)
        if not plot_items:
            plot_items = PLOT_ITEMS

        for item in plot_items:
            item_path = item["path"]
            swath_from_path, rel_path = split_swath_from_path(item_path)
            swath = swath_from_path or default_swath
            if swath not in swath_cache:
                swath_cache[swath] = get_swath_context(h5, swath, pass_start, pass_end)
            ctx = swath_cache[swath]
            lat = ctx["lat"]
            lon = ctx["lon"]
            mask = ctx["mask"]
            height_data = ctx["height_data"]
            height_units = ctx["height_units"]

            if swath_from_path:
                ds_path = swath if rel_path == "" else f"{swath}/{rel_path}"
            else:
                ds_path = f"{swath}/{rel_path}".strip("/")
            if ds_path not in h5:
                print(f"[WARN] missing dataset: {ds_path}")
                continue
            if not isinstance(h5[ds_path], h5py.Dataset):
                print(f"[WARN] not a dataset: {ds_path}")
                continue
            ds = h5[ds_path]
            data = ds[...]
            if mask is not None:
                data = data[mask, ...]
            data = apply_fill(data, ds.attrs)
            if data.dtype.kind not in "fiu":
                print(f"[WARN] non-numeric dataset: {ds_path}")
                continue

            field_name = ds_path.split("/")[-1]
            units = ds.attrs.get("Units", ds.attrs.get("units", ""))
            units_str = to_str(units).strip()
            title_prefix = f"SID {sid} | {granule_file}" if granule_file else os.path.basename(granule_path)
            time_note = ""
            if pass_start is not None and pass_end is not None:
                time_note = f"\n{pass_start} to {pass_end} (UTC) | swath={swath}"
            title_base = f"{title_prefix}\n{field_name}{time_note}"

            bad_geo = ~np.isfinite(lat) | ~np.isfinite(lon)

            kind = item.get("kind", None) or infer_kind(ds_path, data)

            if kind == "zfactor":
                for ch_idx, ch_name in [(0, "Ku"), (1, "Ka")]:
                    zdata = squeeze_channel(data, ch_idx)
                    zdata = zdata.astype(np.float32, copy=False)
                    zdata, _ = reduce_vertical(zdata, "max")
                    if zdata.shape != lat.shape:
                        print(f"[WARN] shape mismatch for plotting: {ds_path} {zdata.shape} vs {lat.shape}")
                        continue
                    zdata[bad_geo] = np.nan
                    label_core = f"{field_name} max ({ch_name})"
                    label = f"{label_core} ({units_str})" if units_str else label_core
                    out_name = f"{field_name}_{ch_name}.png"
                    out_path = os.path.join(save_dir, out_name)
                    plot_field(
                        lat,
                        lon,
                        zdata,
                        title_base + f"\nchannel={ch_name} | vertical=max",
                        label,
                        out_path,
                        choose_colormap(ZFACTOR_CMAP),
                        vmin=ZFACTOR_VMIN,
                        vmax=ZFACTOR_VMAX,
                    )
                continue

            if kind == "height":
                hdata = data.astype(np.float32, copy=False)
                hdata, _ = reduce_vertical(hdata, "max")
                hdata = height_to_km(hdata, units_str)
                if hdata.shape != lat.shape:
                    print(f"[WARN] shape mismatch for plotting: {ds_path} {hdata.shape} vs {lat.shape}")
                    continue
                hdata[bad_geo] = np.nan
                label_core = f"{field_name} max over vertical"
                label = f"{label_core} (km)" if units_str else label_core
                out_name = f"{field_name}.png"
                out_path = os.path.join(save_dir, out_name)
                plot_field(
                    lat,
                    lon,
                    hdata,
                    title_base + "\nvertical=max",
                    label,
                    out_path,
                    choose_colormap(DEFAULT_CMAP),
                )
                continue

            if kind == "height2d":
                hdata = data.astype(np.float32, copy=False)
                hdata = height_to_km(hdata, units_str)
                if hdata.shape != lat.shape:
                    print(f"[WARN] shape mismatch for plotting: {ds_path} {hdata.shape} vs {lat.shape}")
                    continue
                hdata[bad_geo] = np.nan
                label_core = field_name
                label = f"{label_core} (km)" if units_str else label_core
                out_name = f"{field_name}.png"
                out_path = os.path.join(save_dir, out_name)
                plot_field(
                    lat,
                    lon,
                    hdata,
                    title_base,
                    label,
                    out_path,
                    choose_colormap(DEFAULT_CMAP),
                )
                continue

            if kind == "bin":
                bdata = data.astype(np.float32, copy=False)
                if height_data is not None:
                    h_at = map_bin_to_height(bdata, height_data)
                    h_at = height_to_km(h_at, height_units)
                    if h_at.shape != lat.shape:
                        print(f"[WARN] shape mismatch for plotting: {ds_path} {h_at.shape} vs {lat.shape}")
                        continue
                    h_at[bad_geo] = np.nan
                    label = f"{field_name} height (km)"
                    out_name = f"{field_name}_height_km.png"
                    out_path = os.path.join(save_dir, out_name)
                    plot_field(
                        lat,
                        lon,
                        h_at,
                        title_base + "\n(bin -> height)",
                        label,
                        out_path,
                        choose_colormap(DEFAULT_CMAP),
                    )
                else:
                    bdata[bad_geo] = np.nan
                    if bdata.shape != lat.shape:
                        print(f"[WARN] shape mismatch for plotting: {ds_path} {bdata.shape} vs {lat.shape}")
                        continue
                    label = f"{field_name} (bin)"
                    out_name = f"{field_name}_bin.png"
                    out_path = os.path.join(save_dir, out_name)
                    plot_field(
                        lat,
                        lon,
                        bdata,
                        title_base,
                        label,
                        out_path,
                        choose_colormap(DEFAULT_CMAP),
                    )
                continue

            if kind == "flag":
                fdata = data.astype(np.float32, copy=False)
                if fdata.shape != lat.shape:
                    print(f"[WARN] shape mismatch for plotting: {ds_path} {fdata.shape} vs {lat.shape}")
                    continue
                fdata[bad_geo] = np.nan
                valid = np.isfinite(fdata)
                if np.any(valid):
                    vmax = np.nanmax(fdata)
                    vmin = np.nanmin(fdata)
                else:
                    vmin, vmax = 0.0, 1.0
                nlevels = int(vmax - vmin + 1)
                nlevels = max(nlevels, 2)
                cmap = ListedColormap(plt.get_cmap("tab10").colors[:nlevels])
                out_name = f"{field_name}.png"
                out_path = os.path.join(save_dir, out_name)
                plot_field(
                    lat,
                    lon,
                    fdata,
                    title_base,
                    field_name,
                    out_path,
                    cmap,
                    vmin=vmin - 0.5,
                    vmax=vmax + 0.5,
                )
                continue

            data2d = data.astype(np.float32, copy=False)
            if data2d.ndim == 4:
                data2d = np.nanmean(data2d, axis=3)
            if data2d.ndim == 3:
                data2d, _ = reduce_vertical(data2d, "mean")
            if data2d.ndim != 2:
                print(f"[WARN] unsupported shape for plotting: {ds_path} {data.shape}")
                continue
            if data2d.shape != lat.shape:
                print(f"[WARN] shape mismatch for plotting: {ds_path} {data2d.shape} vs {lat.shape}")
                continue
            data2d[bad_geo] = np.nan
            label = f"{field_name} ({units_str})" if units_str else field_name
            out_name = f"{field_name}.png"
            out_path = os.path.join(save_dir, out_name)
            plot_field(
                lat,
                lon,
                data2d,
                title_base,
                label,
                out_path,
                choose_colormap(DEFAULT_CMAP),
            )


if __name__ == "__main__":
    main()
