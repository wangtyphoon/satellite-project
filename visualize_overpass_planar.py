# -*- coding: utf-8 -*-
"""
Plan-view visualization of GPM DPR near-surface radar fields for overpass windows
listed in gpm_passes_from_ibtracs_{year}.csv.

Usage:
  - Edit USER CONFIG below (SID/PASS_ROW or SWATH_OVERRIDE).
  - Run: python3 visualize_overpass_planar.py
"""

import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# =====================================================
# User config
# =====================================================
YEARS = [i for i in range(2020, 2026)]  # IBTrACS season years
PASSES_CSV_TEMPLATE = "gpm_passes_from_ibtracs_{year}.csv"
IBTRACS_CSV_TEMPLATE = "ibtracs_WP_{year}.csv"
DOWNLOAD_DIR_TEMPLATE = "data_gpm_2adpr_{year}"
SAVE_DIR_TEMPLATE = "{year} overpasses"
START = 0
SID = None          # e.g., "2025001N11150" or None to take first row in PASSES_CSV
PASS_ROW = 0      # which pass row (per filtered SID) to plot
SWATH_OVERRIDE = "FS"  # "FS", "NS", "MS", "HS" to override CSV swath

DATASET_CANDIDATES = [
    "SLV/zFactorFinal",
    "SLV/zFactorFinalNearSurface",
    "SLV/zFactorFinalESurface",
    "SLV/precipRateNearSurface",
]
CHANNEL = 0         # use channel 0 when dataset has a channel dimension
VERTICAL_AGG = "mean"  # None to disable (applies when data has vertical bins)

VMIN = -10.0
VMAX = 40.0
COLORMAP = "turbo"
TRACK_TIME_WINDOW_HOURS = 6  # plot storm center within +/- hours of pass start
PASS_BUFFER_MINUTES = 1    # extend pass window by +/- minutes for plotting
OUTPUT_PASSES_CSV = None  # None -> overwrite PASSES_CSV; supports {year} if provided
SHOW_MASK = True
MASK_COLOR = "#bdbdbd"
MASK_ALPHA = 0.5

# =====================================================
# Helpers
# =====================================================

def normalize_swath_name(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s.lstrip("/")

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
    df["lon"] = normalize_lon_360(df["lon"].to_numpy())
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
    lon_i = lon_i % 360.0
    return lat_i, lon_i

def pick_pass_row(passes_df, sid, row_idx):
    if sid is None:
        sub = passes_df
    else:
        sub = passes_df[passes_df["SID"] == sid]
    if len(sub) == 0:
        raise ValueError("No pass rows found for the specified SID (or file is empty).")
    if row_idx >= len(sub):
        raise IndexError(f"Requested PASS_ROW {row_idx} but only {len(sub)} rows available.")
    return sub.iloc[row_idx]

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

def find_dataset_path(h5, swath_prefix, candidates):
    for ds in candidates:
        path = f"{swath_prefix}/{ds}"
        if path in h5:
            return path
    return None

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
        return data, False
    if agg is None:
        raise ValueError("VERTICAL_AGG is None but data has vertical bins.")
    if not np.isfinite(data).any():
        return np.full(data.shape[:2], np.nan, dtype=data.dtype), True
    if agg == "max":
        return np.nanmax(data, axis=2), True
    if agg == "mean":
        return np.nanmean(data, axis=2), True
    raise ValueError(f"Unsupported VERTICAL_AGG {agg}.")

def _wrap_delta_lon_deg(lon, lon0):
    return (lon - lon0 + 180.0) % 360.0 - 180.0

def _latlon_to_local_km(lat, lon, lat0, lon0, radius_km=6371.0):
    dlon = _wrap_delta_lon_deg(lon, lon0)
    x = np.deg2rad(dlon) * radius_km * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * radius_km
    return x, y

def _point_in_effective_swath(lat_row, lon_row, storm_lat, storm_lon, data_row=None, margin_km=0.0):
    valid = np.isfinite(lat_row) & np.isfinite(lon_row)
    if data_row is not None:
        valid &= np.isfinite(data_row)
    if np.count_nonzero(valid) < 2:
        return False
    idx = np.where(valid)[0]
    i0, i1 = idx[0], idx[-1]
    center_idx = lat_row.shape[0] // 2
    if not valid[center_idx]:
        center_idx = idx[len(idx) // 2]
    lat0 = float(lat_row[center_idx])
    lon0 = float(lon_row[center_idx])
    if not (np.isfinite(lat0) and np.isfinite(lon0)):
        return False

    x0, y0 = _latlon_to_local_km(lat_row[i0], lon_row[i0], lat0, lon0)
    x1, y1 = _latlon_to_local_km(lat_row[i1], lon_row[i1], lat0, lon0)
    vx = x1 - x0
    vy = y1 - y0
    vnorm = np.hypot(vx, vy)
    if not np.isfinite(vnorm) or vnorm == 0.0:
        return False

    vhatx = vx / vnorm
    vhaty = vy / vnorm
    left = x0 * vhatx + y0 * vhaty
    right = x1 * vhatx + y1 * vhaty
    if left > right:
        left, right = right, left

    xs, ys = _latlon_to_local_km(storm_lat, storm_lon, lat0, lon0)
    cross = xs * vhatx + ys * vhaty
    return (left - margin_km) <= cross <= (right + margin_km)

def interpolate_track_positions(track_df, target_times):
    t0 = pd.Timestamp("1970-01-01", tz="UTC")
    tt = (track_df["time_utc"] - t0).dt.total_seconds().to_numpy()
    lat = track_df["lat"].astype(float).to_numpy()
    lon = track_df["lon"].astype(float).to_numpy()

    m = np.isfinite(tt) & np.isfinite(lat) & np.isfinite(lon)
    tt = tt[m]
    lat = lat[m]
    lon = lon[m]
    if len(tt) < 2:
        return np.full(len(target_times), np.nan), np.full(len(target_times), np.nan)

    lon_u = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    target = pd.DatetimeIndex(target_times)
    q = (target - t0).total_seconds().to_numpy()
    lat_i = np.interp(q, tt, lat, left=np.nan, right=np.nan)
    lon_i = np.interp(q, tt, lon_u, left=np.nan, right=np.nan)
    lon_i = lon_i % 360.0
    return lat_i, lon_i

def storm_center_within_effective_swath(lat, lon, scan_times, track_df, data=None, margin_km=0.0):
    storm_lat, storm_lon = interpolate_track_positions(track_df, scan_times)
    inside = np.zeros(len(scan_times), dtype=bool)
    for i in range(len(scan_times)):
        if not (np.isfinite(storm_lat[i]) and np.isfinite(storm_lon[i])):
            continue
        data_row = data[i] if data is not None else None
        inside[i] = _point_in_effective_swath(
            lat[i],
            lon[i],
            storm_lat[i],
            storm_lon[i],
            data_row=data_row,
            margin_km=margin_km,
        )
    return inside

def storm_center_within_swath_at_time(
    lat,
    lon,
    scan_times,
    storm_lat,
    storm_lon,
    target_time,
    data=None,
    margin_km=0.0,
):
    if len(scan_times) == 0:
        return False, None
    if not (np.isfinite(storm_lat) and np.isfinite(storm_lon)):
        return False, None
    delta = np.abs((scan_times - target_time).to_numpy()).astype("timedelta64[ns]")
    idx = int(np.argmin(delta.view("int64")))
    data_row = data[idx] if data is not None else None
    inside = _point_in_effective_swath(
        lat[idx],
        lon[idx],
        storm_lat,
        storm_lon,
        data_row=data_row,
        margin_km=margin_km,
    )
    return inside, scan_times[idx]

def centers_to_edges_2d(values):
    if values.ndim != 2:
        raise ValueError("centers_to_edges_2d expects a 2D array.")
    nrow, ncol = values.shape
    edges = np.empty((nrow + 1, ncol + 1), dtype=values.dtype)

    # Interior corners from surrounding centers.
    edges[1:-1, 1:-1] = 0.25 * (
        values[:-1, :-1]
        + values[1:, :-1]
        + values[:-1, 1:]
        + values[1:, 1:]
    )

    # Extrapolate edges along the outer boundary.
    edges[0, 1:-1] = values[0, :-1] + (values[0, :-1] - edges[1, 1:-1])
    edges[-1, 1:-1] = values[-1, :-1] + (values[-1, :-1] - edges[-2, 1:-1])
    edges[1:-1, 0] = values[:-1, 0] + (values[:-1, 0] - edges[1:-1, 1])
    edges[1:-1, -1] = values[:-1, -1] + (values[:-1, -1] - edges[1:-1, -2])

    # Corner extrapolation.
    edges[0, 0] = values[0, 0] + (values[0, 0] - edges[1, 1])
    edges[0, -1] = values[0, -1] + (values[0, -1] - edges[1, -2])
    edges[-1, 0] = values[-1, 0] + (values[-1, 0] - edges[-2, 1])
    edges[-1, -1] = values[-1, -1] + (values[-1, -1] - edges[-2, -2])
    return edges

def normalize_longitudes(lon):
    return (lon + 180.0) % 360.0 - 180.0

def normalize_lon_360(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return lon % 360.0

def unwrap_longitudes_per_scanline(lon):
    lon = np.array(lon, dtype=np.float64, copy=True)
    if lon.ndim != 2:
        return normalize_longitudes(lon)
    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon), axis=1))
    center_idx = lon.shape[1] // 2
    lon0 = lon_unwrapped[:, center_idx]
    lon0_wrapped = normalize_longitudes(lon0)
    lon_wrapped = (lon_unwrapped - lon0[:, None] + 180.0) % 360.0 - 180.0 + lon0_wrapped[:, None]
    return lon_wrapped.astype(np.float32)

# =====================================================
# Main
# =====================================================

def _resolve_output_csv(default_csv, season_year):
    if OUTPUT_PASSES_CSV is None:
        return default_csv
    if "{year}" in OUTPUT_PASSES_CSV:
        return OUTPUT_PASSES_CSV.format(year=season_year)
    return OUTPUT_PASSES_CSV


def run_for_year(season_year: int):
    passes_csv = PASSES_CSV_TEMPLATE.format(year=season_year)
    ibtracs_csv = IBTRACS_CSV_TEMPLATE.format(year=season_year)
    download_dir = DOWNLOAD_DIR_TEMPLATE.format(year=season_year)
    save_dir = SAVE_DIR_TEMPLATE.format(year=season_year)

    passes_df = pd.read_csv(passes_csv)
    os.makedirs(save_dir, exist_ok=True)
    if len(passes_df) == 0:
        raise SystemExit("Passes CSV is empty; run gpm_overpass_finder.py first.")
    if "pass_mid_inside_effective_swath" not in passes_df.columns:
        passes_df["pass_mid_inside_effective_swath"] = pd.Series([pd.NA] * len(passes_df), dtype="boolean")
    if "pass_mid_inside_effective_swath_geo" not in passes_df.columns:
        passes_df["pass_mid_inside_effective_swath_geo"] = pd.Series([pd.NA] * len(passes_df), dtype="boolean")
    if "pass_mid_inside_effective_swath_nearest_scan_utc" not in passes_df.columns:
        passes_df["pass_mid_inside_effective_swath_nearest_scan_utc"] = pd.Series([pd.NaT] * len(passes_df))

    for i in range(START,len(passes_df)):
        row = pick_pass_row(passes_df, SID, i)
        row_idx = row.name
        sid = row["SID"]
        pass_start = pd.to_datetime(row["pass_start_utc"], utc=True)
        pass_end = pd.to_datetime(row["pass_end_utc"], utc=True)
        granule_file = row["granule_file"]
        swath_from_csv = normalize_swath_name(row.get("swath", None))
        swath_pref = normalize_swath_name(SWATH_OVERRIDE) or swath_from_csv

        granule_path = os.path.join(download_dir, granule_file)
        if not os.path.exists(granule_path):
            raise FileNotFoundError(f"Granule not found: {granule_path}")

        with h5py.File(granule_path, "r") as h5:
            swath = resolve_swath(h5, swath_pref)
            lat = h5[f"{swath}/Latitude"][...].astype(np.float32)
            lon = h5[f"{swath}/Longitude"][...].astype(np.float32)
            scan_times = read_scan_times(h5, swath)

            data_path = find_dataset_path(h5, swath, DATASET_CANDIDATES)
            if data_path is None:
                raise ValueError(f"No dataset found under {swath} for {DATASET_CANDIDATES}.")

            ds = h5[data_path]
            data = ds[...]
            print("[DEBUG] swath:", swath)
            print("[DEBUG] lat shape:", lat.shape, "lon shape:", lon.shape)
            print("[DEBUG] data_path:", data_path, "data shape:", data.shape)
            attrs = {k: ds.attrs[k] for k in ds.attrs.keys()}

        data = squeeze_field(data, CHANNEL).astype(np.float32)

        fill = attrs.get("_FillValue", None)
        if fill is not None:
            try:
                data[data == float(fill)] = np.nan
            except Exception:
                pass

        pass_start_plot = pass_start - pd.Timedelta(minutes=PASS_BUFFER_MINUTES)
        pass_end_plot = pass_end + pd.Timedelta(minutes=PASS_BUFFER_MINUTES)
        mask = (scan_times >= pass_start_plot) & (scan_times <= pass_end_plot)
        if not np.any(mask):
            delta = np.abs((scan_times - pass_start).to_numpy()).astype("timedelta64[ns]")
            idx0 = int(np.argmin(delta.view("int64")))
            mask = np.zeros(len(scan_times), dtype=bool)
            mask[idx0] = True
            print("No scanlines in plot window; using nearest scanline only.")

        scan_times = scan_times[mask]
        lat = lat[mask, :]
        lon = lon[mask, :]
        lon = unwrap_longitudes_per_scanline(lon)
        lon = normalize_lon_360(lon)  # 把 -170 轉成 190
        data = data[mask, :]
        data, reduced = reduce_vertical(data, VERTICAL_AGG)

        bad_geo = ~np.isfinite(lat) | ~np.isfinite(lon)
        data[bad_geo] = np.nan

        field_name = data_path.split("/")[-1]
        units = attrs.get("Units", attrs.get("units", ""))
        units_str = to_str(units).strip()
        if reduced:
            label_core = f"{field_name} {VERTICAL_AGG} over vertical"
        else:
            label_core = field_name
        label = f"{label_core} ({units_str})" if units_str else label_core

        cmap = choose_colormap(COLORMAP)

        lon_edges = centers_to_edges_2d(lon)
        lat_edges = centers_to_edges_2d(lat)

        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.pcolormesh(
            lon_edges,
            lat_edges,
            data,
            shading="auto",
            cmap=cmap,
            vmin=VMIN,
            vmax=VMAX,
        )
        if SHOW_MASK:
            mask_nan = ~np.isfinite(data)
            if np.any(mask_nan):
                mask_plot = np.ma.masked_where(~mask_nan, mask_nan.astype(np.float32))
                mask_cmap = ListedColormap([MASK_COLOR])
                ax.pcolormesh(
                    lon_edges,
                    lat_edges,
                    mask_plot,
                    shading="auto",
                    cmap=mask_cmap,
                    vmin=0.0,
                    vmax=1.0,
                    alpha=MASK_ALPHA,
                    zorder=3,
                )
        fig.colorbar(im, ax=ax, label=label)

        try:
            track_df = load_track_for_sid(ibtracs_csv, sid)
            pass_mid = pass_start + (pass_end - pass_start) / 2
            t0 = pass_mid - pd.Timedelta(hours=TRACK_TIME_WINDOW_HOURS)
            t1 = pass_mid + pd.Timedelta(hours=TRACK_TIME_WINDOW_HOURS)
            track_window = track_df[(track_df["time_utc"] >= t0) & (track_df["time_utc"] <= t1)]
            inside = storm_center_within_effective_swath(lat, lon, scan_times, track_df, data=data)
            inside_count = int(np.count_nonzero(inside))
            if inside_count > 0:
                print(f"[INFO] Storm center within effective swath: {inside_count}/{len(inside)} scanlines.")
            else:
                print("[INFO] Storm center outside effective swath for this window.")
            if len(track_window) > 0:
                ax.plot(
                    track_window["lon"],
                    track_window["lat"],
                    "--o",
                    color="black",
                    markersize=3,
                    linewidth=1.0,
                    label="Storm center (±window)",
                )
            if len(track_df) > 0:
                interp_lat, interp_lon = interpolate_track_position(track_df, pass_mid)
                if np.isfinite(interp_lat) and np.isfinite(interp_lon):
                    inside_mid_geo, _ = storm_center_within_swath_at_time(
                        lat,
                        lon,
                        scan_times,
                        interp_lat,
                        interp_lon,
                        pass_mid,
                        data=None,
                    )
                    inside_mid, nearest_time = storm_center_within_swath_at_time(
                        lat,
                        lon,
                        scan_times,
                        interp_lat,
                        interp_lon,
                        pass_mid,
                        data=data,
                    )
                    passes_df.at[row_idx, "pass_mid_inside_effective_swath"] = bool(inside_mid)
                    passes_df.at[row_idx, "pass_mid_inside_effective_swath_geo"] = bool(inside_mid_geo)
                    passes_df.at[row_idx, "pass_mid_inside_effective_swath_nearest_scan_utc"] = (
                        nearest_time.isoformat() if nearest_time is not None else pd.NaT
                    )
                    if nearest_time is not None:
                        status = "inside" if inside_mid else "outside"
                        status_geo = "inside" if inside_mid_geo else "outside"
                        print(
                            f"[INFO] Storm center at pass_mid is {status} effective swath "
                            f"(nearest scan: {nearest_time})."
                        )
                        print(f"[INFO] Storm center at pass_mid is {status_geo} geo-only swath.")
                    ax.plot(
                        interp_lon,
                        interp_lat,
                        marker="x",
                        color="yellow",
                        markersize=8,
                        mew=2,
                        label="Storm center (interp @ pass_mid)",
                    )
        except Exception as exc:
            print(f"Storm center overlay skipped: {exc}")

        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        title_note = f" | {VERTICAL_AGG} vertical" if reduced else ""
        ax.set_title(
            f"SID {sid} | {granule_file}\n"
            f"{pass_start_plot} to {pass_end_plot} (UTC) | swath={swath}{title_note}"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        plt.savefig(f"{save_dir}/gpm_overpass_{sid}_row{i}.png", dpi=150)
        # plt.show()
        plt.close(fig)

    out_csv = _resolve_output_csv(passes_csv, season_year)
    passes_df.to_csv(out_csv, index=False)


def main():
    for season_year in YEARS:
        print(f"\n=== Processing season {season_year} ===")
        run_for_year(season_year)


if __name__ == "__main__":
    main()
