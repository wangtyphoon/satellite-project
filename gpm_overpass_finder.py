# -*- coding: utf-8 -*-
"""
IBTrACS -> find GPM DPR (2ADPR) swath overpass times (pixel/scanline-level)
No argparse; edit config variables below.

Requires:
  pip install earthaccess pandas numpy h5py requests

Notes:
- Uses CMR (via earthaccess) to coarse-filter candidate granules by temporal+bounding_box.
- Then opens each HDF5 granule to compute scanline-level min distance to interpolated storm center.
"""

import os
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import h5py
import earthaccess


# =====================================================
# User config (edit here)
# =====================================================
YEARS = [i for i in range(2015, 2022)]  # IBTrACS season years
IBTRACS_CSV_TEMPLATE = "ibtracs_WP_{year}.csv"
OUT_PASSES_CSV_TEMPLATE = "gpm_passes_from_ibtracs_{year}.csv"

# GPM product to search (DPR L2A precip profile)
GPM_SHORT_NAME = f"GPM_2ADPR"

# Coarse search buffers
TIME_BUFFER_HOURS = 3                  # extend storm lifetime search window by +/- hours
RADIUS_KM = 245                    # "scanned" threshold around storm center
BBOX_BUFFER_DEG = 0.5                 # bbox buffer added around track bbox (degrees)

# Download settings
DO_DOWNLOAD = True
DOWNLOAD_DIR_TEMPLATE = "data_gpm_2adpr_{year}"  # local dir to store DPR 2A granules

# Cleanup settings
CLEANUP_UNLISTED = True               # delete 2A files not referenced by OUT_PASSES_CSV
CLEANUP_PREFIXES = ("2A.", "2A-")     # file name prefixes treated as 2A granules

# Limit for testing (set None for full run)
MAX_STORMS = None                      # e.g., 3 for quick test
MAX_GRANULES_PER_STORM = None          # e.g., 50 for quick test


# =====================================================
# Helpers
# =====================================================

def _to_utc_datetime(s):
    # IBTrACS ISO_TIME is usually like "YYYY-MM-DD HH:MM:SS"
    t = pd.to_datetime(s, errors="coerce", utc=True)
    return t

def _unwrap_lon_deg(lon_deg):
    """Unwrap longitudes to avoid dateline jumps, return continuous lon in degrees."""
    lon_rad = np.deg2rad(lon_deg.astype(float))
    lon_unwrapped = np.unwrap(lon_rad)
    return np.rad2deg(lon_unwrapped)

def _wrap_lon_deg(lon_deg):
    """Wrap to [-180, 180)."""
    x = (lon_deg + 180.0) % 360.0 - 180.0
    return x

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine (km).
    lat1/lon1 are scalars; lat2/lon2 can be arrays.
    """
    R = 6371.0
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

def interpolate_track(track_df, target_times_utc):
    """
    Linear interpolation of storm center (lat, lon) to arbitrary UTC times.

    track_df must have columns: time_utc, lat, lon (lon in [-180,180] ok)
    target_times_utc: pandas.DatetimeIndex (UTC)
    """
    # Convert times to seconds since epoch
    t0 = pd.Timestamp("1970-01-01", tz="UTC")
    tt = (track_df["time_utc"] - t0).dt.total_seconds().to_numpy()

    lat = track_df["lat"].astype(float).to_numpy()
    lon = track_df["lon"].astype(float).to_numpy()

    # Unwrap lon to avoid jumps, interpolate, then wrap back
    lon_u = _unwrap_lon_deg(lon)

    q = (target_times_utc - t0).total_seconds().to_numpy()

    # Guard: remove NaNs
    m = np.isfinite(tt) & np.isfinite(lat) & np.isfinite(lon_u)
    tt2, lat2, lonu2 = tt[m], lat[m], lon_u[m]
    if len(tt2) < 2:
        return np.full(len(q), np.nan), np.full(len(q), np.nan)

    lat_i = np.interp(q, tt2, lat2, left=np.nan, right=np.nan)
    lon_i = np.interp(q, tt2, lonu2, left=np.nan, right=np.nan)
    lon_i = _wrap_lon_deg(lon_i)
    return lat_i, lon_i

def find_first_existing_path(h5, candidates):
    for p in candidates:
        if p in h5:
            return p
    return None

def is_granule_valid(path):
    """
    Lightweight integrity check to decide whether to reuse an existing download.
    - must exist and be non-empty
    - must be readable by h5py
    - must contain at least one of the expected swath latitude datasets
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with h5py.File(path, "r") as h5:
            for sw in ("/NS/Latitude", "/MS/Latitude", "/HS/Latitude"):
                if sw in h5:
                    return True
    except Exception:
        return False
    return False

def read_scan_times(h5, swath_prefix):
    """
    Read scanline times for a swath, return pandas.DatetimeIndex (UTC), length = nscan.
    Typical DPR paths: /NS/ScanTime/Year, Month, DayOfMonth, Hour, Minute, Second, MilliSecond
    """
    st_base = f"{swath_prefix}/ScanTime"
    # common field names
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

    # Build datetime safely
    # Arrays are typically shape (nscan,)
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

def cleanup_unlisted_granules(download_dir, keep_files, prefixes):
    """
    Remove 2A granules under download_dir that are not listed in keep_files.
    keep_files should contain basenames (not full paths).
    """
    if not os.path.isdir(download_dir):
        return []
    removed = []
    for name in os.listdir(download_dir):
        path = os.path.join(download_dir, name)
        if not os.path.isfile(path):
            continue
        if not name.startswith(prefixes):
            continue
        if name in keep_files:
            continue
        try:
            os.remove(path)
            removed.append(name)
        except OSError:
            continue
    return removed

def compute_overpass_windows(h5_path, track_df, radius_km=250.0):
    """
    For a single granule, compute time windows where swath is within radius_km of storm center.
    Returns list of dict records: pass_start, pass_end, min_dist_km, swath, file
    """
    records = []
    with h5py.File(h5_path, "r") as h5:
        # Prefer Normal Scan (NS); fallback to MS/HS if needed
        swath_candidates = ["/FS", "/NS", "/MS", "/HS"]

        swath = None
        lat_path = None
        lon_path = None

        for s in swath_candidates:
            lp = find_first_existing_path(h5, [f"{s}/Latitude"])
            op = find_first_existing_path(h5, [f"{s}/Longitude"])
            st = find_first_existing_path(h5, [f"{s}/ScanTime/Year"])
            if lp and op and st:
                swath = s
                lat_path, lon_path = lp, op
                break

        if swath is None:
            return records  # cannot parse

        lat = h5[lat_path][...]  # (nscan, nray)
        lon = h5[lon_path][...]  # (nscan, nray)
        scan_times = read_scan_times(h5, swath)  # (nscan,)

        # Interpolate storm center to each scanline time
        storm_lat, storm_lon = interpolate_track(track_df, scan_times)

        # Compute distance between storm center and scan center per scanline
        nscan, nray = lat.shape
        center_idx = nray // 2
        center_dist = np.full(nscan, np.nan, dtype=float)

        for i in range(nscan):
            if not np.isfinite(storm_lat[i]) or not np.isfinite(storm_lon[i]):
                continue
            lat_line = lat[i, :]
            lon_line = lon[i, :]
            # mask invalid
            m = np.isfinite(lat_line) & np.isfinite(lon_line)
            if not np.any(m):
                continue
            c_lat = lat_line[center_idx]
            c_lon = lon_line[center_idx]
            # fall back to mean of valid beams if the central beam is invalid
            if not (np.isfinite(c_lat) and np.isfinite(c_lon)):
                c_lat = float(np.nanmean(lat_line[m]))
                c_lon = float(np.nanmean(lon_line[m]))
            if not (np.isfinite(c_lat) and np.isfinite(c_lon)):
                continue
            center_dist[i] = haversine_km(storm_lat[i], storm_lon[i], c_lat, c_lon)

        hit = np.isfinite(center_dist) & (center_dist <= radius_km)
        if not np.any(hit):
            return records

        # Convert hit scanlines into contiguous windows
        idx = np.where(hit)[0]
        # group contiguous
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
                continue
            # close a window [start, prev]
            seg = slice(start, prev + 1)
            records.append(
                {
                    "granule_file": os.path.basename(h5_path),
                    "swath": swath.strip("/"),
                    "pass_start_utc": scan_times[start].isoformat(),
                    "pass_end_utc": scan_times[prev].isoformat(),
                    # minimum center-to-storm distance within this window
                    "min_dist_km": float(np.nanmin(center_dist[seg])),
                }
            )
            start = k
            prev = k

        # last window
        seg = slice(start, prev + 1)
        records.append(
            {
                "granule_file": os.path.basename(h5_path),
                "swath": swath.strip("/"),
                "pass_start_utc": scan_times[start].isoformat(),
                "pass_end_utc": scan_times[prev].isoformat(),
                "min_dist_km": float(np.nanmin(center_dist[seg])),
            }
        )

    return records


# =====================================================
# Main
# =====================================================

def run_for_year(season_year: int):
    ibtracs_csv = IBTRACS_CSV_TEMPLATE.format(year=season_year)
    out_passes_csv = OUT_PASSES_CSV_TEMPLATE.format(year=season_year)
    download_dir = DOWNLOAD_DIR_TEMPLATE.format(year=season_year)

    # 1) Load IBTrACS
    df = pd.read_csv(ibtracs_csv, low_memory=False)

    # Robust column pick
    time_col = "ISO_TIME" if "ISO_TIME" in df.columns else None
    lat_col = "LAT" if "LAT" in df.columns else ("USA_LAT" if "USA_LAT" in df.columns else None)
    lon_col = "LON" if "LON" in df.columns else ("USA_LON" if "USA_LON" in df.columns else None)
    sid_col = "SID" if "SID" in df.columns else None

    if not (time_col and lat_col and lon_col and sid_col):
        raise ValueError(f"Missing required columns. Need SID + time + lat + lon. "
                         f"Found: {df.columns.tolist()[:30]} ...")

    df["time_utc"] = _to_utc_datetime(df[time_col])
    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["time_utc", "lat", "lon", sid_col]).copy()

    os.makedirs(download_dir, exist_ok=True)

    pass_rows = []

    storms = list(df.groupby(sid_col))
    if MAX_STORMS is not None:
        storms = storms[:MAX_STORMS]

    for sid, sdf in storms:
        sdf = sdf.sort_values("time_utc").copy()

        tmin = sdf["time_utc"].min() - pd.Timedelta(hours=TIME_BUFFER_HOURS)
        tmax = sdf["time_utc"].max() + pd.Timedelta(hours=TIME_BUFFER_HOURS)

        # Track bbox + buffer
        lat_min = float(sdf["lat"].min() - BBOX_BUFFER_DEG)
        lat_max = float(sdf["lat"].max() + BBOX_BUFFER_DEG)

        # Handle lon bbox carefully (dateline): use unwrapped then min/max then wrap.
        lon_u = _unwrap_lon_deg(sdf["lon"].to_numpy())
        lon_min_u = float(np.nanmin(lon_u) - BBOX_BUFFER_DEG)
        lon_max_u = float(np.nanmax(lon_u) + BBOX_BUFFER_DEG)
        # choose representative wrapped bbox around the track; if span too large, fallback global-ish
        span = lon_max_u - lon_min_u
        if span >= 350:
            lon_min, lon_max = -180.0, 180.0
        else:
            lon_min = float(_wrap_lon_deg(lon_min_u))
            lon_max = float(_wrap_lon_deg(lon_max_u))
            # If wrapping causes inversion, fallback to -180..180
            if lon_min > lon_max:
                lon_min, lon_max = -180.0, 180.0

        print(f"\n=== SID {sid} ===")
        print(f"Time window (UTC): {tmin} -> {tmax}")
        print(f"BBox: ({lon_min:.2f}, {lat_min:.2f}, {lon_max:.2f}, {lat_max:.2f})")

        # 3) CMR search granules (coarse filter)
        granules = earthaccess.search_data(
            short_name=GPM_SHORT_NAME,
            temporal=(tmin.isoformat(), tmax.isoformat()),
            bounding_box=(lon_min, lat_min, lon_max, lat_max),
        )

        if MAX_GRANULES_PER_STORM is not None:
            granules = granules[:MAX_GRANULES_PER_STORM]

        print(f"Candidate granules: {len(granules)}")
        if len(granules) == 0:
            continue

        # 4) Download (reuse valid local files, download missing/invalid ones)
        paths = []
        to_download = []
        for g in granules:
            fn = g.data_links()[0].split("/")[-1]
            p = os.path.join(download_dir, fn)
            if is_granule_valid(p):
                paths.append(p)
            else:
                to_download.append((g, p))

        if DO_DOWNLOAD and to_download:
            granules_needed = [g for g, _ in to_download]
            downloaded = earthaccess.download(
                granules_needed,
                local_path=download_dir,
                threads=1,
                show_progress=False,
            )
            for p in downloaded:
                sp = str(p)
                if is_granule_valid(sp):
                    paths.append(sp)
                else:
                    print(f"Downloaded but failed integrity check, skipping: {os.path.basename(sp)}")

        if len(paths) == 0:
            print("No local files available after download step.")
            continue

        # Track df for interpolation
        track_df = sdf[["time_utc", "lat", "lon"]].copy()

        # 5) Pixel/scanline-level refine: find real overpass windows
        for fp in paths:
            try:
                recs = compute_overpass_windows(fp, track_df, radius_km=RADIUS_KM)
            except Exception as e:
                print(f"Skip (parse error) {os.path.basename(fp)}: {e}")
                continue

            for r in recs:
                r["SID"] = sid
                pass_rows.append(r)

        print(f"Overpass windows found so far: {len(pass_rows)}")

        # Persist progress after each storm
        out_partial = pd.DataFrame(pass_rows)
        if len(out_partial) == 0:
            out_partial.to_csv(out_passes_csv, index=False, encoding="utf-8-sig")
        else:
            out_partial = out_partial.sort_values(["SID", "pass_start_utc", "granule_file"]).reset_index(drop=True)
            out_partial.to_csv(out_passes_csv, index=False, encoding="utf-8-sig")
        print(f"Progress saved to {out_passes_csv}")

    out = pd.DataFrame(pass_rows)
    if len(out) == 0:
        print("\nNo overpass windows found. Consider increasing RADIUS_KM or buffers.")
        out.to_csv(out_passes_csv, index=False, encoding="utf-8-sig")
        if CLEANUP_UNLISTED:
            removed = cleanup_unlisted_granules(download_dir, set(), CLEANUP_PREFIXES)
            print(f"Cleanup removed {len(removed)} unlisted 2A files.")
        return

    out = out.sort_values(["SID", "pass_start_utc", "granule_file"]).reset_index(drop=True)
    out.to_csv(out_passes_csv, index=False, encoding="utf-8-sig")
    print("\nSaved:", out_passes_csv)
    print(out.head(20).to_string(index=False))

    if CLEANUP_UNLISTED:
        keep = set(out["granule_file"].dropna().astype(str))
        removed = cleanup_unlisted_granules(download_dir, keep, CLEANUP_PREFIXES)
        print(f"Cleanup removed {len(removed)} unlisted 2A files.")


def main():
    earthaccess.login()
    for season_year in YEARS:
        print(f"\n=== Processing season {season_year} ===")
        run_for_year(season_year)


if __name__ == "__main__":
    main()
