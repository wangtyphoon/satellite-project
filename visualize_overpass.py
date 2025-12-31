# -*- coding: utf-8 -*-
"""
Quick-look visualization to confirm overpass windows contain a storm.

What it does:
- Loads IBTrACS track for a given SID.
- Loads the overpass window (from gpm_passes_from_ibtracs_2025.csv).
- Opens the corresponding DPR granule to extract the scan centerline for the
  hit window (between pass_start_utc and pass_end_utc).
- Plots storm track (nearby hours) and the GPM scan centerline in lon/lat.

Usage:
- Edit USER CONFIG below: choose SID, row index (if multiple windows per SID),
  and file paths.
- Run: python visualize_overpass.py
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# =====================================================
# User config
# =====================================================

IBTRACS_CSV = "ibtracs_WP_2024.csv"
PASSES_CSV = "gpm_passes_from_ibtracs_2024.csv"
DOWNLOAD_DIR = "data_gpm_2adpr"

SID = None          # e.g., "2025001N11150" or None to take first row in PASSES_CSV
PASS_ROW = 3        # which pass row (per filtered SID) to plot
TRACK_TIME_WINDOW_HOURS = 6  # plot track within +/- hours of pass start

# =====================================================
# Helpers (copied/lightly adapted from test.py)
# =====================================================

def _to_utc_datetime(s):
    t = pd.to_datetime(s, errors="coerce", utc=True)
    return t

def _unwrap_lon_deg(lon_deg):
    lon_rad = np.deg2rad(lon_deg.astype(float))
    lon_unwrapped = np.unwrap(lon_rad)
    return np.rad2deg(lon_unwrapped)

def _wrap_lon_deg(lon_deg):
    return (lon_deg + 180.0) % 360.0 - 180.0

def find_first_existing_path(h5, candidates):
    for p in candidates:
        if p in h5:
            return p
    return None

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

def is_granule_valid(path):
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

def pick_pass_row(passes_df, sid, row_idx):
    if sid is None:
        sub = passes_df
    else:
        sub = passes_df[passes_df["SID"] == sid]
    if len(sub) == 0:
        raise ValueError("No pass rows found for the specified SID (or file is empty).")
    sub = sub.reset_index(drop=True)
    if row_idx >= len(sub):
        raise IndexError(f"Requested PASS_ROW {row_idx} but only {len(sub)} rows available.")
    return sub.loc[row_idx]

def extract_scan_centerline(granule_path, pass_start, pass_end):
    with h5py.File(granule_path, "r") as h5:
        swath_candidates = ["/NS", "/MS", "/HS"]
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
            raise ValueError("No swath latitude/longitude found in granule.")

        lat = h5[lat_path][...]  # (nscan, nray)
        lon = h5[lon_path][...]
        scan_times = read_scan_times(h5, swath)

    # pick scanlines within the pass window
    m = (scan_times >= pass_start) & (scan_times <= pass_end)
    if not np.any(m):
        raise ValueError("No scanlines fall inside the requested window.")

    lat = lat[m, :]
    lon = lon[m, :]
    scan_times = scan_times[m]

    center_idx = lat.shape[1] // 2
    center_lat = lat[:, center_idx]
    center_lon = lon[:, center_idx]
    return scan_times, center_lat, center_lon

# =====================================================
# Main
# =====================================================

def main():
    passes_df = pd.read_csv(PASSES_CSV)
    if len(passes_df) == 0:
        raise SystemExit("Passes CSV is empty; run test.py first.")

    # choose the pass row
    row = pick_pass_row(passes_df, SID, PASS_ROW)
    sid = row["SID"]
    pass_start = pd.to_datetime(row["pass_start_utc"], utc=True)
    pass_end = pd.to_datetime(row["pass_end_utc"], utc=True)
    granule_file = row["granule_file"]

    granule_path = os.path.join(DOWNLOAD_DIR, granule_file)
    if not is_granule_valid(granule_path):
        raise FileNotFoundError(f"Granule missing or invalid: {granule_path}")

    # load track
    track_df = load_track_for_sid(IBTRACS_CSV, sid)
    t0 = pass_start - timedelta(hours=TRACK_TIME_WINDOW_HOURS)
    t1 = pass_start + timedelta(hours=TRACK_TIME_WINDOW_HOURS)
    track_window = track_df[(track_df["time_utc"] >= t0) & (track_df["time_utc"] <= t1)]

    # closest best-track point to pass_start
    time_diff = (track_df["time_utc"] - pass_start).abs()
    nearest_idx = time_diff.idxmin()
    nearest_row = track_df.loc[nearest_idx]
    nearest_time = nearest_row["time_utc"]

    print("\n=== Pass info ===")
    print(f"SID: {sid}")
    print(f"Granule: {granule_file}")
    print(f"Pass start UTC: {pass_start}")
    print(f"Pass end   UTC: {pass_end}")
    print(f"Nearest best-track time: {nearest_time}")
    print(f"Nearest best-track position: lat={nearest_row['lat']:.3f}, lon={nearest_row['lon']:.3f}")

    # extract scan centerline inside window
    scan_times, center_lat, center_lon = extract_scan_centerline(granule_path, pass_start, pass_end)

    # plot
    plt.figure(figsize=(8, 6))
    plt.title(f"SID {sid} | {granule_file}\\n{pass_start} to {pass_end}")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")

    # storm track around window
    plt.plot(track_window["lon"], track_window["lat"], "-o", color="tab:red", markersize=3, label="Storm track (± window)")
    # whole track as faint line
    plt.plot(track_df["lon"], track_df["lat"], "-", color="tab:red", alpha=0.3, linewidth=1, label="Storm track (full)")

    # GPM scan centerline during pass
    plt.plot(center_lon, center_lat, "-", color="tab:blue", linewidth=2, label="GPM scan centerline (window)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
