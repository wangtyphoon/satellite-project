#!/usr/bin/env python3
"""
Plot the ground-track intersection between GPM DPR and CloudSat.

Dependencies:
  - numpy
  - h5py
  - matplotlib
  - pyhdf
"""

from __future__ import annotations

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pyhdf.HDF import HDF, HC
from pyhdf.VS import VS


GPM_PATH = "2A.GPM.DPR.V9-20211125.20190803-S033403-E050636.030841.V07A.HDF5"
CS_PATH = "2019215032540_70655_CS_2B-GEOPROF_GRANULE_P1_R05_E09_F00.hdf"

MAX_DISTANCE_KM = 30.0
MAX_TIME_DIFF_MIN = 20.0
TIME_MARGIN_MIN = 20.0

OUTPUT_PNG = "gpm_cloudsat_intersection.png"
SHOW_PLOT = True


def read_vdata_1field(path: str, vname: str) -> np.ndarray:
    hdf = HDF(path, HC.READ)
    vs = hdf.vstart()
    vd = vs.attach(vname)
    try:
        vd.setfields(vname)
    except Exception:
        fields = vd.inquire()[2]
        vd.setfields(fields)
    nrecs = vd.inquire()[0]
    data = vd.read(nrecs)
    vd.detach()
    vs.end()
    hdf.close()
    return np.asarray([r[0] for r in data], dtype=np.float64)


def build_profile_datetimes(path: str) -> np.ndarray:
    t_prof = read_vdata_1field(path, "Profile_time")
    base = np.datetime64("1993-01-01T00:00:00", "ns")

    try:
        utc_start_arr = read_vdata_1field(path, "UTC_start")
        utc_start = float(utc_start_arr[0]) if utc_start_arr.size > 0 else None
    except Exception:
        utc_start = None

    try:
        tai_start_arr = read_vdata_1field(path, "TAI_start")
        tai_start = float(tai_start_arr[0]) if tai_start_arr.size > 0 else None
    except Exception:
        tai_start = None

    if tai_start is not None:
        tai_dt = base + np.timedelta64(int(round(tai_start * 1e9)), "ns")
        if utc_start is not None:
            date = str(tai_dt)[:10]
            start_dt = np.datetime64(f"{date}T00:00:00", "ns") + np.timedelta64(
                int(round(utc_start * 1e9)), "ns"
            )
        else:
            start_dt = tai_dt
    elif utc_start is not None:
        start_dt = base + np.timedelta64(int(round(utc_start * 1e9)), "ns")
    else:
        start_dt = base

    t_prof_ns = np.rint(t_prof * 1e9).astype("int64")
    dt = start_dt + t_prof_ns.astype("timedelta64[ns]")
    return dt.astype("datetime64[ns]")


def build_scan_datetimes(path: str, group: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        st = f[f"{group}/ScanTime"]
        year = st["Year"][...].astype(int)
        month = st["Month"][...].astype(int)
        day = st["DayOfMonth"][...].astype(int)
        hour = st["Hour"][...].astype(int)
        minute = st["Minute"][...].astype(int)
        second = st["Second"][...].astype(int)
        msec = st["MilliSecond"][...].astype(int)

    dt = np.array(
        [
            np.datetime64(
                f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}.{ms:03d}"
            )
            for y, mo, d, h, mi, s, ms in zip(year, month, day, hour, minute, second, msec)
        ],
        dtype="datetime64[ms]",
    )
    return dt


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def nearest_scan_indices(scan_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(scan_times, target_times)
    idx = np.clip(idx, 1, scan_times.size - 1)
    left = scan_times[idx - 1]
    right = scan_times[idx]
    choose_right = (right - target_times) < (target_times - left)
    return np.where(choose_right, idx, idx - 1)


def spatial_best_scan_indices(
    cs_time: np.ndarray,
    cs_lat: np.ndarray,
    cs_lon: np.ndarray,
    gpm_time: np.ndarray,
    gpm_lat_center: np.ndarray,
    gpm_lon_center: np.ndarray,
    max_time_diff_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each CloudSat profile, search GPM scans within the time window and
    choose the scan whose center beam is spatially closest.
    """
    window = np.timedelta64(int(max_time_diff_min * 60), "s")
    scan_idx = np.full(cs_time.size, -1, dtype=int)
    time_diff_sec = np.full(cs_time.size, -1, dtype=np.int64)

    for i, t in enumerate(cs_time):
        if not np.isfinite(cs_lat[i]) or not np.isfinite(cs_lon[i]):
            continue
        start = int(np.searchsorted(gpm_time, t - window, side="left"))
        end = int(np.searchsorted(gpm_time, t + window, side="right"))
        if end <= start:
            continue
        d = haversine_km(
            cs_lat[i],
            cs_lon[i],
            gpm_lat_center[start:end],
            gpm_lon_center[start:end],
        )
        j = int(np.nanargmin(d))
        s = start + j
        scan_idx[i] = s
        time_diff_sec[i] = np.abs((t - gpm_time[s]).astype("timedelta64[s]").astype(np.int64))

    return scan_idx, time_diff_sec


def find_intersections(
    cs_time: np.ndarray,
    cs_lat: np.ndarray,
    cs_lon: np.ndarray,
    gpm_time: np.ndarray,
    gpm_lat: np.ndarray,
    gpm_lon: np.ndarray,
    gpm_lat_center: np.ndarray,
    gpm_lon_center: np.ndarray,
    max_distance_km: float,
    max_time_diff_min: float,
) -> dict:
    scan_idx, time_diff_sec = spatial_best_scan_indices(
        cs_time,
        cs_lat,
        cs_lon,
        gpm_time,
        gpm_lat_center,
        gpm_lon_center,
        max_time_diff_min=max_time_diff_min,
    )
    time_ok = (scan_idx >= 0) & (time_diff_sec <= int(max_time_diff_min * 60))

    match_cs_idx = []
    match_gpm_scan = []
    match_gpm_beam = []
    match_dist_km = []
    match_time_sec = []

    for i, s in enumerate(scan_idx):
        if not time_ok[i]:
            continue
        if not np.isfinite(cs_lat[i]) or not np.isfinite(cs_lon[i]):
            continue

        lat_row = gpm_lat[int(s), :]
        lon_row = gpm_lon[int(s), :]
        valid = np.isfinite(lat_row) & np.isfinite(lon_row)
        if not np.any(valid):
            continue

        d = haversine_km(cs_lat[i], cs_lon[i], lat_row, lon_row)
        d = np.where(valid, d, np.nan)
        if not np.any(np.isfinite(d)):
            continue

        j = int(np.nanargmin(d))
        if np.isfinite(d[j]) and d[j] <= max_distance_km:
            match_cs_idx.append(i)
            match_gpm_scan.append(int(s))
            match_gpm_beam.append(j)
            match_dist_km.append(float(d[j]))
            match_time_sec.append(int(time_diff_sec[i]))

    return {
        "cs_idx": np.asarray(match_cs_idx, dtype=int),
        "gpm_scan": np.asarray(match_gpm_scan, dtype=int),
        "gpm_beam": np.asarray(match_gpm_beam, dtype=int),
        "dist_km": np.asarray(match_dist_km, dtype=np.float32),
        "time_sec": np.asarray(match_time_sec, dtype=np.int32),
    }

def _fmt_dt64(x: np.datetime64) -> str:
    # 統一用毫秒顯示（避免 ns 太長）
    return str(x.astype("datetime64[ms]"))

def print_time_summary(name: str, t: np.ndarray, n: int = 5) -> None:
    print(f"\n=== {name} time summary ===")
    print(f"dtype: {t.dtype}, size: {t.size}")
    if t.size == 0:
        print("(empty)")
        return
    t_sorted = np.sort(t)
    print(f"start: {_fmt_dt64(t_sorted[0])} UTC")
    print(f"end  : {_fmt_dt64(t_sorted[-1])} UTC")
    dt_sec = (t_sorted[-1] - t_sorted[0]).astype("timedelta64[s]").astype(int)
    print(f"span : {dt_sec} s  (~{dt_sec/60:.2f} min)")
    print(f"first {n}: {[ _fmt_dt64(x) for x in t_sorted[:n] ]}")
    print(f"last  {n}: {[ _fmt_dt64(x) for x in t_sorted[-n:] ]}")

def plot_intersection(
    cs_lat: np.ndarray,
    cs_lon: np.ndarray,
    gpm_lat_center: np.ndarray,
    gpm_lon_center: np.ndarray,
    match_cs_lat: np.ndarray,
    match_cs_lon: np.ndarray,
    match_gpm_lat: np.ndarray,
    match_gpm_lon: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(cs_lon, cs_lat, color="black", linewidth=1.0, label="CloudSat track")
    ax.plot(
        gpm_lon_center,
        gpm_lat_center,
        color="tab:blue",
        linewidth=1.2,
        label="GPM (center beam)",
    )
    if match_cs_lat.size > 0:
        ax.scatter(match_cs_lon, match_cs_lat, s=18, color="red", label="Intersection (CloudSat)")
        ax.scatter(match_gpm_lon, match_gpm_lat, s=18, color="orange", label="Intersection (GPM)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    if OUTPUT_PNG:
        fig.savefig(OUTPUT_PNG, dpi=150)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    cs_lat = read_vdata_1field(CS_PATH, "Latitude")
    cs_lon = read_vdata_1field(CS_PATH, "Longitude")
    cs_time = build_profile_datetimes(CS_PATH)

    with h5py.File(GPM_PATH, "r") as f:
        if "FS" in f:
            group = "FS"
        elif "NS" in f:
            group = "NS"
        else:
            raise KeyError("GPM file missing FS/NS group.")
        gpm_lat = f[f"{group}/Latitude"][...].astype(np.float32)
        gpm_lon = f[f"{group}/Longitude"][...].astype(np.float32)

    gpm_time = build_scan_datetimes(GPM_PATH, group)

    t0 = gpm_time.min() - np.timedelta64(int(TIME_MARGIN_MIN * 60), "s")
    t1 = gpm_time.max() + np.timedelta64(int(TIME_MARGIN_MIN * 60), "s")
    keep = (cs_time >= t0) & (cs_time <= t1)
    cs_lat = cs_lat[keep]
    cs_lon = cs_lon[keep]
    cs_time = cs_time[keep]

    center_beam = gpm_lat.shape[1] // 2
    gpm_lat_center = gpm_lat[:, center_beam]
    gpm_lon_center = gpm_lon[:, center_beam]

    matches = find_intersections(
        cs_time,
        cs_lat,
        cs_lon,
        gpm_time,
        gpm_lat,
        gpm_lon,
        gpm_lat_center,
        gpm_lon_center,
        max_distance_km=MAX_DISTANCE_KM,
        max_time_diff_min=MAX_TIME_DIFF_MIN,
    )

    if matches["cs_idx"].size > 0:
        match_cs_lat = cs_lat[matches["cs_idx"]]
        match_cs_lon = cs_lon[matches["cs_idx"]]
        match_gpm_lat = gpm_lat[matches["gpm_scan"], matches["gpm_beam"]]
        match_gpm_lon = gpm_lon[matches["gpm_scan"], matches["gpm_beam"]]
    else:
        match_cs_lat = np.array([])
        match_cs_lon = np.array([])
        match_gpm_lat = np.array([])
        match_gpm_lon = np.array([])

    cs_start = str(cs_time.min())[:19] if cs_time.size else "n/a"
    cs_end = str(cs_time.max())[:19] if cs_time.size else "n/a"
    title = (
        "GPM vs CloudSat Intersection Track\n"
        f"CloudSat {cs_start} to {cs_end} UTC | max {MAX_DISTANCE_KM:.0f} km, "
        f"{MAX_TIME_DIFF_MIN:.1f} min"
    )

    plot_intersection(
        cs_lat,
        cs_lon,
        gpm_lat_center,
        gpm_lon_center,
        match_cs_lat,
        match_cs_lon,
        match_gpm_lat,
        match_gpm_lon,
        title,
    )

    print(f"Matched points: {matches['cs_idx'].size}")
    print_time_summary("CloudSat", cs_time)
    print_time_summary("GPM", gpm_time)

if __name__ == "__main__":
    main()
