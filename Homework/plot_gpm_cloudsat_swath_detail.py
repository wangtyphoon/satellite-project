#!/usr/bin/env python3
"""
Plot GPM swath and CloudSat track with intersection details.

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

MAX_DISTANCE_KM = 10.0
MAX_TIME_DIFF_MIN = 20.0
TIME_MARGIN_MIN = 20.0

OUTPUT_PNG = "gpm_cloudsat_swath_detail.png"
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


def spatial_best_scan_indices(
    cs_time: np.ndarray,
    cs_lat: np.ndarray,
    cs_lon: np.ndarray,
    gpm_time: np.ndarray,
    gpm_lat_center: np.ndarray,
    gpm_lon_center: np.ndarray,
    max_time_diff_min: float,
) -> tuple[np.ndarray, np.ndarray]:
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


def decimate(arr: np.ndarray, step: int) -> np.ndarray:
    if step <= 1:
        return arr
    return arr[::step]


def plot_swath_and_track(
    gpm_lat: np.ndarray,
    gpm_lon: np.ndarray,
    gpm_scan_keep: np.ndarray,
    cs_lat: np.ndarray,
    cs_lon: np.ndarray,
    match_cs_lat: np.ndarray,
    match_cs_lon: np.ndarray,
    match_gpm_lat: np.ndarray,
    match_gpm_lon: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # 1. 繪製 GPM Swath (模仿上圖的粉紅色點陣)
    # 這裡調整抽樣步長 (step)，如果點太密就調大數字
    step_scan = 1  # 沿著掃描方向的抽樣
    step_beam = 1   # 沿著 Beam (橫向) 的抽樣
    gpm_lat_plot = gpm_lat[gpm_scan_keep, :]
    gpm_lon_plot = gpm_lon[gpm_scan_keep, :]
    gpm_lat_excl = gpm_lat[~gpm_scan_keep, :]
    gpm_lon_excl = gpm_lon[~gpm_scan_keep, :]
    
    if gpm_lat_excl.size > 0:
        ax.scatter(
            gpm_lat_excl[::step_scan, ::step_beam],
            gpm_lon_excl[::step_scan, ::step_beam],
            s=1,
            color="lightgray",
            alpha=0.5,
            label="GPM Filtered (Excluded)",
        )

    ax.scatter(
        gpm_lat_plot[::step_scan, ::step_beam], 
        gpm_lon_plot[::step_scan, ::step_beam], 
        s=1, 
        color='magenta', 
        alpha=0.6, 
        label="GPM Swath (Points)"
    )

    # 2. 繪製 CloudSat Track (黑色實線)
    ax.plot(cs_lat, cs_lon, color="black", linewidth=2.0, label="CloudSat Track")

    # 3. 繪製匹配成功的點 (模仿上圖的綠色星號)
    if match_cs_lat.size > 0:
        ax.scatter(
            match_cs_lat, 
            match_cs_lon, 
            s=50, 
            marker='*', 
            color="lime", 
            edgecolors='green', 
            linewidths=0.5,
            zorder=5, 
            label="Matched Intersections"
        )

    # 4. 座標軸調整 (依據上圖習慣，Latitude 在 X 軸，Longitude 在 Y 軸)
    ax.set_xlabel("Latitude", fontsize=12)
    ax.set_ylabel("Longitude", fontsize=12)
    ax.set_title("GPM Swath and CloudSat Track Intersection")
    
    # 設置顯示範圍，聚焦在有資料的地方
    if match_cs_lat.size > 0:
        ax.set_xlim(np.nanmin(match_cs_lat)-1, np.nanmax(match_cs_lat)+1)
        ax.set_ylim(np.nanmin(match_cs_lon)-1, np.nanmax(match_cs_lon)+1)

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")

    if OUTPUT_PNG:
        fig.savefig(OUTPUT_PNG, dpi=200)
    if SHOW_PLOT:
        plt.show()


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

    gpm_scan_keep = np.zeros(gpm_lat.shape[0], dtype=bool)
    if matches["gpm_scan"].size > 0:
        gpm_scan_keep[matches["gpm_scan"]] = True

    plot_swath_and_track(
        gpm_lat,
        gpm_lon,
        gpm_scan_keep,
        cs_lat,
        cs_lon,
        match_cs_lat,
        match_cs_lon,
        match_gpm_lat,
        match_gpm_lon,
    )

    print(f"Matched points: {matches['cs_idx'].size}")


if __name__ == "__main__":
    main()
