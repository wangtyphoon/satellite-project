#!/usr/bin/env python3
"""
Plot a GPM DPR reflectivity section sampled along the CloudSat track.

Two methods are supported:
  - nearest: use the nearest GPM footprint in the best-matching scan.
  - interp:  inverse-distance-weighted interpolation using k nearest beams in the scan.
"""

from __future__ import annotations

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pyhdf.HDF import HDF, HC
from pyhdf.VS import VS


GPM_PATH = "2A.GPM.DPR.V9-20211125.20190803-S033403-E050636.030841.V07A.HDF5"
CS_PATH = "2019215032540_70655_CS_2B-GEOPROF_GRANULE_P1_R05_E09_F00.hdf"

METHOD = "nearest"  # "nearest" or "interp"
INTERP_NEIGHBORS = 4

MAX_DISTANCE_KM = 10.0
MAX_TIME_DIFF_MIN = 20.0
TIME_MARGIN_MIN = 20.0

PROFILE_STEP = 1
MATCH_ONLY = True

CHANNEL = 0
VMIN = -10.0
VMAX = 40.0

OUTPUT_PNG = "gpm_cloudsat_track_section.png"
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
    return dt.astype("datetime64[ns]")


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


def apply_lat_lon_axes(ax, x, lat_vals, lon_vals):
    if lat_vals.size == 0:
        return ax

    nticks = min(6, lat_vals.size)
    tick_idx = np.linspace(0, lat_vals.size - 1, num=nticks, dtype=int)
    tick_idx = np.unique(tick_idx)

    ax.set_xticks(x[tick_idx])
    ax.set_xticklabels([f"{lat_vals[i]:.2f}" for i in tick_idx])
    ax.set_xlabel("Latitude (deg)")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(x[tick_idx])
    ax_top.set_xticklabels([f"{lon_vals[i]:.2f}" for i in tick_idx])
    ax_top.set_xlabel("Longitude (deg)")
    return ax_top


def read_gpm_data(path: str, channel: int) -> dict:
    with h5py.File(path, "r") as f:
        if "FS" in f:
            group = "FS"
        elif "NS" in f:
            group = "NS"
        else:
            raise KeyError("GPM file missing FS/NS group.")

        z_ds = f[f"{group}/SLV/zFactorFinal"]
        z = z_ds[..., channel].astype(np.float32)
        fill = z_ds.attrs.get("_FillValue", -9999.9)
        height = f[f"{group}/PRE/height"][...].astype(np.float32)
        lat = f[f"{group}/Latitude"][...].astype(np.float32)
        lon = f[f"{group}/Longitude"][...].astype(np.float32)

    z[z == fill] = np.nan
    return {
        "group": group,
        "z": z,
        "height": height,
        "lat": lat,
        "lon": lon,
    }


def extract_section(
    cs_time: np.ndarray,
    cs_lat: np.ndarray,
    cs_lon: np.ndarray,
    gpm_time: np.ndarray,
    gpm_lat: np.ndarray,
    gpm_lon: np.ndarray,
    gpm_z: np.ndarray,
    gpm_height: np.ndarray,
    max_distance_km: float,
    max_time_diff_min: float,
    method: str,
    interp_neighbors: int,
):
    n_profiles = cs_time.size
    nbin = gpm_z.shape[2]

    z_section = np.full((n_profiles, nbin), np.nan, dtype=np.float32)
    h_section = np.full((n_profiles, nbin), np.nan, dtype=np.float32)
    dist_km = np.full(n_profiles, np.nan, dtype=np.float32)
    time_sec = np.full(n_profiles, np.nan, dtype=np.float32)
    scan_idx = np.full(n_profiles, -1, dtype=int)
    beam_idx = np.full(n_profiles, -1, dtype=int)

    nbeam = gpm_lat.shape[1]
    center_beam = nbeam // 2
    gpm_lat_center = gpm_lat[:, center_beam]
    gpm_lon_center = gpm_lon[:, center_beam]

    window = np.timedelta64(int(max_time_diff_min * 60), "s")
    max_time_sec = int(max_time_diff_min * 60)

    for i, t in enumerate(cs_time):
        if not np.isfinite(cs_lat[i]) or not np.isfinite(cs_lon[i]):
            continue

        start = int(np.searchsorted(gpm_time, t - window, side="left"))
        end = int(np.searchsorted(gpm_time, t + window, side="right"))
        if end <= start:
            continue

        d_center = haversine_km(
            cs_lat[i],
            cs_lon[i],
            gpm_lat_center[start:end],
            gpm_lon_center[start:end],
        )
        if not np.any(np.isfinite(d_center)):
            continue
        s = start + int(np.nanargmin(d_center))

        dt = np.abs((t - gpm_time[s]).astype("timedelta64[s]").astype(np.int64))
        if dt > max_time_sec:
            continue

        lat_row = gpm_lat[s, :]
        lon_row = gpm_lon[s, :]
        d_beam = haversine_km(cs_lat[i], cs_lon[i], lat_row, lon_row)
        valid = np.isfinite(d_beam)
        if not np.any(valid):
            continue

        if method == "nearest":
            j = int(np.nanargmin(d_beam))
            d_min = float(d_beam[j])
            if not np.isfinite(d_min) or d_min > max_distance_km:
                continue
            z_section[i, :] = gpm_z[s, j, :]
            h_section[i, :] = gpm_height[s, j, :]
            dist_km[i] = d_min
            time_sec[i] = float(dt)
            scan_idx[i] = s
            beam_idx[i] = j
        elif method == "interp":
            d_beam = np.where(valid, d_beam, np.inf)
            n_valid = int(np.sum(np.isfinite(d_beam)))
            if n_valid == 0:
                continue
            k = min(interp_neighbors, n_valid)
            idx_k = np.argpartition(d_beam, k - 1)[:k]
            dist_k = d_beam[idx_k]
            keep = np.isfinite(dist_k) & (dist_k <= max_distance_km)
            if not np.any(keep):
                continue
            idx_k = idx_k[keep]
            dist_k = dist_k[keep]

            weights = 1.0 / np.maximum(dist_k, 1e-6)
            w = weights[:, None]

            z_profiles = gpm_z[s, idx_k, :]
            h_profiles = gpm_height[s, idx_k, :]

            z_mask = np.isfinite(z_profiles)
            w_z = w * z_mask
            z_section[i, :] = np.nansum(z_profiles * w_z, axis=0) / np.nansum(w_z, axis=0)

            h_mask = np.isfinite(h_profiles)
            w_h = w * h_mask
            h_section[i, :] = np.nansum(h_profiles * w_h, axis=0) / np.nansum(w_h, axis=0)

            k_best = int(np.nanargmin(dist_k))
            dist_km[i] = float(dist_k[k_best])
            time_sec[i] = float(dt)
            scan_idx[i] = s
            beam_idx[i] = int(idx_k[k_best])
        else:
            raise ValueError(f"Unknown method: {method}")

    return z_section, h_section, dist_km, time_sec, scan_idx, beam_idx


def main() -> None:
    cs_lat = read_vdata_1field(CS_PATH, "Latitude")
    cs_lon = read_vdata_1field(CS_PATH, "Longitude")
    cs_time = build_profile_datetimes(CS_PATH)

    gpm = read_gpm_data(GPM_PATH, CHANNEL)
    gpm_time = build_scan_datetimes(GPM_PATH, gpm["group"])

    t0 = gpm_time.min() - np.timedelta64(int(TIME_MARGIN_MIN * 60), "s")
    t1 = gpm_time.max() + np.timedelta64(int(TIME_MARGIN_MIN * 60), "s")
    keep = (cs_time >= t0) & (cs_time <= t1)
    cs_lat = cs_lat[keep]
    cs_lon = cs_lon[keep]
    cs_time = cs_time[keep]

    if PROFILE_STEP > 1:
        cs_lat = cs_lat[::PROFILE_STEP]
        cs_lon = cs_lon[::PROFILE_STEP]
        cs_time = cs_time[::PROFILE_STEP]

    z_section, h_section, dist_km, time_sec, scan_idx, beam_idx = extract_section(
        cs_time,
        cs_lat,
        cs_lon,
        gpm_time,
        gpm["lat"],
        gpm["lon"],
        gpm["z"],
        gpm["height"],
        max_distance_km=MAX_DISTANCE_KM,
        max_time_diff_min=MAX_TIME_DIFF_MIN,
        method=METHOD,
        interp_neighbors=INTERP_NEIGHBORS,
    )

    match = np.isfinite(dist_km)
    if MATCH_ONLY:
        cs_lat = cs_lat[match]
        cs_lon = cs_lon[match]
        cs_time = cs_time[match]
        z_section = z_section[match, :]
        h_section = h_section[match, :]
        dist_km = dist_km[match]
        time_sec = time_sec[match]
        scan_idx = scan_idx[match]
        beam_idx = beam_idx[match]

    if z_section.size == 0 or np.all(~np.isfinite(z_section)):
        raise RuntimeError("No matched profiles found. Try relaxing time/distance thresholds.")

    height_mean = np.nanmean(h_section, axis=0)
    x = np.arange(z_section.shape[0])

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.pcolormesh(x, height_mean, z_section.T, shading="auto", vmin=VMIN, vmax=VMAX,cmap='jet')
    fig.colorbar(im, label="zFactorFinal (dBZ)")
    apply_lat_lon_axes(ax, x, cs_lat, cs_lon)
    ax.set_ylim(0, np.nanmax(height_mean))

    t_start = str(cs_time.min())[:19] if cs_time.size else "n/a"
    t_end = str(cs_time.max())[:19] if cs_time.size else "n/a"
    ax.set_title(
        "GPM DPR Reflectivity Along CloudSat Track\n"
        f"{t_start} to {t_end} UTC | method={METHOD}, "
        f"max {MAX_DISTANCE_KM:.1f} km, {MAX_TIME_DIFF_MIN:.1f} min"
    )
    ax.set_ylabel("Height (m)")
    fig.tight_layout()

    if OUTPUT_PNG:
        fig.savefig(OUTPUT_PNG, dpi=200)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    print(f"Matched profiles: {z_section.shape[0]}")
    print(f"Mean dist (km): {np.nanmean(dist_km):.2f}")
    print(f"Mean time diff (s): {np.nanmean(time_sec):.1f}")


if __name__ == "__main__":
    main()
