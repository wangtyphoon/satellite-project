#!/usr/bin/env python3
"""
Plot GPM heating-rate sections sampled along the CloudSat track.

Products:
  - GPM 2HSLH: total latent heating
  - GPM 2HCSH: eddy heating (vertical + horizontal) and LW radiative heating
"""

from __future__ import annotations

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pyhdf.HDF import HDF, HC
from pyhdf.VS import VS


GPM_2HSLH_PATH = "2A.GPM.DPR.GPM-SLH.20190803-S033403-E050636.030841.V07A.HDF5"
GPM_2HCSH_PATH = "2B.GPM.DPRGMI.2HCSHv7-0.20190803-S033403-E050636.030841.V07A.HDF5"
CS_PATH = "2019215032540_70655_CS_2B-GEOPROF_GRANULE_P1_R05_E09_F00.hdf"

HSLH_LATENT_VAR = "Swath/latentHeating"
HCSH_LATENT_VAR = "Swath/latentHeating"
HCSH_VEDDY_VAR = "Swath/vEddyHeating"
HCSH_HEDDY_VAR = "Swath/hEddyHeating"
HCSH_LW_VAR = "Swath/lwRadiativeHeating"

METHOD = "interp"  # "nearest" or "interp"
INTERP_NEIGHBORS = 8

MAX_DISTANCE_KM = 10.0
MAX_TIME_DIFF_MIN = 20.0
TIME_MARGIN_MIN = 20.0

PROFILE_STEP = 1
MATCH_ONLY = True

LAYER_BOTTOM_KM = 0.0
LAYER_TOP_KM = 20.0
LAYER_HEIGHT_KM = None  # Optional explicit array (length = nlayer)

LATENT_VMIN = None
LATENT_VMAX = None
EDDY_VMIN = None
EDDY_VMAX = None
LW_VMIN = None
LW_VMAX = None

OUTPUT_PNG = "gpm_cloudsat_heating_section.png"
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


def build_scan_datetimes(path: str, group: str = "Swath") -> np.ndarray:
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


def _coerce_fill_values(attrs: dict) -> list[float]:
    vals = []
    for key in ("_FillValue", "CodeMissingValue"):
        if key not in attrs:
            continue
        val = attrs[key]
        if isinstance(val, np.ndarray):
            if val.size == 0:
                continue
            val = val.flat[0]
        vals.append(float(val))
    return vals


def _mask_fill(data: np.ndarray, attrs: dict) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    for fv in _coerce_fill_values(attrs):
        out[out == fv] = np.nan
    return out


def read_swath_file(path: str, var_paths: dict[str, str]) -> dict:
    with h5py.File(path, "r") as f:
        lat = f["Swath/Latitude"][...].astype(np.float32)
        lon = f["Swath/Longitude"][...].astype(np.float32)
        data = {}
        attrs = {}
        for name, vp in var_paths.items():
            ds = f[vp]
            data[name] = _mask_fill(ds[...], ds.attrs)
            attrs[name] = {k: ds.attrs[k] for k in ds.attrs.keys()}
    time = build_scan_datetimes(path, "Swath")
    return {"lat": lat, "lon": lon, "time": time, "data": data, "attrs": attrs}


def compute_match_indices(
    cs_time: np.ndarray,
    cs_lat: np.ndarray,
    cs_lon: np.ndarray,
    swath_time: np.ndarray,
    swath_lat: np.ndarray,
    swath_lon: np.ndarray,
    max_distance_km: float,
    max_time_diff_min: float,
    method: str,
    interp_neighbors: int,
):
    n_profiles = cs_time.size
    nbeam = swath_lat.shape[1]
    center_beam = nbeam // 2
    k = interp_neighbors if method == "interp" else 1

    scan_idx = np.full(n_profiles, -1, dtype=int)
    beam_idx = np.full(n_profiles, -1, dtype=int)
    beam_k_idx = np.full((n_profiles, k), -1, dtype=int)
    beam_k_w = np.zeros((n_profiles, k), dtype=np.float32)
    dist_km = np.full(n_profiles, np.nan, dtype=np.float32)
    time_sec = np.full(n_profiles, np.nan, dtype=np.float32)

    window = np.timedelta64(int(max_time_diff_min * 60), "s")
    max_time_sec = int(max_time_diff_min * 60)

    for i, t in enumerate(cs_time):
        if not np.isfinite(cs_lat[i]) or not np.isfinite(cs_lon[i]):
            continue

        start = int(np.searchsorted(swath_time, t - window, side="left"))
        end = int(np.searchsorted(swath_time, t + window, side="right"))
        if end <= start:
            continue

        d_center = haversine_km(
            cs_lat[i],
            cs_lon[i],
            swath_lat[start:end, center_beam],
            swath_lon[start:end, center_beam],
        )
        if not np.any(np.isfinite(d_center)):
            continue
        s = start + int(np.nanargmin(d_center))

        dt = np.abs((t - swath_time[s]).astype("timedelta64[s]").astype(np.int64))
        if dt > max_time_sec:
            continue

        lat_row = swath_lat[s, :]
        lon_row = swath_lon[s, :]
        d_beam = haversine_km(cs_lat[i], cs_lon[i], lat_row, lon_row)
        valid = np.isfinite(d_beam)
        if not np.any(valid):
            continue

        if method == "nearest":
            j = int(np.nanargmin(d_beam))
            d_min = float(d_beam[j])
            if not np.isfinite(d_min) or d_min > max_distance_km:
                continue
            scan_idx[i] = s
            beam_idx[i] = j
            beam_k_idx[i, 0] = j
            beam_k_w[i, 0] = 1.0
            dist_km[i] = d_min
            time_sec[i] = float(dt)
        elif method == "interp":
            d_beam = np.where(valid, d_beam, np.inf)
            n_valid = int(np.sum(np.isfinite(d_beam)))
            if n_valid == 0:
                continue
            kk = min(k, n_valid)
            idx_k = np.argpartition(d_beam, kk - 1)[:kk]
            dist_k = d_beam[idx_k]
            keep = np.isfinite(dist_k) & (dist_k <= max_distance_km)
            if not np.any(keep):
                continue
            idx_k = idx_k[keep]
            dist_k = dist_k[keep]
            weights = 1.0 / np.maximum(dist_k, 1e-6)

            scan_idx[i] = s
            beam_idx[i] = int(idx_k[int(np.nanargmin(dist_k))])
            beam_k_idx[i, : idx_k.size] = idx_k
            beam_k_w[i, : idx_k.size] = weights
            dist_km[i] = float(dist_k[np.nanargmin(dist_k)])
            time_sec[i] = float(dt)
        else:
            raise ValueError(f"Unknown method: {method}")

    return scan_idx, beam_idx, beam_k_idx, beam_k_w, dist_km, time_sec


def sample_swath_vars(
    swath_vars: dict[str, np.ndarray],
    scan_idx: np.ndarray,
    beam_k_idx: np.ndarray,
    beam_k_w: np.ndarray,
) -> dict[str, np.ndarray]:
    n_profiles = scan_idx.size
    out = {}
    for name, data in swath_vars.items():
        nlayer = data.shape[2]
        section = np.full((n_profiles, nlayer), np.nan, dtype=np.float32)
        for i in range(n_profiles):
            s = scan_idx[i]
            if s < 0:
                continue
            idx_k = beam_k_idx[i]
            w = beam_k_w[i]
            valid = idx_k >= 0
            if not np.any(valid):
                continue
            idx_k = idx_k[valid]
            w = w[valid]
            profiles = data[s, idx_k, :]
            w2 = w[:, None]
            w_mask = w2 * np.isfinite(profiles)
            denom = np.nansum(w_mask, axis=0)
            num = np.nansum(profiles * w_mask, axis=0)
            section[i, :] = np.where(denom > 0, num / denom, np.nan)
        out[name] = section
    return out


def _units(attrs: dict) -> str:
    for key in ("Units", "units"):
        if key in attrs:
            val = attrs[key]
            if isinstance(val, bytes):
                return val.decode("utf-8", "ignore")
            return str(val)
    return ""


def _ensure_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def main() -> None:
    _ensure_file(CS_PATH)
    _ensure_file(GPM_2HCSH_PATH)
    _ensure_file(GPM_2HSLH_PATH)

    cs_lat = read_vdata_1field(CS_PATH, "Latitude")
    cs_lon = read_vdata_1field(CS_PATH, "Longitude")
    cs_time = build_profile_datetimes(CS_PATH)

    ref = read_swath_file(GPM_2HCSH_PATH, {"latent": HCSH_LATENT_VAR})
    hcs = read_swath_file(
        GPM_2HCSH_PATH,
        {
            "v_eddy": HCSH_VEDDY_VAR,
            "h_eddy": HCSH_HEDDY_VAR,
            "lw": HCSH_LW_VAR,
        },
    )
    hsl = read_swath_file(GPM_2HSLH_PATH, {"latent": HSLH_LATENT_VAR})

    if (
        hsl["lat"].shape != ref["lat"].shape
        or hsl["lon"].shape != ref["lon"].shape
        or hsl["time"].shape != ref["time"].shape
    ):
        raise ValueError("2HSLH and 2HCSH grids do not match; update matching logic.")

    t0 = ref["time"].min() - np.timedelta64(int(TIME_MARGIN_MIN * 60), "s")
    t1 = ref["time"].max() + np.timedelta64(int(TIME_MARGIN_MIN * 60), "s")
    keep = (cs_time >= t0) & (cs_time <= t1)
    cs_lat = cs_lat[keep]
    cs_lon = cs_lon[keep]
    cs_time = cs_time[keep]

    if PROFILE_STEP > 1:
        cs_lat = cs_lat[::PROFILE_STEP]
        cs_lon = cs_lon[::PROFILE_STEP]
        cs_time = cs_time[::PROFILE_STEP]

    scan_idx, beam_idx, beam_k_idx, beam_k_w, dist_km, time_sec = compute_match_indices(
        cs_time,
        cs_lat,
        cs_lon,
        ref["time"],
        ref["lat"],
        ref["lon"],
        max_distance_km=MAX_DISTANCE_KM,
        max_time_diff_min=MAX_TIME_DIFF_MIN,
        method=METHOD,
        interp_neighbors=INTERP_NEIGHBORS,
    )

    hsl_section = sample_swath_vars(hsl["data"], scan_idx, beam_k_idx, beam_k_w)
    hcs_section = sample_swath_vars(hcs["data"], scan_idx, beam_k_idx, beam_k_w)

    total_latent = hsl_section["latent"]
    eddy_heat = hcs_section["v_eddy"] + hcs_section["h_eddy"]
    lw_heat = hcs_section["lw"]

    match = np.isfinite(dist_km)
    if MATCH_ONLY:
        cs_lat = cs_lat[match]
        cs_lon = cs_lon[match]
        cs_time = cs_time[match]
        total_latent = total_latent[match, :]
        eddy_heat = eddy_heat[match, :]
        lw_heat = lw_heat[match, :]
        dist_km = dist_km[match]
        time_sec = time_sec[match]

    if total_latent.size == 0:
        raise RuntimeError("No matched profiles found. Try relaxing thresholds.")

    nlayer = total_latent.shape[1]
    if LAYER_HEIGHT_KM is not None:
        if len(LAYER_HEIGHT_KM) != nlayer:
            raise ValueError("LAYER_HEIGHT_KM length does not match nlayer.")
        y = np.asarray(LAYER_HEIGHT_KM)
        y_label = "Height (km)"
    elif LAYER_BOTTOM_KM is not None and LAYER_TOP_KM is not None:
        y = np.linspace(LAYER_BOTTOM_KM, LAYER_TOP_KM, nlayer)
        y_label = "Height (km)"
    else:
        y = np.arange(nlayer)
        y_label = "Vertical level"

    x = np.arange(total_latent.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    im0 = axes[0].pcolormesh(x, y, total_latent.T, shading="auto", vmin=LATENT_VMIN, vmax=3,cmap='jet')
    units_latent = _units(hsl["attrs"]["latent"])
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.set_label(f"Total latent heating ({units_latent})" if units_latent else "Total latent heating")
    axes[0].set_ylabel(y_label)

    im1 = axes[1].pcolormesh(x, y, eddy_heat.T, shading="auto", vmin=EDDY_VMIN, vmax=EDDY_VMAX,cmap='jet')
    units_eddy = _units(hcs["attrs"]["v_eddy"])
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.set_label(
        f"V+H eddy heating ({units_eddy})" if units_eddy else "V+H eddy heating"
    )
    axes[1].set_ylabel(y_label)

    im2 = axes[2].pcolormesh(x, y, lw_heat.T, shading="auto", vmin=LW_VMIN, vmax=LW_VMAX,cmap='jet')
    units_lw = _units(hcs["attrs"]["lw"])
    cb2 = fig.colorbar(im2, ax=axes[2])
    cb2.set_label(f"LW radiative heating ({units_lw})" if units_lw else "LW radiative heating")
    axes[2].set_ylabel(y_label)

    apply_lat_lon_axes(axes[2], x, cs_lat, cs_lon)

    t_start = str(cs_time.min())[:19] if cs_time.size else "n/a"
    t_end = str(cs_time.max())[:19] if cs_time.size else "n/a"
    axes[0].set_title(
        "GPM Heating Rates Along CloudSat Track\n"
        f"{t_start} to {t_end} UTC | method={METHOD}, "
        f"max {MAX_DISTANCE_KM:.1f} km, {MAX_TIME_DIFF_MIN:.1f} min"
    )

    fig.tight_layout()
    if OUTPUT_PNG:
        fig.savefig(OUTPUT_PNG, dpi=200)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    print(f"Matched profiles: {total_latent.shape[0]}")
    print(f"Mean dist (km): {np.nanmean(dist_km):.2f}")
    print(f"Mean time diff (s): {np.nanmean(time_sec):.1f}")


if __name__ == "__main__":
    main()

# def describe_dataset(h5_path: str, dset_path: str):
#     import h5py, numpy as np
#     with h5py.File(h5_path, "r") as f:
#         ds = f[dset_path]
#         print("Path:", dset_path)
#         print("Shape:", ds.shape, "dtype:", ds.dtype)
#         print("Attrs:")
#         for k in ds.attrs.keys():
#             v = ds.attrs[k]
#             if isinstance(v, bytes):
#                 v = v.decode("utf-8", "ignore")
#             elif isinstance(v, np.ndarray) and v.size <= 20:
#                 v = v.tolist()
#             print(f"  {k}: {v}")

# describe_dataset(GPM_2HSLH_PATH, "Swath/latentHeating")
# describe_dataset(GPM_2HCSH_PATH, "Swath/latentHeating")
# describe_dataset(GPM_2HCSH_PATH, "Swath/vEddyHeating")
# describe_dataset(GPM_2HCSH_PATH, "Swath/hEddyHeating")
# describe_dataset(GPM_2HCSH_PATH, "Swath/lwRadiativeHeating")
