import math

import h5py
import numpy as np
import matplotlib.pyplot as plt

from cloudsat import read_vdata_1field


FILE_PATH = "3B-HHR.MS.MRG.3IMERG.20190803-S040000-E042959.0240.V07B.HDF5"

# 100x100 km region around the center (half-size on each side).
CENTER_LAT = -4.5
CENTER_LON = 145.0 
HALF_SIZE_KM = 100.0

OUTPUT_PNG = "imerg_surface_rain_rate.png"

CLOUDSAT_PATH = "2019215032540_70655_CS_2B-GEOPROF_GRANULE_P1_R05_E09_F00.hdf"
GPM_PATH = "2A.GPM.DPR.V9-20211125.20190803-S033403-E050636.030841.V07A.HDF5"


def _subset_indices(lat, lon, center_lat, center_lon, half_size_km):
    if center_lat is None or center_lon is None:
        return slice(None), slice(None), None, None
    half_lat_deg = half_size_km / 111.0
    lon_scale = max(0.1, math.cos(math.radians(center_lat)))
    half_lon_deg = half_size_km / (111.0 * lon_scale)
    lat_min = center_lat - half_lat_deg
    lat_max = center_lat + half_lat_deg
    lon_min = center_lon - half_lon_deg
    lon_max = center_lon + half_lon_deg
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Center/region outside dataset domain.")
    return slice(lat_idx.min(), lat_idx.max() + 1), slice(lon_idx.min(), lon_idx.max() + 1), half_lat_deg, half_lon_deg


def _filter_track(lat, lon, lat_min, lat_max, lon_min, lon_max):
    valid = np.isfinite(lat) & np.isfinite(lon)
    in_box = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    keep = valid & in_box
    return lat[keep], lon[keep]


def _read_gpm_center_track(path: str):
    with h5py.File(path, "r") as f:
        if "FS" in f:
            group = "FS"
        elif "NS" in f:
            group = "NS"
        else:
            raise KeyError("GPM file missing FS/NS group.")
        lat = f[f"{group}/Latitude"][...].astype(np.float32)
        lon = f[f"{group}/Longitude"][...].astype(np.float32)

    center_beam = lat.shape[1] // 2
    return lat[:, center_beam], lon[:, center_beam]


def main():
    with h5py.File(FILE_PATH, "r") as ds:
        grid = ds["Grid"]
        lat = grid["lat"][:]
        lon = grid["lon"][:]
        precip = grid["precipitation"][0]
        fill_value = grid["precipitation"].attrs.get("_FillValue")
        units_attr = grid["precipitation"].attrs.get("Units", "")
        units = (
            units_attr.decode("utf-8", "ignore")
            if isinstance(units_attr, (bytes, bytearray))
            else str(units_attr)
        )

    lat_sl, lon_sl, half_lat_deg, half_lon_deg = _subset_indices(
        lat, lon, CENTER_LAT, CENTER_LON, HALF_SIZE_KM
    )
    lat_sub = lat[lat_sl]
    lon_sub = lon[lon_sl]
    lat_min = CENTER_LAT - half_lat_deg
    lat_max = CENTER_LAT + half_lat_deg
    lon_min = CENTER_LON - half_lon_deg
    lon_max = CENTER_LON + half_lon_deg

    data = precip[lon_sl, lat_sl].T
    if fill_value is not None:
        data = np.where(data == fill_value, np.nan, data)

    cs_lat = read_vdata_1field(CLOUDSAT_PATH, "Latitude")
    cs_lon = read_vdata_1field(CLOUDSAT_PATH, "Longitude")
    cs_lat, cs_lon = _filter_track(cs_lat, cs_lon, lat_min, lat_max, lon_min, lon_max)

    gpm_lat, gpm_lon = _read_gpm_center_track(GPM_PATH)
    gpm_lat, gpm_lon = _filter_track(gpm_lat, gpm_lon, lat_min, lat_max, lon_min, lon_max)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.pcolormesh(lon_sub, lat_sub, data, cmap="turbo", shading="auto")
    ax.scatter([CENTER_LON], [CENTER_LAT], color="k", s=20, zorder=3)
    if cs_lat.size:
        ax.plot(cs_lon, cs_lat, color="black", linewidth=1.2, label="CloudSat track")
    if gpm_lat.size:
        ax.plot(gpm_lon, gpm_lat, color="tab:blue", linewidth=1.2, label="GPM track")
    if half_lat_deg is not None and half_lon_deg is not None:
        rect = plt.Rectangle(
            (CENTER_LON - half_lon_deg, CENTER_LAT - half_lat_deg),
            2 * half_lon_deg,
            2 * half_lat_deg,
            fill=False,
            color="k",
            lw=1.2,
        )
        ax.add_patch(rect)

    ax.set_title("IMERG Surface Rain Rate")
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    label = f"Rain rate ({units})" if units else "Rain rate"
    fig.colorbar(im, ax=ax, label=label)
    if cs_lat.size or gpm_lat.size:
        ax.legend(loc="best")
    fig.savefig(OUTPUT_PNG, dpi=150)
    print("Saved:", OUTPUT_PNG)


if __name__ == "__main__":
    main()
