import math

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt


FILE_PATH = "merg_2019080304_4km-pixel.nc4"

# Set the storm center and radius to match your assignment figure.
# If CENTER_LAT/LON is None, the script will use the full domain (no circle).
CENTER_LAT = -4.5
CENTER_LON = 145.0
RADIUS_KM = 75

# Cold cloud threshold for deep convection (K).
TB_THRESHOLD_K = 220.0

OUTPUT_PNG = "mergeir_tb_compare.png"


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _subset_indices(lat, lon, center_lat, center_lon, radius_km):
    if center_lat is None or center_lon is None:
        return slice(None), slice(None), None
    radius_deg = radius_km / 111.0
    lat_min = center_lat - radius_deg
    lat_max = center_lat + radius_deg
    lon_min = center_lon - radius_deg
    lon_max = center_lon + radius_deg
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Center/radius outside dataset domain.")
    return slice(lat_idx.min(), lat_idx.max() + 1), slice(lon_idx.min(), lon_idx.max() + 1), radius_deg


def main():
    ds = nc.Dataset(FILE_PATH)
    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]
    time_var = ds.variables["time"]
    time = time_var[:]
    tb = ds.variables["Tb"]

    lat_sl, lon_sl, radius_deg = _subset_indices(lat, lon, CENTER_LAT, CENTER_LON, RADIUS_KM)
    lat_sub = lat[lat_sl]
    lon_sub = lon[lon_sl]

    lons2d, lats2d = np.meshgrid(lon_sub, lat_sub)
    if CENTER_LAT is not None and CENTER_LON is not None:
        dist_km = np.vectorize(_haversine_km)(lats2d, lons2d, CENTER_LAT, CENTER_LON)
        circle_mask = dist_km <= RADIUS_KM
    else:
        circle_mask = None

    tb_t0 = tb[0, lat_sl, lon_sl].filled(np.nan)
    tb_t1 = tb[1, lat_sl, lon_sl].filled(np.nan)

    if circle_mask is not None:
        tb_t0_c = np.where(circle_mask, tb_t0, np.nan)
        tb_t1_c = np.where(circle_mask, tb_t1, np.nan)
    else:
        tb_t0_c = tb_t0
        tb_t1_c = tb_t1

    mean0 = np.nanmean(tb_t0_c)
    mean1 = np.nanmean(tb_t1_c)
    cold0 = np.nanmean(tb_t0_c < TB_THRESHOLD_K)
    cold1 = np.nanmean(tb_t1_c < TB_THRESHOLD_K)
    min0 = np.nanmin(tb_t0_c) if np.isfinite(tb_t0_c).any() else np.nan
    min1 = np.nanmin(tb_t1_c) if np.isfinite(tb_t1_c).any() else np.nan

    if mean1 < mean0 and cold1 > cold0:
        label = "Intensifying"
    elif mean1 > mean0 and cold1 < cold0:
        label = "Weakening"
    else:
        label = "Mixed/Uncertain"

    try:
        time_vals = nc.num2date(time, units=time_var.units)
    except Exception:
        time_vals = time
    print("Time0:", time_vals[0], "Time1:", time_vals[1])
    if hasattr(time_vals[0], "strftime"):
        time_labels = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_vals[:2]]
    else:
        time_labels = [str(t) for t in time_vals[:2]]
    print(f"Mean Tb: {mean0:.2f} K -> {mean1:.2f} K")
    print(f"Cold cloud fraction (<{TB_THRESHOLD_K:.0f}K): {cold0:.3f} -> {cold1:.3f}")
    if np.isfinite(min0) and np.isfinite(min1):
        if min1 <= min0:
            print(f"Coldest cloud-top Tb: {min0:.2f} K -> {min1:.2f} K (cooling)")
        else:
            print(f"Coldest cloud-top Tb: {min0:.2f} K -> {min1:.2f} K (warming)")
    else:
        print("Coldest cloud-top Tb: N/A (no valid data in selection)")
    print("Decision:", label)

    vmin, vmax = 180, 320
    cmap = "turbo"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, data, title in [
        (axes[0], tb_t0, f"Time 0 ({time_labels[0]})"),
        (axes[1], tb_t1, f"Time 1 ({time_labels[1]})"),
    ]:
        im = ax.pcolormesh(lon_sub, lat_sub, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        if CENTER_LAT is not None and CENTER_LON is not None:
            circle = plt.Circle((CENTER_LON, CENTER_LAT), radius_deg*0.8, fill=False, color="k", lw=1.5)
            ax.add_patch(circle)

    fig.colorbar(im, ax=axes, label="Tb (K)")
    fig.suptitle(f"IR Brightness Temperature ({label})")
    fig.savefig(OUTPUT_PNG, dpi=150)
    print("Saved:", OUTPUT_PNG)


if __name__ == "__main__":
    main()
