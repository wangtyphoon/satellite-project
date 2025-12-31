import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------
# Optional: Cartopy for maps
# -------------------------
USE_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    USE_CARTOPY = False
    print("Cartopy not found. Using standard Matplotlib plotting.")

# -------------------------
# Constants
# -------------------------
MU_EARTH = 398600.4418  # km^3/s^2
WGS84_A = 6378.137      # km (Earth Radius)
WGS84_F = 1.0 / 298.257223563
J2 = 1.08263e-3         # J2 Zonal Harmonic

# -------------------------
# TLE Data (Jan 1, 2015)
# -------------------------
CLOUDSAT_TLE = (
    "1 29107U 06016A   15001.13125000  .00000000  00000-0  00000-0 0  9991",
    "2 29107  98.2170 330.8200 0000824  91.6200  14.5700 14.57000000 46515",
)

GPM_TLE = (
    "1 39574U 14009A   15001.13125000  .00000000  00000-0  00000-0 0  9992",
    "2 39574  65.0000  45.0000 0010000   0.0000   0.0000 15.50000000 1000",
)

# -------------------------
# Time & Coordinate Helpers
# -------------------------
def tle_epoch_to_datetime(line1: str) -> datetime:
    """Parse TLE epoch string to datetime object."""
    epoch_str = line1[18:32].strip()
    yy = int(epoch_str[0:2])
    day = float(epoch_str[2:])
    year = 2000 + yy if yy < 57 else 1900 + yy
    day_int = int(np.floor(day))
    frac = day - day_int
    dt0 = datetime(year, 1, 1) + timedelta(days=day_int - 1, seconds=frac * 86400.0)
    return dt0

def julian_date(dt: datetime) -> float:
    """Calculate Julian Date from datetime."""
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + (A // 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return float(jd)

def gmst_radians(jd_ut1: float) -> float:
    """Calculate GMST in radians (IAU-82 approx)."""
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    return 2.0 * np.pi * ((gmst_sec % 86400.0) / 86400.0)

def eci_to_ecef(x, y, z, jd):
    """Rotate TEME/ECI coordinates to ECEF."""
    theta = gmst_radians(jd)
    c, s = np.cos(theta), np.sin(theta)
    x_new =  x * c + y * s
    y_new = -x * s + y * c
    return x_new, y_new, z

def split_dateline(lon_deg, lat_deg, jump_threshold=180.0):
    """Split tracks that cross the dateline for clean plotting."""
    lon = np.asarray(lon_deg)
    lat = np.asarray(lat_deg)
    jumps = np.abs(np.diff(lon)) > jump_threshold
    cut_idx = np.where(jumps)[0] + 1
    segments = np.split(np.arange(len(lon)), cut_idx)
    out = []
    for s in segments:
        if len(s) >= 2:
            out.append((lon[s], lat[s]))
    return out

def great_circle_distance_km(lat1, lon1, lat2, lon2, radius_km=WGS84_A):
    """Great-circle distance between points (deg). Supports array inputs for lat2/lon2."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return radius_km * c

# -------------------------
# J2 Propagation Logic (The Fix)
# -------------------------
def recover_semimajor_axis(n_rev_day, ecc, inc_rad):
    """
    Recover 'a' from TLE mean motion 'n' accounting for J2.
    Critical for getting the correct orbital period.
    """
    n_rad_s = n_rev_day * 2.0 * np.pi / 86400.0
    a_meas = (MU_EARTH / (n_rad_s**2))**(1.0/3.0) # Initial guess (Kepler)
    
    a = a_meas
    for _ in range(10):
        # Secular variation approximation
        p = a * (1 - ecc**2)
        term = 1.5 * J2 * (WGS84_A / p)**2 * np.sqrt(1 - ecc**2) * (1 - 1.5 * np.sin(inc_rad)**2)
        n_kepler = n_rad_s / (1 + term)
        a_new = (MU_EARTH / (n_kepler**2))**(1.0/3.0)
        
        if abs(a_new - a) < 1e-8:
            a = a_new
            break
        a = a_new
        
    return a, n_rad_s

def propagate_j2(tle1, tle2, duration_min, step_sec=60, start_offset_sec=0.0):
    """
    Propagate orbit using J2 perturbed model.
    start_offset_sec lets us begin before/after the TLE epoch (can be negative).
    Returns epoch_dt, lons, lats, times_sec (relative to TLE epoch).
    """
    # 1. Parse TLE
    epoch_dt = tle_epoch_to_datetime(tle1)
    print("TLE Epoch (UTC):", epoch_dt)
    parts = tle2.split()
    inc = np.deg2rad(float(parts[2]))
    raan0 = np.deg2rad(float(parts[3]))
    ecc = float("0." + parts[4])
    argp0 = np.deg2rad(float(parts[5]))
    M0 = np.deg2rad(float(parts[6]))
    n_rev_day = float(parts[7])

    # 2. Recover physics parameters (The Fix)
    a, n_mean_tle = recover_semimajor_axis(n_rev_day, ecc, inc)
    
    # 3. Calculate Secular Rates (J2 Precession)
    p = a * (1 - ecc**2)
    n_kepler = np.sqrt(MU_EARTH / a**3)
    
    # RAAN precession (Nodal precession) - moves orbital plane
    raan_dot = -1.5 * n_kepler * J2 * (WGS84_A / p)**2 * np.cos(inc)
    
    # Argument of Perigee precession
    argp_dot = 0.75 * n_kepler * J2 * (WGS84_A / p)**2 * (4 - 5 * np.sin(inc)**2)

    lats, lons = [], []
    total_sec = duration_min * 60.0
    times = np.arange(start_offset_sec, start_offset_sec + total_sec + 1, step_sec)

    for t in times:
        # Update orbital elements
        raan = raan0 + raan_dot * t
        argp = argp0 + argp_dot * t
        M = M0 + n_mean_tle * t
        
        # Kepler Equation
        E = M if ecc < 0.8 else np.pi
        for _ in range(15):
            f = E - ecc * np.sin(E) - M
            fp = 1.0 - ecc * np.cos(E)
            dE = -f / fp
            E += dE
            if abs(dE) < 1e-10: break
            
        # Position in Orbital Plane (PQW)
        r_val = a * (1 - ecc * np.cos(E))
        
        sin_nu = (np.sqrt(1 - ecc**2) * np.sin(E)) / (1 - ecc * np.cos(E))
        cos_nu = (np.cos(E) - ecc) / (1 - ecc * np.cos(E))
        u = np.arctan2(sin_nu, cos_nu) + argp  # Argument of Latitude
        
        # PQW -> TEME (Inertial)
        x_node = r_val * np.cos(u)
        y_node = r_val * np.sin(u)
        
        x_eci = x_node * np.cos(raan) - y_node * np.cos(inc) * np.sin(raan)
        y_eci = x_node * np.sin(raan) + y_node * np.cos(inc) * np.cos(raan)
        z_eci = y_node * np.sin(inc)
        
        # TEME -> ECEF
        current_dt = epoch_dt + timedelta(seconds=float(t))
        jd = julian_date(current_dt)
        x_ecef, y_ecef, z_ecef = eci_to_ecef(x_eci, y_eci, z_eci, jd)
        
        # ECEF -> Geodetic (Spherical approx for plotting)
        lon = np.arctan2(y_ecef, x_ecef)
        hyp = np.sqrt(x_ecef**2 + y_ecef**2)
        lat = np.arctan2(z_ecef, hyp)
        
        lats.append(np.degrees(lat))
        lons.append((np.degrees(lon) + 540) % 360 - 180)

    return epoch_dt, np.array(lons), np.array(lats), times


def find_intersections(lat_c, lon_c, lat_g, lon_g, times_sec, step_sec,
                       window_minutes=20.0, max_distance_km=200.0):
    """
    Find time spans where CloudSat and GPM pass within a time window (±window_minutes)
    and spatial threshold (great-circle distance <= max_distance_km).
    Returns list of (start_idx, end_idx, matched_idx_array, min_dist_array).
    """
    n = len(times_sec)
    window_steps = int(np.ceil(window_minutes * 60.0 / step_sec))
    nearest_j = np.zeros(n, dtype=int)
    min_dist = np.zeros(n, dtype=float)

    for i in range(n):
        j0 = max(0, i - window_steps)
        j1 = min(n, i + window_steps + 1)
        dists = great_circle_distance_km(lat_c[i], lon_c[i], lat_g[j0:j1], lon_g[j0:j1])
        k = int(np.argmin(dists))
        nearest_j[i] = j0 + k
        min_dist[i] = float(dists[k])

    close_mask = min_dist <= max_distance_km
    intersections = []
    i = 0
    while i < n:
        if not close_mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and close_mask[i + 1]:
            i += 1
        end = i
        intersections.append((start, end, nearest_j[start:end + 1], min_dist[start:end + 1]))
        i += 1

    return intersections, min_dist, nearest_j

# -------------------------
# Main Execution
# -------------------------
def main():
    # ------------ Plot 1: short segment (about 3 orbits) ------------
    short_duration_min = 317  # ~3.2 orbits as before
    print("Propagating short segment (~3 orbits)...")
    ep_c_short, lon_c_s, lat_c_s, _ = propagate_j2(CLOUDSAT_TLE[0], CLOUDSAT_TLE[1], short_duration_min)
    ep_g_short, lon_g_s, lat_g_s, _ = propagate_j2(GPM_TLE[0], GPM_TLE[1], short_duration_min)

    segs_c_s = split_dateline(lon_c_s, lat_c_s)
    segs_g_s = split_dateline(lon_g_s, lat_g_s)

    fig = plt.figure(figsize=(13, 7))
    if USE_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        transform_args = {'transform': ccrs.PlateCarree()}
    else:
        ax = plt.gca()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        transform_args = {}

    first = True
    for lo, la in segs_c_s:
        ax.plot(lo, la, color='tab:blue', linewidth=1.6,
                label='CloudSat (J2)' if first else None, **transform_args)
        first = False
    first = True
    for lo, la in segs_g_s:
        ax.plot(lo, la, color='tab:orange', linewidth=1.6, linestyle='--',
                label='GPM (J2)' if first else None, **transform_args)
        first = False

    ax.plot(lon_c_s[0], lat_c_s[0], 'o', color='tab:blue', label='CloudSat start', **transform_args)
    ax.plot(lon_c_s[-1], lat_c_s[-1], 'x', color='tab:blue', label='CloudSat end', **transform_args)
    ax.plot(lon_g_s[0], lat_g_s[0], 'o', color='tab:orange', label='GPM start', **transform_args)
    ax.plot(lon_g_s[-1], lat_g_s[-1], 'x', color='tab:orange', label='GPM end', **transform_args)

    ax.set_title(
        f"CloudSat & GPM Ground Tracks (~3 orbits)\n"
        f"Start epoch (UTC): {ep_c_short:%Y-%m-%d %H:%M:%S}"
    )
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='lower left')
    plt.tight_layout()
    out_short = "short_ground_tracks.png"
    plt.savefig(out_short, dpi=150)
    print(f"Saved plot to {out_short}")
    plt.show()

    # # ------------ Plot 2: full month intersections ------------
    # window_minutes = 20.0
    # max_distance_km = 200.0  # spatial proximity threshold for "same region"
    # step_sec = 60
    # start_dt = datetime(2015, 1, 1, 0, 0, 0)
    # end_dt = datetime(2015, 2, 1, 0, 0, 0)
    # epoch_dt = tle_epoch_to_datetime(CLOUDSAT_TLE[0])

    # start_offset_sec = (start_dt - epoch_dt).total_seconds()
    # duration_min = int(np.ceil((end_dt - start_dt).total_seconds() / 60.0))

    # print("Propagating CloudSat for January 2015...")
    # ep_c, lon_c, lat_c, times_c = propagate_j2(
    #     CLOUDSAT_TLE[0], CLOUDSAT_TLE[1], duration_min,
    #     step_sec=step_sec, start_offset_sec=start_offset_sec
    # )

    # print("Propagating GPM for January 2015...")
    # ep_g, lon_g, lat_g, times_g = propagate_j2(
    #     GPM_TLE[0], GPM_TLE[1], duration_min,
    #     step_sec=step_sec, start_offset_sec=start_offset_sec
    # )

    # n = min(len(times_c), len(times_g))
    # lon_c, lat_c, times_c = lon_c[:n], lat_c[:n], times_c[:n]
    # lon_g, lat_g, times_g = lon_g[:n], lat_g[:n], times_g[:n]

    # intersections, min_dist, nearest_j = find_intersections(
    #     lat_c, lon_c, lat_g, lon_g, times_c, step_sec,
    #     window_minutes=window_minutes, max_distance_km=max_distance_km
    # )

    # print(f"Found {len(intersections)} intersection windows (<= {max_distance_km} km within ±{window_minutes} min).")
    # for idx, (s, e, match_idx, _) in enumerate(intersections, 1):
    #     dt_start = epoch_dt + timedelta(seconds=float(times_c[s]))
    #     dt_end = epoch_dt + timedelta(seconds=float(times_c[e]))
    #     print(f"{idx:02d}: {dt_start:%Y-%m-%d %H:%M} to {dt_end:%Y-%m-%d %H:%M} "
    #           f"({(e - s + 1) * step_sec / 60:.1f} min)")

    # segs_c = split_dateline(lon_c, lat_c)
    # segs_g = split_dateline(lon_g, lat_g)

    # fig = plt.figure(figsize=(13, 7))

    # if USE_CARTOPY:
    #     ax = plt.axes(projection=ccrs.PlateCarree())
    #     ax.set_global()
    #     ax.coastlines(linewidth=0.8)
    #     ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
    #     gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
    #     gl.top_labels = False
    #     gl.right_labels = False
    #     transform_args = {'transform': ccrs.PlateCarree()}
    # else:
    #     ax = plt.gca()
    #     ax.set_xlim(-180, 180)
    #     ax.set_ylim(-90, 90)
    #     ax.grid(True, linestyle=':', alpha=0.6)
    #     ax.set_xlabel("Longitude")
    #     ax.set_ylabel("Latitude")
    #     transform_args = {}

    # first = True
    # for lo, la in segs_c:
    #     ax.plot(lo, la, color='tab:blue', linewidth=1.0, alpha=0.5,
    #             label='CloudSat (J2)' if first else None, **transform_args)
    #     first = False

    # first = True
    # for lo, la in segs_g:
    #     ax.plot(lo, la, color='tab:orange', linewidth=1.0, alpha=0.5, linestyle='--',
    #             label='GPM (J2)' if first else None, **transform_args)
    #     first = False

    # ax.plot(lon_c[0], lat_c[0], 'o', color='tab:blue', label='CloudSat start', **transform_args)
    # ax.plot(lon_c[-1], lat_c[-1], 'x', color='tab:blue', label='CloudSat end', **transform_args)
    # ax.plot(lon_g[0], lat_g[0], 'o', color='tab:orange', label='GPM start', **transform_args)
    # ax.plot(lon_g[-1], lat_g[-1], 'x', color='tab:orange', label='GPM end', **transform_args)

    # first_c = True
    # first_g = True
    # for s, e, match_idx, _ in intersections:
    #     c_lo = lon_c[s:e + 1]
    #     c_la = lat_c[s:e + 1]
    #     g_start = int(np.min(match_idx))
    #     g_end = int(np.max(match_idx))
    #     g_lo = lon_g[g_start:g_end + 1]
    #     g_la = lat_g[g_start:g_end + 1]

    #     for lo, la in split_dateline(c_lo, c_la):
    #         ax.plot(lo, la, color='crimson', linewidth=2.2,
    #                 label='Intersection path (CloudSat)' if first_c else None, **transform_args)
    #         first_c = False
    #     for lo, la in split_dateline(g_lo, g_la):
    #         ax.plot(lo, la, color='limegreen', linewidth=2.2, linestyle='-',
    #                 label='Intersection path (GPM)' if first_g else None, **transform_args)
    #         first_g = False

    # ax.set_title(
    #     f"CloudSat / GPM Near-Simultaneous Overpasses (Jan 2015)\n"
    #     f"<= {max_distance_km} km and ±{window_minutes:.0f} min temporal window"
    # )
    # handles, labels = ax.get_legend_handles_labels()
    # uniq = dict(zip(labels, handles))
    # ax.legend(uniq.values(), uniq.keys(), loc='lower left')

    # out_file = "jan2015_intersections.png"
    # plt.tight_layout()
    # plt.savefig(out_file, dpi=150)
    # print(f"Saved plot to {out_file}")
    # plt.show()

if __name__ == "__main__":
    main()
