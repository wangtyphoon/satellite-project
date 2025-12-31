import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------
# Optional: cartopy for coastlines
# -------------------------
USE_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    USE_CARTOPY = False


# -------------------------
# Given TLEs (Jan 1, 2015)
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
# Constants
# -------------------------
MU_EARTH = 398600.4418        # km^3 / s^2
OMEGA_EARTH = 7.2921150e-5    # rad / s  (Earth rotation rate)
WGS84_A = 6378.137            # km
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


# -------------------------
# Time helpers
# -------------------------
def tle_epoch_to_datetime(line1: str) -> datetime:
    """
    Parse TLE epoch from line 1: YYDDD.DDDDDDDD -> datetime (UTC approx).
    Example: 15001.13125000 -> 2015, day 1 + fractional day.
    """
    epoch_str = line1[18:32].strip()
    yy = int(epoch_str[0:2])
    day = float(epoch_str[2:])

    year = 2000 + yy if yy < 57 else 1900 + yy  # standard TLE convention
    day_int = int(np.floor(day))
    frac = day - day_int

    dt0 = datetime(year, 1, 1) + timedelta(days=day_int - 1, seconds=frac * 86400.0)
    return dt0


def julian_date(dt: datetime) -> float:
    """
    Compute Julian Date from UTC datetime (Gregorian calendar).
    """
    y = dt.year
    m = dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0

    if m <= 2:
        y -= 1
        m += 12

    A = y // 100
    B = 2 - A + (A // 4)

    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return float(jd)


def gmst_radians(jd_ut1: float) -> float:
    """
    Approx GMST (radians) from Julian date.
    Sufficient for ground-track visualization homework.
    """
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    gmst_sec = gmst_sec % 86400.0
    return 2.0 * np.pi * (gmst_sec / 86400.0)


# -------------------------
# Orbit parsing and propagation (Two-body Kepler)
# -------------------------
def parse_tle_line2(line2: str):
    """
    Parse TLE line 2 fields needed for two-body propagation.
    Returns i, raan, e, argp, M0, n (all radians except n in rev/day).
    """
    parts = line2.split()
    # parts: [2, satnum, i, raan, e, argp, M, n, revno]
    inc_deg = float(parts[2])
    raan_deg = float(parts[3])
    e_str = parts[4]
    ecc = float("0." + e_str)  # TLE omits decimal point
    argp_deg = float(parts[5])
    M_deg = float(parts[6])
    n_rev_day = float(parts[7])

    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    M0 = np.deg2rad(M_deg)

    return inc, raan, ecc, argp, M0, n_rev_day


def mean_motion_to_semimajor_axis(n_rev_day: float) -> float:
    """
    Convert mean motion (rev/day) -> semimajor axis a (km) using two-body relation.
    """
    n_rad_s = n_rev_day * 2.0 * np.pi / 86400.0
    a = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)
    return a


def solve_kepler(M, e, max_iter=30, tol=1e-12):
    """
    Solve M = E - e sin E for eccentric anomaly E using Newton-Raphson.
    Works well for small e typical of LEO.
    """
    # initial guess
    E = M if e < 0.8 else np.pi

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def r1(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, s],
                     [0,-s, c]])

def r3(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, s, 0],
                     [-s, c, 0],
                     [ 0, 0, 1]])


def eci_position_from_elements(t_sec, inc, raan, e, argp, M0, n_rev_day):
    """
    Two-body propagation from TLE mean elements:
    - M(t) = M0 + n t
    - solve Kepler for E
    - get nu, r
    - PQW -> ECI rotation
    """
    n_rad_s = n_rev_day * 2.0 * np.pi / 86400.0
    a = mean_motion_to_semimajor_axis(n_rev_day)

    M = (M0 + n_rad_s * t_sec) % (2.0 * np.pi)
    E = solve_kepler(M, e)

    # radius
    r = a * (1.0 - e * np.cos(E))

    # true anomaly
    sinv = (np.sqrt(1 - e * e) * np.sin(E)) / (1 - e * np.cos(E))
    cosv = (np.cos(E) - e) / (1 - e * np.cos(E))
    nu = np.arctan2(sinv, cosv)

    # PQW position
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])

    # PQW -> ECI
    Q = r3(-raan) @ r1(-inc) @ r3(-argp)
    r_eci = Q @ r_pqw
    return r_eci  # km


def eci_to_ecef(r_eci_km, jd):
    """
    ECI -> ECEF using GMST rotation about z-axis.
    """
    theta = gmst_radians(jd)
    return r3(theta) @ r_eci_km


def ecef_to_geodetic_wgs84(r_ecef_km):
    """
    ECEF (km) -> geodetic lat/lon (deg) on WGS84 using iteration.
    """
    x, y, z = r_ecef_km
    lon = np.arctan2(y, x)
    p = np.sqrt(x*x + y*y)

    lat = np.arctan2(z, p * (1 - WGS84_E2))  # initial guess

    for _ in range(15):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat*sin_lat)
        h = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1 - WGS84_E2 * (N / (N + h))))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    lat_deg = np.degrees(lat)
    lon_deg = (np.degrees(lon) + 540) % 360 - 180  # wrap to [-180, 180]
    return lat_deg, lon_deg


def split_dateline(lon_deg, lat_deg, jump_threshold=180.0):
    lon = np.asarray(lon_deg)
    lat = np.asarray(lat_deg)
    jumps = np.abs(np.diff(lon)) > jump_threshold
    cut_idx = np.where(jumps)[0] + 1
    idx_segments = np.split(np.arange(len(lon)), cut_idx)
    segments = []
    for s in idx_segments:
        if len(s) >= 2:
            segments.append((lon[s], lat[s]))
    return segments


def ground_track_from_tle(tle1, tle2, duration_minutes, step_seconds=60):
    epoch_dt = tle_epoch_to_datetime(tle1)
    print("TLE Epoch (UTC):", epoch_dt)
    inc, raan, e, argp, M0, n_rev_day = parse_tle_line2(tle2)

    times = np.arange(0, duration_minutes * 60 + 1, step_seconds, dtype=float)
    lats = []
    lons = []

    for tsec in times:
        dt = epoch_dt + timedelta(seconds=float(tsec))
        jd = julian_date(dt)

        r_eci = eci_position_from_elements(tsec, inc, raan, e, argp, M0, n_rev_day)
        r_ecef = eci_to_ecef(r_eci, jd)
        lat, lon = ecef_to_geodetic_wgs84(r_ecef)
        lats.append(lat)
        lons.append(lon)

    return epoch_dt, np.array(lons), np.array(lats), n_rev_day


def main():
    # periods from mean motion
    _, _, _, _, _, n_cloud = parse_tle_line2(CLOUDSAT_TLE[1])
    _, _, _, _, _, n_gpm = parse_tle_line2(GPM_TLE[1])
    p_cloud_min = 1440.0 / n_cloud
    p_gpm_min = 1440.0 / n_gpm

    duration_min = int(np.ceil(3.2 * max(p_cloud_min, p_gpm_min)))  # ensure >= 3 orbits
    step_seconds = 60  # 1-minute sampling

    epoch_c, lon_c, lat_c, _ = ground_track_from_tle(*CLOUDSAT_TLE, duration_min, step_seconds)
    epoch_g, lon_g, lat_g, _ = ground_track_from_tle(*GPM_TLE, duration_min, step_seconds)

    segs_c = split_dateline(lon_c, lat_c)
    segs_g = split_dateline(lon_g, lat_g)

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=(13, 6))

    if USE_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        first = True
        for lo, la in segs_c:
            ax.plot(lo, la, linewidth=1.6, transform=ccrs.PlateCarree(),
                    label="CloudSat (two-body Kepler)" if first else None)
            first = False

        first = True
        for lo, la in segs_g:
            ax.plot(lo, la, linewidth=1.6, linestyle="--", transform=ccrs.PlateCarree(),
                    label="GPM (two-body Kepler)" if first else None)
            first = False

        # mark start/end points to indicate direction
        ax.scatter(lon_c[0], lat_c[0], color="tab:red", s=30, marker="o", transform=ccrs.PlateCarree(), label="CloudSat start")
        ax.scatter(lon_c[-1], lat_c[-1], color="tab:red", s=30, marker="x", transform=ccrs.PlateCarree(), label="CloudSat end")
        ax.scatter(lon_g[0], lat_g[0], color="tab:green", s=30, marker="o", transform=ccrs.PlateCarree(), label="GPM start")
        ax.scatter(lon_g[-1], lat_g[-1], color="tab:green", s=30, marker="x", transform=ccrs.PlateCarree(), label="GPM end")
    else:
        ax = plt.gca()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.grid(True, linewidth=0.3, alpha=0.6)

        for lo, la in segs_c:
            ax.plot(lo, la, linewidth=1.6, label="CloudSat (two-body Kepler)")
            break
        for lo, la in segs_g:
            ax.plot(lo, la, linewidth=1.6, linestyle="--", label="GPM (two-body Kepler)")
            break

        # plot all segments without repeated legend labels
        for lo, la in segs_c[1:]:
            ax.plot(lo, la, linewidth=1.6)
        for lo, la in segs_g[1:]:
            ax.plot(lo, la, linewidth=1.6, linestyle="--")

        ax.scatter(lon_c[0], lat_c[0], color="tab:red", s=30, marker="o", label="CloudSat start")
        ax.scatter(lon_c[-1], lat_c[-1], color="tab:red", s=30, marker="x", label="CloudSat end")
        ax.scatter(lon_g[0], lat_g[0], color="tab:green", s=30, marker="o", label="GPM start")
        ax.scatter(lon_g[-1], lat_g[-1], color="tab:green", s=30, marker="x", label="GPM end")

    ax.set_title(
        "Ground Tracks from TLE (Two-body Kepler propagation; no SGP4 library)\n"
        f"Start epoch (UTC) {epoch_c:%Y-%m-%d %H:%M:%S} | Duration ≈ {duration_min} min "
        f"| CloudSat period ≈ {p_cloud_min:.1f} min, GPM period ≈ {p_gpm_min:.1f} min"
    )

    # avoid duplicate legend entries after adding markers
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower left")

    plt.tight_layout()
    outpng = "q1_ground_tracks_kepler.png"
    plt.savefig(outpng, dpi=200)
    plt.show()
    print(f"Saved: {outpng}")
    if not USE_CARTOPY:
        print("Note: cartopy not available -> plotted without coastlines.")

if __name__ == "__main__":
    main()
