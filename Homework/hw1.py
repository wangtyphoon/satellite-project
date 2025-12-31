import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

from skyfield.api import EarthSatellite, load, wgs84

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# -------------------------
# TLEs (Jan 1, 2015)
# -------------------------
CLOUDSAT_TLE = (
    "1 29107U 06016A   15001.13125000  .00000000  00000-0  00000-0 0  9991",
    "2 29107  98.2170 330.8200 0000824  91.6200  14.5700 14.57000000 46515",
)

GPM_TLE = (
    "1 39574U 14009A   15001.13125000  .00000000  00000-0  00000-0 0  9992",
    "2 39574  65.0000  45.0000 0010000   0.0000   0.0000 15.50000000 1000",
)


def period_minutes_from_sat(sat: EarthSatellite) -> float:
    """
    Skyfield/SGP4 internal mean motion no_kozai is in rad/min.
    Period (min) = 2π / n(rad/min)
    """
    n_rad_per_min = sat.model.no_kozai
    return 2.0 * np.pi / n_rad_per_min


def split_dateline(lon_deg, lat_deg, jump_threshold=180.0):
    """
    將跨越日界線(±180)造成的線段跳躍拆開，避免地圖上畫出一條穿越整個地球的直線。
    """
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


def propagate_ground_track(sat: EarthSatellite, ts, duration_minutes: int, step_minutes: int = 1):
    """
    從 TLE 的 epoch 開始，推算 duration_minutes 內每 step_minutes 的子星點(lat/lon)。
    """
    minutes = np.arange(0, duration_minutes + 1, step_minutes)
    t0 = ts.utc(sat.epoch.utc_datetime())
    t = t0 + minutes / 1440.0  # Time + (days)

    geocentric = sat.at(t)
    sub = wgs84.subpoint(geocentric)

    lat = sub.latitude.degrees
    lon = sub.longitude.degrees
    return lon, lat


def propagate_ground_track_span(sat: EarthSatellite, ts, start_dt, end_dt, step_minutes: int = 1):
    """
    Propagate from start_dt to end_dt with fixed step (minutes).
    """
    times = []
    cur = start_dt
    step = timedelta(minutes=step_minutes)
    while cur <= end_dt:
        times.append(cur)
        cur = cur + step
    t = ts.utc(times)
    sub = wgs84.subpoint(sat.at(t))
    lat = sub.latitude.degrees
    lon = sub.longitude.degrees
    return lon, lat


def great_circle_distance_km(lat1, lon1, lat2, lon2, radius_km=6378.137):
    """
    Great-circle distance between scalar lat/lon (deg); lat2/lon2 can be arrays.
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return radius_km * c


def find_intersections(lat_c, lon_c, lat_g, lon_g, step_minutes,
                       window_minutes=20.0, max_distance_km=200.0):
    """
    Find windows where CloudSat and GPM are within ±window_minutes and <= max_distance_km.
    Returns list of (start_idx, end_idx, nearest_idx_array, min_dist_array).
    """
    n = len(lat_c)
    window_steps = int(np.ceil(window_minutes / step_minutes))
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


def main():
    ts = load.timescale()

    cloudsat = EarthSatellite(*CLOUDSAT_TLE, name="CloudSat", ts=ts)
    gpm = EarthSatellite(*GPM_TLE, name="GPM", ts=ts)

    # 至少 3 圈：用 period 推估，抓稍微多一點（3.2 圈）更保險
    p_cloud = period_minutes_from_sat(cloudsat)
    p_gpm = period_minutes_from_sat(gpm)
    duration = int(np.ceil(3.2 * max(p_cloud, p_gpm)))  # minutes
    step = 1  # 1 minute resolution

    # 推算 ground track
    lon_c, lat_c = propagate_ground_track(cloudsat, ts, duration, step)
    lon_g, lat_g = propagate_ground_track(gpm, ts, duration, step)
    # 拆日界線跳點
    segs_c = split_dateline(lon_c, lat_c)
    segs_g = split_dateline(lon_g, lat_g)

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=(13, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # 畫 CloudSat（只在第一段加 label，避免 legend 重複）
    first = True
    for lon_s, lat_s in segs_c:
        ax.plot(
            lon_s, lat_s,
            transform=ccrs.PlateCarree(),
            linewidth=1.6,
            label="CloudSat" if first else None,
        )
        first = False

    # 畫 GPM
    first = True
    for lon_s, lat_s in segs_g:
        ax.plot(
            lon_s, lat_s,
            transform=ccrs.PlateCarree(),
            linewidth=1.6,
            linestyle="--",
            label="GPM" if first else None,
        )
        first = False

    # 標題：含 epoch 與軌道圈數資訊
    epoch_dt = cloudsat.epoch.utc_datetime()
    ax.set_title(
        f"Ground Tracks from TLE Epoch (UTC) {epoch_dt:%Y-%m-%d %H:%M:%S}\n"
        f"Duration ≈ {duration} min  |  CloudSat period ≈ {p_cloud:.1f} min,  GPM period ≈ {p_gpm:.1f} min",
        fontsize=12
    )

    ax.legend(loc="lower left")
    plt.tight_layout()

    # mark start/end points to show track direction
    ax.scatter(lon_c[0], lat_c[0], color="tab:red", s=30, marker="o", transform=ccrs.PlateCarree(), label="CloudSat start")
    ax.scatter(lon_c[-1], lat_c[-1], color="tab:red", s=30, marker="x", transform=ccrs.PlateCarree(), label="CloudSat end")
    ax.scatter(lon_g[0], lat_g[0], color="tab:green", s=30, marker="o", transform=ccrs.PlateCarree(), label="GPM start")
    ax.scatter(lon_g[-1], lat_g[-1], color="tab:green", s=30, marker="x", transform=ccrs.PlateCarree(), label="GPM end")

    # avoid duplicate legend entries after adding markers
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower left")

    outpng = "q1_ground_tracks.png"
    plt.savefig(outpng, dpi=200)
    plt.show()
    print(f"Saved: {outpng}")

    # -------------------------
    # Plot 2: month-long intersections (Jan 2015)
    # -------------------------
    window_minutes = 20.0
    max_distance_km = 122.5
    step_minutes = 1/60
    start_dt = datetime(2015, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2015, 2, 1, 0, 0, 0, tzinfo=timezone.utc)

    print("Propagating CloudSat for January 2015 (Skyfield SGP4)...")
    lon_c_m, lat_c_m = propagate_ground_track_span(cloudsat, ts, start_dt, end_dt, step_minutes)
    print("Propagating GPM for January 2015 (Skyfield SGP4)...")
    lon_g_m, lat_g_m = propagate_ground_track_span(gpm, ts, start_dt, end_dt, step_minutes)

    n_m = min(len(lat_c_m), len(lat_g_m))
    lon_c_m, lat_c_m = lon_c_m[:n_m], lat_c_m[:n_m]
    lon_g_m, lat_g_m = lon_g_m[:n_m], lat_g_m[:n_m]

    intersections, _, _ = find_intersections(
        lat_c_m, lon_c_m, lat_g_m, lon_g_m, step_minutes,
        window_minutes=window_minutes, max_distance_km=max_distance_km
    )
    print(f"Found {len(intersections)} intersection windows (<= {max_distance_km} km within ±{window_minutes} min).")
    lens = np.array([e - s + 1 for (s, e, _, _) in intersections])
    print("Intersection window lengths (min/max/median):", lens.min(), lens.max(), np.median(lens))
    print("Count of length==1:", np.sum(lens == 1), "/", len(lens))

    segs_c_m = split_dateline(lon_c_m, lat_c_m)
    segs_g_m = split_dateline(lon_g_m, lat_g_m)

    fig2 = plt.figure(figsize=(13, 6))
    ax2 = plt.axes(projection=ccrs.PlateCarree())
    ax2.set_global()

    ax2.coastlines(linewidth=0.8)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
    gl2 = ax2.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
    gl2.top_labels = False
    gl2.right_labels = False

    # first = True
    # for lon_s, lat_s in segs_c_m:
    #     ax2.plot(
    #         lon_s, lat_s,
    #         transform=ccrs.PlateCarree(),
    #         linewidth=0.8,
    #         alpha=0.5,
    #         label="CloudSat (SGP4)" if first else None,
    #     )
    #     first = False

    # first = True
    # for lon_s, lat_s in segs_g_m:
    #     ax2.plot(
    #         lon_s, lat_s,
    #         transform=ccrs.PlateCarree(),
    #         linewidth=0.8,
    #         alpha=0.5,
    #         linestyle="--",
    #         label="GPM (SGP4)" if first else None,
    #     )
    #     first = False

    # highlight intersections
    first_c = True
    first_g = True
    for s, e, match_idx, _ in intersections:
        c_lo = lon_c_m[s:e + 1]
        c_la = lat_c_m[s:e + 1]
        g_start = int(np.min(match_idx))
        g_end = int(np.max(match_idx))
        g_lo = lon_g_m[g_start:g_end + 1]
        g_la = lat_g_m[g_start:g_end + 1]

        for lo, la in split_dateline(c_lo, c_la):
            ax2.plot(
                lo, la,
                transform=ccrs.PlateCarree(),
                color="crimson",
                linewidth=2.0,
                label="Intersection path (CloudSat)" if first_c else None,
            )
            first_c = False
        for lo, la in split_dateline(g_lo, g_la):
            ax2.plot(
                lo, la,
                transform=ccrs.PlateCarree(),
                color="limegreen",
                linewidth=2.0,
                label="Intersection path (GPM)" if first_g else None,
            )
            first_g = False

    ax2.set_title(
        "CloudSat / GPM Near-Simultaneous Overpasses (Jan 2015)\n"
        f"SGP4 (Skyfield) | <= {max_distance_km} km and ±{window_minutes:.0f} min",
        fontsize=12
    )

    handles2, labels2 = ax2.get_legend_handles_labels()
    uniq2 = dict(zip(labels2, handles2))
    ax2.legend(uniq2.values(), uniq2.keys(), loc="lower left")
    plt.tight_layout()

    outpng2 = "jan2015_intersections_skyfield.png"
    plt.savefig(outpng2, dpi=200)
    plt.show()
    print(f"Saved: {outpng2}")


if __name__ == "__main__":
    main()
