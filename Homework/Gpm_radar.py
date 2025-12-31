
import h5py
import numpy as np
import matplotlib.pyplot as plt


path = "2A.GPM.DPR.V9-20211125.20190803-S033403-E050636.030841.V07A.HDF5"


def print_hdf5_tree(h5obj, prefix="", max_depth=3, _depth=0):
    if _depth > max_depth:
        return
    for key in h5obj.keys():
        obj = h5obj[key]
        if isinstance(obj, h5py.Dataset):
            shape = "x".join(str(s) for s in obj.shape)
            print(f"{prefix}{key} [Dataset] shape={shape} dtype={obj.dtype}")
        else:
            print(f"{prefix}{key} [Group]")
            print_hdf5_tree(obj, prefix=prefix + "  ", max_depth=max_depth, _depth=_depth + 1)


def read_dataset(path, dataset_path):
    with h5py.File(path, "r") as f:
        ds = f[dataset_path]
        data = ds[...]
        attrs = {k: ds.attrs[k] for k in ds.attrs.keys()}
    return data, attrs


def build_scan_datetimes(path):
    with h5py.File(path, "r") as f:
        st = f["FS/ScanTime"]
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


def plot_gpm_reflectivity_section(
    path,
    target_time_utc,
    window_min=2.0,
    beam_index=24,
    channel=0,
    vmin=-10.0,
    vmax=40.0,
):
    with h5py.File(path, "r") as f:
        z_ds = f["FS/SLV/zFactorFinal"]
        z = z_ds[..., channel].astype(np.float32)  # (nscan, nbeam, nbin)
        height = f["FS/PRE/height"][...].astype(np.float32)
        lat = f["FS/Latitude"][...].astype(np.float32)
        lon = f["FS/Longitude"][...].astype(np.float32)
        fill = z_ds.attrs.get("_FillValue", -9999.9)

    z[z == fill] = np.nan

    t64 = build_scan_datetimes(path)  # length = nscan
    t0 = np.datetime64(target_time_utc)
    dt_ns = np.abs(t64 - t0).astype("timedelta64[ns]").astype(np.int64)
    idx0 = int(np.argmin(dt_ns))

    half = np.timedelta64(int(window_min * 60), "s")
    mask = (t64 >= (t0 - half)) & (t64 <= (t0 + half))
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.array([idx0])

    z_sel = z[idx, beam_index, :]  # (nsel, nbin)
    h_sel = height[idx, beam_index, :]
    y = np.nanmean(h_sel, axis=0)

    x = np.arange(idx.size)
    lat_sel = lat[idx, beam_index]
    lon_sel = lon[idx, beam_index]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.pcolormesh(x, y, z_sel.T, shading="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, label="zFactorFinal (dBZ)")
    apply_lat_lon_axes(ax, x, lat_sel, lon_sel)
    ax.set_ylim(0, np.nanmax(y))

    t_start = str(t64[idx[0]])[:19]
    t_end = str(t64[idx[-1]])[:19]
    ax.set_title(
        f"GPM DPR Reflectivity Section (beam={beam_index}, ch={channel})\n"
        f"{t_start} to {t_end} (UTC), center={str(t0)[:19]}"
    )
    ax.set_ylabel("Height (m)")
    fig.tight_layout()
    plt.show()

    return idx, t64[idx], lat_sel, lon_sel


if __name__ == "__main__":
    with h5py.File(path, "r") as f:
        print("Top-level groups:", list(f.keys()))
        print_hdf5_tree(f, max_depth=2)

    lat, lat_attrs = read_dataset(path, "FS/Latitude")
    lon, lon_attrs = read_dataset(path, "FS/Longitude")

    print("\nLatitude sample:", lat[:3, :3])
    print("Longitude sample:", lon[:3, :3])
    print("Latitude attrs keys:", list(lat_attrs.keys())[:5])
    plot_gpm_reflectivity_section(
            path,
            target_time_utc="2019-08-03T03:55:55",
            window_min=2.0,
            beam_index=24,
            channel=0,
        )
    
    # for i in range(48):
    #     plot_gpm_reflectivity_section(
    #         path,
    #         target_time_utc="2019-08-03T03:55:55",
    #         window_min=20.0,
    #         beam_index=i,
    #         channel=0,
    #     )
