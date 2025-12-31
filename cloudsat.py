import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF, HC
from pyhdf.VS import VS

def read_sds(path: str, name: str) -> np.ndarray:
    f = SD(path, SDC.READ)
    a = f.select(name)[:]
    return np.asarray(a)

def read_vdata_1field(path: str, vname: str) -> np.ndarray:
    hdf = HDF(path, HC.READ)
    vs = hdf.vstart()
    vd = vs.attach(vname)
    # 多數 CloudSat 單欄位 Vdata：欄位名=Vdata 名稱；若失敗看 vd.inquire()[2] 改成那個 fields 名
    try:
        vd.setfields(vname)
    except Exception:
        fields = vd.inquire()[2]  # e.g. "Profile_time"
        vd.setfields(fields)
    nrecs = vd.inquire()[0]
    data = vd.read(nrecs)
    vd.detach()
    vs.end()
    hdf.close()
    return np.asarray([r[0] for r in data], dtype=np.float64)

def build_profile_datetimes(path: str) -> np.ndarray:
    t_prof = read_vdata_1field(path, "Profile_time")  # usually length = nray

    # 1993-01-01 為 CloudSat 常用基準（TAI_start/UTC_start 也採這個 reference）
    base = datetime(1993, 1, 1)

    # 嘗試讀 UTC_start（通常是一整天的 UTC 秒數）
    try:
        utc_start_arr = read_vdata_1field(path, "UTC_start")
        utc_start = float(utc_start_arr[0]) if utc_start_arr.size > 0 else None
    except Exception:
        utc_start = None

    # 嘗試讀 TAI_start（用來還原絕對日期）
    try:
        tai_start_arr = read_vdata_1field(path, "TAI_start")
        tai_start = float(tai_start_arr[0]) if tai_start_arr.size > 0 else None
    except Exception:
        tai_start = None

    # 建立起始 datetime
    if tai_start is not None:
        # 先用 TAI_start 拿到日期，再用 UTC_start 套時間，對齊檔名中的起始時間
        tai_dt = base + timedelta(seconds=tai_start)
        if utc_start is not None:
            start_dt = datetime.combine(tai_dt.date(), datetime.min.time()) + timedelta(seconds=utc_start)
        else:
            start_dt = tai_dt
    elif utc_start is not None:
        # 只剩 UTC_start 就假設也是從 1993-01-01 起算
        start_dt = base + timedelta(seconds=utc_start)
    else:
        # 完全沒有 anchor，只好把 profile_time 視為從 1993-01-01 開始的秒數
        start_dt = base

    # 轉 datetime array
    dt = np.array([start_dt + timedelta(seconds=float(s)) for s in t_prof], dtype="datetime64[ns]")
    return dt

def apply_cpr_cloud_mask(
    Ze_dbz: np.ndarray,
    cpr_mask: np.ndarray,
    keep_min_class: int = 20,
) -> np.ndarray:
    """
    Ze_dbz: (nray, nbin) dBZ (含 NaN)
    cpr_mask: CPR_Cloud_mask，通常 (nray, nbin)
    keep_min_class:
      - 預設 20：只保留「較可信的雲/回波」等級（通常能大幅減少雜點）
      - 若你想保留更薄的雲，可改成 10
    """
    Z = Ze_dbz.copy()

    # CPR_Cloud_mask 常見是 uint8/int16 的分類值；先確保 shape 對齊
    assert cpr_mask.shape == Z.shape, f"mask shape {cpr_mask.shape} != Ze shape {Z.shape}"

    # 只保留 mask >= 閾值；其餘視為雜訊/不可信（設 NaN）
    valid = (cpr_mask >= keep_min_class)
    Z[~valid] = np.nan
    return Z

def apply_lat_lon_axes(ax, x: np.ndarray, lat_vals: np.ndarray, lon_vals: np.ndarray):
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

def plot_reflectivity_section(
    path: str,
    target_time_utc: str,
    window_min: float = 2.0,
    vmin: float = -30.0,
    vmax: float = 30.0,
):
    # --- 讀資料 ---
    Ze_raw = read_sds(path, "Radar_Reflectivity").astype(np.float32)  # (nray, nbin)
    H  = read_sds(path, "Height").astype(np.float32)             # (nray, nbin) or (nbin,) 依產品而定
    cpr_mask = read_sds(path, "CPR_Cloud_mask").astype(np.int16)

    lat = read_vdata_1field(path, "Latitude")
    lon = read_vdata_1field(path, "Longitude")
    t64 = build_profile_datetimes(path)  # datetime64[ns], length nray

    # --- 對齊檢查 ---
    nray = Ze_raw.shape[0]
    assert lat.size == nray and lon.size == nray and t64.size == nray, "lat/lon/time 沒有對齊 nray"

    # --- 缺值處理（至少把 -8192 / -8888 遮罩）---
    Ze = Ze_raw.copy()
    Ze[(Ze == -8192) | (Ze == -8888)] = np.nan
    Ze *= 0.01  # 原始值為 dBZ*100，換算成 dBZ

    # --- 選時間窗 ---
    t0 = np.datetime64(target_time_utc)  # e.g. "2017-07-30T03:34:55"
    dt_ns = np.abs(t64 - t0).astype("timedelta64[ns]").astype(np.int64)
    idx0 = int(np.argmin(dt_ns))  # 最接近的那條 ray

    half = np.timedelta64(int(window_min * 60), "s")
    mask = (t64 >= (t0 - half)) & (t64 <= (t0 + half))
    idx = np.where(mask)[0]

    # 若時間窗內沒資料，就退而求其次：畫最接近的單條 profile
    if idx.size == 0:
        idx = np.array([idx0])

    Ze_sel = Ze[idx, :]  # (nsel, nbin)

    # Height 可能是 (nray, nbin) 或 (nbin,)
    if H.ndim == 2:
        H_sel = H[idx, :]
        y = np.nanmean(H_sel, axis=0)  # 用平均高度當 y 軸（通常每條 ray 高度幾乎一樣）
    else:
        y = H  # (nbin,)

    # x 軸仍用 along-track index，但上下軸顯示緯度/經度
    x = np.arange(idx.size)
    lat_sel = lat[idx]
    lon_sel = lon[idx]


    # --- 畫圖 ---
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.pcolormesh(x, y, Ze_sel.T, shading="auto", vmin=vmin, vmax=vmax,cmap='jet')
    fig.colorbar(im, label="Radar Reflectivity (dBZ)")
    apply_lat_lon_axes(ax, x, lat_sel, lon_sel)
    ax.set_ylim(0, np.nanmax(y))

    # 標註資訊
    t_start = str(t64[idx[0]])[:19]
    t_end   = str(t64[idx[-1]])[:19]
    ax.set_title(f"CloudSat Reflectivity Section\n{t_start} to {t_end} (UTC), center={str(t0)[:19]}")
    ax.set_ylabel("Height")
    fig.tight_layout()
    plt.show()

    # 濾除雜訊版本（依 CPR_Cloud_mask）
    cpr_sel = cpr_mask[idx, :]
    Ze_filt = apply_cpr_cloud_mask(Ze_sel, cpr_sel, keep_min_class=20)

    fig_f, ax_f = plt.subplots(figsize=(12, 4))
    im_f = ax_f.pcolormesh(x, y, Ze_filt.T, shading="auto", vmin=vmin, vmax=vmax,cmap='jet')
    fig_f.colorbar(im_f, label="Radar Reflectivity (dBZ)")
    apply_lat_lon_axes(ax_f, x, lat_sel, lon_sel)
    ax_f.set_ylim(0, np.nanmax(y))
    ax_f.set_title(
        f"CloudSat Reflectivity Section (Filtered)\n{t_start} to {t_end} (UTC), center={str(t0)[:19]}"
    )
    ax_f.set_ylabel("Height")
    fig_f.tight_layout()
    plt.show()

    # 回傳索引與對應 lat/lon/time 方便你後續做交會比對
    return idx, t64[idx], lat[idx], lon[idx]

path = "2019215032540_70655_CS_2B-GEOPROF_GRANULE_P1_R05_E09_F00.hdf"

idx, tt, la, lo = plot_reflectivity_section(
    path,
    target_time_utc="2019-08-03T04:13:46",
    window_min=0.58
)

print("selected rays:", idx.size)
print("time head/tail:", tt[0], tt[-1])
print("lat/lon head:", la[0], lo[0])

