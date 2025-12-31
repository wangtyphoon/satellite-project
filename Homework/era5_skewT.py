import os
import cdsapi
import xarray as xr
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
import numpy as np

# 設定參數
CENTER_LAT = -4.5
CENTER_LON = 145.0
DATE = "2019-08-03"
TIMES = ["00:00", "06:00"]  # 你要畫的兩個時刻

OUT_NC = "era5_sounding_data.nc"


def download_era5_data():
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "temperature",
                "relative_humidity",
                "specific_humidity",          # <<< 新增：q
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": [
                "100", "150", "200", "250", "300", "400", "500",
                "600", "700", "800", "850", "900", "925", "950", "1000"
            ],
            "year": "2019",
            "month": "08",
            "day": "03",
            "time": TIMES,
            # CDS 的 area 順序是 [N, W, S, E]；同一點就都填一樣即可
            "area": [CENTER_LAT, CENTER_LON, CENTER_LAT, CENTER_LON],
        },
        OUT_NC,
    )


def _make_skewt_plot(p, t, td, u, v, title, filename, dew_label):
    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=45)

    skew.plot(p, t, "r", linewidth=2, label="Temperature")
    skew.plot(p, td, "g", linewidth=2, label=dew_label)
    skew.plot_barbs(p, u, v)

    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-40, 40)

    skew.plot_dry_adiabats(alpha=0.25)
    skew.plot_moist_adiabats(alpha=0.25)
    skew.plot_mixing_lines(alpha=0.25)

    plt.title(title, fontsize=14)
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_sounding_two_versions(ds, time_index):
    # 取出該時刻 + 最近點，並 squeeze 掉長度為 1 的 lat/lon
    data = (
        ds.isel(valid_time=time_index)
        .sel(latitude=CENTER_LAT, longitude=CENTER_LON, method="nearest")
        .squeeze()
    )

    # 壓力 (hPa)
    p = data.pressure_level.values * units.hPa

    # 溫度 (K -> °C)
    t = (data.t.values - 273.15) * units.degC

    # RH (0~1)
    rh = (data.r.values / 100.0) * units.dimensionless

    # q (kg/kg) -> dimensionless（MetPy 可接受）
    q = data.q.values * units("kg/kg")

    # 風 (m/s)
    u = data.u.values * units("m/s")
    v = data.v.values * units("m/s")
    print(rh)
    # 露點：版本 1（由 RH 算）
    td_rh = mpcalc.dewpoint_from_relative_humidity(t, rh)

    # 露點：版本 2（由 q 算）
    # 注意：這裡需要 pressure + temperature + specific_humidity
    td_q = mpcalc.dewpoint_from_specific_humidity(p, t, q)

    # 時間字串
    time_val = ds.valid_time.values[time_index]
    t_str = np.datetime_as_string(time_val, unit="h")  # e.g. 2019-08-03T00
    title_base = f"ERA5 Sounding at ({CENTER_LAT}, {CENTER_LON})\n{t_str} UTC"

    # 兩張圖分別存檔
    fname_rh = f"sounding_{t_str.replace(':','').replace('T','_')}_dewRH.png"
    fname_q  = f"sounding_{t_str.replace(':','').replace('T','_')}_dewQ.png"

    _make_skewt_plot(
        p, t, td_rh, u, v,
        title=f"{title_base}\nDewpoint from RH",
        filename=fname_rh,
        dew_label="Dewpoint (from RH)"
    )
    _make_skewt_plot(
        p, t, td_q, u, v,
        title=f"{title_base}\nDewpoint from q",
        filename=fname_q,
        dew_label="Dewpoint (from q)"
    )

    # 你若想快速比較數值差異，也可以印出幾層
    for lev in [1000, 850, 700, 500, 300, 200, 100]:
        tt = (data.t.sel(pressure_level=lev).values - 273.15)
        tdrh_lev = td_rh[np.where(data.pressure_level.values == lev)[0][0]].m
        tdq_lev  = td_q[np.where(data.pressure_level.values == lev)[0][0]].m
        print(f"{lev:4d} hPa: T={tt:6.1f} °C | Td(RH)={tdrh_lev:6.1f} °C | Td(q)={tdq_lev:6.1f} °C")


def main():
    if not os.path.exists(OUT_NC):
        print("Downloading data from CDS...")
        download_era5_data()

    ds = xr.open_dataset(OUT_NC)
    print("Dataset dimensions:", ds.dims)
    print("Dataset variables:", list(ds.data_vars))

    # 確認有 q
    if "q" not in ds.data_vars:
        raise RuntimeError("Dataset missing 'q' (specific_humidity). Please re-download with 'specific_humidity'.")

    for i in range(len(ds.valid_time)):
        plot_sounding_two_versions(ds, i)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
