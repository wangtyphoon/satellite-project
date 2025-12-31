# -*- coding: utf-8 -*-
"""
Download IBTrACS v04r01 Western Pacific (WP) list CSV
and extract SEASON = 2025 to a new CSV.

Author: ChatGPT
"""

import os
import requests
import pandas as pd

# =====================================================
# 使用者設定區（只需要改這裡）
# =====================================================

USE_LAST_3_YEARS = False        # True：只下載近三年（檔案較小）
YEARS = [i for i in range(2015,2022)]                  # 可放多個年份，例如 [2022, 2023, 2024, 2025]
RAW_CSV = "ibtracs_raw.csv"
CHUNKSIZE = 500_000             # None 或整數；資料很大時建議設 >0
PARSE_ISO_TIME = True           # 是否將 ISO_TIME 轉成 datetime
MIN_WMO_WIND = 35            # 只保留 WMO_WIND >= 35；設為 None 可關閉

# =====================================================
# 固定參數（通常不需更動）
# =====================================================

BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/"
)

URL_WP = BASE_URL + "ibtracs.WP.list.v04r01.csv"
URL_LAST3Y = BASE_URL + "ibtracs.last3years.list.v04r01.csv"


# =====================================================
# 函式定義
# =====================================================

def download_csv(url: str, out_path: str):
    """Download CSV using streaming."""
    print(f"Downloading: {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Saved raw file: {out_path}")


def filter_season(in_csv: str, out_csv: str, season_year: int):
    """Filter SEASON == season_year and write to output CSV."""

    def process(df):
        df = df[df["SEASON"].astype(str) == str(season_year)].copy()
        if MIN_WMO_WIND is not None and "WMO_WIND" in df.columns:
            wmo = pd.to_numeric(df["USA_WIND"], errors="coerce")
            df = df[wmo >= MIN_WMO_WIND]
        if PARSE_ISO_TIME and "ISO_TIME" in df.columns:
            df["ISO_TIME"] = pd.to_datetime(
                df["ISO_TIME"], errors="coerce", utc=True
            )
        return df

    total = 0
    first = True

    if CHUNKSIZE and CHUNKSIZE > 0:
        print(f"Reading CSV in chunks (chunksize={CHUNKSIZE})")
        for chunk in pd.read_csv(in_csv, chunksize=CHUNKSIZE, low_memory=False):
            out = process(chunk)
            if len(out) == 0:
                continue

            out.to_csv(
                out_csv,
                mode="w" if first else "a",
                header=first,
                index=False,
                encoding="utf-8-sig",
            )
            first = False
            total += len(out)
    else:
        print("Reading CSV in full (no chunking)")
        df = pd.read_csv(in_csv, low_memory=False)
        out = process(df)
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        total = len(out)

    print(f"Output rows: {total}")
    print(f"Saved filtered CSV: {out_csv}")


# =====================================================
# 主程式
# =====================================================

def main():
    url = URL_LAST3Y if USE_LAST_3_YEARS else URL_WP

    if not os.path.exists(RAW_CSV):
        download_csv(url, RAW_CSV)
    else:
        print(f"Raw CSV already exists, skip download: {RAW_CSV}")

    for season_year in YEARS:
        out_csv = f"ibtracs_WP_{season_year}.csv"
        filter_season(RAW_CSV, out_csv, season_year)


if __name__ == "__main__":
    main()
