#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download GPM 2HSLH granules based on the DPR granule list in gpm_passes_swath_true.csv.
Only performs download (no overpass search); see gpm_overpass_finder_v2.py for reference.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import earthaccess


CSV_PATH = Path(__file__).resolve().parent / "gpm_passes_swath_true.csv"
GRANULE_COL = "granule_file"
PASS_TIME_COL = "pass_time_utc"

SHORT_NAME = "GPM_2HSLH"
DOWNLOAD_DIR_TEMPLATE = "data_gpm_2hslh_{year}"
TIME_PAD_MIN = 10


_DPR_NAME_RE = re.compile(
    r"""
    2A\.GPM\.DPR\.
    [^.]+
    \.
    (?P<date>\d{8})
    -S(?P<start>\d{6})
    -E(?P<end>\d{6})
    \.(?P<orbit>\d{6})
    \.(?P<ver>V\d{2}[A-Z])
    \.HDF5
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class GranuleKey:
    date: str
    start: str
    end: str
    orbit: str
    ver: str

    @property
    def year(self) -> str:
        return self.date[:4]

    @property
    def hslh_name(self) -> str:
        return (
            f"2A.GPM.DPR.GPM-SLH.{self.date}-S{self.start}-E{self.end}."
            f"{self.orbit}.{self.ver}.HDF5"
        )

    @property
    def fragment(self) -> str:
        return f"{self.date}-S{self.start}-E{self.end}.{self.orbit}"


def _parse_dpr_name(name: str) -> Optional[GranuleKey]:
    m = _DPR_NAME_RE.search(name)
    if not m:
        return None
    return GranuleKey(
        date=m.group("date"),
        start=m.group("start"),
        end=m.group("end"),
        orbit=m.group("orbit"),
        ver=m.group("ver"),
    )


def _granule_filename(granule) -> Optional[str]:
    try:
        return granule.data_links()[0].split("/")[-1]
    except Exception:
        return None


def _pick_best(granules: Iterable, key: GranuleKey):
    for g in granules:
        fname = _granule_filename(g)
        if fname and key.fragment in fname:
            return g
    return next(iter(granules), None)


def _is_valid(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        import h5py  # optional integrity check
    except Exception:
        return True
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def _search_granule(key: GranuleKey, pass_time: Optional[pd.Timestamp]):
    # 1) exact granule name search
    granules = earthaccess.search_data(short_name=SHORT_NAME, granule_name=key.hslh_name)
    if granules:
        return _pick_best(granules, key)

    # 2) fallback: temporal search around pass time
    if pass_time is None or pd.isna(pass_time):
        return None
    t0 = pass_time - pd.Timedelta(minutes=TIME_PAD_MIN)
    t1 = pass_time + pd.Timedelta(minutes=TIME_PAD_MIN)
    granules = earthaccess.search_data(
        short_name=SHORT_NAME,
        temporal=(t0.isoformat(), t1.isoformat()),
    )
    if not granules:
        return None
    return _pick_best(granules, key)


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    if GRANULE_COL not in df.columns:
        raise ValueError(f"Missing column {GRANULE_COL} in {CSV_PATH}")

    df[PASS_TIME_COL] = pd.to_datetime(df.get(PASS_TIME_COL), errors="coerce", utc=True)
    basenames = df[GRANULE_COL].dropna().astype(str).map(lambda v: Path(v).name)
    names = basenames.unique().tolist()
    pass_time_map = (
        pd.DataFrame({"basename": basenames, "pass_time": df.loc[basenames.index, PASS_TIME_COL]})
        .dropna(subset=["basename"])
        .groupby("basename", sort=False)["pass_time"]
        .first()
        .to_dict()
    )

    earthaccess.login()

    downloaded = []
    skipped = []
    missing = []
    bad_names = []

    for name in names:
        key = _parse_dpr_name(name)
        if not key:
            bad_names.append(name)
            continue

        download_dir = Path(DOWNLOAD_DIR_TEMPLATE.format(year=key.year))
        download_dir.mkdir(parents=True, exist_ok=True)
        target_path = download_dir / key.hslh_name

        if _is_valid(str(target_path)):
            skipped.append(target_path.name)
            continue

        pass_time = pass_time_map.get(name)
        granule = _search_granule(key, pass_time)
        if not granule:
            missing.append(key.hslh_name)
            continue

        results = earthaccess.download([granule], local_path=str(download_dir), threads=1, show_progress=True)
        if results:
            downloaded.append(Path(results[0]).name)
        else:
            missing.append(key.hslh_name)

    print("\nDone.")
    print(f"Downloaded: {len(downloaded)}")
    print(f"Already present: {len(skipped)}")
    print(f"Missing in CMR: {len(missing)}")
    print(f"Unparsed DPR names: {len(bad_names)}")
    if missing:
        print("Missing examples:", missing[:10])
    if bad_names:
        print("Unparsed examples:", bad_names[:10])


if __name__ == "__main__":
    main()
