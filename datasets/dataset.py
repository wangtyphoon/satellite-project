#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge rows from GPM passes where pass_mid_inside_effective_swath_geo is True.
Outputs a single CSV across multiple years.
"""

from __future__ import annotations

import glob
import os

import pandas as pd


# =====================================================
# Config
# =====================================================
GPM_PASSES_GLOB = "../gpm_passes_from_ibtracs_*.csv"
OUT_CSV = "gpm_passes_swath_true.csv"

SWATH_COL = "pass_mid_inside_effective_swath_geo"


def _normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower().map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "1.0": True,
                "0.0": False,
                "yes": True,
                "no": False,
            }
        )
    return series.astype(float) == 1.0


def _load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if SWATH_COL not in df.columns:
        raise KeyError(f"Missing column {SWATH_COL} in {path}")
    swath = _normalize_bool(df[SWATH_COL])
    return df.loc[swath == True].copy()


def main() -> None:
    files = sorted(glob.glob(GPM_PASSES_GLOB))
    if not files:
        raise FileNotFoundError(f"No files match {GPM_PASSES_GLOB}")

    filtered = []
    for path in files:
        df = _load_and_filter(path)
        df["source"] = os.path.basename(path)
        filtered.append(df)

    out_df = pd.concat(filtered, ignore_index=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
