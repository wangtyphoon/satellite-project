#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count rows within 250 km for the first source file in gpm_passes_swath_true.csv.
"""

from __future__ import annotations

import pandas as pd


IN_CSV = "gpm_passes_swath_true.csv"
DIST_COL = "min_dist_km"
SOURCE_COL = "source"
DIST_THRESHOLD_KM = 250.0


def main() -> None:
    df = pd.read_csv(IN_CSV)
    if SOURCE_COL not in df.columns:
        raise KeyError(f"Missing column {SOURCE_COL} in {IN_CSV}")
    if DIST_COL not in df.columns:
        raise KeyError(f"Missing column {DIST_COL} in {IN_CSV}")

    first_source = sorted(df[SOURCE_COL].dropna().unique())[0]
    df_first = df.loc[df[SOURCE_COL] == first_source]
    count = int((df_first[DIST_COL] <= DIST_THRESHOLD_KM).sum())

    print(f"First source: {first_source}")
    print(f"Rows within {DIST_THRESHOLD_KM:.0f} km: {count}")


if __name__ == "__main__":
    main()
