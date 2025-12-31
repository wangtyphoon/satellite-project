#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute RI stats for passes where pass_mid_inside_effective_swath_geo is True.
Outputs a CSV with overall and per-file counts.
"""

from __future__ import annotations

import glob
import os
import pandas as pd


# =====================================================
# Config
# =====================================================
GPM_PASSES_GLOB = "gpm_passes_from_ibtracs_*.csv"
OUT_CSV = "gpm_passes_ri_stats.csv"

# Thresholds (24h intensity change)
RI_THRESHOLD = 30.0          # rapid intensification: > +30 kt
RW_THRESHOLD = -30.0         # rapid weakening: < -30 kt
POS_THRESHOLD = 0.0          # intensifying
NEG_THRESHOLD = 0.0          # weakening

SWATH_COL = "pass_mid_inside_effective_swath_geo"
DELTA_COL = "delta_24h"

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


def _count_stats(df: pd.DataFrame) -> dict:
    if SWATH_COL not in df.columns or DELTA_COL not in df.columns:
        return {
            "swath_true": 0,
            "ri_count": 0,
            "pos_count": 0,
            "rw_count": 0,
            "neg_count": 0,
        }
    swath = _normalize_bool(df[SWATH_COL])
    mask_swath = swath == True
    ri_mask = mask_swath & (df[DELTA_COL] > RI_THRESHOLD)
    pos_mask = mask_swath & (df[DELTA_COL] > POS_THRESHOLD)
    rw_mask = mask_swath & (df[DELTA_COL] < RW_THRESHOLD)
    neg_mask = mask_swath & (df[DELTA_COL] <= NEG_THRESHOLD)
    return {
        "swath_true": int(mask_swath.sum()),
        "ri_count": int(ri_mask.sum()),
        "pos_count": int(pos_mask.sum()),
        "rw_count": int(rw_mask.sum()),
        "neg_count": int(neg_mask.sum()),
    }


def main() -> None:
    files = sorted(glob.glob(GPM_PASSES_GLOB))
    if not files:
        raise FileNotFoundError(f"No files match {GPM_PASSES_GLOB}")

    rows = []
    for path in files:
        df = pd.read_csv(path)
        stats = _count_stats(df)
        stats["source"] = os.path.basename(path)
        rows.append(stats)

    # Overall
    all_df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    overall = _count_stats(all_df)
    overall["source"] = "ALL"
    rows.append(overall)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
