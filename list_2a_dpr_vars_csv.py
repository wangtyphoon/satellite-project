#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export variable names from a GPM 2A DPR HDF5 file to CSV."""

import os

import h5py
import pandas as pd

# =====================================================
# User config
# =====================================================
H5_PATH = r"data_gpm_2adpr_2015\2A.GPM.DPR.V9-20211125.20150116-S205515-E222747.005029.V07A.HDF5"  # set to the 2A DPR .HDF5 file path
PREFIX = None  # optional: only list paths under this prefix (e.g., "FS")
OUT_CSV = "vars_2a_dpr.csv"  # output CSV path
    

def build_table(h5, prefix=None):
    rows = []

    def visitor(name, obj):
        if prefix and not name.startswith(prefix.rstrip("/") + "/") and name != prefix.rstrip("/"):
            return
        rows.append({"path": name})

    h5.visititems(visitor)
    rows.sort(key=lambda r: r["path"])
    return rows


def main():
    if not H5_PATH:
        raise SystemExit("Set H5_PATH to the 2A DPR HDF5 file path.")
    if not os.path.exists(H5_PATH):
        raise SystemExit(f"File not found: {H5_PATH}")

    with h5py.File(H5_PATH, "r") as h5:
        rows = build_table(h5, PREFIX)

    if len(rows) == 0:
        raise SystemExit("No variables found for the requested prefix.")

    df = pd.DataFrame(rows)
    parts = df["path"].str.strip("/").str.split("/")
    max_depth = int(parts.map(len).max())
    for i in range(max_depth):
        df[f"level_{i+1}"] = parts.str.get(i)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
