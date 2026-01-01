#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect 2HSLH HDF5 file keys and dataset shapes.
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt


H5_PATH = (
    "data_gpm_2hslh_2015/"
    "2A.GPM.DPR.GPM-SLH.20150209-S014240-E031510.005390.V07A.HDF5"
)

SWATH_DATASETS = [
    "latentHeating",
    "Q2",
]
LEVEL_DATASETS = [
    "nearSurfLevel",
    "meltLevel",
]

NEARSURF_PLOT_OUT = "nearSurfLevel_2d.png"


def main() -> None:
    with h5py.File(H5_PATH, "r") as h5:
        print(f"HDF5 file: {H5_PATH}")
        keys = []
        h5.visit(keys.append)
        print(f"Total keys: {len(keys)}")
        for key in keys:
            obj = h5[key]
            if isinstance(obj, h5py.Dataset):
                print(f"{key} | shape={obj.shape} | dtype={obj.dtype}")
            else:
                print(f"{key} | group")
        if "Swath" in h5:
            print("\nSwath level/height/bin candidates:")
            for name, obj in h5["Swath"].items():
                if not isinstance(obj, h5py.Dataset):
                    continue
                lname = name.lower()
                if "level" in lname or "height" in lname or "bin" in lname:
                    print(f"Swath/{name} | shape={obj.shape} | dtype={obj.dtype}")

        print("\nSwath dataset attributes (latentHeating/Q2 if present):")
        for name in SWATH_DATASETS:
            path = f"Swath/{name}"
            if path not in h5:
                print(f"{path} | missing")
                continue
            ds = h5[path]
            print(f"{path} | attrs={sorted(ds.attrs.keys())}")
            for k in sorted(ds.attrs.keys()):
                print(f"  - {k}: {ds.attrs[k]}")

        if "Swath" in h5:
            missing = [f"Swath/{n}" for n in LEVEL_DATASETS if f"Swath/{n}" not in h5]
            if missing:
                print(f"\nMissing level datasets: {missing}")
            else:
                ns_ds = h5["Swath/nearSurfLevel"]
                ml_ds = h5["Swath/meltLevel"]
                ns = ns_ds[...]
                ml = ml_ds[...]
                ns_fill = ns_ds.attrs.get("_FillValue", ns_ds.attrs.get("missing_value", None))
                ml_fill = ml_ds.attrs.get("_FillValue", ml_ds.attrs.get("missing_value", None))
                valid = np.isfinite(ns) & np.isfinite(ml)
                if ns_fill is not None:
                    valid &= ~np.isclose(ns, float(ns_fill))
                if ml_fill is not None:
                    valid &= ~np.isclose(ml, float(ml_fill))
                flat = np.argwhere(valid)
                if flat.size == 0:
                    print("\nNo valid nearSurfLevel/meltLevel samples found.")
                else:
                    i, j = flat[0]
                    ns_val = float(ns[i, j])
                    ml_val = float(ml[i, j])
                    delta_km = (ns_val - ml_val) * 0.25
                    print("\nSample melt layer estimate:")
                    print(f"  nearSurfLevel[{i},{j}]={ns_val}")
                    print(f"  meltLevel[{i},{j}]={ml_val}")
                    print(f"  (nearSurfLevel - meltLevel) * 0.25 km = {delta_km:.2f} km")

                ns_plot = ns.astype(float, copy=False)
                if ns_fill is not None:
                    ns_plot = np.where(np.isclose(ns_plot, float(ns_fill)), np.nan, ns_plot)
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.imshow(ns_plot, origin="lower", aspect="auto", cmap="viridis")
                ax.set_title("nearSurfLevel (2D)")
                ax.set_xlabel("Ray index")
                ax.set_ylabel("Scan index")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                plt.show()
                print(f"Wrote {NEARSURF_PLOT_OUT}")


if __name__ == "__main__":
    main()


import h5py
import numpy as np

HDF5 = "data_gpm_2hslh_2015/2A.GPM.DPR.GPM-SLH.20150209-S014240-E031510.005390.V07A.HDF5"
BIN_DZ_KM = 0.25

with h5py.File(HDF5, "r") as f:
    near_surf = f["Swath/nearSurfLevel"][:]          # int16 (應該是)
    melt = f["Swath/meltLevel"][:]                   # int16
    pr = f["Swath/nearSurfacePrecipRate"][:]         # float32
    lh = f["Swath/latentHeating"][:]                 # (nscan,nray,nlayer)
    lh_fill = f["Swath/latentHeating"].attrs.get("_FillValue", -9999.9)

nlayer = lh.shape[2]  # 80

# 1) 有降水 + latentHeating 至少有一層不是 fill（避免無檢索像素）
has_precip = np.isfinite(pr) & (pr > 0)

lh_valid_any = np.any(lh != lh_fill, axis=2)  # (nscan,nray)
candidate = has_precip & lh_valid_any

# 2) 合理範圍遮罩（bin index 必須在 [0, nlayer-1]）
near_ok = (near_surf >= 0) & (near_surf < nlayer)
melt_ok = (melt >= 0) & (melt < nlayer)

good = candidate & near_ok & melt_ok

print("good pixel fraction:", good.mean())

# 3) 隨便取一個 good 像素算 melt height（相對地表）
if np.any(good):
    i, j = np.argwhere(good)[0]
    melt_height_km = (melt[i, j] - near_surf[i, j]) * BIN_DZ_KM
    print("sample good pixel:", (i, j))
    print("  pr:", pr[i, j])
    print("  nearSurfLevel:", int(near_surf[i, j]))
    print("  meltLevel:", int(melt[i, j]))
    print("  melt height above surface (km):", float(melt_height_km))
else:
    print("No valid pixels found with current criteria.")
