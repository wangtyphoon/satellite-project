from pathlib import Path

from sklearn.cluster import DBSCAN,HDBSCAN 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('gpm_passes_swath_true.csv')
#df = df[(df['intensity_bst'] > 45) & (df['delta_24h'] > 0)].copy()
summary_path = Path("plots_radar_feature_target") / "summary_r2_intensity_bst_all.csv"
summary_df = pd.read_csv(summary_path)
r2_threshold = 0.25
corr_threshold = 0.9

summary_df = summary_df.loc[summary_df["r2"] >= r2_threshold, ["feature", "r2"]].copy()
summary_df = summary_df[summary_df["feature"].isin(df.columns)]
summary_df.sort_values("r2", ascending=False, inplace=True)

candidate_cols = summary_df["feature"].tolist()
candidate_df = df.loc[:, candidate_cols].apply(pd.to_numeric, errors="coerce")
corr = candidate_df.corr(method="pearson")

feature_cols = []
for feature in candidate_cols:
    if not feature_cols:
        feature_cols.append(feature)
        continue
    if corr.loc[feature, feature_cols].abs().max() > corr_threshold:
        continue
    feature_cols.append(feature)
print(f"Selected {len(feature_cols)} features after correlation filtering.")
print("Clustering features:", feature_cols)
if not feature_cols:
    raise ValueError("No features found with r2 threshold that exist in the dataset.")

features = df[feature_cols].copy()
features = features.dropna()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.values)

dbscan = DBSCAN(min_samples=5, eps=0.05, metric="cosine")
labels_dbscan = dbscan.fit_predict(features_scaled)

hdb = HDBSCAN(min_cluster_size=10, min_samples=5, metric="cosine")
labels_hdbscan = hdb.fit_predict(features_scaled)

df.loc[features.index, "cluster_dbscan"] = labels_dbscan
df.loc[features.index, "cluster_hdbscan"] = labels_hdbscan

output_path = Path("gpm_passes_swath_true_hdbscan_bst.csv")
df.to_csv(output_path, index=False)
print(f"HDBSCAN results saved to {output_path}")

print("DBSCAN cluster counts:")
print(df["cluster_dbscan"].value_counts(dropna=False))
print("HDBSCAN cluster counts:")
print(df["cluster_hdbscan"].value_counts(dropna=False))

# Output mean delta_24h and intensity_bst by cluster
cluster_means_dbscan = (
    df.loc[features.index, ["cluster_dbscan", "delta_24h", "intensity_bst"]]
    .groupby("cluster_dbscan", dropna=False)
    .mean()
)
print("DBSCAN cluster means:")
print(cluster_means_dbscan)

cluster_means_hdbscan = (
    df.loc[features.index, ["cluster_hdbscan", "delta_24h", "intensity_bst"]]
    .groupby("cluster_hdbscan", dropna=False)
    .mean()
)
print("HDBSCAN cluster means:")
print(cluster_means_hdbscan)

print("DBSCAN vs HDBSCAN cross-tab:")
print(pd.crosstab(df.loc[features.index, "cluster_dbscan"],
                  df.loc[features.index, "cluster_hdbscan"],
                  dropna=False))

CLUSTER_METHOD = "hdbscan"  # "dbscan"
cluster_col = f"cluster_{CLUSTER_METHOD}"
df.loc[features.index, "cluster"] = df.loc[features.index, cluster_col]
cluster_means = cluster_means_dbscan if CLUSTER_METHOD == "dbscan" else cluster_means_hdbscan

NPY_DIR = Path(__file__).parent / "zFactorFinal"
NPY_PREFIX = "1_max_"
INCLUDE_NOISE = True

cluster_rows = df.loc[features.index]
cluster_ids = sorted(cluster_rows["cluster"].dropna().unique())
if not INCLUDE_NOISE:
    cluster_ids = [cid for cid in cluster_ids if not np.isclose(cid, -1)]

for cluster_id in cluster_ids:
    cluster_subset = cluster_rows[np.isclose(cluster_rows["cluster"], cluster_id)]
    granule_stems = (
        cluster_subset["granule_file"]
        .dropna()
        .astype(str)
        .map(lambda value: Path(value).stem)
        .unique()
    )

    npy_paths = []
    missing = []
    for stem in granule_stems:
        npy_path = NPY_DIR / f"{NPY_PREFIX}{stem}.npy"
        if npy_path.exists():
            npy_paths.append(npy_path)
        else:
            missing.append(npy_path.name)

    if not npy_paths:
        print(f"No npy files found for cluster {cluster_id:g}.")
        continue

    sum_data = None
    count_data = None
    for npy_path in npy_paths:
        data = np.load(npy_path)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.shape} in {npy_path}")
        if sum_data is None:
            sum_data = np.zeros_like(data, dtype=float)
            count_data = np.zeros_like(data, dtype=float)
        valid = np.isfinite(data)
        sum_data[valid] += data[valid]
        count_data[valid] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        composite = sum_data / count_data
    composite[count_data == 0] = np.nan

    composite_out = NPY_DIR / f"composite_cluster_{cluster_id:g}_max.npy"
    composite_png = NPY_DIR / f"composite_cluster_{cluster_id:g}_max.png"
    np.save(composite_out, composite)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.imshow(composite, origin="lower", cmap="jet", vmax=35, vmin=0)
    ax.set_title(
        f"Composite (cluster {cluster_id:g}, n={len(npy_paths)}),\n"
        f"delta_24h mean={cluster_means.loc[cluster_id, 'delta_24h']:.2f}, "
        f"intensity_bst mean={cluster_means.loc[cluster_id, 'intensity_bst']:.2f}"
    )
    fig.colorbar(im, ax=ax, label="zFactorFinal")
    fig.tight_layout()
    fig.savefig(composite_png, bbox_inches="tight")

    if missing:
        print(f"Cluster {cluster_id:g} missing {len(missing)} npy files (first 10): {missing[:10]}")
# from scipy.ndimage import gaussian_filter
# # --- after you have `composite` (2D with NaN) ---

# SIGMA = 1.5            # 以 grid cell 為單位；例如 grid=5 km，sigma=1.5 ~ 7.5 km 的平滑尺度
# TRUNCATE = 3.0         # kernel 半徑 ~ truncate*sigma
# MIN_WEIGHT = 1e-3      # 權重閾值，避免除以極小值造成噪聲

# valid = np.isfinite(composite)
# data0 = np.where(valid, composite, 0.0).astype(float)
# w0 = valid.astype(float)

# data_f = gaussian_filter(data0, sigma=SIGMA, mode="constant", cval=0.0, truncate=TRUNCATE)
# w_f = gaussian_filter(w0,   sigma=SIGMA, mode="constant", cval=0.0, truncate=TRUNCATE)

# with np.errstate(invalid="ignore", divide="ignore"):
#     composite_smooth = data_f / np.maximum(w_f, MIN_WEIGHT)

# # 權重太小的地方視為無資料，維持 NaN（避免 NaN 外圈被「補出值」）
# composite_smooth[w_f < MIN_WEIGHT] = np.nan

# # --- save / plot ---
# SMOOTH_OUT = NPY_DIR / f"composite_cluster_{TARGET_CLUSTER:g}_max_smooth_sigma{SIGMA:g}.npy"
# SMOOTH_PNG = NPY_DIR / f"composite_cluster_{TARGET_CLUSTER:g}_max_smooth_sigma{SIGMA:g}.png"
# np.save(SMOOTH_OUT, composite_smooth)

# fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
# im = ax.imshow(composite_smooth, origin="lower", cmap="jet")
# ax.set_title(f"Composite smooth (cluster {TARGET_CLUSTER:g}, n={len(npy_paths)}, sigma={SIGMA:g})")
# fig.colorbar(im, ax=ax, label="zFactorFinal")
# fig.tight_layout()
fig.show()
