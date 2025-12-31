from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "gpm_passes_swath_true.csv"

FEATURE_POOL = [
    "zFactorFinal_max_r100",
    "zFactorFinal_max_r100_245",
    "zFactorFinal_mean_r100",
    "zFactorFinal_mean_r100_245",
    "zFactorFinal_mean_r25_symmetry",
    "zFactorFinal_max_r25_symmetry",
    "zFactorFinal_mean_r75_symmetry",
    "zFactorFinal_max_r75_symmetry",
    "zFactorFinal_mean_r150_symmetry",
    "zFactorFinal_max_r150_symmetry",
    
]

TARGET_COLS = ["delta_24h", "intensity_bst"]

MIN_FEATURES = 3
EPS_VALUES = np.linspace(0.05, 0.5, 10)
MIN_SAMPLES_VALUES = [10, 12, 14, 16, 18, 20]
METRIC = "cosine"
TOP_N = 20


def variance_explained(y: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    if mask.sum() < 2:
        return 0.0
    y = y[mask]
    labels = labels[mask]
    overall_mean = y.mean()
    ss_total = np.sum((y - overall_mean) ** 2)
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for cluster_id in np.unique(labels):
        cluster_vals = y[labels == cluster_id]
        if cluster_vals.size == 0:
            continue
        cluster_mean = cluster_vals.mean()
        ss_between += cluster_vals.size * (cluster_mean - overall_mean) ** 2
    return float(ss_between / ss_total)


def cluster_count_factor(n_clusters: int) -> float:
    if n_clusters < 2:
        return 0.0
    if n_clusters <= 6:
        return 1.0
    return 6.0 / n_clusters


def evaluate_config(features: np.ndarray, targets: np.ndarray, eps: float, min_samples: int) -> dict:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    dbscan = DBSCAN(min_samples=min_samples, eps=eps, metric=METRIC)
    labels = dbscan.fit_predict(features_scaled)
    n_rows = labels.size
    noise_frac = float(np.sum(labels == -1) / n_rows)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {}

    r2_intensity = variance_explained(targets[:, 0], labels)
    r2_delta = variance_explained(targets[:, 1], labels)
    base_score = 0.5 * (r2_intensity + r2_delta)
    score = base_score * (1.0 - noise_frac) * cluster_count_factor(n_clusters)

    return {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "noise_frac": noise_frac,
        "r2_intensity": r2_intensity,
        "r2_delta": r2_delta,
        "score": score,
    }


def run_search(df: pd.DataFrame) -> pd.DataFrame:
    available_features = [col for col in FEATURE_POOL if col in df.columns]
    missing = [col for col in FEATURE_POOL if col not in df.columns]
    if missing:
        print(f"Missing features in CSV (skipped): {missing}")
    if not available_features:
        raise ValueError("No available features found in the CSV.")

    results = []
    max_features = len(available_features)
    for size in range(MIN_FEATURES, max_features + 1):
        for cols in combinations(available_features, size):
            subset = df[list(cols) + TARGET_COLS].dropna()
            if len(subset) < max(40, MIN_SAMPLES_VALUES[0] * 3):
                continue

            features = subset[list(cols)].to_numpy()
            targets = subset[TARGET_COLS].to_numpy()

            for eps in EPS_VALUES:
                for min_samples in MIN_SAMPLES_VALUES:
                    result = evaluate_config(features, targets, float(eps), int(min_samples))
                    if not result:
                        continue
                    result.update(
                        {
                            "features": "|".join(cols),
                            "n_features": len(cols),
                            "n_rows": len(subset),
                        }
                    )
                    results.append(result)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No valid DBSCAN configurations found.")

    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
    return results_df


def summarize_best(df: pd.DataFrame, best_row: pd.Series) -> None:
    cols = best_row["features"].split("|")
    subset = df[cols + TARGET_COLS].dropna()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(subset[cols].to_numpy())

    dbscan = DBSCAN(
        min_samples=int(best_row["min_samples"]),
        eps=float(best_row["eps"]),
        metric=METRIC,
    )
    labels = dbscan.fit_predict(features_scaled)
    subset = subset.assign(cluster=labels)

    print("\nBest configuration")
    print(best_row.to_string())
    print("\nCluster counts (including noise=-1)")
    print(subset["cluster"].value_counts(dropna=False).sort_index())
    print("\nCluster means (intensity_bst, delta_24h)")
    print(subset.groupby("cluster")[TARGET_COLS].mean())


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    results_df = run_search(df)

    out_csv = Path(__file__).resolve().parent / "auto_cluster_results.csv"
    results_df.to_csv(out_csv, index=False)

    print(f"Saved results: {out_csv}")
    print("\nTop results")
    print(results_df.head(TOP_N).to_string(index=False))

    summarize_best(df, results_df.iloc[0])


if __name__ == "__main__":
    main()
