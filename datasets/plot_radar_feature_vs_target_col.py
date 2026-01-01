#!/usr/bin/env python3
"""Correlation matrix and heatmap for radar features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    # Config
    csv_path = Path("gpm_passes_swath_true.csv")
    target = "intensity_bst"
    outdir = Path("plots_radar_feature_target")
    r2_threshold = 0.1

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df1 = df.loc[:, 'zFactorFinal_max_r100':]
    feature_cols = [c for c in df1.columns]
    df = df[(df['intensity_bst'] > 55) & (df['delta_24h'] > 0)].copy()

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    outdir.mkdir(parents=True, exist_ok=True)

    target_series = pd.to_numeric(df[target], errors="coerce")
    selected_features = []
    for feature in feature_cols:
        x = pd.to_numeric(df[feature], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(target_series)
        if mask.sum() < 2:
            continue
        r = np.corrcoef(x[mask], target_series[mask])[0, 1]
        if np.isfinite(r) and (r ** 2) >= r2_threshold:
            selected_features.append(feature)

    corr_cols = selected_features + [target]
    corr_df = df.loc[:, corr_cols].apply(pd.to_numeric, errors="coerce")
    corr_df = corr_df.dropna(how="all")

    corr = corr_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()

    heatmap_path = outdir / "correlation_heatmap.png"
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.show()

    corr_path = outdir / "correlation_matrix.csv"
    corr.to_csv(corr_path)

    print(f"Saved heatmap to: {heatmap_path}")
    print(f"Correlation matrix: {corr_path}")


if __name__ == "__main__":
    main()
