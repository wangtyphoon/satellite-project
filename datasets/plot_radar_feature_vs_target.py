#!/usr/bin/env python3
"""Scatter plots with linear regression and R^2 for radar features vs target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main() -> None:
    # Config
    csv_path = Path("gpm_passes_swath_true.csv")
    target = "delta_24h"
    feature_prefix = "stormtop_"
    features = None
    outdir = Path("plots_radar_feature_target")
    max_points = 2000

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df1 = df.loc[:, 'zFactorFinal_max_r100':]
    feature_cols = [c for c in df1.columns]
    #df = df[(df['intensity_bst'] > 55) & (df['delta_24h'] > 0)].copy()

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    # if features:
    #     feature_cols = [c for c in features if c in df.columns]
    #     missing = [c for c in features if c not in df.columns]
    #     if missing:
    #         raise ValueError(f"Missing feature columns: {missing}")
    # else:
    #     feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    
    #if not feature_cols:
    #   raise ValueError("No feature columns found.")

    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    y_all = pd.to_numeric(df[target], errors="coerce")

    for feature in feature_cols:
        x_all = pd.to_numeric(df[feature], errors="coerce")
        mask = np.isfinite(x_all) & np.isfinite(y_all)
        if mask.sum() < 2:
            summary_rows.append({
                "feature": feature,
                "n": int(mask.sum()),
                "slope": float("nan"),
                "intercept": float("nan"),
                "r2": float("nan"),
            })
            continue

        x = x_all[mask].to_numpy()
        y = y_all[mask].to_numpy()

        if max_points and len(x) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x), size=max_points, replace=False)
            x_plot = x[idx]
            y_plot = y[idx]
        else:
            x_plot = x
            y_plot = y

        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r2 = compute_r2(y, y_pred)

        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        y_line = slope * x_line + intercept

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.scatter(x_plot, y_plot, s=12, alpha=0.6, edgecolors="none")
        ax.plot(x_line, y_line, color="red", linewidth=2)
        ax.set_title(f"{feature} vs {target}\nR^2={r2:.3f}, n={len(x)}")
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        fig.tight_layout()

        out_path = outdir / f"{feature}_vs_{target}.png"
        #fig.savefig(out_path, bbox_inches="tight")
        #plt.close(fig)
        plt.show()
        summary_rows.append({
            "feature": feature,
            "n": int(len(x)),
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values("r2", ascending=False, inplace=True, na_position="last")
    summary_path = outdir / f"summary_r2_{target}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved plots to: {outdir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
