"""
PCA Dimensionality Reduction Module for AEP Energy Consumption ML Platform.
Implements:
1. Scree Plot & Cumulative Explained Variance analysis.
2. 2D Principal Component Projection (PC1 vs. PC2) color-coded by High Demand vs Normal Demand.
3. Feature loadings and variance decomposition metrics.
Generates:
- static/charts/pca_scree_variance.png
- static/charts/pca_2d_projection.png
"""

import json
from pathlib import Path
import time
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "AEP_preprocessed.csv"
CHARTS_DIR = BASE_DIR / "static" / "charts"
CACHE_PATH = CHARTS_DIR / "pca_cache.json"

FEATURES = ["hour", "month", "dayofweek", "lag_1", "lag_24"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y_clf = df["High_Demand"]
    y_reg = df["AEP_MW"]
    return df, X, y_clf, y_reg


def compute_pca_summary(force: bool = False) -> Dict[str, Any]:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    scree_chart = CHARTS_DIR / "pca_scree_variance.png"
    proj_chart = CHARTS_DIR / "pca_2d_projection.png"

    if not force and CACHE_PATH.exists() and scree_chart.exists() and proj_chart.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass

    df, X, y_clf, y_reg = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA with all components
    pca = PCA(n_components=len(FEATURES))
    pca.fit(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    # --- 1. Scree Plot & Cumulative Variance Chart ---
    plt.figure(figsize=(8.5, 4.8), dpi=160)
    comps = [f"PC{i+1}" for i in range(len(FEATURES))]
    bars = plt.bar(comps, var_ratio * 100, color="#0284c7", alpha=0.85, width=0.55, label="Individual Explained Variance (%)")
    plt.plot(comps, cum_var * 100, "o-", color="#10b981", lw=2.4, markersize=7, label="Cumulative Explained Variance (%)")
    plt.axhline(y=90, color="#dc2626", linestyle="--", lw=1.5, label="90% Variance Threshold (Retained in 2 PCs)")

    for bar, pct in zip(bars, var_ratio * 100):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{pct:.1f}%", ha="center", fontsize=9.5, fontweight="bold")

    for i, c_pct in enumerate(cum_var * 100):
        plt.text(i, c_pct - 5.5, f"{c_pct:.1f}%", ha="center", fontsize=9.5, fontweight="bold", color="#059669")

    plt.title("PCA Scree Plot: Individual & Cumulative Explained Variance", fontsize=11.5, fontweight="bold", pad=12)
    plt.xlabel("Principal Components (Orthogonal Subspace)", fontsize=10.5, fontweight="bold")
    plt.ylabel("Explained Variance Ratio (%)", fontsize=10.5, fontweight="bold")
    plt.ylim(0, 108)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    plt.savefig(scree_chart, bbox_inches="tight")
    plt.close()

    # --- 2. 2D PCA Projection Scatter Plot ---
    # Transform observations onto PC1 and PC2
    pca_2d = PCA(n_components=2)
    coords = pca_2d.fit_transform(X_scaled)

    # Sample for plotting clarity without overwhelming vector renderer
    sample_indices = np.random.RandomState(42).choice(len(coords), size=min(15000, len(coords)), replace=False)
    sample_coords = coords[sample_indices]
    sample_labels = y_clf.iloc[sample_indices].values

    plt.figure(figsize=(9, 5.2), dpi=160)
    normal_mask = sample_labels == 0
    high_mask = sample_labels == 1

    plt.scatter(
        sample_coords[normal_mask, 0],
        sample_coords[normal_mask, 1],
        c="#10b981",
        alpha=0.35,
        s=12,
        label="Normal Demand (< 15,310 MW)",
        edgecolors="none",
    )
    plt.scatter(
        sample_coords[high_mask, 0],
        sample_coords[high_mask, 1],
        c="#ef4444",
        alpha=0.45,
        s=16,
        label="High Demand Alert (>= 15,310 MW)",
        edgecolors="none",
    )

    plt.title("2D PCA Manifold Projection: High-Demand vs. Normal Grid Regimes", fontsize=11.5, fontweight="bold", pad=12)
    plt.xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% Variance - Overall Load Magnitude)", fontsize=10.5, fontweight="bold")
    plt.ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% Variance - Diurnal & Temporal Contrast)", fontsize=10.5, fontweight="bold")
    plt.axvline(0, color="#64748b", linestyle=":", lw=1, alpha=0.6)
    plt.axhline(0, color="#64748b", linestyle=":", lw=1, alpha=0.6)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(proj_chart, bbox_inches="tight")
    plt.close()

    # Loadings table
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(FEATURES))],
        index=FEATURES,
    )

    loadings_list = []
    for feat in FEATURES:
        loadings_list.append({
            "feature": feat,
            "pc1": round(float(loadings.loc[feat, "PC1"]), 4),
            "pc2": round(float(loadings.loc[feat, "PC2"]), 4),
            "pc3": round(float(loadings.loc[feat, "PC3"]), 4),
        })

    summary = {
        "dataset_rows": len(df),
        "pc1_variance": round(float(var_ratio[0] * 100), 2),
        "pc2_variance": round(float(var_ratio[1] * 100), 2),
        "cumulative_variance_2pc": round(float(cum_var[1] * 100), 2),
        "features": FEATURES,
        "loadings": loadings_list,
        "charts": {
            "scree": "charts/pca_scree_variance.png",
            "projection": "charts/pca_2d_projection.png",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(CACHE_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print("Computing PCA Summary...")
    res = compute_pca_summary(force=True)
    print("PC1 Variance:", res["pc1_variance"], "%")
    print("PC2 Variance:", res["pc2_variance"], "%")
    print("Cumulative 2 PCs:", res["cumulative_variance_2pc"], "%")
