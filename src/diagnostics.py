"""
Diagnostics: cluster purity, per-HLA metrics, and visualisations.

Provides three diagnostic analyses run after Level 1:
  1. Cluster purity  — log(pos/neg) per cluster, quantifying binder dominance.
  2. Per-HLA purity  — mean/median/min/max of cluster purity across all
                        clusters each allele appears in.
  3. Binder/non-binder cluster counts per HLA — how many clusters an allele
     "wins" vs "loses" after Level 1 binarisation.

All computations are vectorised with numpy/pandas to avoid memory
explosion on 18M+ aggregated pairs. No Python loops over clusters or HLAs.
"""
import numpy as np
import pandas as pd
from src.config import PipelineConfig
import time

# lazy import matplotlib only when plotting
_MPL_BACKEND_SET = False


def _setup_mpl():
    """Set matplotlib to non-interactive backend (once)."""
    global _MPL_BACKEND_SET
    if not _MPL_BACKEND_SET:
        import matplotlib
        matplotlib.use("Agg")
        _MPL_BACKEND_SET = True


# ============================================================================
# Cluster purity (point 3)
# ============================================================================

def compute_cluster_purity(
    agg: pd.DataFrame, n_hla: int, cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Per-cluster purity metrics (vectorised, no loops over clusters).

    For each cluster c:
      - n_hlas:       number of distinct alleles observed
      - total_pos:    sum of n_pos across all alleles
      - total_neg:    sum of n_neg across all alleles
      - binder_ratio: total_pos / (total_pos + total_neg)
      - purity_score: log((n_pos + eps) / (n_neg + eps))
                      Positive = binder-dominated; negative = non-binder-dominated.

    Returns:
        DataFrame indexed by cluster_id
    """
    print("[diag] Computing cluster purity...")
    t0 = time.time()

    eps = 1e-6
    g = agg.groupby("cluster_id", sort=False)

    cluster_stats = pd.DataFrame({
        "n_hlas": g["hla_idx"].nunique(),
        "total_pos": g["n_pos"].sum(),
        "total_neg": g["n_neg"].sum(),
    })
    cluster_stats["total_obs"] = (
        cluster_stats["total_pos"] + cluster_stats["total_neg"]
    )
    cluster_stats["binder_ratio"] = (
        cluster_stats["total_pos"] / cluster_stats["total_obs"]
    )
    cluster_stats["purity_score"] = np.log(
        (cluster_stats["total_pos"] + eps) / (cluster_stats["total_neg"] + eps)
    )

    cluster_stats.index.name = "cluster_id"
    cluster_stats = cluster_stats.reset_index()

    print(f"  Clusters: {len(cluster_stats):,}")
    print(f"  Purity score: mean={cluster_stats['purity_score'].mean():.3f}, "
          f"median={cluster_stats['purity_score'].median():.3f}")
    print(f"  ({time.time()-t0:.1f}s)")

    return cluster_stats


# ============================================================================
# Per-HLA purity (point 4)
# ============================================================================

def compute_hla_purity(
    agg: pd.DataFrame,
    cluster_purity: pd.DataFrame,
    n_hla: int,
    hla_names: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Per-HLA purity: for each allele, compute summary statistics over the
    purity scores of all clusters that allele appears in.

    Implemented via merge + groupby (no Python loops over HLAs).

    Returns:
        DataFrame with [hla_idx, hla_name, mean_purity, median_purity,
                         min_purity, max_purity, n_clusters]
    """
    print("[diag] Computing per-HLA purity...")
    t0 = time.time()

    # merge cluster purity onto each (cluster, HLA) row
    merged = agg[["cluster_id", "hla_idx"]].merge(
        cluster_purity[["cluster_id", "purity_score"]],
        on="cluster_id", how="left",
    )

    g = merged.groupby("hla_idx", sort=True)["purity_score"]
    hla_purity = pd.DataFrame({
        "hla_idx": np.arange(n_hla),
        "hla_name": hla_names,
    })

    stats = g.agg(["mean", "median", "min", "max", "count"]).reset_index()
    stats.columns = ["hla_idx", "mean_purity", "median_purity",
                     "min_purity", "max_purity", "n_clusters"]

    hla_purity = hla_purity.merge(stats, on="hla_idx", how="left")
    hla_purity = hla_purity.fillna(0)

    print(f"  Mean purity across HLAs: {hla_purity['mean_purity'].mean():.3f}")
    print(f"  ({time.time()-t0:.1f}s)")

    return hla_purity


# ============================================================================
# Binder / non-binder cluster counts per HLA (point 5)
# ============================================================================

def compute_hla_cluster_counts(
    agg: pd.DataFrame,
    n_hla: int,
    hla_names: np.ndarray,
    p_h: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    For each allele, count how many clusters are classified as binder vs
    non-binder using the b_call from Level 1.

    Fully vectorised via groupby.

    Returns:
        DataFrame with [hla_idx, hla_name, p_h, n_binder_clusters,
                         n_nonbinder_clusters, n_total_clusters, binder_fraction]
    """
    print("[diag] Computing binder/non-binder cluster counts per HLA...")
    t0 = time.time()

    g = agg.groupby("hla_idx")
    binder_clusters = g["b_call"].sum().astype(np.int32)
    total_clusters = g["b_call"].count().astype(np.int32)

    counts = pd.DataFrame({
        "hla_idx": np.arange(n_hla),
        "hla_name": hla_names,
        "p_h": p_h,
    })

    bc = binder_clusters.reset_index()
    bc.columns = ["hla_idx", "n_binder_clusters"]
    tc = total_clusters.reset_index()
    tc.columns = ["hla_idx", "n_total_clusters"]

    counts = counts.merge(bc, on="hla_idx", how="left")
    counts = counts.merge(tc, on="hla_idx", how="left")
    counts = counts.fillna(0)
    counts["n_nonbinder_clusters"] = (
        counts["n_total_clusters"] - counts["n_binder_clusters"]
    ).astype(np.int32)
    counts["binder_fraction"] = np.where(
        counts["n_total_clusters"] > 0,
        counts["n_binder_clusters"] / counts["n_total_clusters"],
        0.0,
    )

    print(f"  Mean binder clusters per HLA: "
          f"{counts['n_binder_clusters'].mean():.1f}")
    print(f"  Mean non-binder clusters per HLA: "
          f"{counts['n_nonbinder_clusters'].mean():.1f}")
    print(f"  ({time.time()-t0:.1f}s)")

    return counts


# ============================================================================
# Plotting
# ============================================================================

def plot_cluster_purity_hist(cluster_purity: pd.DataFrame, out_dir):
    """Histogram of cluster purity scores — binder-dominated vs non-binder."""
    _setup_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    scores = cluster_purity["purity_score"].values
    scores_clipped = np.clip(scores, -10, 10)  # clip outliers for viz
    ax.hist(scores_clipped, bins=100, color="#2196F3", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Cluster Purity Score  [log(pos/neg)]", fontsize=12)
    ax.set_ylabel("Number of Clusters", fontsize=12)
    ax.set_title("Cluster Purity Distribution", fontsize=14)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="neutral (50/50)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_purity_histogram.png", dpi=150)
    plt.close()


def plot_hla_purity_hist(hla_purity: pd.DataFrame, out_dir):
    """Histogram of per-HLA mean purity scores."""
    _setup_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = hla_purity["mean_purity"].dropna().values
    ax.hist(vals, bins=60, color="#4CAF50", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Mean Purity Score per HLA", fontsize=12)
    ax.set_ylabel("Number of HLAs", fontsize=12)
    ax.set_title("Per-HLA Mean Purity Distribution", fontsize=14)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "hla_purity_histogram.png", dpi=150)
    plt.close()


def plot_hla_cluster_counts(counts: pd.DataFrame, out_dir):
    """
    Two visualisations for binder/non-binder cluster counts:
      1. Horizontal bar plot: top 30 HLAs by total clusters
      2. Histogram: distribution of binder fraction across all HLAs
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    # ── bar plot (top 30 HLAs by total clusters) ──
    top = counts.nlargest(30, "n_total_clusters").copy()
    top = top.sort_values("n_total_clusters", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top))
    ax.barh(y, top["n_binder_clusters"], color="#4CAF50",
            label="Binder clusters", alpha=0.85)
    ax.barh(y, top["n_nonbinder_clusters"], left=top["n_binder_clusters"],
            color="#F44336", label="Non-binder clusters", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top["hla_name"].values, fontsize=7)
    ax.set_xlabel("Number of Clusters", fontsize=12)
    ax.set_title("Binder vs Non-binder Clusters per HLA (top 30)", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "hla_cluster_counts_barplot.png", dpi=150)
    plt.close()

    # ── histogram of binder fraction ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts["binder_fraction"].values, bins=50,
            color="#FF9800", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Binder Cluster Fraction per HLA", fontsize=12)
    ax.set_ylabel("Number of HLAs", fontsize=12)
    ax.set_title("Distribution of Binder Cluster Fraction", fontsize=14)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "hla_binder_fraction_histogram.png", dpi=150)
    plt.close()


# ============================================================================
# Main entry point
# ============================================================================

def run_diagnostics(
    agg: pd.DataFrame,
    n_hla: int,
    hla_names: np.ndarray,
    p_h: np.ndarray,
    cfg: PipelineConfig,
):
    """
    Run all diagnostics after Level 1.
    Saves CSVs and PNG plots to outputs/diagnostics/.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTICS: Purity & Cluster Metrics")
    print("=" * 60)

    out = cfg.output_dir / "diagnostics"

    # point 3: cluster purity
    cluster_purity = compute_cluster_purity(agg, n_hla, cfg)
    cluster_purity.to_csv(out / "cluster_purity.csv", index=False)

    # point 4: per-HLA purity
    hla_purity = compute_hla_purity(agg, cluster_purity, n_hla, hla_names, cfg)
    hla_purity.to_csv(out / "hla_purity.csv", index=False)

    # point 5: binder / non-binder cluster counts
    counts = compute_hla_cluster_counts(agg, n_hla, hla_names, p_h, cfg)
    counts.to_csv(out / "hla_cluster_counts.csv", index=False)

    # ── generate plots ──
    print("[diag] Generating plots...")
    plot_cluster_purity_hist(cluster_purity, out)
    plot_hla_purity_hist(hla_purity, out)
    plot_hla_cluster_counts(counts, out)

    print(f"[diag] Saved to {out}/")
    return cluster_purity, hla_purity, counts