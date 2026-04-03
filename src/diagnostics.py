"""
Diagnostics: per-(cluster, allele) purity, per-allele summaries, and plots.

The purity metric φ_{ch} is defined per observed (cluster, allele) pair:
    φ_{ch} = log((n_{ch}^+ + ε) / (n_{ch}^- + ε))

Positive φ_{ch} means binder-dominated; negative means non-binder-dominated.

Per-allele summaries (mean, std, median, min, max of φ_{ch} across clusters)
characterise the typical signal quality each allele contributes.

All computations are vectorised — no Python loops over 21M+ pairs.
"""
import numpy as np
import pandas as pd
from src.config import PipelineConfig
import time

_MPL_BACKEND_SET = False


def _setup_mpl():
    global _MPL_BACKEND_SET
    if not _MPL_BACKEND_SET:
        import matplotlib
        matplotlib.use("Agg")
        _MPL_BACKEND_SET = True


# ============================================================================
# Per-(cluster, allele) purity: φ_{ch}
# ============================================================================

def compute_pair_purity(agg: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Compute purity φ_{ch} = log((n_pos + ε) / (n_neg + ε)) for every
    observed (cluster, allele) pair. Each row of agg IS one such pair.

    Returns:
        agg with added 'phi_ch' column (float32)
    """
    print("[diag] Computing per-(cluster, allele) purity φ_{ch}...")
    t0 = time.time()

    eps = 1e-6
    n_pos = agg["n_pos"].values.astype(np.float64)
    n_neg = agg["n_neg"].values.astype(np.float64)

    phi = np.log((n_pos + eps) / (n_neg + eps))
    agg["phi_ch"] = phi.astype(np.float32)

    n_binder_dom = np.sum(phi > 0)
    n_nonbinder_dom = np.sum(phi < 0)
    print(f"  Pairs: {len(agg):,}  "
          f"binder-dominated (φ>0): {n_binder_dom:,}  "
          f"non-binder-dominated (φ<0): {n_nonbinder_dom:,}")
    print(f"  φ: mean={phi.mean():.3f}, median={np.median(phi):.3f}")
    print(f"  ({time.time()-t0:.1f}s)")

    return agg


# ============================================================================
# Per-allele purity summary
# ============================================================================

def compute_hla_purity(
    agg: pd.DataFrame,
    n_hla: int,
    hla_names: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Per-allele purity: mean, std, median, min, max of φ_{ch} across all
    clusters c in which allele h is observed.
    """
    print("[diag] Computing per-allele purity summaries...")
    t0 = time.time()

    g = agg.groupby("hla_idx", sort=True)["phi_ch"]
    stats = g.agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
    stats.columns = ["hla_idx", "mean_phi", "std_phi", "median_phi",
                     "min_phi", "max_phi", "n_clusters"]

    hla_purity = pd.DataFrame({
        "hla_idx": np.arange(n_hla),
        "hla_name": hla_names,
    })
    hla_purity = hla_purity.merge(stats, on="hla_idx", how="left")
    hla_purity = hla_purity.fillna(0)

    print(f"  Mean purity across alleles: {hla_purity['mean_phi'].mean():.3f}")
    print(f"  Std purity across alleles:  {hla_purity['std_phi'].mean():.3f}")
    print(f"  ({time.time()-t0:.1f}s)")

    return hla_purity


# ============================================================================
# Binder / non-binder cluster counts per allele
# ============================================================================

def compute_hla_cluster_counts(
    agg: pd.DataFrame,
    n_hla: int,
    hla_names: np.ndarray,
    p_h: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Binder vs non-binder cluster counts per allele (from Level 1 b_call)."""
    print("[diag] Computing binder/non-binder cluster counts per allele...")
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
    print(f"  Mean binder clusters:     {counts['n_binder_clusters'].mean():.1f}")
    print(f"  Mean non-binder clusters: {counts['n_nonbinder_clusters'].mean():.1f}")
    print(f"  ({time.time()-t0:.1f}s)")
    return counts


# ============================================================================
# Plotting
# ============================================================================

def plot_pair_purity_hist(agg, out_dir):
    _setup_mpl()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = np.clip(agg["phi_ch"].values, -15, 15)
    ax.hist(vals, bins=120, color="#2196F3", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.set_xlabel(r"Purity $\varphi_{ch}$  [log(pos/neg)]", fontsize=12)
    ax.set_ylabel("Number of (cluster, allele) pairs", fontsize=12)
    ax.set_title(r"Distribution of $\varphi_{ch}$", fontsize=14)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="neutral")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pair_purity_histogram.png", dpi=150)
    plt.close()


def plot_hla_purity_hist(hla_purity, out_dir):
    _setup_mpl()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = hla_purity["mean_phi"].dropna().values
    ax.hist(vals, bins=60, color="#4CAF50", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.set_xlabel(r"Mean purity $\bar{\varphi}_h$", fontsize=12)
    ax.set_ylabel("Number of alleles", fontsize=12)
    ax.set_title("Per-allele mean purity distribution", fontsize=14)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "hla_purity_histogram.png", dpi=150)
    plt.close()


def plot_hla_cluster_counts(counts, out_dir):
    _setup_mpl()
    import matplotlib.pyplot as plt
    top = counts.nlargest(30, "n_total_clusters").copy()
    top = top.sort_values("n_total_clusters", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top))
    ax.barh(y, top["n_binder_clusters"], color="#4CAF50",
            label="Binder", alpha=0.85)
    ax.barh(y, top["n_nonbinder_clusters"], left=top["n_binder_clusters"],
            color="#F44336", label="Non-binder", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top["hla_name"].values, fontsize=7)
    ax.set_xlabel("Number of clusters", fontsize=12)
    ax.set_title("Binder vs non-binder clusters (top 30 alleles)", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "hla_cluster_counts_barplot.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts["binder_fraction"].values, bins=50,
            color="#FF9800", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Binder cluster fraction", fontsize=12)
    ax.set_ylabel("Number of alleles", fontsize=12)
    ax.set_title("Distribution of binder cluster fraction", fontsize=14)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "hla_binder_fraction_histogram.png", dpi=150)
    plt.close()


# ============================================================================
# Entry point
# ============================================================================

def run_diagnostics(agg, n_hla, hla_names, p_h, cfg):
    print("\n" + "=" * 60)
    print("DIAGNOSTICS: Purity & Cluster Metrics")
    print("=" * 60)

    out = cfg.output_dir / "diagnostics"

    agg = compute_pair_purity(agg, cfg)
    hla_purity = compute_hla_purity(agg, n_hla, hla_names, cfg)
    hla_purity.to_csv(out / "hla_purity.csv", index=False)
    counts = compute_hla_cluster_counts(agg, n_hla, hla_names, p_h, cfg)
    counts.to_csv(out / "hla_cluster_counts.csv", index=False)

    print("[diag] Generating plots...")
    plot_pair_purity_hist(agg, out)
    plot_hla_purity_hist(hla_purity, out)
    plot_hla_cluster_counts(counts, out)

    print(f"[diag] Saved to {out}/")
    return hla_purity, counts