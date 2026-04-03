#!/usr/bin/env python3
"""
Per-HLA Analysis and Visualization.

Reads Stage 1 outputs and produces:
  1. Per-HLA boxplots of gamma, n_tilde, lambda, p_tilde distributions
     ALL alleles shown (alleles without propagated data shown as empty)
  2. Per-HLA bar chart of log(n_pos+1) and log(n_neg+1)
  3. CSV summaries

Usage:
  python analyze_per_hla.py --stage1-dir outputs/anchor_all_08/stage1/

Integration:
  from analyze_per_hla import run_per_hla_analysis
  run_per_hla_analysis(stage1_dir)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================================
# Data loading
# ============================================================================

def load_level1(stage1_dir: Path) -> pd.DataFrame:
    l1 = stage1_dir / "level1"
    for name in ["level1_results_gibbs_final.parquet", "level1_results.parquet"]:
        path = l1 / name
        if path.exists():
            print(f"[load] Level 1: {path}")
            return pd.read_parquet(path)
    raise FileNotFoundError(f"No level1 results in {l1}")


def load_propagated(stage1_dir: Path) -> pd.DataFrame:
    path = stage1_dir / "level3" / "propagated_labels.parquet"
    if path.exists():
        print(f"[load] Propagated: {path}")
        return pd.read_parquet(path)
    print("[load] No propagated labels found")
    return pd.DataFrame()


def load_hla_index(stage1_dir: Path):
    path = stage1_dir / "level2" / "hla_index.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_noise_params(stage1_dir: Path):
    l1 = stage1_dir / "level1"
    for name in ["noise_params_gibbs_final.csv", "noise_params.csv"]:
        path = l1 / name
        if path.exists():
            return pd.read_csv(path)
    return None


# ============================================================================
# Summaries
# ============================================================================

def compute_per_hla_observed(agg, hla_names):
    print("[analysis] Per-HLA observed summaries...")
    t0 = time.time()
    n_hla = len(hla_names)

    g = agg.groupby("hla_idx")
    gamma_stats = g["gamma"].agg(["mean", "std", "median", "min", "max", "count"])
    gamma_stats.columns = [f"gamma_{c}" for c in gamma_stats.columns]

    pos_neg = g.agg(
        total_pos=("n_pos", "sum"),
        total_neg=("n_neg", "sum"),
        total_obs=("n_total", "sum"),
        n_clusters=("cluster_id", "nunique"),
    )

    summary = pd.DataFrame({"hla_idx": np.arange(n_hla), "hla_name": hla_names})
    summary = summary.merge(gamma_stats.reset_index(), on="hla_idx", how="left")
    summary = summary.merge(pos_neg.reset_index(), on="hla_idx", how="left")
    summary = summary.fillna(0)
    summary = summary.sort_values("n_clusters", ascending=False).reset_index(drop=True)
    print(f"  ({time.time()-t0:.1f}s)")
    return summary


def compute_per_hla_propagated(prop_df, hla_names):
    if prop_df.empty:
        return pd.DataFrame()
    print("[analysis] Per-HLA propagated summaries...")
    t0 = time.time()
    g = prop_df.groupby("hla_idx")
    stats = g.agg(
        n_propagated=("p_tilde", "count"),
        ptilde_mean=("p_tilde", "mean"),
        ptilde_std=("p_tilde", "std"),
        ptilde_median=("p_tilde", "median"),
        ntilde_mean=("n_tilde", "mean"),
        ntilde_std=("n_tilde", "std"),
        ntilde_median=("n_tilde", "median"),
        lambda_mean=("lambda_w", "mean"),
        lambda_std=("lambda_w", "std"),
        lambda_median=("lambda_w", "median"),
        n_positive=("p_tilde", lambda x: (x > 0.5).sum()),
    ).reset_index()
    stats["hla_name"] = hla_names[stats["hla_idx"].values]
    print(f"  ({time.time()-t0:.1f}s)")
    return stats


# ============================================================================
# Plotting helpers
# ============================================================================

DPI = 200
SCATTER_ALPHA = 0.15
SCATTER_SIZE = 4
MAX_SCATTER_PTS = 300


def _make_boxplot(data_lists, labels, ylabel, title, fname, out_dir,
                  log_scale=True, color="#2196F3", scatter_color="#1565C0"):
    """Generic boxplot for many HLAs with proper sizing and log y-axis."""
    n = len(labels)
    width = max(24, n * 0.14)

    fig, ax = plt.subplots(figsize=(width, 8))

    # filter out empty arrays for boxplot (but keep positions)
    non_empty_idx = [i for i, d in enumerate(data_lists) if len(d) > 0]
    non_empty_data = [data_lists[i] for i in non_empty_idx]

    if non_empty_data:
        bp = ax.boxplot(non_empty_data,
                        positions=non_empty_idx,
                        widths=0.6,
                        patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="red", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

    # scatter data points
    for i, vals in enumerate(data_lists):
        if len(vals) == 0:
            continue
        sub = vals if len(vals) <= MAX_SCATTER_PTS else np.random.choice(
            vals, MAX_SCATTER_PTS, replace=False)
        jitter = np.random.normal(0, 0.13, len(sub))
        ax.scatter(i + jitter, sub, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
                   c=scatter_color, zorder=0, rasterized=True)

    if log_scale:
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:.4f}" if abs(x) < 0.01 else f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}"))

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90,
                       fontsize=max(2.5, min(5.5, 280 / n)), ha="center")
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1, n)

    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=DPI)
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================================
# Main plots
# ============================================================================

def plot_gamma_boxplot(agg, summary, out_dir):
    """Boxplot of gamma per HLA, sorted by cluster count descending."""
    print("[plot] gamma boxplot...")
    hla_order = summary["hla_idx"].values
    hla_labels = summary["hla_name"].values

    data = []
    for hla_idx in hla_order:
        vals = agg.loc[agg["hla_idx"] == hla_idx, "gamma"].values
        if len(vals) > 5000:
            vals = np.random.choice(vals, 5000, replace=False)
        data.append(vals.astype(np.float64))

    _make_boxplot(data, hla_labels,
                  ylabel=r"$\gamma_{ch}$",
                  title=r"Posterior binding probability $\gamma_{ch}$ per allele "
                        r"(sorted by cluster count $\rightarrow$)",
                  fname="per_hla_gamma_boxplot.png",
                  out_dir=out_dir, log_scale=True)


def plot_propagated_boxplots(prop_df, summary, hla_names, out_dir):
    """Boxplots of n_tilde, lambda, p_tilde — ALL alleles shown."""
    if prop_df.empty:
        print("[plot] No propagated data — skipping")
        return

    print("[plot] propagated metrics boxplots (all alleles)...")

    # Use the same allele order as gamma plot (by cluster count)
    hla_order = summary["hla_idx"].values
    hla_labels = summary["hla_name"].values

    for metric, ylabel, title_str, fname, color, sc in [
        ("n_tilde", r"$\tilde{n}_{ch}$",
         r"Effective sample size $\tilde{n}_{ch}$",
         "per_hla_ntilde_boxplot.png", "#4CAF50", "#2E7D32"),
        ("lambda_w", r"$\lambda_{ch}$",
         r"Confidence weight $\lambda_{ch}$",
         "per_hla_lambda_boxplot.png", "#FF9800", "#E65100"),
        ("p_tilde", r"$\tilde{p}_{ch}$",
         r"Propagated binding probability $\tilde{p}_{ch}$",
         "per_hla_ptilde_boxplot.png", "#9C27B0", "#6A1B9A"),
    ]:
        data = []
        for hla_idx in hla_order:
            mask = prop_df["hla_idx"] == hla_idx
            if mask.any():
                vals = prop_df.loc[mask, metric].values
                if len(vals) > 5000:
                    vals = np.random.choice(vals, 5000, replace=False)
                data.append(vals.astype(np.float64))
            else:
                data.append(np.array([], dtype=np.float64))

        use_log = metric == "n_tilde"  # log scale only for n_tilde
        _make_boxplot(data, hla_labels,
                      ylabel=ylabel,
                      title=f"{title_str} per allele (sorted by cluster count; "
                            f"empty = not a propagation target)",
                      fname=fname, out_dir=out_dir,
                      log_scale=use_log, color=color, scatter_color=sc)


def plot_pos_neg_bars(summary, out_dir):
    """Bar chart: log(n_pos+1) up in blue, -log(n_neg+1) down in red."""
    print("[plot] pos/neg bar chart...")
    hla_labels = summary["hla_name"].values
    n = len(hla_labels)

    log_pos = np.log(summary["total_pos"].values + 1)
    log_neg = np.log(summary["total_neg"].values + 1)

    width = max(24, n * 0.14)
    fig, ax = plt.subplots(figsize=(width, 7))

    x = np.arange(n)
    ax.bar(x, log_pos, color="#2196F3", alpha=0.85,
           label=r"$\log(n_h^+ + 1)$", width=0.75)
    ax.bar(x, -log_neg, color="#F44336", alpha=0.85,
           label=r"$-\log(n_h^- + 1)$", width=0.75)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(hla_labels, rotation=90,
                       fontsize=max(2.5, min(5.5, 280 / n)), ha="center")
    ax.set_ylabel("log(count + 1)", fontsize=12)
    ax.set_title("Per-allele positive (blue ↑) and negative (red ↓) observation counts "
                 r"(sorted by cluster count $\rightarrow$)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-1, n)
    plt.tight_layout()
    plt.savefig(out_dir / "per_hla_pos_neg_bars.png", dpi=DPI)
    plt.close()
    print(f"  Saved: per_hla_pos_neg_bars.png")


# ============================================================================
# Entry point
# ============================================================================

def run_per_hla_analysis(stage1_dir):
    stage1_dir = Path(stage1_dir)
    out_dir = stage1_dir / "per_hla_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PER-HLA ANALYSIS")
    print("=" * 60)

    agg = load_level1(stage1_dir)
    prop_df = load_propagated(stage1_dir)
    hla_index = load_hla_index(stage1_dir)
    noise = load_noise_params(stage1_dir)

    hla_names = (hla_index["hla_name"].values if hla_index is not None
                 else np.array([f"HLA_{i}" for i in range(agg["hla_idx"].max()+1)]))
    n_hla = len(hla_names)
    print(f"  Alleles: {n_hla}, Observed: {len(agg):,}, Propagated: {len(prop_df):,}")

    obs_summary = compute_per_hla_observed(agg, hla_names)
    if noise is not None:
        obs_summary = obs_summary.merge(
            noise[["hla_idx", "alpha", "beta", "p_h", "tau_h"]],
            on="hla_idx", how="left")
    obs_summary.to_csv(out_dir / "per_hla_observed_summary.csv", index=False)

    prop_summary = pd.DataFrame()
    if not prop_df.empty:
        prop_summary = compute_per_hla_propagated(prop_df, hla_names)
        prop_summary.to_csv(out_dir / "per_hla_propagated_summary.csv", index=False)

    plot_gamma_boxplot(agg, obs_summary, out_dir)
    plot_propagated_boxplots(prop_df, obs_summary, hla_names, out_dir)
    plot_pos_neg_bars(obs_summary, out_dir)

    print(f"\n[per-HLA] Outputs: {out_dir}/")
    gc.collect()
    return obs_summary, prop_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-dir", required=True, type=Path)
    run_per_hla_analysis(p.parse_args().stage1_dir)


if __name__ == "__main__":
    main()