"""
Level 2: Pairwise HLA Similarity from Co-occurrence.

Builds contingency tables for all HLA pairs using sparse matrix
multiplication, runs Fisher's exact test, applies BH-FDR correction,
and outputs the similarity matrix S_{hh'}.

Additionally produces:
  - HLA×HLA matrices for OR, p-value, and adjusted p-value
  - Three heatmaps (log-scaled) for visual inspection
  - Per-HLA association counts (tested / significant)

Equations reference: Eq. 11-14 from the paper.

Performance notes:
  - Contingency tables are computed via sparse matrix multiplication
    (P^T @ P, P^T @ N, etc.), which is O(nnz) and avoids all-pairs loops.
  - Fisher's exact tests are parallelised via joblib.
  - HLA×HLA matrices are ~361×361 = 130K entries — trivially small.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact
from scipy.stats import false_discovery_control
from src.config import PipelineConfig
from joblib import Parallel, delayed
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# lazy matplotlib setup (only when plotting)
_MPL_BACKEND_SET = False


def _setup_mpl():
    """Set matplotlib to non-interactive backend (once)."""
    global _MPL_BACKEND_SET
    if not _MPL_BACKEND_SET:
        import matplotlib
        matplotlib.use("Agg")
        _MPL_BACKEND_SET = True


# ============================================================================
# Sparse call matrices
# ============================================================================

def build_sparse_call_matrices(
    agg: pd.DataFrame,
    n_clusters: int,
    n_hla: int,
) -> tuple:
    """
    Build two sparse CSC matrices from binarised calls:
      P[c, h] = 1 if b_ch = 1 (positive call)
      N[c, h] = 1 if b_ch = 0 AND observed (negative call)

    CSC format is optimal for column slicing and for the transposed
    matrix multiplication used to compute contingency tables.

    Returns:
        (P, N) — sparse CSC matrices of shape (n_clusters, n_hla)
    """
    print("[L2] Building sparse call matrices...")
    t0 = time.time()

    c_idx = agg["cluster_id"].values
    h_idx = agg["hla_idx"].values
    b = agg["b_call"].values

    # positive calls: clusters called as binders
    pos_mask = b == 1
    P = sparse.csc_matrix(
        (np.ones(pos_mask.sum(), dtype=np.float32),
         (c_idx[pos_mask], h_idx[pos_mask])),
        shape=(n_clusters, n_hla),
    )

    # negative calls: observed but NOT called as binders
    neg_mask = b == 0
    N = sparse.csc_matrix(
        (np.ones(neg_mask.sum(), dtype=np.float32),
         (c_idx[neg_mask], h_idx[neg_mask])),
        shape=(n_clusters, n_hla),
    )

    print(f"  P nnz: {P.nnz:,}  N nnz: {N.nnz:,}  ({time.time()-t0:.1f}s)")
    return P, N


# ============================================================================
# Contingency tables via sparse matmul
# ============================================================================

def compute_contingency_matrices(P, N) -> tuple:
    """
    Compute all pairwise contingency tables simultaneously via
    sparse matrix multiplication. For HLA pair (h, h'):
      a = P[:,h]^T @ P[:,h']    (both positive)
      b = P[:,h]^T @ N[:,h']    (h pos, h' neg)
      c = N[:,h]^T @ P[:,h']    (h neg, h' pos)
      d = N[:,h]^T @ N[:,h']    (both negative)

    Result: four dense (H x H) matrices — small since H ~ 300–400.

    Returns:
        (a_mat, b_mat, c_mat, d_mat, total) — each shape (n_hla, n_hla)
    """
    print("[L2] Computing contingency tables via sparse matmul...")
    t0 = time.time()

    a_mat = (P.T @ P).toarray().astype(np.int32)
    b_mat = (P.T @ N).toarray().astype(np.int32)
    c_mat = (N.T @ P).toarray().astype(np.int32)
    d_mat = (N.T @ N).toarray().astype(np.int32)

    total = a_mat + b_mat + c_mat + d_mat
    print(f"  Max shared clusters per pair: {total.max()}")
    print(f"  ({time.time()-t0:.1f}s)")
    return a_mat, b_mat, c_mat, d_mat, total


# ============================================================================
# Fisher's exact test
# ============================================================================

def _fisher_one_pair(a, b, c, d):
    """Run Fisher's exact test for one HLA pair. Returns (OR, p-value)."""
    table = np.array([[a, b], [c, d]])
    try:
        odds_ratio, p_val = fisher_exact(table, alternative="two-sided")
    except Exception:
        odds_ratio, p_val = 1.0, 1.0
    return odds_ratio, p_val


def _compute_conservative_or(a_vals, b_vals, c_vals, d_vals):
    """
    Solution 3: Uncertainty-aware odds ratios.

    Compute the lower bound of the 95% CI for log(OR) using the
    Haldane-Anscombe correction (+0.5 to all cells).

    OR_conservative = max(1, exp(log(OR) - 1.96 * SE(log OR)))

    This prevents rare alleles from generating massive propagation
    weights due to small-sample coincidences.

    Returns:
        ndarray of conservative OR values (≥ 1.0)
    """
    # Haldane-Anscombe correction: add 0.5 to prevent zero cells
    ac = a_vals + 0.5
    bc = b_vals + 0.5
    cc = c_vals + 0.5
    dc = d_vals + 0.5

    log_or = np.log(ac * dc / (bc * cc))
    se = np.sqrt(1.0/ac + 1.0/bc + 1.0/cc + 1.0/dc)

    # lower bound of 95% CI
    lb = log_or - 1.96 * se

    # conservative OR: must be > 1 to contribute as positive weight
    or_conservative = np.maximum(1.0, np.exp(lb))

    return or_conservative


def run_fisher_tests(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    c_mat: np.ndarray,
    d_mat: np.ndarray,
    total: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Run Fisher's exact test for all HLA pairs with enough shared clusters.
    Uses joblib for parallelisation across pairs.

    Only tests pairs in the upper triangle with
    total[i,j] >= min_shared_clusters.

    Returns:
        DataFrame with columns [h1, h2, a, b, c, d, OR, pvalue, shared]
    """
    print("[L2] Running Fisher's exact tests...")
    t0 = time.time()
    n_hla = a_mat.shape[0]

    # ── collect valid pairs (upper triangle, enough shared clusters) ──
    pairs = []
    for i in range(n_hla):
        for j in range(i + 1, n_hla):
            if total[i, j] >= cfg.min_shared_clusters:
                pairs.append((i, j))

    print(f"  Valid HLA pairs: {len(pairs):,} / {n_hla*(n_hla-1)//2:,}")

    if not pairs:
        print("  WARNING: No valid pairs found!")
        return pd.DataFrame()

    # ── extract contingency values ──
    h1s = np.array([p[0] for p in pairs], dtype=np.int16)
    h2s = np.array([p[1] for p in pairs], dtype=np.int16)
    a_vals = a_mat[h1s, h2s]
    b_vals = b_mat[h1s, h2s]
    c_vals = c_mat[h1s, h2s]
    d_vals = d_mat[h1s, h2s]

    # ── parallel Fisher's exact tests ──
    results = Parallel(n_jobs=cfg.n_jobs, backend="loky", verbose=0)(
        delayed(_fisher_one_pair)(a_vals[k], b_vals[k], c_vals[k], d_vals[k])
        for k in range(len(pairs))
    )

    ors = np.array([r[0] for r in results], dtype=np.float64)
    pvals = np.array([r[1] for r in results], dtype=np.float64)

    # replace inf/nan OR with 1.0 (no association)
    ors = np.where(np.isfinite(ors), ors, 1.0)

    result_df = pd.DataFrame({
        "h1": h1s, "h2": h2s,
        "a": a_vals, "b": b_vals, "c": c_vals, "d": d_vals,
        "OR": ors, "pvalue": pvals,
        "shared": a_vals + b_vals + c_vals + d_vals,
    })

    # Solution 3: conservative OR (lower bound of 95% CI with Haldane correction)
    result_df["OR_conservative"] = _compute_conservative_or(
        a_vals.astype(np.float64), b_vals.astype(np.float64),
        c_vals.astype(np.float64), d_vals.astype(np.float64),
    )

    print(f"  Fisher tests done ({time.time()-t0:.1f}s)")
    return result_df


# ============================================================================
# FDR correction
# ============================================================================

def apply_fdr_correction(
    result_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    Adds columns 'pvalue_adj' and 'significant' to the DataFrame.
    """
    print("[L2] Applying BH-FDR correction...")
    if result_df.empty:
        return result_df

    pval_adj = false_discovery_control(
        result_df["pvalue"].values, method="bh"
    )
    result_df["pvalue_adj"] = pval_adj
    result_df["significant"] = pval_adj < cfg.fdr_threshold

    n_sig = result_df["significant"].sum()
    print(f"  Significant pairs: {n_sig:,} / {len(result_df):,} "
          f"(FDR < {cfg.fdr_threshold})")
    return result_df


# ============================================================================
# Similarity matrix
# ============================================================================

def build_similarity_matrix(
    result_df: pd.DataFrame,
    n_hla: int,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Build symmetric HLA similarity matrix S_{hh'} (Eq. 14).

    When cfg.use_conservative_or is True (Solution 3), uses the lower-bound
    OR instead of the point estimate. This prevents rare-allele inflation
    of propagation weights.

    S = log(OR) if significant and OR > 1, else 0.

    Returns:
        ndarray of shape (n_hla, n_hla)
    """
    print("[L2] Building similarity matrix...")
    S = np.zeros((n_hla, n_hla), dtype=np.float64)

    if result_df.empty:
        return S

    sig = result_df[result_df["significant"]].copy()

    # choose which OR to use for weights
    if cfg.use_conservative_or and "OR_conservative" in sig.columns:
        or_col = "OR_conservative"
        print("  Using conservative OR (lower-bound 95% CI)")
    else:
        or_col = "OR"
        print("  Using point-estimate OR")

    or_vals = np.clip(sig[or_col].values, 1e-10, None)
    log_or = np.log(or_vals)

    h1 = sig["h1"].values
    h2 = sig["h2"].values
    S[h1, h2] = log_or
    S[h2, h1] = log_or  # symmetric

    n_pos = np.sum(log_or > 0)
    n_neg = np.sum(log_or < 0)
    n_zero = np.sum(np.isclose(log_or, 0))
    print(f"  Non-zero entries: {np.count_nonzero(S):,} "
          f"(positive: {n_pos}, zero/neutral: {n_zero}, negative: {n_neg})")
    return S


# ============================================================================
# HLA×HLA matrices and heatmaps (point 8)
# ============================================================================

def build_hla_matrices(result_df, n_hla, hla_names, out_dir):
    """
    Build and save symmetric HLA×HLA matrices for OR, p-value, adj p-value.
    Saved as both .npy (fast reload) and .csv (human-readable).
    Diagonal is NaN (self-comparison undefined).
    """
    print("[L2] Building HLA×HLA matrices...")

    # initialise with neutral values (OR=1 means no effect, pval=1 means not tested)
    or_mat = np.ones((n_hla, n_hla), dtype=np.float64)
    pval_mat = np.ones((n_hla, n_hla), dtype=np.float64)
    padj_mat = np.ones((n_hla, n_hla), dtype=np.float64)

    if not result_df.empty:
        h1 = result_df["h1"].values
        h2 = result_df["h2"].values

        # fill both triangles (symmetric)
        or_mat[h1, h2] = result_df["OR"].values
        or_mat[h2, h1] = result_df["OR"].values
        pval_mat[h1, h2] = result_df["pvalue"].values
        pval_mat[h2, h1] = result_df["pvalue"].values
        if "pvalue_adj" in result_df.columns:
            padj_mat[h1, h2] = result_df["pvalue_adj"].values
            padj_mat[h2, h1] = result_df["pvalue_adj"].values

    # diagonal = self (undefined)
    np.fill_diagonal(or_mat, np.nan)
    np.fill_diagonal(pval_mat, np.nan)
    np.fill_diagonal(padj_mat, np.nan)

    # save as numpy arrays
    np.save(out_dir / "hla_or_matrix.npy", or_mat)
    np.save(out_dir / "hla_pvalue_matrix.npy", pval_mat)
    np.save(out_dir / "hla_pvalue_adj_matrix.npy", padj_mat)

    # save as labeled CSV (small enough: ~361×361)
    for name, mat in [("OR", or_mat), ("pvalue", pval_mat),
                      ("pvalue_adj", padj_mat)]:
        df = pd.DataFrame(mat, index=hla_names, columns=hla_names)
        df.to_csv(out_dir / f"hla_{name}_matrix.csv")

    return or_mat, pval_mat, padj_mat


def plot_hla_heatmaps(or_mat, pval_mat, padj_mat, hla_names, out_dir):
    """
    Three heatmaps on log scale:
      1. log10(OR) — symmetric diverging colormap
      2. -log10(p-value) — sequential, higher = more significant
      3. -log10(adjusted p-value) — sequential

    Axis labels are shown only when n_hla <= 50.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    n = len(hla_names)
    show_labels = n <= 50

    for title, mat, fname, cmap in [
        ("log\u2081\u2080(Odds Ratio)", or_mat, "heatmap_OR.png", "RdBu_r"),
        ("-log\u2081\u2080(p-value)", pval_mat, "heatmap_pvalue.png", "YlOrRd"),
        ("-log\u2081\u2080(adj. p-value)", padj_mat,
         "heatmap_pvalue_adj.png", "YlOrRd"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_mat = mat.copy()

        if "OR" in fname:
            # log10 of OR, symmetric colour scale
            plot_mat = np.where(np.isnan(plot_mat), 1.0, plot_mat)
            plot_mat = np.log10(np.clip(plot_mat, 1e-10, None))
            vmax = np.nanpercentile(np.abs(plot_mat[plot_mat != 0]), 95)
            if vmax == 0:
                vmax = 1.0
            im = ax.imshow(plot_mat, cmap=cmap, vmin=-vmax, vmax=vmax,
                           aspect="auto", interpolation="nearest")
        else:
            # -log10(pvalue): higher = more significant
            plot_mat = np.where(np.isnan(plot_mat), 1.0, plot_mat)
            plot_mat = -np.log10(np.clip(plot_mat, 1e-300, 1.0))
            im = ax.imshow(plot_mat, cmap=cmap, aspect="auto",
                           interpolation="nearest")

        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=14)

        if show_labels:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(hla_names, rotation=90, fontsize=5)
            ax.set_yticklabels(hla_names, fontsize=5)
        else:
            ax.set_xlabel(f"Allele index (n={n})")
            ax.set_ylabel(f"Allele index (n={n})")

        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    print(f"  Heatmaps saved to {out_dir}/")


# ============================================================================
# Per-HLA association counts (point 9)
# ============================================================================

def compute_hla_associations(result_df, n_hla, hla_names, out_dir):
    """
    For each HLA: count how many other HLAs had any association tested,
    and how many were significant. Saved as CSV + histogram plot.

    Returns:
        DataFrame with [hla_idx, hla_name, n_tested, n_significant]
    """
    print("[L2] Computing per-HLA association counts...")
    t0 = time.time()

    tested = np.zeros(n_hla, dtype=np.int32)
    significant = np.zeros(n_hla, dtype=np.int32)

    if not result_df.empty:
        h1 = result_df["h1"].values
        h2 = result_df["h2"].values
        # each pair contributes to both HLAs
        np.add.at(tested, h1, 1)
        np.add.at(tested, h2, 1)

        if "significant" in result_df.columns:
            sig_mask = result_df["significant"].values
            np.add.at(significant, h1[sig_mask], 1)
            np.add.at(significant, h2[sig_mask], 1)

    assoc_df = pd.DataFrame({
        "hla_idx": np.arange(n_hla),
        "hla_name": hla_names,
        "n_tested": tested,
        "n_significant": significant,
    })
    assoc_df.to_csv(out_dir / "hla_associations.csv", index=False)

    print(f"  Mean tested per HLA: {tested.mean():.1f}")
    print(f"  Mean significant per HLA: {significant.mean():.1f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # ── plot: two histograms of log(count) ──
    _setup_mpl()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, label, color in [
        (axes[0], tested, "Tested associations", "#2196F3"),
        (axes[1], significant, "Significant associations", "#E91E63"),
    ]:
        vals_pos = vals[vals > 0]
        if len(vals_pos) > 0:
            ax.hist(np.log10(vals_pos + 1), bins=40, color=color,
                    edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_xlabel("log\u2081\u2080(n + 1)", fontsize=12)
        ax.set_ylabel("Number of HLAs", fontsize=12)
        ax.set_title(label, fontsize=13)

    plt.tight_layout()
    plt.savefig(out_dir / "hla_associations_histogram.png", dpi=150)
    plt.close()

    return assoc_df


# ============================================================================
# Full Level 2 pipeline
# ============================================================================

def run_level2(
    agg: pd.DataFrame,
    n_hla: int,
    hla_names: np.ndarray,
    cfg: PipelineConfig,
) -> tuple:
    """
    Full Level 2 pipeline:
      1. Build sparse call matrices from Level 1 binary calls
      2. Compute contingency tables (sparse matmul)
      3. Fisher's exact tests (parallel)
      4. BH-FDR correction
      5. Build similarity matrix
      6. Build HLA×HLA matrices + heatmaps
      7. Compute per-HLA association counts

    Returns:
        (S_matrix, pairwise_df)
    """
    print("\n" + "=" * 60)
    print("LEVEL 2: Pairwise HLA Similarity")
    print("=" * 60)

    n_clusters = agg["cluster_id"].max() + 1

    # step 1-2: sparse matrices and contingency tables
    P, N = build_sparse_call_matrices(agg, n_clusters, n_hla)
    a_mat, b_mat, c_mat, d_mat, total = compute_contingency_matrices(P, N)
    del P, N  # free memory

    # step 3: Fisher's exact tests
    result_df = run_fisher_tests(a_mat, b_mat, c_mat, d_mat, total, cfg)
    del a_mat, b_mat, c_mat, d_mat, total

    # step 4: FDR correction
    result_df = apply_fdr_correction(result_df, cfg)

    # step 5: similarity matrix
    S = build_similarity_matrix(result_df, n_hla, cfg)

    # ── save core outputs ──
    out = cfg.output_dir / "level2"
    np.save(out / "similarity_matrix.npy", S)
    pd.DataFrame({"hla_idx": np.arange(n_hla), "hla_name": hla_names}).to_csv(
        out / "hla_index.csv", index=False
    )
    from src.io_utils import save_df
    save_df(result_df, out / "pairwise_tests.parquet")

    # step 6: HLA×HLA matrices and heatmaps
    or_mat, pval_mat, padj_mat = build_hla_matrices(
        result_df, n_hla, hla_names, out)
    plot_hla_heatmaps(or_mat, pval_mat, padj_mat, hla_names, out)

    # step 7: per-HLA association counts
    compute_hla_associations(result_df, n_hla, hla_names, out)

    print(f"[L2] Saved to {out}/")
    return S, result_df