"""
Level 2: Pairwise HLA Similarity from Co-occurrence.
Builds contingency tables for all HLA pairs using sparse matrix
multiplication, runs Fisher's exact test, applies BH-FDR correction,
and outputs the similarity matrix S_{hh'}.
Equations reference: Eq. 11-14 from the paper.
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
def build_sparse_call_matrices(
    agg: pd.DataFrame,
    n_clusters: int,
    n_hla: int,
) -> tuple:
    """
    Build two sparse CSC matrices from binarized calls:
      P[c, h] = 1 if b_ch = 1 (positive call)
      N[c, h] = 1 if b_ch = 0 AND observed (negative call)
    CSC format is optimal for column slicing and matrix multiplication.
    """
    print("[L2] Building sparse call matrices...")
    t0 = time.time()
    c_idx = agg["cluster_id"].values
    h_idx = agg["hla_idx"].values
    b = agg["b_call"].values
    # positive calls
    pos_mask = b == 1
    P = sparse.csc_matrix(
        (np.ones(pos_mask.sum(), dtype=np.float32),
         (c_idx[pos_mask], h_idx[pos_mask])),
        shape=(n_clusters, n_hla),
    )
    # negative calls (observed but not positive)
    neg_mask = b == 0
    N = sparse.csc_matrix(
        (np.ones(neg_mask.sum(), dtype=np.float32),
         (c_idx[neg_mask], h_idx[neg_mask])),
        shape=(n_clusters, n_hla),
    )
    print(f"  P nnz: {P.nnz:,}  N nnz: {N.nnz:,}  ({time.time()-t0:.1f}s)")
    return P, N
def compute_contingency_matrices(P, N) -> tuple:
    """
    Compute all pairwise contingency tables simultaneously via
    sparse matrix multiplication. For HLA pair (h, h'):
      a = P[:,h]^T @ P[:,h']    (both positive)
      b = P[:,h]^T @ N[:,h']    (h pos, h' neg)
      c = N[:,h]^T @ P[:,h']    (h neg, h' pos)
      d = N[:,h]^T @ N[:,h']    (both negative)
    Result: four dense (H x H) matrices.
    """
    print("[L2] Computing contingency tables via sparse matmul...")
    t0 = time.time()
    # (H x H) dense matrices — small since H ~ 300
    a_mat = (P.T @ P).toarray().astype(np.int32)
    b_mat = (P.T @ N).toarray().astype(np.int32)
    c_mat = (N.T @ P).toarray().astype(np.int32)
    d_mat = (N.T @ N).toarray().astype(np.int32)
    total = a_mat + b_mat + c_mat + d_mat
    print(f"  Max shared clusters per pair: {total.max()}")
    print(f"  ({time.time()-t0:.1f}s)")
    return a_mat, b_mat, c_mat, d_mat, total
def _fisher_one_pair(a, b, c, d):
    """Run Fisher's exact test for one HLA pair. Returns (OR, p-value)."""
    table = np.array([[a, b], [c, d]])
    try:
        odds_ratio, p_val = fisher_exact(table, alternative="two-sided")
    except Exception:
        odds_ratio, p_val = 1.0, 1.0
    return odds_ratio, p_val
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
    Returns DataFrame with columns [h1, h2, a, b, c, d, OR, pvalue, shared].
    """
    print("[L2] Running Fisher's exact tests...")
    t0 = time.time()
    n_hla = a_mat.shape[0]
    # collect valid pairs (upper triangle, enough shared clusters)
    pairs = []
    for i in range(n_hla):
        for j in range(i + 1, n_hla):
            if total[i, j] >= cfg.min_shared_clusters:
                pairs.append((i, j))
    print(f"  Valid HLA pairs: {len(pairs):,} / {n_hla*(n_hla-1)//2:,}")
    if not pairs:
        print("  WARNING: No valid pairs found!")
        return pd.DataFrame()
    # extract contingency values for all pairs
    h1s = np.array([p[0] for p in pairs], dtype=np.int16)
    h2s = np.array([p[1] for p in pairs], dtype=np.int16)
    a_vals = a_mat[h1s, h2s]
    b_vals = b_mat[h1s, h2s]
    c_vals = c_mat[h1s, h2s]
    d_vals = d_mat[h1s, h2s]
    # parallel Fisher's exact tests
    results = Parallel(n_jobs=cfg.n_jobs, backend="loky", verbose=0)(
        delayed(_fisher_one_pair)(a_vals[k], b_vals[k], c_vals[k], d_vals[k])
        for k in range(len(pairs))
    )
    ors = np.array([r[0] for r in results], dtype=np.float64)
    pvals = np.array([r[1] for r in results], dtype=np.float64)
    # replace inf/nan OR with 1.0
    ors = np.where(np.isfinite(ors), ors, 1.0)
    result_df = pd.DataFrame({
        "h1": h1s, "h2": h2s,
        "a": a_vals, "b": b_vals, "c": c_vals, "d": d_vals,
        "OR": ors, "pvalue": pvals,
        "shared": a_vals + b_vals + c_vals + d_vals,
    })
    print(f"  Fisher tests done ({time.time()-t0:.1f}s)")
    return result_df
def apply_fdr_correction(
    result_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to p-values."""
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
def build_similarity_matrix(
    result_df: pd.DataFrame,
    n_hla: int,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Build symmetric HLA similarity matrix S_{hh'} (Eq. 14).
    S = log(OR) if significant and OR > 0, else 0.
    """
    print("[L2] Building similarity matrix...")
    S = np.zeros((n_hla, n_hla), dtype=np.float64)
    if result_df.empty:
        return S
    sig = result_df[result_df["significant"]].copy()
    # log(OR), clamp OR > 0 to avoid log(0)
    or_vals = np.clip(sig["OR"].values, 1e-10, None)
    log_or = np.log(or_vals)
    h1 = sig["h1"].values
    h2 = sig["h2"].values
    S[h1, h2] = log_or
    S[h2, h1] = log_or  # symmetric
    n_pos = np.sum(log_or > 0)
    n_neg = np.sum(log_or < 0)
    print(f"  Non-zero entries: {np.count_nonzero(S):,} "
          f"(positive: {n_pos}, negative: {n_neg})")
    return S
def run_level2(
    agg: pd.DataFrame,
    n_hla: int,
    hla_names: np.ndarray,
    cfg: PipelineConfig,
) -> tuple:
    """
    Full Level 2 pipeline:
      1. Build sparse call matrices
      2. Compute contingency tables (sparse matmul)
      3. Fisher's exact tests (parallel)
      4. BH-FDR correction
      5. Build similarity matrix
    Returns: (S_matrix, pairwise_df)
    """
    print("\n" + "="*60)
    print("LEVEL 2: Pairwise HLA Similarity")
    print("="*60)
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
    # ── save outputs ──
    out = cfg.output_dir / "level2"
    np.save(out / "similarity_matrix.npy", S)
    pd.DataFrame({"hla_idx": np.arange(n_hla), "hla_name": hla_names}).to_csv(
        out / "hla_index.csv", index=False
    )
    from src.io_utils import save_df
    save_df(result_df, out / "pairwise_tests.parquet")
    print(f"[L2] Saved to {out}/")
    return S, result_df
