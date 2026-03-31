"""
Level 1: Scoring Observed Cluster-HLA Pairs.
Binary noise model that computes posterior binding probabilities gamma_ch
for each observed (cluster, HLA) pair, distinguishing between
contradictory evidence and absence of evidence.
Equations reference: Eq. 5-10 from the paper.
"""
import numpy as np
import pandas as pd
from src.config import PipelineConfig
import time
def estimate_noise_params(
    agg: pd.DataFrame,
    n_hla: int,
    cfg: PipelineConfig,
) -> tuple:
    """
    Estimate per-HLA false negative (alpha_h) and false positive (beta_h)
    rates from high-confidence clusters.
    alpha_h: fraction of non-binder labels in near-unanimous binder clusters.
    beta_h:  fraction of binder labels in near-unanimous non-binder clusters.
    Returns: (alpha, beta) each ndarray of shape (n_hla,).
    """
    print("[L1] Estimating noise parameters...")
    t0 = time.time()
    alpha_h = np.full(n_hla, cfg.alpha_default, dtype=np.float64)
    beta_h = np.full(n_hla, cfg.beta_default, dtype=np.float64)
    n_pos = agg["n_pos"].values
    n_neg = agg["n_neg"].values
    n_total = agg["n_total"].values
    hla_idx = agg["hla_idx"].values
    min_n = cfg.noise_min_cluster_size
    purity = cfg.noise_purity_threshold
    # ── estimate alpha_h (false negative rate) ──
    # from clusters where almost all labels are binders
    binder_mask = (n_total >= min_n) & (n_pos / n_total > purity)
    if binder_mask.any():
        # for each HLA, sum non-binder labels in binder-dominated clusters
        fn_num = np.zeros(n_hla, dtype=np.float64)   # false neg count
        fn_den = np.zeros(n_hla, dtype=np.float64)   # total in those clusters
        idx_b = hla_idx[binder_mask]
        np.add.at(fn_num, idx_b, n_neg[binder_mask])
        np.add.at(fn_den, idx_b, n_total[binder_mask])
        has_data = fn_den > 0
        alpha_h[has_data] = fn_num[has_data] / fn_den[has_data]
    # ── estimate beta_h (false positive rate) ──
    # from clusters where almost all labels are non-binders
    nonbinder_mask = (n_total >= min_n) & (n_neg / n_total > purity)
    if nonbinder_mask.any():
        fp_num = np.zeros(n_hla, dtype=np.float64)
        fp_den = np.zeros(n_hla, dtype=np.float64)
        idx_nb = hla_idx[nonbinder_mask]
        np.add.at(fp_num, idx_nb, n_pos[nonbinder_mask])
        np.add.at(fp_den, idx_nb, n_total[nonbinder_mask])
        has_data = fp_den > 0
        beta_h[has_data] = fp_num[has_data] / fp_den[has_data]
    # clamp to reasonable ranges
    alpha_h = np.clip(alpha_h, 0.001, 0.40)
    beta_h = np.clip(beta_h, 0.001, 0.15)
    n_empirical_a = np.sum(alpha_h != cfg.alpha_default)
    n_empirical_b = np.sum(beta_h != cfg.beta_default)
    print(f"  alpha: empirical for {n_empirical_a}/{n_hla} HLAs, "
          f"mean={alpha_h.mean():.4f}")
    print(f"  beta:  empirical for {n_empirical_b}/{n_hla} HLAs, "
          f"mean={beta_h.mean():.4f}")
    print(f"  ({time.time()-t0:.1f}s)")
    return alpha_h, beta_h
def compute_posteriors(
    agg: pd.DataFrame,
    alpha_h: np.ndarray,
    beta_h: np.ndarray,
    p_h: np.ndarray,
) -> np.ndarray:
    """
    Compute posterior binding probability gamma_ch for every observed
    (cluster, HLA) pair using the binary noise model (Eq. 9).
    gamma_ch = pi_h * L1 / (pi_h * L1 + (1-pi_h) * L0)
    where:
      L1 = (1 - alpha_h)^n+ * alpha_h^n-    (true binder likelihood)
      L0 = beta_h^n+ * (1 - beta_h)^n-      (true non-binder likelihood)
    All computation is fully vectorized over the entire agg dataframe.
    Returns: ndarray of gamma values, same length as agg.
    """
    print("[L1] Computing posterior binding probabilities...")
    t0 = time.time()
    # gather per-row parameters via HLA index
    h = agg["hla_idx"].values
    n_pos = agg["n_pos"].values.astype(np.float64)
    n_neg = agg["n_neg"].values.astype(np.float64)
    a = alpha_h[h]    # false negative rate per row
    b = beta_h[h]     # false positive rate per row
    pi = p_h[h]       # prior per row
    # compute log-likelihoods (log-space for numerical stability)
    log_L1 = n_pos * np.log(1.0 - a) + n_neg * np.log(a)
    log_L0 = n_pos * np.log(b) + n_neg * np.log(1.0 - b)
    # log-posterior using log-sum-exp trick
    log_num = np.log(pi) + log_L1
    log_den_term = np.log(1.0 - pi) + log_L0
    # gamma = exp(log_num) / (exp(log_num) + exp(log_den_term))
    # = sigmoid(log_num - log_den_term)
    log_odds = log_num - log_den_term
    gamma = 1.0 / (1.0 + np.exp(-log_odds))
    # handle edge cases
    gamma = np.clip(gamma, 1e-10, 1.0 - 1e-10)
    print(f"  gamma: mean={gamma.mean():.4f}, median={np.median(gamma):.4f}, "
          f">0.5: {(gamma > 0.5).sum():,}/{len(gamma):,}")
    print(f"  ({time.time()-t0:.1f}s)")
    return gamma
def binarize_calls(
    gamma: np.ndarray,
    hla_idx: np.ndarray,
    p_h: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Convert posterior gamma_ch to binary call b_ch (Eq. 10).
    b_ch = 1 if gamma_ch > tau_h, else 0
    tau_h = tau_multiplier * p_h (default: 2 * base rate)
    """
    print("[L1] Binarizing binding calls...")
    tau = cfg.tau_multiplier * p_h[hla_idx]
    b = (gamma > tau).astype(np.int8)
    print(f"  Positive calls: {b.sum():,}/{len(b):,} "
          f"({100*b.mean():.2f}%)")
    return b
def run_level1(
    agg: pd.DataFrame,
    n_hla: int,
    p_h: np.ndarray,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Full Level 1 pipeline: estimate noise -> posteriors -> binarize.
    Adds columns [gamma, b_call] to agg and returns it.
    Also saves noise params and results to outputs/level1/.
    """
    print("\n" + "="*60)
    print("LEVEL 1: Binary Noise Model")
    print("="*60)
    # step 1: estimate noise parameters
    alpha_h, beta_h = estimate_noise_params(agg, n_hla, cfg)
    # step 2: compute posteriors
    gamma = compute_posteriors(agg, alpha_h, beta_h, p_h)
    agg["gamma"] = gamma.astype(np.float32)
    # step 3: binarize
    b_call = binarize_calls(gamma, agg["hla_idx"].values, p_h, cfg)
    agg["b_call"] = b_call
    # ── save outputs ──
    out = cfg.output_dir / "level1"
    # noise parameters
    noise_df = pd.DataFrame({
        "hla_idx": np.arange(n_hla),
        "alpha": alpha_h,
        "beta": beta_h,
        "p_h": p_h,
    })
    noise_df.to_csv(out / "noise_params.csv", index=False)
    # main results
    from src.io_utils import save_df
    save_df(agg, out / "level1_results.parquet")
    print(f"[L1] Saved to {out}/")
    return agg, alpha_h, beta_h
