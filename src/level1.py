"""
Level 1: Scoring Observed Cluster-HLA Pairs.

Binary noise model that computes posterior binding probabilities gamma_ch
for each observed (cluster, HLA) pair, distinguishing between
contradictory evidence and absence of evidence.

The key insight: a continuous Beta-Binomial model treats "tested and
found negative" the same as "not tested," which is wrong. The binary
noise model with latent theta_ch ∈ {0,1} properly separates these
four situations:
  A: contradictory evidence → gamma pulled toward prior
  B: positive evidence only → gamma high
  C: negative evidence only → gamma very low
  D: no evidence            → gamma = prior

Equations reference: Eq. 5-10 from the paper.
"""
import numpy as np
import pandas as pd
from src.config import PipelineConfig
import time


# ============================================================================
# Noise parameter estimation
# ============================================================================

def estimate_noise_params(
    agg: pd.DataFrame,
    n_hla: int,
    cfg: PipelineConfig,
) -> tuple:
    """
    Estimate per-allele false negative (alpha_h) and false positive (beta_h)
    rates from high-confidence clusters.

    Strategy:
      alpha_h: fraction of non-binder labels in near-unanimous binder clusters
               (clusters where >90% of labels are binders and n >= 5).
               Represents missed detections due to low abundance or poor ionisation.

      beta_h:  fraction of binder labels in near-unanimous non-binder clusters
               (clusters where >90% of labels are non-binders and n >= 5).
               Represents contaminant co-purification.

    For alleles with too few high-confidence clusters, global defaults are used:
      alpha_default = 0.10, beta_default = 0.02

    Values are clamped to:
      alpha_h ∈ [0.001, 0.40]
      beta_h  ∈ [0.001, 0.15]

    Returns:
        (alpha_h, beta_h) — each ndarray of shape (n_hla,)
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
        # for each allele, sum non-binder labels in binder-dominated clusters
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

    # ── clamp to reasonable ranges ──
    alpha_h = np.clip(alpha_h, 0.001, 0.40)
    beta_h = np.clip(beta_h, 0.001, 0.15)

    # ── report ──
    n_empirical_a = np.sum(~np.isclose(alpha_h, cfg.alpha_default))
    n_empirical_b = np.sum(~np.isclose(beta_h, cfg.beta_default))
    print(f"  alpha: empirical for {n_empirical_a}/{n_hla} alleles, "
          f"mean={alpha_h.mean():.4f}")
    print(f"  beta:  empirical for {n_empirical_b}/{n_hla} alleles, "
          f"mean={beta_h.mean():.4f}")
    print(f"  ({time.time()-t0:.1f}s)")
    return alpha_h, beta_h


# ============================================================================
# Posterior computation
# ============================================================================

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

    Computation is in log-space for numerical stability:
      log_odds = log(pi_h) + log(L1) - log(1-pi_h) - log(L0)
      gamma = sigmoid(log_odds)

    All computation is fully vectorized over the entire agg DataFrame.

    Returns:
        ndarray of gamma values, same length as agg
    """
    print("[L1] Computing posterior binding probabilities...")
    t0 = time.time()

    # ── gather per-row parameters via allele index ──
    h = agg["hla_idx"].values
    n_pos = agg["n_pos"].values.astype(np.float64)
    n_neg = agg["n_neg"].values.astype(np.float64)

    a = alpha_h[h]    # false negative rate per row
    b = beta_h[h]     # false positive rate per row
    pi = p_h[h]       # prior per row

    # ── log-likelihoods ──
    log_L1 = n_pos * np.log(1.0 - a) + n_neg * np.log(a)
    log_L0 = n_pos * np.log(b) + n_neg * np.log(1.0 - b)

    # ── log-posterior via sigmoid ──
    log_num = np.log(pi) + log_L1
    log_den_term = np.log(1.0 - pi) + log_L0
    log_odds = log_num - log_den_term

    # clip to avoid overflow in exp()
    gamma = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -500, 500)))

    # ── handle edge cases ──
    gamma = np.clip(gamma, 1e-10, 1.0 - 1e-10)

    print(f"  gamma: mean={gamma.mean():.4f}, median={np.median(gamma):.4f}, "
          f">0.5: {(gamma > 0.5).sum():,}/{len(gamma):,}")
    print(f"  ({time.time()-t0:.1f}s)")
    return gamma


# ============================================================================
# Binarisation
# ============================================================================

def binarize_calls(
    gamma: np.ndarray,
    hla_idx: np.ndarray,
    p_h: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Convert posterior gamma_ch to binary call b_ch (Eq. 10).

    b_ch = 1 if gamma_ch > tau_h, else 0

    where tau_h = min(tau_multiplier * p_h, tau_max).

    The hard cap tau_max prevents the threshold from becoming unreachable
    for high-p_h alleles. Without it, an allele with p_h = 0.40 would
    need gamma > 0.80, which is overly stringent.

    Returns:
        ndarray of int8 binary calls
    """
    print("[L1] Binarizing binding calls...")

    # ── compute per-row threshold with cap ──
    tau = cfg.tau_multiplier * p_h[hla_idx]
    tau = np.minimum(tau, cfg.tau_max)

    b = (gamma > tau).astype(np.int8)

    # ── report threshold stats ──
    tau_vals = np.minimum(cfg.tau_multiplier * p_h, cfg.tau_max)
    print(f"  tau_h: mean={tau_vals.mean():.4f}, "
          f"median={np.median(tau_vals):.4f}, "
          f"range=[{tau_vals.min():.4f}, {tau_vals.max():.4f}]")
    print(f"  tau_max cap applied to "
          f"{(cfg.tau_multiplier * p_h > cfg.tau_max).sum()}"
          f"/{len(p_h)} alleles")
    print(f"  Positive calls: {b.sum():,}/{len(b):,} ({100*b.mean():.2f}%)")
    return b


# ============================================================================
# Full Level 1 pipeline
# ============================================================================

def run_level1(
    agg: pd.DataFrame,
    n_hla: int,
    p_h: np.ndarray,
    cfg: PipelineConfig,
) -> tuple:
    """
    Full Level 1 pipeline: estimate noise → posteriors → binarize.

    Adds columns [gamma, b_call] to agg and returns it.
    Also saves noise params and results to outputs/level1/.

    Returns:
        (agg_with_gamma_and_bcall, alpha_h, beta_h)
    """
    print("\n" + "=" * 60)
    print("LEVEL 1: Binary Noise Model")
    print("=" * 60)

    # step 1: estimate noise parameters from high-confidence clusters
    alpha_h, beta_h = estimate_noise_params(agg, n_hla, cfg)

    # step 2: compute posterior binding probabilities
    gamma = compute_posteriors(agg, alpha_h, beta_h, p_h)
    agg["gamma"] = gamma.astype(np.float32)

    # step 3: binarize calls for Level 2 co-occurrence analysis
    b_call = binarize_calls(gamma, agg["hla_idx"].values, p_h, cfg)
    agg["b_call"] = b_call

    # ── save outputs ──
    out = cfg.output_dir / "level1"

    # noise parameters (one row per allele)
    noise_df = pd.DataFrame({
        "hla_idx": np.arange(n_hla),
        "alpha": alpha_h,
        "beta": beta_h,
        "p_h": p_h,
        "tau_h": np.minimum(cfg.tau_multiplier * p_h, cfg.tau_max),
    })
    noise_df.to_csv(out / "noise_params.csv", index=False)

    # full results (one row per observed cluster-allele pair)
    from src.io_utils import save_df
    save_df(agg, out / "level1_results.parquet")

    print(f"[L1] Saved to {out}/")
    return agg, alpha_h, beta_h