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
    rates using Beta-prior regularisation (Solution 2).

    Instead of hard-coded defaults for data-poor alleles, we place conjugate
    Beta priors over alpha_h and beta_h:
        alpha_h ~ Beta(a_alpha, b_alpha)    → E[alpha] = a/(a+b)
        beta_h  ~ Beta(a_beta, b_beta)      → E[beta]  = a/(a+b)

    The posterior mean given observed data becomes:
        alpha_h = (a_alpha + FN_h) / (a_alpha + b_alpha + N_binder_h)
        beta_h  = (a_beta  + FP_h) / (a_beta  + b_beta  + N_nonbinder_h)

    For well-observed alleles, the data dominates. For data-poor alleles,
    the posterior reverts to the prior mean — no hard-coded defaults needed.

    Returns:
        (alpha_h, beta_h) — each ndarray of shape (n_hla,)
    """
    print("[L1] Estimating noise parameters (Bayesian)...")
    t0 = time.time()

    n_pos = agg["n_pos"].values
    n_neg = agg["n_neg"].values
    n_total = agg["n_total"].values
    hla_idx = agg["hla_idx"].values
    min_n = cfg.noise_min_cluster_size
    purity = cfg.noise_purity_threshold

    # Beta prior hyperparameters
    a_a, b_a = cfg.alpha_prior_a, cfg.alpha_prior_b  # for alpha (FN rate)
    a_b, b_b = cfg.beta_prior_a, cfg.beta_prior_b    # for beta  (FP rate)

    # ── alpha_h: false negative rate ──
    # Estimated from near-unanimous binder clusters (>90% binder, n>=5).
    # FN_h = non-binder labels in these clusters (false negatives).
    # N_binder_h = total labels in these clusters.
    fn_num = np.zeros(n_hla, dtype=np.float64)
    fn_den = np.zeros(n_hla, dtype=np.float64)
    binder_mask = (n_total >= min_n) & (n_pos / n_total > purity)
    if binder_mask.any():
        idx_b = hla_idx[binder_mask]
        np.add.at(fn_num, idx_b, n_neg[binder_mask])
        np.add.at(fn_den, idx_b, n_total[binder_mask])

    # Beta posterior mean: (a_alpha + FN) / (a_alpha + b_alpha + N_binder)
    alpha_h = (a_a + fn_num) / (a_a + b_a + fn_den)

    # ── beta_h: false positive rate ──
    # Estimated from near-unanimous non-binder clusters.
    fp_num = np.zeros(n_hla, dtype=np.float64)
    fp_den = np.zeros(n_hla, dtype=np.float64)
    nonbinder_mask = (n_total >= min_n) & (n_neg / n_total > purity)
    if nonbinder_mask.any():
        idx_nb = hla_idx[nonbinder_mask]
        np.add.at(fp_num, idx_nb, n_pos[nonbinder_mask])
        np.add.at(fp_den, idx_nb, n_total[nonbinder_mask])

    beta_h = (a_b + fp_num) / (a_b + b_b + fp_den)

    # ── clamp to safe ranges ──
    alpha_h = np.clip(alpha_h, 0.001, 0.40)
    beta_h = np.clip(beta_h, 0.001, 0.15)

    # ── report ──
    n_data_a = np.sum(fn_den > 0)
    n_data_b = np.sum(fp_den > 0)
    print(f"  alpha: data for {n_data_a}/{n_hla} alleles, "
          f"mean={alpha_h.mean():.4f}, "
          f"prior mean={a_a/(a_a+b_a):.4f}")
    print(f"  beta:  data for {n_data_b}/{n_hla} alleles, "
          f"mean={beta_h.mean():.4f}, "
          f"prior mean={a_b/(a_b+b_b):.4f}")
    print(f"  ({time.time()-t0:.1f}s)")
    return alpha_h, beta_h


def estimate_noise_from_theta(
    agg: pd.DataFrame,
    theta: np.ndarray,
    n_hla: int,
    cfg: PipelineConfig,
) -> tuple:
    """
    Re-estimate noise parameters given hard theta assignments (for Gibbs).

    Given current theta_ch ∈ {0,1}:
      alpha_h = Beta posterior mean from pairs where theta=1 (true binders):
                (a_alpha + Σ_{θ=1} n_neg) / (a_alpha + b_alpha + Σ_{θ=1} n_total)
      beta_h  = Beta posterior mean from pairs where theta=0 (true non-binders):
                (a_beta + Σ_{θ=0} n_pos) / (a_beta + b_beta + Σ_{θ=0} n_total)
    """
    n_pos = agg["n_pos"].values.astype(np.float64)
    n_neg = agg["n_neg"].values.astype(np.float64)
    n_total = agg["n_total"].values.astype(np.float64)
    hla_idx = agg["hla_idx"].values

    a_a, b_a = cfg.alpha_prior_a, cfg.alpha_prior_b
    a_b, b_b = cfg.beta_prior_a, cfg.beta_prior_b

    # ── alpha from theta=1 clusters ──
    binder_mask = theta == 1
    fn_num = np.zeros(n_hla, dtype=np.float64)
    fn_den = np.zeros(n_hla, dtype=np.float64)
    if binder_mask.any():
        np.add.at(fn_num, hla_idx[binder_mask], n_neg[binder_mask])
        np.add.at(fn_den, hla_idx[binder_mask], n_total[binder_mask])
    alpha_h = (a_a + fn_num) / (a_a + b_a + fn_den)

    # ── beta from theta=0 clusters ──
    nonbinder_mask = theta == 0
    fp_num = np.zeros(n_hla, dtype=np.float64)
    fp_den = np.zeros(n_hla, dtype=np.float64)
    if nonbinder_mask.any():
        np.add.at(fp_num, hla_idx[nonbinder_mask], n_pos[nonbinder_mask])
        np.add.at(fp_den, hla_idx[nonbinder_mask], n_total[nonbinder_mask])
    beta_h = (a_b + fp_num) / (a_b + b_b + fp_den)

    alpha_h = np.clip(alpha_h, 0.001, 0.40)
    beta_h = np.clip(beta_h, 0.001, 0.15)

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