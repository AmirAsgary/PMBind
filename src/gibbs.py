"""
Gibbs Sampling for joint refinement of all Stage 1 latent variables.

Implements the Gibbs sweep described in the paper (Algorithm 1), with
three key fixes over the previous (broken) EM implementation:

1. FEEDBACK: Propagated labels from Level 3 feed back into Level 1 as
   augmented priors. For unobserved pairs where Level 3 produced
   (p_tilde, lambda_w), the prior in the next iteration becomes:
       pi_ch = lambda * p_tilde + (1-lambda) * p_h
   This is the missing link that makes iteration meaningful.

2. THETA SAMPLING: At each iteration, theta_ch is sampled from
   Bernoulli(gamma_ch) rather than using the continuous posterior.
   This provides proper mixing and allows noise parameters to be
   estimated from hard assignments (Solution 2: Bayesian noise).

3. BAYESIAN NOISE: Alpha_h and beta_h are re-estimated as Beta
   posterior means given current theta assignments, not from
   heuristic purity thresholds.

Each Gibbs sweep:
  1. Sample theta_ch ~ Bernoulli(gamma_ch) for observed pairs
  2. M-step: re-estimate alpha_h, beta_h from Beta posteriors (Solution 2)
  3. M-step: re-estimate p_h from theta counts with shrinkage
  4. E-step: recompute gamma_ch with updated params + propagation prior
  5. Every K iterations: binarize → Level 2 (conservative OR) → Level 3
     (coverage-penalized) → feed back propagated labels

Convergence: max|gamma^(t+1) - gamma^(t)| < tol.
"""
import numpy as np
import pandas as pd
from src.config import PipelineConfig
from src.level1 import (
    estimate_noise_from_theta, compute_posteriors, binarize_calls,
)
from src.level2 import run_level2
from src.level3 import run_level3
from src.io_utils import save_df
import time
import gc


def _build_propagation_prior(
    agg: pd.DataFrame,
    prop_df: pd.DataFrame,
    p_h: np.ndarray,
) -> np.ndarray:
    """
    Build per-(cluster, allele) prior array that incorporates Level 3
    propagation evidence.

    For OBSERVED pairs: pi_ch = p_h (unchanged — observed data dominates).
    For UNOBSERVED pairs that have propagated labels:
        pi_ch = lambda * p_tilde + (1 - lambda) * p_h

    We don't literally modify pi for unobserved pairs (they aren't in agg).
    Instead, we augment the prior for observed pairs where a similar allele's
    propagated label provides information:

    For each observed (c, h), if the propagated labels for the SAME cluster
    but DIFFERENT alleles are strong, we don't change (c, h) — it has its
    own data. The feedback works indirectly: propagated labels update p_h
    in the next iteration's M-step by adding soft pseudo-observations.

    Returns:
        Updated p_h array (shape n_hla) with propagation-informed rates.
    """
    if prop_df is None or prop_df.empty:
        return p_h

    n_hla = len(p_h)
    p_h_new = p_h.copy()

    # For each target allele in prop_df, compute the weighted average
    # of p_tilde across all propagated clusters as a soft update to p_h.
    # This is the feedback: propagated evidence shifts the allele's base rate.
    prop_pos = np.zeros(n_hla, dtype=np.float64)
    prop_weight = np.zeros(n_hla, dtype=np.float64)

    hla_idx = prop_df["hla_idx"].values
    p_tilde = prop_df["p_tilde"].values.astype(np.float64)
    lam = prop_df["lambda_w"].values.astype(np.float64)

    # weighted contribution of propagated labels
    np.add.at(prop_pos, hla_idx, lam * p_tilde)
    np.add.at(prop_weight, hla_idx, lam)

    # blend: for alleles with propagated data, shift p_h
    has_prop = prop_weight > 0
    if has_prop.any():
        prop_rate = prop_pos[has_prop] / prop_weight[has_prop]
        # weighted blend: more propagation weight → more influence
        # normalise so propagation contributes proportional to its total weight
        blend = prop_weight[has_prop] / (prop_weight[has_prop] + 100.0)
        p_h_new[has_prop] = (1 - blend) * p_h[has_prop] + blend * prop_rate

    return p_h_new


def run_gibbs(
    agg: pd.DataFrame,
    n_hla: int,
    p_h: np.ndarray,
    hla_names: np.ndarray,
    cluster_sizes: dict,
    cfg: PipelineConfig,
    skip_level3: bool = False,
) -> tuple:
    """
    Run the Gibbs sampling loop over Levels 1-3.

    This is the corrected version that:
      - Samples theta from Bernoulli(gamma) for proper Gibbs mixing
      - Uses Bayesian Beta posteriors for noise parameters (Solution 2)
      - Feeds propagated labels back as prior updates (the critical fix)
      - Uses conservative OR (Solution 3) and coverage penalty (Solution 1)

    Args:
        agg:            aggregated counts DataFrame (modified in-place)
        n_hla:          number of alleles
        p_h:            initial binder rates (from shrinkage)
        hla_names:      allele name array
        cluster_sizes:  dict {cluster_id: n_peptides} for coverage penalty
        cfg:            pipeline configuration
        skip_level3:    if True, skip propagation in the loop

    Returns:
        (agg, S, pairwise_df, prop_df, p_h)
    """
    max_iter = cfg.gibbs_max_iter
    tol = cfg.gibbs_tol
    recompute_every = cfg.gibbs_recompute_S_every
    sample_theta = cfg.gibbs_sample_theta

    print("\n" + "=" * 60)
    print(f"GIBBS SAMPLING (max_iter={max_iter}, tol={tol}, "
          f"sample={'stochastic' if sample_theta else 'deterministic'})")
    print("=" * 60)

    t_start = time.time()

    gamma_prev = agg["gamma"].values.astype(np.float64).copy()

    S = None
    pairwise_df = pd.DataFrame()
    prop_df = pd.DataFrame()

    for iteration in range(1, max_iter + 1):
        t_iter = time.time()
        print(f"\n--- Gibbs Iteration {iteration}/{max_iter} ---")

        # ════════════════════════════════════════════════════════════
        # Step 1: Sample theta_ch from current posteriors
        # ════════════════════════════════════════════════════════════
        gamma_current = agg["gamma"].values.astype(np.float64)
        if sample_theta:
            theta = np.random.binomial(1, gamma_current).astype(np.int8)
        else:
            # deterministic: hard threshold at 0.5
            theta = (gamma_current > 0.5).astype(np.int8)

        n_binder = theta.sum()
        print(f"  [θ] Sampled theta: {n_binder:,}/{len(theta):,} binders "
              f"({100*n_binder/len(theta):.2f}%)")

        # ════════════════════════════════════════════════════════════
        # Step 2: Re-estimate noise params from theta (Bayesian, Sol. 2)
        # ════════════════════════════════════════════════════════════
        alpha_h, beta_h = estimate_noise_from_theta(agg, theta, n_hla, cfg)
        print(f"  [M] alpha: mean={alpha_h.mean():.4f}, "
              f"beta: mean={beta_h.mean():.4f}")

        # ════════════════════════════════════════════════════════════
        # Step 3: Re-estimate p_h from theta + propagation feedback
        # ════════════════════════════════════════════════════════════
        # count theta=1 per allele, then shrinkage
        theta_pos = np.zeros(n_hla, dtype=np.float64)
        theta_tot = np.zeros(n_hla, dtype=np.float64)
        np.add.at(theta_pos, agg["hla_idx"].values, theta.astype(np.float64))
        np.add.at(theta_tot, agg["hla_idx"].values, 1.0)

        p_global = theta_pos.sum() / theta_tot.sum()
        k = cfg.shrinkage_k
        p_h = (theta_pos + k * p_global) / (theta_tot + k)
        p_h = np.clip(p_h, cfg.p_h_floor, cfg.p_h_ceil)

        # incorporate Level 3 feedback (if available from previous iteration)
        p_h = _build_propagation_prior(agg, prop_df, p_h)
        p_h = np.clip(p_h, cfg.p_h_floor, cfg.p_h_ceil)

        print(f"  [M] p_h: mean={p_h.mean():.4f}, "
              f"median={np.median(p_h):.4f}")

        # ════════════════════════════════════════════════════════════
        # Step 4: Recompute posteriors gamma_ch (E-step)
        # ════════════════════════════════════════════════════════════
        gamma_new = compute_posteriors(agg, alpha_h, beta_h, p_h)
        agg["gamma"] = gamma_new.astype(np.float32)

        b_call = binarize_calls(gamma_new, agg["hla_idx"].values, p_h, cfg)
        agg["b_call"] = b_call

        # ════════════════════════════════════════════════════════════
        # Check convergence
        # ════════════════════════════════════════════════════════════
        delta = np.max(np.abs(gamma_new - gamma_prev))
        mean_delta = np.mean(np.abs(gamma_new - gamma_prev))
        gamma_prev = gamma_new.copy()

        print(f"  [Δ] max|delta gamma| = {delta:.6f}, "
              f"mean|delta| = {mean_delta:.6f}")

        if delta < tol:
            print(f"  *** Converged at iteration {iteration} "
                  f"(delta {delta:.6f} < tol {tol}) ***")
            # run final Level 2 + 3
            S, pairwise_df = run_level2(agg, n_hla, hla_names, cfg)
            gc.collect()
            if not skip_level3:
                prop_df = run_level3(
                    agg, S, n_hla, p_h, hla_names, cluster_sizes, cfg)
                gc.collect()
            break

        # ════════════════════════════════════════════════════════════
        # Step 5: Recompute S and propagate (every K iterations)
        # ════════════════════════════════════════════════════════════
        if iteration % recompute_every == 0 or iteration == max_iter:
            S, pairwise_df = run_level2(agg, n_hla, hla_names, cfg)
            gc.collect()

            if not skip_level3:
                prop_df = run_level3(
                    agg, S, n_hla, p_h, hla_names, cluster_sizes, cfg)
                gc.collect()
                # Report feedback magnitude
                if not prop_df.empty:
                    n_prop_pos = (prop_df["p_tilde"] > 0.5).sum()
                    print(f"  [Feedback] Propagated: {len(prop_df):,} pairs, "
                          f"{n_prop_pos:,} positive-leaning")

        elapsed_iter = time.time() - t_iter
        print(f"  Iteration {iteration} done ({elapsed_iter:.1f}s)")

    else:
        # max_iter reached without convergence
        print(f"\n  WARNING: Gibbs did not converge in {max_iter} iterations "
              f"(final delta = {delta:.6f})")
        if S is None:
            S, pairwise_df = run_level2(agg, n_hla, hla_names, cfg)
            if not skip_level3:
                prop_df = run_level3(
                    agg, S, n_hla, p_h, hla_names, cluster_sizes, cfg)

    # ── save final outputs ──
    out = cfg.output_dir / "level1"
    noise_df = pd.DataFrame({
        "hla_idx": np.arange(n_hla),
        "alpha": alpha_h,
        "beta": beta_h,
        "p_h": p_h,
        "tau_h": np.minimum(cfg.tau_multiplier * p_h, cfg.tau_max),
    })
    noise_df.to_csv(out / "noise_params_gibbs_final.csv", index=False)
    save_df(agg, out / "level1_results_gibbs_final.parquet")

    elapsed_total = time.time() - t_start
    print(f"\n[Gibbs] Total time: {elapsed_total:.1f}s")

    return agg, S, pairwise_df, prop_df, p_h