"""
Iterative EM refinement of the Stage 1 pipeline.

Implements the deterministic EM variant of the Gibbs sampling scheme
described in the paper (Section "Extension: Iterative Refinement via
Gibbs Sampling", Remark on EM).

Each iteration:
  E-step: Recompute posterior gamma_ch given current noise params and priors.
  M-step: Re-estimate alpha_h, beta_h, p_h from current gamma_ch.
          Optionally recompute similarity S and propagated labels.

Convergence: max|gamma^(t+1) - gamma^(t)| < tol.

The key insight is that single-pass Level 1 → 2 → 3 may produce
inconsistent estimates because each level consumes the predecessor's
output without revision. The EM loop allows Level 3 propagation
evidence to flow back into Level 1 posteriors, yielding globally
self-consistent label assignments.

Typical behaviour: 3-5 iterations suffice. The similarity matrix S
changes minimally after iteration 1, so it can be recomputed every K
iterations (em_recompute_S_every) rather than every iteration.
"""
import numpy as np
import pandas as pd
from src.config import PipelineConfig
from src.level1 import estimate_noise_params, compute_posteriors, binarize_calls
from src.level2 import run_level2
from src.level3 import run_level3
from src.data_loader import compute_global_binder_rates
from src.io_utils import save_df
import time
import gc


def run_em(
    agg: pd.DataFrame,
    n_hla: int,
    p_h: np.ndarray,
    hla_names: np.ndarray,
    cfg: PipelineConfig,
    skip_level3: bool = False,
) -> tuple:
    """
    Run the iterative EM refinement loop over Levels 1-3.

    On each iteration:
      1. E-step: compute gamma_ch = P(theta_ch=1 | data, alpha, beta, p_h)
      2. M-step:
         a. Re-estimate alpha_h, beta_h from current gammas
         b. Re-estimate p_h from current gammas (with shrinkage)
         c. Re-binarize calls b_ch
         d. Every K iterations: recompute S via Level 2 and
            re-propagate labels via Level 3.

    Convergence is checked via max|delta gamma| < em_tol.

    Args:
        agg:           aggregated counts DataFrame (modified in-place)
        n_hla:         number of alleles
        p_h:           initial binder rates (from shrinkage)
        hla_names:     allele name array
        cfg:           pipeline configuration
        skip_level3:   if True, skip propagation in the EM loop

    Returns:
        (agg, S, pairwise_df, prop_df, p_h)
        where agg has updated gamma and b_call columns.
    """
    max_iter = cfg.em_max_iter
    tol = cfg.em_tol
    recompute_every = cfg.em_recompute_S_every

    print("\n" + "=" * 60)
    print(f"EM ITERATIVE REFINEMENT (max_iter={max_iter}, tol={tol})")
    print("=" * 60)

    t_em_start = time.time()

    # ── initial state ──
    # First pass has already been done by run_stage1 before calling us.
    # agg already has 'gamma' and 'b_call' from the initial Level 1.
    gamma_prev = agg["gamma"].values.astype(np.float64).copy()

    S = None
    pairwise_df = pd.DataFrame()
    prop_df = pd.DataFrame()

    for iteration in range(1, max_iter + 1):
        t_iter = time.time()
        print(f"\n--- EM Iteration {iteration}/{max_iter} ---")

        # ════════════════════════════════════════════════════════════
        # M-step (a): Re-estimate noise parameters from current labels
        # ════════════════════════════════════════════════════════════
        # Use current gamma to soft-assign clusters to binder/non-binder.
        # For the noise estimation, we use the binary calls (which come
        # from the previous iteration's gamma).
        alpha_h, beta_h = estimate_noise_params(agg, n_hla, cfg)

        # ════════════════════════════════════════════════════════════
        # M-step (b): Re-estimate binder rates from current gammas
        # ════════════════════════════════════════════════════════════
        # Use a soft version: p_h = (sum of gamma_ch + k*p_global) /
        #                            (n_observed + k)
        pos_soft = np.zeros(n_hla, dtype=np.float64)
        tot_obs = np.zeros(n_hla, dtype=np.float64)
        np.add.at(pos_soft, agg["hla_idx"].values,
                  agg["gamma"].values.astype(np.float64))
        np.add.at(tot_obs, agg["hla_idx"].values, 1.0)

        p_global = pos_soft.sum() / tot_obs.sum()
        k = cfg.shrinkage_k
        p_h = (pos_soft + k * p_global) / (tot_obs + k)
        p_h = np.clip(p_h, cfg.p_h_floor, cfg.p_h_ceil)

        print(f"  [M] Updated p_h: mean={p_h.mean():.4f}, "
              f"median={np.median(p_h):.4f}")

        # ════════════════════════════════════════════════════════════
        # E-step: Recompute posteriors gamma_ch
        # ════════════════════════════════════════════════════════════
        gamma_new = compute_posteriors(agg, alpha_h, beta_h, p_h)
        agg["gamma"] = gamma_new.astype(np.float32)

        # ── binarize with updated priors ──
        b_call = binarize_calls(gamma_new, agg["hla_idx"].values, p_h, cfg)
        agg["b_call"] = b_call

        # ════════════════════════════════════════════════════════════
        # Check convergence
        # ════════════════════════════════════════════════════════════
        delta = np.max(np.abs(gamma_new - gamma_prev))
        mean_delta = np.mean(np.abs(gamma_new - gamma_prev))
        gamma_prev = gamma_new.copy()

        print(f"  [E] max|delta gamma| = {delta:.6f}, "
              f"mean|delta| = {mean_delta:.6f}")

        if delta < tol:
            print(f"  *** Converged at iteration {iteration} "
                  f"(delta {delta:.6f} < tol {tol}) ***")
            # run final Level 2 + 3
            S, pairwise_df = run_level2(agg, n_hla, hla_names, cfg)
            gc.collect()
            if not skip_level3:
                prop_df = run_level3(agg, S, n_hla, p_h, hla_names, cfg)
                gc.collect()
            break

        # ════════════════════════════════════════════════════════════
        # Optionally recompute S and propagate (every K iterations)
        # ════════════════════════════════════════════════════════════
        if iteration % recompute_every == 0 or iteration == max_iter:
            S, pairwise_df = run_level2(agg, n_hla, hla_names, cfg)
            gc.collect()

            if not skip_level3:
                prop_df = run_level3(agg, S, n_hla, p_h, hla_names, cfg)
                gc.collect()

        elapsed_iter = time.time() - t_iter
        print(f"  Iteration {iteration} done ({elapsed_iter:.1f}s)")

    else:
        # max_iter reached without convergence
        print(f"\n  WARNING: EM did not converge in {max_iter} iterations "
              f"(final delta = {delta:.6f})")
        # ensure we have S and pairwise_df from the last recompute
        if S is None:
            S, pairwise_df = run_level2(agg, n_hla, hla_names, cfg)
            if not skip_level3:
                prop_df = run_level3(agg, S, n_hla, p_h, hla_names, cfg)

    # ── save final EM outputs ──
    out = cfg.output_dir / "level1"
    noise_df = pd.DataFrame({
        "hla_idx": np.arange(n_hla),
        "alpha": alpha_h,
        "beta": beta_h,
        "p_h": p_h,
        "tau_h": np.minimum(cfg.tau_multiplier * p_h, cfg.tau_max),
    })
    noise_df.to_csv(out / "noise_params_em_final.csv", index=False)
    save_df(agg, out / "level1_results_em_final.parquet")

    elapsed_total = time.time() - t_em_start
    print(f"\n[EM] Total EM time: {elapsed_total:.1f}s")

    return agg, S, pairwise_df, prop_df, p_h