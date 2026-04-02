"""
Level 3: Label Propagation to Unobserved Pairs.

Propagates binding labels from well-characterised alleles to
under-represented ones using the similarity weights from Level 2.

Solution 1 (Coverage Penalty): The confidence weight lambda_ch is
penalized by the cluster coverage density rho_c = n_c / |c|.
Clusters where only a tiny fraction of peptides have labels contribute
less confidence to propagated estimates, preventing the framework from
extrapolating high confidence from sparse, potentially non-representative
evidence within large diffuse clusters.

Equations reference: Eq. 15-19 from the paper.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from src.config import PipelineConfig
import time
import gc


def build_propagation_weights(
    S: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Build propagation weight matrix W_{hh'} (Eq. 15).
    w_{hh'} = log(OR_{hh'}) if OR > 1 AND significant (S > 0), else 0.
    S already stores log(OR) for significant pairs, 0 otherwise.
    We only keep positive associations (S > 0 means OR > 1).
    """
    print("[L3] Building propagation weights...")
    W = np.where(S > 0, S, 0.0)  # only positive associations
    n_nonzero = np.count_nonzero(W)
    print(f"  Non-zero weights: {n_nonzero} "
          f"(out of {W.shape[0]*(W.shape[0]-1)//2} pairs)")
    return W


def identify_target_hlas(
    agg: pd.DataFrame,
    n_hla: int,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Identify under-represented alleles for label propagation.
    Returns boolean mask of shape (n_hla,).
    """
    obs_per_hla = np.zeros(n_hla, dtype=np.int64)
    np.add.at(obs_per_hla, agg["hla_idx"].values, 1)

    if cfg.propagate_only_rare:
        target = obs_per_hla < cfg.rare_hla_max_obs
    else:
        target = np.ones(n_hla, dtype=bool)

    n_target = target.sum()
    print(f"[L3] Target alleles for propagation: {n_target} / {n_hla}")
    if n_target > 0:
        print(f"  Obs range in targets: "
              f"[{obs_per_hla[target].min()}, {obs_per_hla[target].max()}]")
    return target


def _compute_coverage_penalty(
    agg: pd.DataFrame,
    cluster_sizes: dict,
    n_clusters: int,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Solution 1: Coverage-penalized confidence weighting.

    Compute f(rho_c) = rho_c^tau for each cluster, where:
      rho_c = n_c / |c|   (fraction of cluster peptides that have labels)
      tau = coverage_penalty_tau  (default 0.5 = square root dampening)

    Returns:
        ndarray of shape (n_clusters,) with penalty values in (0, 1].
    """
    tau = cfg.coverage_penalty_tau

    # total labeled observations per cluster (across all alleles)
    n_labeled = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(n_labeled, agg["cluster_id"].values, agg["n_total"].values)

    # cluster sizes from the full cluster mapping
    cluster_size_arr = np.ones(n_clusters, dtype=np.float64)  # default 1 (no penalty)
    for cid, size in cluster_sizes.items():
        if cid < n_clusters:
            cluster_size_arr[cid] = max(size, 1)

    # coverage density: fraction of peptides with any label
    rho = np.minimum(n_labeled / cluster_size_arr, 1.0)

    # power-law penalty
    penalty = np.power(rho, tau)

    n_penalized = np.sum(penalty < 0.5)
    print(f"[L3] Coverage penalty (tau={tau}): "
          f"mean={penalty.mean():.3f}, median={np.median(penalty):.3f}, "
          f"heavily penalized (<0.5): {n_penalized:,}/{n_clusters:,}")

    return penalty


def propagate_labels(
    agg: pd.DataFrame,
    W: np.ndarray,
    target_hlas: np.ndarray,
    n_hla: int,
    p_h: np.ndarray,
    cluster_sizes: dict,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Propagate binding labels to unobserved (cluster, allele) pairs (Eq. 16-19).

    For each target allele h' and each cluster c where h' has no observations
    but other alleles do:
      p_tilde_{ch'} = sum(w_{hh'} * gamma_{ch}) / sum(w_{hh'})
      n_tilde_{ch'} = sum(w_{hh'} * n_{ch} * f(rho_c))     ← coverage penalty
      lambda_{ch'} = n_tilde / (n_tilde + kappa_0)

    The coverage penalty f(rho_c) = rho_c^tau down-weights n_tilde
    for sparsely labeled clusters (Solution 1).

    Returns:
        DataFrame with [cluster_id, hla_idx, p_tilde, n_tilde, lambda_w]
    """
    print("[L3] Propagating labels...")
    t0 = time.time()

    n_clusters = agg["cluster_id"].max() + 1
    c_idx = agg["cluster_id"].values
    h_idx = agg["hla_idx"].values
    gamma_vals = agg["gamma"].values.astype(np.float64)
    n_total_vals = agg["n_total"].values.astype(np.float64)

    # ── Solution 1: compute coverage penalty per cluster ──
    coverage_penalty = _compute_coverage_penalty(
        agg, cluster_sizes, n_clusters, cfg)

    # penalized observation counts: n_ch * f(rho_c)
    penalized_n = n_total_vals * coverage_penalty[c_idx]

    # sparse matrix of gamma values: G[c,h] = gamma_ch
    G = sparse.csc_matrix(
        (gamma_vals, (c_idx, h_idx)),
        shape=(n_clusters, n_hla),
    )
    # sparse matrix of PENALIZED observation counts: Nobs[c,h] = n_ch * f(rho_c)
    Nobs = sparse.csc_matrix(
        (penalized_n, (c_idx, h_idx)),
        shape=(n_clusters, n_hla),
    )
    # observation indicator: O[c,h] = 1 if observed
    O = sparse.csc_matrix(
        (np.ones(len(c_idx), dtype=np.float64), (c_idx, h_idx)),
        shape=(n_clusters, n_hla),
    )

    target_idxs = np.where(target_hlas)[0]
    if len(target_idxs) == 0:
        print("  No target alleles → skipping propagation")
        return pd.DataFrame(columns=[
            "cluster_id", "hla_idx", "p_tilde", "n_tilde", "lambda_w"
        ])

    all_results = []
    batch_size = 10

    for batch_start in range(0, len(target_idxs), batch_size):
        batch_end = min(batch_start + batch_size, len(target_idxs))
        batch_targets = target_idxs[batch_start:batch_end]

        for h_prime in batch_targets:
            # source alleles with positive weight to h'
            w_vec = W[:, h_prime]
            sources = np.where(w_vec > 0)[0]
            if len(sources) == 0:
                continue

            w_sources = w_vec[sources]

            G_sub = G[:, sources]
            Nobs_sub = Nobs[:, sources]
            O_sub = O[:, sources]

            # weighted sums across source alleles
            wg = G_sub @ w_sources       # numerator of p_tilde
            wsum = O_sub @ w_sources     # denominator of p_tilde
            wn = Nobs_sub @ w_sources    # n_tilde (coverage-penalized)

            # valid: h' not observed, at least one source observed
            h_prime_observed = np.array(O[:, h_prime].todense()).ravel() > 0
            has_source = wsum > 0
            valid = (~h_prime_observed) & has_source
            valid_idx = np.where(valid)[0]

            if len(valid_idx) == 0:
                continue

            p_tilde = wg[valid_idx] / wsum[valid_idx]
            n_tilde = wn[valid_idx]
            lambda_w = n_tilde / (n_tilde + cfg.kappa_0)

            res = pd.DataFrame({
                "cluster_id": valid_idx.astype(np.int32),
                "hla_idx": np.full(len(valid_idx), h_prime, dtype=np.int16),
                "p_tilde": p_tilde.astype(np.float32),
                "n_tilde": n_tilde.astype(np.float32),
                "lambda_w": lambda_w.astype(np.float32),
            })
            all_results.append(res)

        if (batch_start // batch_size) % 5 == 0:
            elapsed = time.time() - t0
            pct = 100 * batch_end / len(target_idxs)
            print(f"  Progress: {batch_end}/{len(target_idxs)} "
                  f"target alleles ({pct:.0f}%, {elapsed:.1f}s)")

    if not all_results:
        print("  WARNING: No labels propagated!")
        return pd.DataFrame(columns=[
            "cluster_id", "hla_idx", "p_tilde", "n_tilde", "lambda_w"
        ])

    prop_df = pd.concat(all_results, ignore_index=True)
    del all_results, G, Nobs, O
    gc.collect()

    n_positive = (prop_df["p_tilde"] > 0.5).sum()
    print(f"  Propagated pairs: {len(prop_df):,}")
    print(f"  Positive-leaning (p_tilde > 0.5): {n_positive:,}")
    print(f"  Mean lambda: {prop_df['lambda_w'].mean():.4f}")
    print(f"  ({time.time()-t0:.1f}s)")

    return prop_df


def run_level3(
    agg: pd.DataFrame,
    S: np.ndarray,
    n_hla: int,
    p_h: np.ndarray,
    hla_names: np.ndarray,
    cluster_sizes: dict,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Full Level 3 pipeline:
      1. Build propagation weights from similarity matrix
      2. Identify target (under-represented) alleles
      3. Propagate labels with coverage-penalized confidence

    Returns:
        DataFrame of propagated labels.
    """
    print("\n" + "=" * 60)
    print("LEVEL 3: Label Propagation")
    print("=" * 60)

    W = build_propagation_weights(S, cfg)
    target_hlas = identify_target_hlas(agg, n_hla, cfg)
    prop_df = propagate_labels(
        agg, W, target_hlas, n_hla, p_h, cluster_sizes, cfg)

    # save outputs
    out = cfg.output_dir / "level3"
    if not prop_df.empty:
        from src.io_utils import save_df
        save_df(prop_df, out / "propagated_labels.parquet")

        summary = prop_df.groupby("hla_idx").agg(
            n_propagated=("p_tilde", "count"),
            mean_p_tilde=("p_tilde", "mean"),
            mean_lambda=("lambda_w", "mean"),
            n_positive=("p_tilde", lambda x: (x > 0.5).sum()),
        ).reset_index()
        summary["hla_name"] = hla_names[summary["hla_idx"].values]
        summary.to_csv(out / "propagation_summary.csv", index=False)

    print(f"[L3] Saved to {out}/")
    return prop_df