#!/usr/bin/env python3
"""
Run the full Stage 1 statistical pipeline:
  Level 1  → Binary noise model (gamma_ch posteriors)
  Diag     → Cluster purity, per-HLA metrics, visualisations
  Level 2  → Pairwise HLA similarity + matrices + heatmaps
  Level 3  → Label propagation (p_tilde for rare HLAs)
  EM       → (optional) Iterative refinement until convergence

Usage:
  # Basic run (single pass)
  python run_stage1.py \\
      --observations data/PMDb_class1.parquet \\
      --cluster-dir  anchor_clusters/

  # With EM refinement (5 iterations max)
  python run_stage1.py \\
      --observations data/PMDb_class1.parquet \\
      --cluster-dir  anchor_clusters/ \\
      --em-iter 5

  # Non-human alleles, custom columns
  python run_stage1.py \\
      --observations data/all_species.csv \\
      --cluster-dir  anchor_clusters/ \\
      --no-allele-filter \\
      --peptide-col  peptide_seq \\
      --allele-col   mhc \\
      --label-col    binder

Inputs:
  --observations : Parquet, CSV, or TSV with peptide observations.
                   Must contain columns for peptide sequence, allele,
                   and binding label (column names are configurable).
  --cluster-dir  : Output directory from anchor_cluster.py.
                   Must contain clusters.tsv.

Outputs (written to <cluster-dir>/stage1/):
  stage1/level1/      — noise parameters, posterior gamma values
  stage1/level2/      — pairwise HLA tests, similarity matrix, heatmaps
  stage1/level3/      — propagated labels for rare HLAs
  stage1/diagnostics/ — purity metrics, cluster/HLA plots
"""

import argparse
import time
import gc
from pathlib import Path

from src.config import PipelineConfig
from src.data_loader import (
    load_observations,
    load_cluster_mapping,
    build_aggregated_counts,
    compute_global_binder_rates,
)
from src.level1 import run_level1
from src.level2 import run_level2
from src.level3 import run_level3
from src.diagnostics import run_diagnostics
from src.em_wrapper import run_em


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: Statistical framework for immunopeptidome denoising",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── required paths ──
    p.add_argument(
        "--observations", required=True, type=Path,
        help="Path to observation file (.parquet, .csv, or .tsv)")
    p.add_argument(
        "--cluster-dir", required=True, type=Path,
        help="Path to anchor_cluster.py output directory (contains clusters.tsv)")

    # ── column names (for non-standard input files) ──
    p.add_argument("--peptide-col", default="long_mer",
                    help="Column name for peptide sequences (default: long_mer)")
    p.add_argument("--allele-col", default="allele",
                    help="Column name for allele names (default: allele)")
    p.add_argument("--label-col", default="assigned_label",
                    help="Column name for binding labels (default: assigned_label)")

    # ── allele filtering ──
    p.add_argument(
        "--allele-prefix", default=None,
        help="Keep alleles starting with prefix (default: no filter, keep all)")
    p.add_argument(
        "--no-allele-filter", action="store_true",
        help="Explicitly disable allele filtering (same as omitting --allele-prefix)")

    # ── p_h regularisation ──
    p.add_argument("--shrinkage-k", type=float, default=50.0,
                    help="Beta-prior pseudocount strength (default: 50)")
    p.add_argument("--p-h-ceil", type=float, default=0.95,
                    help="Max allowed p_h after shrinkage (default: 0.95)")
    p.add_argument("--tau-max", type=float, default=0.50,
                    help="Hard cap on binarisation threshold tau_h (default: 0.50)")

    # ── Level 2 ──
    p.add_argument("--n-jobs", type=int, default=12,
                    help="Parallel workers for Fisher tests (default: 12)")
    p.add_argument("--fdr", type=float, default=0.05,
                    help="FDR threshold for BH correction (default: 0.05)")
    p.add_argument("--min-shared", type=int, default=10,
                    help="Min shared clusters for HLA pair testing (default: 10)")

    # ── Level 3 ──
    p.add_argument("--rare-max-obs", type=int, default=5000,
                    help="Max observations for an allele to be 'rare' (default: 5000)")
    p.add_argument("--skip-level3", action="store_true",
                    help="Skip label propagation (Level 3)")

    # ── EM iterative refinement ──
    p.add_argument("--em-iter", type=int, default=0,
                    help="Number of EM iterations (0 = single pass, no EM). "
                         "Recommended: 3-5 for refinement.")
    p.add_argument("--em-tol", type=float, default=1e-3,
                    help="EM convergence tolerance (default: 1e-3)")
    p.add_argument("--em-recompute-S", type=int, default=3,
                    help="Recompute similarity matrix every K EM iterations "
                         "(default: 3). Set to 1 for full recomputation.")

    return p.parse_args()


def validate_inputs(args):
    """Check that required files exist before starting the pipeline."""
    if not args.observations.exists():
        raise FileNotFoundError(
            f"Observations file not found: {args.observations}")
    if not args.cluster_dir.exists():
        raise FileNotFoundError(
            f"Cluster directory not found: {args.cluster_dir}")

    cluster_tsv = args.cluster_dir / "clusters.tsv"
    if not cluster_tsv.exists():
        raise FileNotFoundError(
            f"clusters.tsv not found in {args.cluster_dir}. "
            f"Run anchor_cluster.py first.")


def main():
    args = parse_args()
    validate_inputs(args)

    # ── resolve allele prefix ──
    allele_prefix = args.allele_prefix
    if args.no_allele_filter:
        allele_prefix = None

    # ── build configuration ──
    cfg = PipelineConfig(
        observations_path=args.observations,
        cluster_dir=args.cluster_dir,
        peptide_col=args.peptide_col,
        allele_col=args.allele_col,
        label_col=args.label_col,
        allele_prefix=allele_prefix,
        shrinkage_k=args.shrinkage_k,
        p_h_ceil=args.p_h_ceil,
        tau_max=args.tau_max,
        n_jobs=args.n_jobs,
        fdr_threshold=args.fdr,
        min_shared_clusters=args.min_shared,
        rare_hla_max_obs=args.rare_max_obs,
        em_max_iter=args.em_iter,
        em_tol=args.em_tol,
        em_recompute_S_every=args.em_recompute_S,
    )

    # ── print configuration ──
    print("Pipeline Configuration:")
    print(f"  Observations:      {cfg.observations_path}")
    print(f"  Cluster dir:       {cfg.cluster_dir}")
    print(f"  Output dir:        {cfg.output_dir}")
    print(f"  Columns:           peptide={cfg.peptide_col}, "
          f"allele={cfg.allele_col}, label={cfg.label_col}")
    print(f"  Allele filter:     {cfg.allele_prefix or 'disabled (all alleles)'}")
    print(f"  Shrinkage k:       {cfg.shrinkage_k}")
    print(f"  p_h ceil:          {cfg.p_h_ceil}")
    print(f"  tau_max:           {cfg.tau_max}")
    print(f"  FDR threshold:     {cfg.fdr_threshold}")
    print(f"  min_shared:        {cfg.min_shared_clusters}")
    print(f"  Parallel workers:  {cfg.n_jobs}")
    print(f"  EM iterations:     {cfg.em_max_iter} "
          f"({'single pass' if cfg.em_max_iter == 0 else 'iterative'})")
    if cfg.em_max_iter > 0:
        print(f"  EM tolerance:      {cfg.em_tol}")
        print(f"  EM recompute S:    every {cfg.em_recompute_S_every} iterations")

    t_start = time.time()

    # ══════════════════════════════════════════════════════════════
    # DATA LOADING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    df = load_observations(cfg)
    pep_to_cluster = load_cluster_mapping(cfg)
    agg, hla_names = build_aggregated_counts(df, pep_to_cluster, cfg)

    del df, pep_to_cluster  # free ~4-6 GB
    gc.collect()

    n_hla = len(hla_names)
    p_h = compute_global_binder_rates(agg, n_hla, cfg)

    # ══════════════════════════════════════════════════════════════
    # LEVEL 1: Binary Noise Model (initial pass)
    # ══════════════════════════════════════════════════════════════
    agg, alpha_h, beta_h = run_level1(agg, n_hla, p_h, cfg)
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # DIAGNOSTICS: Purity & Cluster Metrics
    # ══════════════════════════════════════════════════════════════
    run_diagnostics(agg, n_hla, hla_names, p_h, cfg)
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # BRANCH: single-pass vs EM iterative refinement
    # ══════════════════════════════════════════════════════════════
    prop_df = None

    if cfg.em_max_iter > 0:
        # ── EM iterative refinement ──
        # This loop re-runs Levels 1-3 until gamma converges.
        # Level 2 is recomputed every em_recompute_S_every iterations.
        agg, S, pairwise_df, prop_df, p_h = run_em(
            agg, n_hla, p_h, hla_names, cfg,
            skip_level3=args.skip_level3,
        )
        gc.collect()

    else:
        # ── single pass (original pipeline) ──

        # Level 2: HLA Similarity
        S, pairwise_df = run_level2(agg, n_hla, hla_names, cfg)
        gc.collect()

        # Level 3: Label Propagation
        if not args.skip_level3:
            prop_df = run_level3(agg, S, n_hla, p_h, hla_names, cfg)
            gc.collect()

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"STAGE 1 COMPLETE  ({elapsed:.0f}s / {elapsed / 60:.1f}min)")
    print("=" * 60)
    print(f"  Observed pairs (D_obs):     {len(agg):,}")
    if prop_df is not None and not prop_df.empty:
        print(f"  Propagated pairs (D_prop):  {len(prop_df):,}")
    print(f"  Alleles:                    {n_hla}")
    n_sig = (pairwise_df["significant"].sum()
             if not pairwise_df.empty else 0)
    print(f"  Significant allele pairs:   {n_sig:,}")
    if cfg.em_max_iter > 0:
        print(f"  EM iterations:              {cfg.em_max_iter}")
    print(f"\nOutputs in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()