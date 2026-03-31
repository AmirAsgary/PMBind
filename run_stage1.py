#!/usr/bin/env python3
"""
Run the full Stage 1 statistical pipeline:
  Level 1 → Binary noise model (gamma_ch posteriors)
  Level 2 → Pairwise HLA similarity (S_hh' matrix)
  Level 3 → Label propagation (p_tilde for rare HLAs)

Usage:
  python run_stage1.py \\
      --observations data/PMDb_class1.parquet \\
      --cluster-dir  anchor_clusters/

  python run_stage1.py \\
      --observations data/observations.csv \\
      --cluster-dir  anchor_clusters/ \\
      --peptide-col  peptide_seq \\
      --allele-col   hla \\
      --label-col    binder \\
      --skip-level3

Inputs:
  --observations : Parquet, CSV, or TSV with peptide observations.
                   Must contain columns for peptide sequence, HLA allele,
                   and binding label (column names are configurable).
  --cluster-dir  : Output directory from anchor_cluster.py.
                   Must contain clusters.tsv.

Outputs (written to <cluster-dir>/stage1/):
  stage1/level1/  — noise parameters, posterior gamma values
  stage1/level2/  — pairwise HLA tests, similarity matrix
  stage1/level3/  — propagated labels for rare HLAs
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: Statistical framework for immunopeptidome denoising",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── required paths ──
    p.add_argument(
        "--observations", required=True, type=Path,
        help="Path to observation file (.parquet, .csv, or .tsv)")
    p.add_argument(
        "--cluster-dir", required=True, type=Path,
        help="Path to anchor_cluster.py output directory (contains clusters.tsv)")

    # ── column names (for non-standard inputs) ──
    p.add_argument(
        "--peptide-col", default="long_mer",
        help="Column name for peptide sequences (default: long_mer)")
    p.add_argument(
        "--allele-col", default="allele",
        help="Column name for HLA alleles (default: allele)")
    p.add_argument(
        "--label-col", default="assigned_label",
        help="Column name for binding labels (default: assigned_label)")

    # ── filtering & compute ──
    p.add_argument(
        "--hla-prefix", default="HLA",
        help="Keep only alleles starting with this prefix (default: HLA)")
    p.add_argument(
        "--n-jobs", type=int, default=12,
        help="Number of parallel workers for Fisher tests (default: 12)")
    p.add_argument(
        "--fdr", type=float, default=0.05,
        help="FDR threshold for BH correction (default: 0.05)")
    p.add_argument(
        "--min-shared", type=int, default=10,
        help="Minimum shared clusters for HLA pair testing (default: 10)")
    p.add_argument(
        "--rare-max-obs", type=int, default=5000,
        help="Max observations for an HLA to be considered 'rare' (default: 5000)")

    # ── control ──
    p.add_argument(
        "--skip-level3", action="store_true",
        help="Skip label propagation (Level 3)")

    return p.parse_args()


def validate_inputs(args):
    """Check that required files exist before starting the pipeline."""
    if not args.observations.exists():
        raise FileNotFoundError(f"Observations file not found: {args.observations}")
    if not args.cluster_dir.exists():
        raise FileNotFoundError(f"Cluster directory not found: {args.cluster_dir}")

    cluster_tsv = args.cluster_dir / "clusters.tsv"
    if not cluster_tsv.exists():
        raise FileNotFoundError(
            f"clusters.tsv not found in {args.cluster_dir}. "
            f"Run anchor_cluster.py first."
        )


def main():
    args = parse_args()
    validate_inputs(args)

    # ── configure ──
    cfg = PipelineConfig(
        observations_path=args.observations,
        cluster_dir=args.cluster_dir,
        peptide_col=args.peptide_col,
        allele_col=args.allele_col,
        label_col=args.label_col,
        hla_prefix=args.hla_prefix,
        n_jobs=args.n_jobs,
        fdr_threshold=args.fdr,
        min_shared_clusters=args.min_shared,
        rare_hla_max_obs=args.rare_max_obs,
    )

    print("Pipeline Configuration:")
    print(f"  Observations:      {cfg.observations_path}")
    print(f"  Cluster dir:       {cfg.cluster_dir}")
    print(f"  Output dir:        {cfg.output_dir}")
    print(f"  Columns:           peptide={cfg.peptide_col}, "
          f"allele={cfg.allele_col}, label={cfg.label_col}")
    print(f"  HLA prefix filter: {cfg.hla_prefix}")
    print(f"  FDR threshold:     {cfg.fdr_threshold}")
    print(f"  Parallel workers:  {cfg.n_jobs}")

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

    del df, pep_to_cluster
    gc.collect()

    n_hla = len(hla_names)
    p_h = compute_global_binder_rates(agg, n_hla)

    # ══════════════════════════════════════════════════════════════
    # LEVEL 1: Binary Noise Model
    # ══════════════════════════════════════════════════════════════
    agg, alpha_h, beta_h = run_level1(agg, n_hla, p_h, cfg)
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # LEVEL 2: HLA Similarity
    # ══════════════════════════════════════════════════════════════
    S, pairwise_df = run_level2(agg, n_hla, hla_names, cfg)
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # LEVEL 3: Label Propagation
    # ══════════════════════════════════════════════════════════════
    prop_df = None
    if not args.skip_level3:
        prop_df = run_level3(agg, S, n_hla, p_h, hla_names, cfg)
        gc.collect()

    # ── summary ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"STAGE 1 COMPLETE  ({elapsed:.0f}s / {elapsed / 60:.1f}min)")
    print("=" * 60)
    print(f"  Observed pairs (D_obs):     {len(agg):,}")
    if prop_df is not None and not prop_df.empty:
        print(f"  Propagated pairs (D_prop):  {len(prop_df):,}")
    print(f"  HLAs:                       {n_hla}")
    n_sig = (
        pairwise_df["significant"].sum()
        if not pairwise_df.empty else 0
    )
    print(f"  Significant HLA pairs:      {n_sig:,}")
    print(f"\nOutputs in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()