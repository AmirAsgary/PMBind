#!/usr/bin/env python3
"""
Run the full Stage 1 statistical pipeline:
  Level 1 → Binary noise model (gamma_ch posteriors)
  Level 2 → Pairwise HLA similarity (S_hh' matrix)
  Level 3 → Label propagation (p_tilde for rare HLAs)
Usage:
  python run_stage1.py                     # defaults (80% identity)
  python run_stage1.py --identity 60       # use 60% clustering
  python run_stage1.py --skip-level3       # only run L1 + L2
"""
import argparse
import time
import gc
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
def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Statistical framework for immunopeptidome denoising"
    )
    parser.add_argument("--identity", type=int, default=80,
                        choices=[60, 80],
                        help="MMseqs2 clustering identity (60 or 80)")
    parser.add_argument("--skip-level3", action="store_true",
                        help="Skip label propagation (Level 3)")
    parser.add_argument("--hla-prefix", type=str, default="HLA",
                        help="Filter alleles starting with this prefix")
    parser.add_argument("--n-jobs", type=int, default=12,
                        help="Number of parallel workers")
    parser.add_argument("--fdr", type=float, default=0.05,
                        help="FDR threshold for BH correction")
    parser.add_argument("--min-shared", type=int, default=10,
                        help="Minimum shared clusters for HLA pair testing")
    parser.add_argument("--rare-max-obs", type=int, default=5000,
                        help="Max observations for an HLA to be 'rare'")
    args = parser.parse_args()
    # ── configure ──
    cfg = PipelineConfig(
        cluster_identity=args.identity,
        hla_prefix=args.hla_prefix,
        n_jobs=args.n_jobs,
        fdr_threshold=args.fdr,
        min_shared_clusters=args.min_shared,
        rare_hla_max_obs=args.rare_max_obs,
    )
    print("Pipeline Configuration:")
    print(f"  Cluster identity:  {cfg.cluster_identity}%")
    print(f"  HLA prefix filter: {cfg.hla_prefix}")
    print(f"  FDR threshold:     {cfg.fdr_threshold}")
    print(f"  Parallel workers:  {cfg.n_jobs}")
    print(f"  Output dir:        {cfg.output_dir}")
    t_start = time.time()
    # ══════════════════════════════════════════════════════════════
    # DATA LOADING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    df = load_observations(cfg)
    pep_to_cluster = load_cluster_mapping(cfg)
    agg, hla_names = build_aggregated_counts(df, pep_to_cluster, cfg)
    del df, pep_to_cluster  # free ~4-6 GB
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
    if not args.skip_level3:
        prop_df = run_level3(agg, S, n_hla, p_h, hla_names, cfg)
        gc.collect()
    # ── summary ──
    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print(f"STAGE 1 COMPLETE  ({elapsed:.0f}s / {elapsed/60:.1f}min)")
    print("="*60)
    print(f"  Observed pairs (D_obs):     {len(agg):,}")
    if not args.skip_level3 and not prop_df.empty:
        print(f"  Propagated pairs (D_prop):  {len(prop_df):,}")
    print(f"  HLAs:                       {n_hla}")
    print(f"  Significant HLA pairs:      "
          f"{pairwise_df['significant'].sum() if not pairwise_df.empty else 0:,}")
    print(f"\nOutputs in: {cfg.output_dir}/")
if __name__ == "__main__":
    main()
