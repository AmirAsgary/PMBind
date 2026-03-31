"""
Data loading and preprocessing.
Loads parquet observations, MMseqs2 clusters, and builds aggregated
(cluster, HLA) count matrices for downstream pipeline levels.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PipelineConfig
import time
import gc
def load_observations(cfg: PipelineConfig) -> pd.DataFrame:
    """Load parquet with only needed columns, filter to HLA alleles."""
    print(f"[data] Loading parquet: {cfg.parquet_path}")
    t0 = time.time()
    df = pd.read_parquet(cfg.parquet_path, columns=cfg.parquet_columns)
    print(f"  Raw rows: {len(df):,}  ({time.time()-t0:.1f}s)")
    # filter to HLA alleles only
    mask = df["allele"].str.startswith(cfg.hla_prefix)
    df = df[mask].reset_index(drop=True)
    print(f"  After HLA filter: {len(df):,}")
    # ensure label is int8 (0 or 1)
    df["assigned_label"] = df["assigned_label"].fillna(0).astype(np.int8)
    # encode alleles as categorical integers for memory
    df["allele"] = df["allele"].astype("category")
    gc.collect()
    return df
def load_cluster_mapping(cfg: PipelineConfig) -> dict:
    """
    Load MMseqs2 cluster TSV -> dict mapping peptide_seq -> cluster_id (int).
    Format: representative_seq <TAB> member_seq (one line per member).
    """
    print(f"[data] Loading cluster TSV: {cfg.cluster_tsv}")
    t0 = time.time()
    # read with chunks to handle large files
    rep_to_id = {}
    pep_to_cluster = {}
    next_id = 0
    with open(cfg.cluster_tsv, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts
            # assign numeric cluster id based on representative
            if rep not in rep_to_id:
                rep_to_id[rep] = next_id
                next_id += 1
            pep_to_cluster[member] = rep_to_id[rep]
    print(f"  Clusters: {next_id:,}  Peptides mapped: {len(pep_to_cluster):,}"
          f"  ({time.time()-t0:.1f}s)")
    return pep_to_cluster
def build_aggregated_counts(
    df: pd.DataFrame,
    pep_to_cluster: dict,
    cfg: PipelineConfig,
) -> tuple:
    """
    Aggregate peptide-level observations to (cluster, HLA) counts.
    Returns:
        agg: DataFrame with columns [cluster_id, hla_idx, n_pos, n_neg]
        hla_names: ndarray of HLA allele strings (index = hla_idx)
    """
    print("[data] Building aggregated counts...")
    t0 = time.time()
    # map peptides to cluster ids (drop unmapped)
    cluster_ids = df["long_mer"].map(pep_to_cluster)
    valid = cluster_ids.notna()
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"  Warning: {n_dropped:,} peptides not found in cluster mapping")
    df = df[valid].copy()
    df["cluster_id"] = cluster_ids[valid].astype(np.int32)
    gc.collect()
    # encode HLAs as integer indices
    hla_cat = df["allele"].cat
    hla_names = np.array(hla_cat.categories)
    df["hla_idx"] = hla_cat.codes.astype(np.int16)
    # aggregate: count binders and non-binders per (cluster, HLA)
    grouped = df.groupby(["cluster_id", "hla_idx"], observed=True)["assigned_label"]
    n_pos = grouped.sum().astype(np.int32)
    n_total = grouped.count().astype(np.int32)
    # build result dataframe
    agg = pd.DataFrame({
        "cluster_id": n_pos.index.get_level_values(0),
        "hla_idx": n_pos.index.get_level_values(1),
        "n_pos": n_pos.values,
        "n_neg": (n_total.values - n_pos.values).astype(np.int32),
    })
    agg["n_total"] = agg["n_pos"] + agg["n_neg"]
    print(f"  Aggregated pairs: {len(agg):,}  "
          f"HLAs: {len(hla_names)}  ({time.time()-t0:.1f}s)")
    del df
    gc.collect()
    return agg, hla_names
def compute_global_binder_rates(agg: pd.DataFrame, n_hla: int) -> np.ndarray:
    """
    Compute p_h = global binder rate for each HLA.
    Returns: ndarray of shape (n_hla,) with p_h values.
    """
    pos_per_hla = np.zeros(n_hla, dtype=np.float64)
    tot_per_hla = np.zeros(n_hla, dtype=np.float64)
    # vectorized accumulation
    np.add.at(pos_per_hla, agg["hla_idx"].values, agg["n_pos"].values)
    np.add.at(tot_per_hla, agg["hla_idx"].values, agg["n_total"].values)
    # avoid division by zero
    p_h = np.divide(pos_per_hla, tot_per_hla,
                     out=np.full(n_hla, 0.05), where=tot_per_hla > 0)
    print(f"[data] Global binder rates: mean={p_h.mean():.4f}, "
          f"median={np.median(p_h):.4f}, range=[{p_h.min():.4f}, {p_h.max():.4f}]")
    return p_h
