"""
Data loading and preprocessing.

Loads observations (parquet/csv/tsv), anchor-cluster output, and builds
aggregated (cluster, HLA) count matrices for downstream pipeline levels.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PipelineConfig
import time
import gc


def _load_tabular(path: Path, columns: list = None) -> pd.DataFrame:
    """
    Load a tabular file, auto-detecting format from extension.
    Supports: .parquet, .csv, .tsv, .txt (tab-separated)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    elif suffix == ".csv":
        df = pd.read_csv(path, usecols=columns)
        return df
    elif suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t", usecols=columns)
        return df
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}' for {path}. "
            f"Use .parquet, .csv, or .tsv"
        )


def load_observations(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Load observation data and filter to HLA alleles.

    Expected columns (configurable via cfg):
      - peptide_col  : peptide sequence (default: 'long_mer')
      - allele_col   : HLA allele name  (default: 'allele')
      - label_col    : binding label 0/1 (default: 'assigned_label')
    """
    print(f"[data] Loading observations: {cfg.observations_path}")
    t0 = time.time()

    df = _load_tabular(cfg.observations_path, columns=cfg.load_columns)
    print(f"  Raw rows: {len(df):,}  ({time.time()-t0:.1f}s)")

    # rename to canonical internal names if user columns differ
    col_map = {}
    if cfg.peptide_col != "peptide":
        col_map[cfg.peptide_col] = "peptide"
    if cfg.allele_col != "allele":
        col_map[cfg.allele_col] = "allele"
    if cfg.label_col != "label":
        col_map[cfg.label_col] = "label"
    if col_map:
        df = df.rename(columns=col_map)

    # filter to HLA alleles only
    mask = df["allele"].str.startswith(cfg.hla_prefix)
    df = df[mask].reset_index(drop=True)
    print(f"  After HLA filter: {len(df):,}")

    # ensure label is int8 (0 or 1)
    df["label"] = df["label"].fillna(0).astype(np.int8)

    # encode alleles as categorical integers for memory
    df["allele"] = df["allele"].astype("category")

    gc.collect()
    return df


def load_cluster_mapping(cfg: PipelineConfig) -> dict:
    """
    Load anchor_cluster.py output → dict mapping peptide_seq → cluster_id (int).

    Reads clusters.tsv with columns:
      cluster_id | representative_anchor | peptide_header | sequence | anchor

    Returns: dict {sequence_str: cluster_id_int}
    """
    cluster_tsv = cfg.cluster_tsv
    print(f"[data] Loading cluster mapping: {cluster_tsv}")
    t0 = time.time()

    df = pd.read_csv(cluster_tsv, sep="\t", usecols=["cluster_id", "sequence"])

    # parse numeric id from "cluster_0", "cluster_1", etc.
    df["cluster_id_num"] = (
        df["cluster_id"]
        .str.split("_", n=1)
        .str[1]
        .astype(np.int32)
    )

    # build mapping: sequence → cluster_id (int)
    # if duplicates exist (same sequence, same cluster), first wins
    pep_to_cluster = dict(zip(df["sequence"], df["cluster_id_num"]))

    n_clusters = df["cluster_id_num"].nunique()
    print(f"  Clusters: {n_clusters:,}  "
          f"Peptides mapped: {len(pep_to_cluster):,}  "
          f"({time.time()-t0:.1f}s)")

    del df
    gc.collect()
    return pep_to_cluster


def build_aggregated_counts(
    df: pd.DataFrame,
    pep_to_cluster: dict,
    cfg: PipelineConfig,
) -> tuple:
    """
    Aggregate peptide-level observations to (cluster, HLA) counts.

    Returns:
        agg: DataFrame with columns [cluster_id, hla_idx, n_pos, n_neg, n_total]
        hla_names: ndarray of HLA allele strings (index = hla_idx)
    """
    print("[data] Building aggregated counts...")
    t0 = time.time()

    # map peptides to cluster ids (drop unmapped)
    cluster_ids = df["peptide"].map(pep_to_cluster)
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
    grouped = df.groupby(["cluster_id", "hla_idx"], observed=True)["label"]
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

    np.add.at(pos_per_hla, agg["hla_idx"].values, agg["n_pos"].values)
    np.add.at(tot_per_hla, agg["hla_idx"].values, agg["n_total"].values)

    p_h = np.divide(pos_per_hla, tot_per_hla,
                     out=np.full(n_hla, 0.05), where=tot_per_hla > 0)

    print(f"[data] Global binder rates: mean={p_h.mean():.4f}, "
          f"median={np.median(p_h):.4f}, range=[{p_h.min():.4f}, {p_h.max():.4f}]")
    return p_h
