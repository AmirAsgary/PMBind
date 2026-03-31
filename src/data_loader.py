"""
Data loading and preprocessing.

Loads observations (parquet/csv/tsv), anchor-cluster output, and builds
aggregated (cluster, HLA) count matrices for downstream pipeline levels.

Key functions:
  load_observations()      — Load + filter observation data
  load_cluster_mapping()   — Read anchor_cluster.py clusters.tsv
  build_aggregated_counts() — Aggregate to (cluster, allele) level
  compute_global_binder_rates() — Beta-prior shrinkage for p_h

The output of this module is the `agg` DataFrame with columns:
  [cluster_id, hla_idx, n_pos, n_neg, n_total]
which is consumed by all downstream levels.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PipelineConfig
import time
import gc


# ============================================================================
# File I/O helpers
# ============================================================================

def _load_tabular(path: Path, columns: list = None) -> pd.DataFrame:
    """
    Load a tabular file, auto-detecting format from extension.
    Supports: .parquet, .csv, .tsv, .txt (tab-separated).

    Args:
        path: file path
        columns: optional list of columns to load (for parquet efficiency)

    Returns:
        DataFrame with requested columns
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    elif suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    elif suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", usecols=columns)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}' for {path}. "
            f"Use .parquet, .csv, or .tsv"
        )


# ============================================================================
# Observation loading
# ============================================================================

def load_observations(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Load observation data and optionally filter by allele prefix.

    Expected columns (configurable via cfg):
      - peptide_col  : peptide sequence (default: 'long_mer')
      - allele_col   : allele name     (default: 'allele')
      - label_col    : binding label 0/1 (default: 'assigned_label')

    If cfg.allele_prefix is None, ALL alleles are kept (including
    non-human MHC). Set cfg.allele_prefix = "HLA" to restrict to
    human HLA allotypes.

    Returns:
        DataFrame with canonical columns: ['peptide', 'allele', 'label']
        where allele is categorical and label is int8.
    """
    print(f"[data] Loading observations: {cfg.observations_path}")
    t0 = time.time()

    df = _load_tabular(cfg.observations_path, columns=cfg.load_columns)
    print(f"  Raw rows: {len(df):,}  ({time.time()-t0:.1f}s)")

    # ── rename to canonical internal column names ──
    col_map = {cfg.peptide_col: "peptide", cfg.allele_col: "allele",
               cfg.label_col: "label"}
    # only rename if source ≠ target (avoids pandas warning)
    col_map = {k: v for k, v in col_map.items() if k != v}
    if col_map:
        df = df.rename(columns=col_map)

    # ── optional allele prefix filter ──
    # None = keep ALL alleles (human, mouse, etc.)
    if cfg.allele_prefix is not None:
        mask = df["allele"].str.startswith(cfg.allele_prefix)
        df = df[mask].reset_index(drop=True)
        print(f"  After allele filter ('{cfg.allele_prefix}*'): {len(df):,}")
    else:
        print(f"  Allele filter: disabled (keeping all alleles)")

    # ── normalise types ──
    df["label"] = df["label"].fillna(0).astype(np.int8)
    df["allele"] = df["allele"].astype("category")

    # ── report raw label distribution (sanity check) ──
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    print(f"  Labels: {n_pos:,} positive, {n_neg:,} negative "
          f"(raw rate: {n_pos/(n_pos+n_neg):.4f})")

    gc.collect()
    return df


# ============================================================================
# Cluster mapping
# ============================================================================

def load_cluster_mapping(cfg: PipelineConfig) -> dict:
    """
    Load anchor_cluster.py output → dict mapping peptide_seq → cluster_id (int).

    Reads clusters.tsv with columns:
      cluster_id | representative_anchor | peptide_header | sequence | anchor

    The cluster_id strings ("cluster_0", "cluster_1", ...) are parsed into
    integer ids for memory-efficient downstream processing.

    Returns:
        dict {sequence_string: cluster_id_int}
    """
    cluster_tsv = cfg.cluster_tsv
    print(f"[data] Loading cluster mapping: {cluster_tsv}")
    t0 = time.time()

    # only read the two columns we need
    df = pd.read_csv(cluster_tsv, sep="\t", usecols=["cluster_id", "sequence"])

    # parse "cluster_42" → 42
    df["cluster_id_num"] = (
        df["cluster_id"]
        .str.split("_", n=1)
        .str[1]
        .astype(np.int32)
    )

    # build the mapping: sequence → cluster id
    # if a sequence appears in multiple rows (same cluster), first wins
    pep_to_cluster = dict(zip(df["sequence"], df["cluster_id_num"]))

    n_clusters = df["cluster_id_num"].nunique()
    print(f"  Clusters: {n_clusters:,}  "
          f"Peptides mapped: {len(pep_to_cluster):,}  "
          f"({time.time()-t0:.1f}s)")

    del df
    gc.collect()
    return pep_to_cluster


# ============================================================================
# Aggregation
# ============================================================================

def build_aggregated_counts(
    df: pd.DataFrame,
    pep_to_cluster: dict,
    cfg: PipelineConfig,
) -> tuple:
    """
    Aggregate peptide-level observations to (cluster, allele) counts.

    For each (cluster, allele) pair, counts:
      n_pos  = number of binder labels
      n_neg  = number of non-binder labels
      n_total = n_pos + n_neg

    Peptides not found in the cluster mapping are dropped with a warning.

    Returns:
        agg: DataFrame with columns [cluster_id, hla_idx, n_pos, n_neg, n_total]
        hla_names: ndarray of allele strings (index = hla_idx)
    """
    print("[data] Building aggregated counts...")
    t0 = time.time()

    # ── map peptides to cluster ids ──
    cluster_ids = df["peptide"].map(pep_to_cluster)
    valid = cluster_ids.notna()
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        pct = 100 * n_dropped / len(df)
        print(f"  Warning: {n_dropped:,} peptides not in cluster mapping "
              f"({pct:.1f}% of data)")

    df = df[valid].copy()
    df["cluster_id"] = cluster_ids[valid].astype(np.int32)
    gc.collect()

    # ── encode alleles as integer indices ──
    hla_cat = df["allele"].cat
    hla_names = np.array(hla_cat.categories)
    df["hla_idx"] = hla_cat.codes.astype(np.int16)

    # ── aggregate: count binders and non-binders per (cluster, allele) ──
    grouped = df.groupby(["cluster_id", "hla_idx"], observed=True)["label"]
    n_pos = grouped.sum().astype(np.int32)
    n_total = grouped.count().astype(np.int32)

    agg = pd.DataFrame({
        "cluster_id": n_pos.index.get_level_values(0),
        "hla_idx": n_pos.index.get_level_values(1),
        "n_pos": n_pos.values,
        "n_neg": (n_total.values - n_pos.values).astype(np.int32),
    })
    agg["n_total"] = agg["n_pos"] + agg["n_neg"]

    # ── sanity check: aggregated totals should match input ──
    agg_total_pos = agg["n_pos"].sum()
    agg_total_neg = agg["n_neg"].sum()
    print(f"  Aggregated pairs: {len(agg):,}  HLAs: {len(hla_names)}")
    print(f"  Aggregated labels: {agg_total_pos:,} pos, {agg_total_neg:,} neg "
          f"(rate: {agg_total_pos/(agg_total_pos+agg_total_neg):.4f})")
    print(f"  ({time.time()-t0:.1f}s)")

    del df
    gc.collect()
    return agg, hla_names


# ============================================================================
# Global binder rates with Beta-prior shrinkage
# ============================================================================

def compute_global_binder_rates(
    agg: pd.DataFrame, n_hla: int, cfg: PipelineConfig,
) -> np.ndarray:
    """
    Compute p_h with Beta-prior shrinkage to prevent extreme values.

    The raw MLE p_h = n_pos_h / n_total_h can be degenerate for data-poor
    alleles: an allele with 3/3 positive observations gets p_h = 1.0,
    which makes the prior pi_h = 1 and the binarisation threshold
    tau_h = 2 (unreachable), silently disabling the noise model.

    We regularise with a Beta-prior:
        p_h = (n_pos_h + k * p_global) / (n_total_h + k)

    where p_global is the overall binder rate across ALL alleles
    and k is the shrinkage strength. For well-observed alleles
    (n_total >> k), the shrinkage is negligible. For data-poor alleles,
    p_h reverts toward p_global.

    After shrinkage, values are clamped to [p_h_floor, p_h_ceil].

    Args:
        agg: aggregated counts DataFrame
        n_hla: number of alleles
        cfg: pipeline configuration

    Returns:
        ndarray of shape (n_hla,) with regularised p_h values
    """
    # ── accumulate per-allele totals ──
    pos_per_hla = np.zeros(n_hla, dtype=np.float64)
    tot_per_hla = np.zeros(n_hla, dtype=np.float64)

    np.add.at(pos_per_hla, agg["hla_idx"].values, agg["n_pos"].values)
    np.add.at(tot_per_hla, agg["hla_idx"].values, agg["n_total"].values)

    # ── global binder rate (anchor for shrinkage) ──
    p_global = pos_per_hla.sum() / tot_per_hla.sum()

    # ── raw rates (for reporting only) ──
    p_h_raw = np.divide(pos_per_hla, tot_per_hla,
                         out=np.full(n_hla, p_global), where=tot_per_hla > 0)

    # ── Beta-prior shrinkage ──
    # This is the posterior mean of a Beta(k*p_global, k*(1-p_global)) prior
    k = cfg.shrinkage_k
    p_h = (pos_per_hla + k * p_global) / (tot_per_hla + k)

    # ── clamp to safe range ──
    p_h = np.clip(p_h, cfg.p_h_floor, cfg.p_h_ceil)

    # ── reporting ──
    n_raw_extreme = np.sum((p_h_raw > 0.95) | (p_h_raw < 0.05))
    print(f"[data] Global binder rate: {p_global:.4f}")
    print(f"  Raw p_h:  mean={p_h_raw.mean():.4f}, "
          f"median={np.median(p_h_raw):.4f}, "
          f"extreme (>0.95 or <0.05): {n_raw_extreme}/{n_hla}")
    print(f"  Shrunk p_h (k={k}): mean={p_h.mean():.4f}, "
          f"median={np.median(p_h):.4f}, "
          f"range=[{p_h.min():.4f}, {p_h.max():.4f}]")
    print(f"  Obs per allele: mean={tot_per_hla.mean():.0f}, "
          f"median={np.median(tot_per_hla):.0f}, "
          f"min={tot_per_hla.min():.0f}")

    return p_h