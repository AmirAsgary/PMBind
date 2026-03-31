"""
Configuration for the Cluster-Based Bayesian Label Denoising pipeline.

All paths, hyperparameters, and constants in one place.
Paths are set dynamically from CLI arguments — nothing is hardcoded.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # ── paths (set by CLI — no defaults) ──
    observations_path: Path = None   # parquet / csv / tsv with peptide observations
    cluster_dir: Path = None         # anchor_cluster.py output directory
    output_dir: Path = None          # set in __post_init__ → cluster_dir/stage1

    # ── HLA filtering ──
    hla_prefix: str = "HLA"

    # ── observation columns ──
    #   peptide_col : column containing peptide sequences (join key to clusters)
    #   allele_col  : column containing HLA allele names
    #   label_col   : column containing binding labels (0/1)
    peptide_col: str = "long_mer"
    allele_col: str = "allele"
    label_col: str = "assigned_label"

    # ── Level 1: binary noise model ──
    alpha_default: float = 0.10
    beta_default: float = 0.02
    noise_min_cluster_size: int = 5
    noise_purity_threshold: float = 0.90

    # ── Level 1: binarisation ──
    tau_multiplier: float = 2.0

    # ── Level 2: similarity ──
    fdr_threshold: float = 0.05
    min_shared_clusters: int = 10
    n_jobs: int = 12

    # ── Level 3: propagation ──
    kappa_0: float = 10.0
    propagate_only_rare: bool = True
    rare_hla_max_obs: int = 5000

    def __post_init__(self):
        # resolve to Path objects
        if self.observations_path is not None:
            self.observations_path = Path(self.observations_path)
        if self.cluster_dir is not None:
            self.cluster_dir = Path(self.cluster_dir)

        # output lives inside the cluster directory
        if self.cluster_dir is not None:
            self.output_dir = self.cluster_dir / "stage1"
        elif self.output_dir is None:
            self.output_dir = Path("./outputs/stage1")

        self.output_dir = Path(self.output_dir)

        # derived paths inside cluster_dir
        self.cluster_tsv = (
            self.cluster_dir / "clusters.tsv" if self.cluster_dir else None
        )

        # ensure output subdirs exist
        for sub in ["level1", "level2", "level3"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    # convenience: columns to load if the input is parquet
    @property
    def load_columns(self):
        return [self.peptide_col, self.allele_col, self.label_col]