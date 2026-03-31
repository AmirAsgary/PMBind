"""
Configuration for the Cluster-Based Bayesian Label Denoising pipeline.
All paths, hyperparameters, and constants in one place.
"""
from dataclasses import dataclass, field
from pathlib import Path
@dataclass
class PipelineConfig:
    # ── paths ──
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./outputs")
    parquet_path: Path = None  # set in __post_init__
    fasta_path: Path = None
    # ── clustering ──
    cluster_identity: int = 80  # 60 or 80 (maps to mmseqs60/mmseqs80)
    cluster_tsv: Path = None    # set in __post_init__
    # ── HLA filtering ──
    hla_prefix: str = "HLA"  # keep only alleles starting with this
    # ── Level 1: binary noise model ──
    alpha_default: float = 0.10   # default false negative rate
    beta_default: float = 0.02    # default false positive rate
    noise_min_cluster_size: int = 5       # min observations to estimate noise
    noise_purity_threshold: float = 0.90  # fraction for "near-unanimous" clusters
    # ── Level 1: binarisation ──
    tau_multiplier: float = 2.0  # tau_h = tau_multiplier * pi_h
    # ── Level 2: similarity ──
    fdr_threshold: float = 0.05    # Benjamini-Hochberg FDR
    min_shared_clusters: int = 10  # minimum shared clusters for a valid pair
    n_jobs: int = 12               # parallel workers
    # ── Level 3: propagation ──
    kappa_0: float = 10.0  # confidence hyperparameter
    propagate_only_rare: bool = True  # only propagate to under-represented HLAs
    rare_hla_max_obs: int = 5000      # HLAs with fewer obs are "rare"
    # ── memory ──
    parquet_columns: list = field(default_factory=lambda: [
        "long_mer", "allele", "assigned_label"
    ])
    def __post_init__(self):
        self.parquet_path = self.data_dir / "PMDb_2025_11_18_class1.parquet"
        self.fasta_path = self.data_dir / "peptides.fasta"
        ident = self.cluster_identity
        self.cluster_tsv = (
            self.data_dir / f"mmseqs{ident}" / f"cluster{ident}_cluster.tsv"
        )
        # ensure output dirs exist
        for sub in ["level1", "level2", "level3"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
