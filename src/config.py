"""
Configuration for the Cluster-Based Bayesian Label Denoising pipeline.

All paths, hyperparameters, and constants in one place.
Paths are set dynamically from CLI arguments — nothing is hardcoded.

Key design decisions:
  - allele_prefix=None means NO filtering (processes all species).
  - p_h is regularised via Beta-prior shrinkage (shrinkage_k) to prevent
    degenerate posteriors for data-poor alleles.
  - tau_h is capped at tau_max to ensure all alleles have a reachable
    binarisation threshold.
  - A diagnostics/ subdirectory stores purity metrics and plots.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # ── paths (set by CLI — no defaults) ──────────────────────────────
    observations_path: Path = None   # parquet / csv / tsv with peptide observations
    cluster_dir: Path = None         # anchor_cluster.py output directory
    output_dir: Path = None          # set in __post_init__ → cluster_dir/stage1

    # ── allele filtering ──────────────────────────────────────────────
    # None = keep ALL alleles (human + non-human).
    # Set to "HLA" to restrict to human HLA alleles only.
    allele_prefix: str = None

    # ── observation column names ──────────────────────────────────────
    # These map your input file's column names to internal names.
    #   peptide_col : column containing peptide sequences (join key to clusters)
    #   allele_col  : column containing allele names (e.g. HLA-A*02:01, H-2-Kb)
    #   label_col   : column containing binding labels (0/1)
    peptide_col: str = "long_mer"
    allele_col: str = "allele"
    label_col: str = "assigned_label"

    # ── p_h estimation (Beta-prior shrinkage) ─────────────────────────
    # Prevents degenerate p_h for data-poor alleles.
    # p_h = (n_pos + k * p_global) / (n_total + k)
    shrinkage_k: float = 50.0     # pseudocount strength (higher = more shrinkage)
    p_h_floor: float = 0.001      # minimum allowed p_h after shrinkage
    p_h_ceil: float = 0.95        # maximum allowed p_h after shrinkage

    # ── Level 1: binary noise model ───────────────────────────────────
    # Noise parameters estimated from high-confidence clusters, with
    # Beta-prior regularisation (Solution 2: Bayesian noise treatment).
    # Prior encodes global expectation: E[alpha] ≈ a/(a+b).
    alpha_prior_a: float = 2.0    # Beta(2, 18) → E[alpha] ≈ 0.10
    alpha_prior_b: float = 18.0
    beta_prior_a: float = 1.0     # Beta(1, 49) → E[beta] ≈ 0.02
    beta_prior_b: float = 49.0
    noise_min_cluster_size: int = 5       # min observations to estimate noise
    noise_purity_threshold: float = 0.90  # fraction for "near-unanimous" clusters

    # ── Level 1: binarisation ─────────────────────────────────────────
    # tau_h = min(tau_multiplier * p_h, tau_max)
    # The cap prevents unreachable thresholds for high-p_h alleles.
    tau_multiplier: float = 2.0   # base multiplier for adaptive threshold
    tau_max: float = 0.50         # hard cap on tau_h

    # ── Level 2: pairwise HLA similarity ──────────────────────────────
    fdr_threshold: float = 0.05    # Benjamini-Hochberg FDR
    min_shared_clusters: int = 10  # minimum shared clusters for a valid pair
    n_jobs: int = 12               # parallel workers for Fisher tests
    use_conservative_or: bool = True   # Solution 3: use lower-bound OR with
                                       # Haldane-Anscombe correction instead of
                                       # point estimate, to prevent rare-allele
                                       # inflation of propagation weights.

    # ── Level 3: label propagation ────────────────────────────────────
    kappa_0: float = 10.0          # confidence hyperparameter for lambda_ch
    propagate_only_rare: bool = True   # only propagate to under-represented alleles
    rare_hla_max_obs: int = 5000       # alleles with fewer obs are "rare"
    coverage_penalty_tau: float = 0.5  # Solution 1: power-law exponent for coverage
                                       # penalty f(rho_c) = rho_c^tau.  Penalises
                                       # extrapolation from sparsely labeled clusters.

    # ── Gibbs sampling ──────────────────────────────────────────────
    gibbs_max_iter: int = 10       # maximum Gibbs iterations (0 = single-pass)
    gibbs_tol: float = 1e-3        # convergence tolerance (max |delta gamma|)
    gibbs_recompute_S_every: int = 3  # recompute similarity every K iterations
    gibbs_sample_theta: bool = True   # True = stochastic Gibbs; False = deterministic
                                      # (use posterior mean instead of Bernoulli draw)

    def __post_init__(self):
        """Resolve paths and create output directories."""
        if self.observations_path is not None:
            self.observations_path = Path(self.observations_path)
        if self.cluster_dir is not None:
            self.cluster_dir = Path(self.cluster_dir)

        # output lives inside the cluster directory: cluster_dir/stage1/
        if self.cluster_dir is not None:
            self.output_dir = self.cluster_dir / "stage1"
        elif self.output_dir is None:
            self.output_dir = Path("./outputs/stage1")

        self.output_dir = Path(self.output_dir)

        # derived path: the clusters.tsv from anchor_cluster.py
        self.cluster_tsv = (
            self.cluster_dir / "clusters.tsv" if self.cluster_dir else None
        )

        # ensure output subdirectories exist
        for sub in ["level1", "level2", "level3", "diagnostics"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def load_columns(self):
        """Columns to read from the observation file (for parquet efficiency)."""
        return [self.peptide_col, self.allele_col, self.label_col]