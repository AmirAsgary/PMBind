# Cluster-Based Bayesian Label Denoising for Immunopeptidomics

A two-stage framework for cleaning, augmenting, and learning from noisy
immunopeptidomic data. Stage 1 (this repository) is a purely statistical
pipeline that denoises peptide–allele binding labels, discovers pairwise
allele similarity, and propagates labels to under-represented alleles—all
built on BLOSUM62-aware anchor-residue clustering with iterative Gibbs
sampling refinement.

---

## Overview

Mass spectrometry–based immunopeptidomics produces millions of
peptide–allele binding observations, but these data suffer from three
systematic problems:

1. **Label noise** — contaminants labeled as binders, true binders missed
2. **Class imbalance** — ~97% non-binders, ~3% binders
3. **Allele bias** — a handful of well-studied alleles dominate; ~50% of
   alleles have so few observations that naive statistics are degenerate

This pipeline addresses all three:

| Component | What it does | Key output |
|-----------|-------------|------------|
| **Level 1** | Binary noise model with Bayesian Beta-prior noise estimation: computes posterior γ_{ch} for each (cluster, allele) pair | `gamma`, `b_call` per pair |
| **Diagnostics** | Per-(cluster, allele) purity φ_{ch}, per-allele summaries, binder/non-binder counts | CSVs + histograms |
| **Level 2** | Pairwise allele similarity via Fisher's exact test with uncertainty-aware conservative odds ratios | Similarity matrix S, heatmaps |
| **Level 3** | Label propagation with coverage-penalized confidence to prevent extrapolation from sparsely labeled clusters | Propagated labels with calibrated λ |
| **Gibbs** | Iterative refinement of all latent variables with proper feedback from Level 3 back into Level 1 | Globally self-consistent labels |

---

## Project Structure

```
.
├── run_stage1.py                      # Main entry point
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # All hyperparameters and paths
│   ├── data_loader.py                 # Load observations + clusters
│   ├── level1.py                      # Binary noise model → posterior γ_{ch}
│   ├── level2.py                      # Fisher tests, conservative OR, heatmaps
│   ├── level3.py                      # Coverage-penalized label propagation
│   ├── diagnostics.py                 # Per-pair purity φ_{ch}, per-allele summaries
│   ├── gibbs.py                       # Gibbs sampling with feedback loop
│   ├── io_utils.py                    # Parquet/CSV save/load helpers
│   │
│   ├── anchor_cluster.py             # BLOSUM-aware anchor clustering (standalone)
│   ├── generate_report.py            # Clustering statistics report
│   │
│   ├── anchor_cluster_rs/            # Optional Rust backend for clustering
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml
│   │   └── src/lib.rs
│   │
│   └── installations/
│       └── install_rust_backend.sh
│
└── data/                              # Your input data (not tracked)
    ├── peptides.fasta
    └── observations.parquet
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scipy joblib matplotlib pyarrow
```

### Step 1: Cluster peptides by anchor similarity

```bash
python src/anchor_cluster.py \
    -i data/peptides.fasta \
    -o data/anchor_clusters \
    -t 0.6
```

### Step 2: Run Stage 1 (single pass)

```bash
python run_stage1.py \
    --observations data/observations.parquet \
    --cluster-dir  data/anchor_clusters/
```

### Step 2b: Run Stage 1 with Gibbs refinement (recommended)

```bash
python run_stage1.py \
    --observations data/observations.parquet \
    --cluster-dir  data/anchor_clusters/ \
    --gibbs-iter 10
```

Outputs are written to `data/anchor_clusters/stage1/`.

---

## Anchor Clustering

Groups peptides by the similarity of their MHC-I anchor residues
(first 3 + last 3 amino acids) using a BLOSUM62-normalized metric
with double weight on positions 2 and Ω.

```bash
python src/anchor_cluster.py -i INPUT -o OUTDIR -t THRESHOLD
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | *required* | Input FASTA file |
| `-o, --outdir` | `anchor_clusters` | Output directory |
| `-t, --threshold` | `0.6` | BLOSUM similarity threshold (0.0–1.0) |
| `--min-cluster-size` | `2` | Min members for per-cluster FASTA |
| `--n-front` | `3` | N-terminal anchor length |
| `--n-back` | `3` | C-terminal anchor length |

**Threshold guide:**

| Value | Effect |
|-------|--------|
| 0.8 | Strict — mostly exact matches + very conservative substitutions |
| 0.6 | Moderate — allows 1–2 conservative substitutions (recommended) |
| 0.4 | Relaxed — broader groups for exploratory analysis |

**Optional Rust acceleration** (10–100× faster clustering):

```bash
bash src/installations/install_rust_backend.sh
```

---

## Stage 1 Pipeline

### Required Arguments

| Flag | Description |
|------|-------------|
| `--observations` | Observation file (`.parquet`, `.csv`, or `.tsv`) |
| `--cluster-dir` | Output directory from `anchor_cluster.py` (must contain `clusters.tsv`) |

### Column Mapping

| Flag | Default | Description |
|------|---------|-------------|
| `--peptide-col` | `long_mer` | Column with peptide sequences |
| `--allele-col` | `allele` | Column with allele names |
| `--label-col` | `assigned_label` | Column with binding labels (0/1) |

### Allele Filtering

| Flag | Default | Description |
|------|---------|-------------|
| `--allele-prefix` | `None` | Keep alleles starting with prefix (e.g., `HLA`). Default: no filter — all species processed. |
| `--no-allele-filter` | — | Explicitly disable filtering |

### Prior Regularisation

| Flag | Default | Description |
|------|---------|-------------|
| `--shrinkage-k` | `50` | Beta-prior pseudocount strength for p_h. Higher = more shrinkage toward global mean. |
| `--p-h-ceil` | `0.95` | Maximum allowed p_h after shrinkage |
| `--tau-max` | `0.50` | Hard cap on binarisation threshold τ_h |

### Level 2: Pairwise Similarity

| Flag | Default | Description |
|------|---------|-------------|
| `--n-jobs` | `12` | Parallel workers for Fisher's exact tests |
| `--fdr` | `0.05` | FDR threshold for Benjamini-Hochberg correction |
| `--min-shared` | `10` | Minimum shared clusters for a valid allele pair |

### Level 3: Label Propagation

| Flag | Default | Description |
|------|---------|-------------|
| `--rare-max-obs` | `5000` | Alleles with fewer observations are "rare" |
| `--skip-level3` | — | Skip label propagation entirely |

### Gibbs Sampling

| Flag | Default | Description |
|------|---------|-------------|
| `--gibbs-iter` | `0` | Number of Gibbs iterations. 0 = single pass. Recommended: 5–10. |
| `--gibbs-tol` | `1e-3` | Convergence tolerance: max\|Δγ\| across all pairs |
| `--gibbs-recompute-S` | `3` | Recompute similarity matrix every K iterations |
| `--gibbs-deterministic` | — | Use posterior means instead of Bernoulli sampling |

---

## Output Structure

```
anchor_clusters/stage1/
│
├── level1/
│   ├── noise_params.csv                  # α_h, β_h, p_h, τ_h per allele
│   ├── level1_results.parquet            # γ_{ch}, b_call, φ_{ch} for every pair
│   ├── noise_params_gibbs_final.csv      # (if Gibbs) converged parameters
│   └── level1_results_gibbs_final.parquet
│
├── level2/
│   ├── similarity_matrix.npy             # S_{hh'} matrix
│   ├── hla_index.csv                     # Allele index ↔ name mapping
│   ├── pairwise_tests.parquet            # Fisher results + OR_conservative
│   ├── hla_OR_matrix.csv                 # Allele × Allele odds ratios
│   ├── hla_pvalue_matrix.csv             # Raw p-values
│   ├── hla_pvalue_adj_matrix.csv         # BH-adjusted p-values
│   ├── heatmap_OR.png                    # log₁₀(OR) heatmap
│   ├── heatmap_pvalue.png                # -log₁₀(p) heatmap
│   ├── heatmap_pvalue_adj.png            # -log₁₀(adj p) heatmap
│   ├── hla_associations.csv              # Per-allele: n_tested, n_significant
│   └── hla_associations_histogram.png
│
├── level3/
│   ├── propagated_labels.parquet         # p̃_{ch}, ñ_{ch}, λ_{ch}
│   └── propagation_summary.csv           # Per-allele propagation stats
│
├── diagnostics/
│   ├── hla_purity.csv                    # Per-allele: mean/std/median/min/max of φ_{ch}
│   ├── hla_cluster_counts.csv            # Binder vs non-binder clusters per allele
│   ├── pair_purity_histogram.png         # Distribution of φ_{ch}
│   ├── hla_purity_histogram.png          # Distribution of φ̄_h
│   ├── hla_cluster_counts_barplot.png    # Top 30 alleles stacked bar
│   └── hla_binder_fraction_histogram.png
│
└── stage1_YYYYMMDD_HHMMSS.log           # Full pipeline log
```

---

## Key Design Decisions

### BLOSUM-aware anchor clustering

MHC-I binding specificity is determined primarily by anchor residues at
positions 2 and Ω. Full-length tools like MMseqs2 don't distinguish
anchor from non-anchor positions. On short peptides (8–14 aa), MMseqs2's
internal estimators break down — our tests showed it silently disables
identity filtering for 6-residue masked sequences. The BLOSUM62-normalized
similarity with position weighting produces biologically interpretable
clusters by construction.

### Beta-prior noise estimation

Noise parameters α_h (false negative rate) and β_h (false positive rate)
are estimated via conjugate Beta posteriors:

```
α_h | data ~ Beta(a_α + FN_h, b_α + correct_detections_h)
```

For well-observed alleles, the data dominates. For data-poor alleles, the
posterior smoothly reverts to the prior mean — no hard-coded defaults, no
discontinuities.

### Conservative odds ratios

The point-estimate OR can be wildly inflated for rare alleles with small
contingency table cells. We use the Haldane-Anscombe correction (+0.5 to
all cells) and take the lower bound of the 95% CI:

```
OR_conservative = max(1, exp(log(OR*) - 1.96 × SE(log OR*)))
```

If the lower bound ≤ 1, the weight is set to zero — no propagation
between alleles unless we are statistically confident.

### Coverage-penalized propagation

Large clusters where only a few peptides have labels can produce
misleadingly high confidence. The effective sample size is penalized:

```
ñ_{ch'} = Σ w_{hh'} · n_{ch} · f(ρ_c)    where f(ρ_c) = ρ_c^0.5
```

A cluster with 4% label coverage gets its contribution reduced 5×.

### Per-pair purity φ_{ch}

Purity is defined per observed (cluster, allele) pair, not per cluster:

```
φ_{ch} = log((n_pos + ε) / (n_neg + ε))
```

Per-allele summaries (mean, std, median) of φ_{ch} characterise each
allele's signal quality across its clusters.

### Gibbs sampling with feedback

The single-pass pipeline (L1 → L2 → L3) is feed-forward. The Gibbs
sampler closes the loop: propagated labels feed back into allele-level
priors p_h, which shift γ_{ch}, which shift the binarised calls, which
shift the similarity matrix. Each block is sampled from its tractable
conjugate conditional:

- **θ_{ch}**: Bernoulli(γ_{ch}) — the Level 1 posterior
- **α_h**: Beta posterior from true-binder clusters
- **β_h**: Beta posterior from true-non-binder clusters
- **p_h**: Beta posterior with shrinkage + propagation feedback

---

## Examples

### Basic single-pass run

```bash
python src/anchor_cluster.py -i data/peptides.fasta -o data/clusters -t 0.6
python run_stage1.py \
    --observations data/observations.parquet \
    --cluster-dir data/clusters
```

### Gibbs refinement (recommended for production)

```bash
python run_stage1.py \
    --observations data/observations.parquet \
    --cluster-dir data/clusters \
    --gibbs-iter 10 \
    --gibbs-recompute-S 3
```

### Non-human alleles (mouse, etc.)

```bash
python run_stage1.py \
    --observations data/mouse_data.csv \
    --cluster-dir data/clusters \
    --no-allele-filter \
    --peptide-col sequence \
    --allele-col mhc_allele \
    --label-col binder
```

### Strict clustering + low FDR

```bash
python src/anchor_cluster.py -i data/peptides.fasta -o data/strict -t 0.8
python run_stage1.py \
    --observations data/observations.parquet \
    --cluster-dir data/strict \
    --fdr 0.01 --min-shared 20
```

### Deterministic Gibbs (EM-equivalent, reproducible)

```bash
python run_stage1.py \
    --observations data/observations.parquet \
    --cluster-dir data/clusters \
    --gibbs-iter 10 \
    --gibbs-deterministic
```

### Run MSA on anchor clusters

```bash
for f in data/clusters/fasta/cluster_*.fasta; do
    muscle -in "$f" -out "${f%.fasta}.aln"
done
```

---

## Performance

| Dataset | Anchor Clustering | Stage 1 (single pass) | Stage 1 (10 Gibbs iters) |
|---------|------------------|-----------------------|--------------------------|
| 7K peptides | <1s | ~10s | ~1 min |
| 2.5M peptides | ~3 min (Python) / ~15s (Rust) | ~5 min | ~30 min |
| 44M observations, 475 alleles | ~5 min | ~2 min | ~2 min (converges in <20 iters) |

The dominant cost in Gibbs iterations is Level 2 (Fisher's exact tests).
With `--gibbs-recompute-S 3`, similarity is only recomputed every 3rd
iteration; the per-iteration L1 updates take ~1s.

---

## Dependencies

**Required:**

```
numpy
pandas
scipy
joblib
matplotlib
```

**Optional:**

```
pyarrow          # faster parquet I/O
rust + maturin   # faster anchor clustering (install via src/installations/)
```

---

## Citation

If you use this framework, please cite:

> [Authors]. Cluster-Based Bayesian Label Denoising, HLA Similarity
> Estimation, and Noise-Aware Binding Prediction from Immunopeptidomic
> Data. [Year].

---

## License

[Specify your license here]