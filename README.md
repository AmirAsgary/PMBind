# Cluster-Based Bayesian Label Denoising for Immunopeptidomics

A two-stage framework for cleaning, augmenting, and learning from noisy
immunopeptidomic data. Stage 1 (this repository) is a purely statistical
pipeline that denoises peptide–HLA binding labels, discovers pairwise
allele similarity, and propagates labels to under-represented alleles—all
built on BLOSUM62-aware anchor-residue clustering.

---

## Overview

Mass spectrometry–based immunopeptidomics produces millions of
peptide–allele binding observations, but these data suffer from three
systematic problems:

1. **Label noise** — contaminants labeled as binders, true binders missed
2. **Class imbalance** — ~95% non-binders, ~5% binders
3. **Allele bias** — a handful of well-studied alleles dominate the data

This pipeline addresses all three through a modular three-level analysis:

| Level | What it does | Key output |
|-------|-------------|------------|
| **Level 1** | Binary noise model: computes posterior binding probability γ_ch for each (cluster, allele) pair | `gamma`, `b_call` per pair |
| **Diagnostics** | Cluster purity, per-allele purity, binder/non-binder counts | CSVs + histograms |
| **Level 2** | Pairwise allele similarity via Fisher's exact test on co-occurrence | Similarity matrix S, heatmaps |
| **Level 3** | Label propagation from well-characterized to rare alleles | Propagated labels with confidence |
| **EM** | (Optional) Iterative refinement of all levels until convergence | Globally self-consistent labels |

---

## Project Structure

```
.
├── run_stage1.py                      # Main entry point for the Stage 1 pipeline
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # All hyperparameters and path configuration
│   ├── data_loader.py                 # Load observations + clusters, Beta-prior shrinkage
│   ├── level1.py                      # Binary noise model → posterior γ_ch
│   ├── level2.py                      # Fisher tests, similarity matrix, heatmaps
│   ├── level3.py                      # Label propagation to unobserved pairs
│   ├── diagnostics.py                 # Purity metrics and visualizations
│   ├── em_wrapper.py                  # Iterative EM refinement loop
│   ├── io_utils.py                    # Parquet/CSV save/load helpers
│   │
│   ├── anchor_cluster.py             # BLOSUM-aware anchor clustering (standalone)
│   ├── generate_report.py            # Clustering statistics report
│   │
│   ├── anchor_cluster_rs/            # Optional Rust backend for clustering
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml
│   │   └── src/
│   │       └── lib.rs
│   │
│   └── installations/
│       └── install_rust_backend.sh   # One-command Rust backend installer
│
└── data/                              # Your input data (not tracked)
    ├── peptides.fasta
    └── PMDb_class1.parquet
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install numpy pandas scipy joblib matplotlib pyarrow
```

### Step 1: Cluster peptides by anchor similarity

```bash
python src/anchor_cluster.py \
    -i data/peptides.fasta \
    -o data/anchor_clusters \
    -t 0.6
```

This groups peptides by the similarity of their MHC-I anchor residues
(first 3 + last 3 amino acids) using a BLOSUM62-normalized metric with
double weight on positions 2 and Ω.

### Step 2: Run the Stage 1 statistical pipeline

```bash
python run_stage1.py \
    --observations data/PMDb_class1.parquet \
    --cluster-dir  data/anchor_clusters/
```

Outputs are written to `data/anchor_clusters/stage1/`.

---

## Detailed Usage

### Anchor Clustering

```bash
python src/anchor_cluster.py \
    -i INPUT_FASTA \
    -o OUTPUT_DIR \
    -t THRESHOLD \
    --min-cluster-size MIN_SIZE \
    --n-front 3 \
    --n-back 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | *required* | Input FASTA file (full-length peptides) |
| `-o, --outdir` | `anchor_clusters` | Output directory |
| `-t, --threshold` | `0.6` | BLOSUM similarity threshold (0.0–1.0) |
| `--min-cluster-size` | `2` | Min members for per-cluster FASTA output |
| `--n-front` | `3` | N-terminal anchor length |
| `--n-back` | `3` | C-terminal anchor length |

**Threshold guide:**

| Value | Behaviour | Use case |
|-------|-----------|----------|
| 0.8 | Strict — mostly exact + very conservative substitutions | Fine-grained motif resolution |
| 0.6 | Moderate — allows 1–2 conservative substitutions (A↔V, L↔I) | General purpose (recommended) |
| 0.4 | Relaxed — broader groups | Exploratory analysis |

**Outputs:**

```
anchor_clusters/
├── clusters.tsv            # Full mapping: cluster_id → peptide → anchor
├── cluster_summary.tsv     # Cluster sizes sorted descending
├── summary.txt             # Run parameters and statistics
└── fasta/
    ├── cluster_0.fasta     # Per-cluster FASTA (ready for MSA)
    ├── cluster_1.fasta
    └── ...
```

**Optional Rust acceleration:**

```bash
bash src/installations/install_rust_backend.sh
```

Requires: `curl` (to install Rust), `pip` (to install maturin). The
Python script auto-detects the Rust backend and uses it when available.

### Stage 1 Pipeline

```bash
python run_stage1.py \
    --observations PATH \
    --cluster-dir  PATH \
    [options]
```

#### Required Arguments

| Flag | Description |
|------|-------------|
| `--observations` | Observation file (`.parquet`, `.csv`, or `.tsv`) |
| `--cluster-dir` | Output directory from `anchor_cluster.py` (must contain `clusters.tsv`) |

#### Column Mapping

The observation file must contain three columns: peptide sequence, allele
name, and binding label. Column names are configurable:

| Flag | Default | Description |
|------|---------|-------------|
| `--peptide-col` | `long_mer` | Column with peptide sequences |
| `--allele-col` | `allele` | Column with allele names |
| `--label-col` | `assigned_label` | Column with binding labels (0/1) |

#### Allele Filtering

| Flag | Default | Description |
|------|---------|-------------|
| `--allele-prefix` | `None` | Keep alleles starting with prefix (e.g., `HLA`). Default: no filter (all species). |
| `--no-allele-filter` | — | Explicitly disable filtering |

#### Regularization

| Flag | Default | Description |
|------|---------|-------------|
| `--shrinkage-k` | `50` | Beta-prior pseudocount strength. Higher = more shrinkage toward the global mean. |
| `--p-h-ceil` | `0.95` | Maximum allowed p_h after shrinkage |
| `--tau-max` | `0.50` | Hard cap on binarization threshold τ_h |

#### Level 2: Pairwise Similarity

| Flag | Default | Description |
|------|---------|-------------|
| `--n-jobs` | `12` | Parallel workers for Fisher's exact tests |
| `--fdr` | `0.05` | FDR threshold for Benjamini-Hochberg correction |
| `--min-shared` | `10` | Minimum shared clusters for a valid allele pair |

#### Level 3: Label Propagation

| Flag | Default | Description |
|------|---------|-------------|
| `--rare-max-obs` | `5000` | Alleles with fewer observations are "rare" |
| `--skip-level3` | — | Skip label propagation entirely |

#### EM Iterative Refinement

| Flag | Default | Description |
|------|---------|-------------|
| `--em-iter` | `0` | Number of EM iterations. 0 = single pass (no EM). Recommended: 3–5. |
| `--em-tol` | `1e-3` | Convergence tolerance: max\|Δγ\| across all pairs |
| `--em-recompute-S` | `3` | Recompute similarity matrix every K iterations |

#### Output Structure

```
anchor_clusters/stage1/
├── level1/
│   ├── noise_params.csv              # α_h, β_h, p_h, τ_h per allele
│   ├── level1_results.parquet        # γ_ch and b_call for every pair
│   ├── noise_params_em_final.csv     # (if EM) converged parameters
│   └── level1_results_em_final.parquet
│
├── level2/
│   ├── similarity_matrix.npy         # S_{hh'} matrix
│   ├── hla_index.csv                 # Allele index ↔ name mapping
│   ├── pairwise_tests.parquet        # All Fisher test results
│   ├── hla_OR_matrix.csv             # Allele × Allele odds ratios
│   ├── hla_pvalue_matrix.csv         # Raw p-values
│   ├── hla_pvalue_adj_matrix.csv     # BH-adjusted p-values
│   ├── heatmap_OR.png                # log₁₀(OR) heatmap
│   ├── heatmap_pvalue.png            # -log₁₀(p) heatmap
│   ├── heatmap_pvalue_adj.png        # -log₁₀(adj p) heatmap
│   ├── hla_associations.csv          # Per-allele: n_tested, n_significant
│   └── hla_associations_histogram.png
│
├── level3/
│   ├── propagated_labels.parquet     # p̃_ch, ñ_ch, λ_ch
│   └── propagation_summary.csv       # Per-allele propagation stats
│
└── diagnostics/
    ├── cluster_purity.csv            # Per-cluster purity score
    ├── cluster_purity_histogram.png
    ├── hla_purity.csv                # Per-allele mean/median/min/max purity
    ├── hla_purity_histogram.png
    ├── hla_cluster_counts.csv        # Binder vs non-binder clusters per allele
    ├── hla_cluster_counts_barplot.png
    └── hla_binder_fraction_histogram.png
```

---

## Examples

### Basic run with defaults

```bash
python src/anchor_cluster.py -i data/peptides.fasta -o data/clusters -t 0.6
python run_stage1.py --observations data/PMDb_class1.parquet --cluster-dir data/clusters
```

### Non-human alleles (mouse, etc.)

```bash
python run_stage1.py \
    --observations data/mouse_data.csv \
    --cluster-dir  data/clusters \
    --no-allele-filter \
    --peptide-col  sequence \
    --allele-col   mhc_allele \
    --label-col    binder
```

### With EM refinement (recommended for production)

```bash
python run_stage1.py \
    --observations data/PMDb_class1.parquet \
    --cluster-dir  data/clusters \
    --em-iter 5 \
    --em-tol 1e-3
```

### Strict clustering + low FDR

```bash
python src/anchor_cluster.py -i data/peptides.fasta -o data/clusters_strict -t 0.8
python run_stage1.py \
    --observations data/PMDb_class1.parquet \
    --cluster-dir  data/clusters_strict \
    --fdr 0.01 \
    --min-shared 20
```

### Run MSA on anchor clusters

```bash
for f in data/clusters/fasta/cluster_*.fasta; do
    muscle -in "$f" -out "${f%.fasta}.aln"
done
```

---

## Key Design Decisions

### Why anchor-based clustering instead of MMseqs2?

MHC-I binding specificity is determined primarily by anchor residues at
positions 2 and Ω (the C-terminus). Full-length sequence identity tools
like MMseqs2 do not distinguish anchor from non-anchor positions. On
short peptides (8–14 aa), MMseqs2's internal score-per-column estimator
breaks down entirely—our empirical tests showed that its linclust module
silently disables sequence-identity filtering for masked 6-residue
sequences, producing clusters with arbitrarily dissimilar anchors.

The BLOSUM62-normalized anchor similarity with position weighting
(P2 and PΩ get 2× weight) produces biologically interpretable clusters
by construction.

### Why Beta-prior shrinkage for p_h?

In typical immunopeptidomic databases, ~50% of alleles have so few
observations that the raw binder rate p_h is degenerate (0.0 or 1.0).
When p_h = 1.0, the binarization threshold τ_h = 2 × p_h = 2.0 is
unreachable, silently disabling the noise model for that allele. The
Beta-prior shrinkage `p_h = (n_pos + k·p_global) / (n_total + k)` pulls
extreme estimates toward the global mean, and the τ_max cap ensures all
alleles have a reachable threshold.

### Why EM refinement?

The single-pass pipeline (L1 → L2 → L3) is a feed-forward chain: each
level consumes the predecessor's output without the opportunity to revise
earlier estimates. The EM loop allows Level 3 propagation evidence to
flow back into Level 1 posteriors, yielding globally self-consistent
label assignments. In practice, 3–5 iterations suffice for convergence.

### Why min_shared_clusters = 10?

Fisher's exact test on a 2×2 contingency table with very small cell
counts has low statistical power and produces unstable odds ratios (OR
can be ∞ or 0). The threshold of 10 shared clusters ensures each cell
has a reasonable expected count. Setting it to 5 is defensible for
exploratory analysis—the BH-FDR correction handles the increased multiple
testing burden—but may introduce noisy associations.

---

## Performance

| Dataset | Anchor Clustering | Stage 1 (single pass) | Stage 1 (5 EM iters) |
|---------|------------------|-----------------------|----------------------|
| 7K peptides | <1s | ~10s | ~50s |
| 2.5M peptides | ~160s (Python) / ~15s (Rust) | ~5 min | ~25 min |
| 5M peptides, 361 alleles | ~5 min | ~15 min | ~1 hr |

The dominant cost in Stage 1 is Level 2 (Fisher's exact tests). With
`--em-recompute-S 3`, the similarity matrix is only recomputed every 3rd
EM iteration, reducing the cost substantially.

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

**Optional (for faster I/O):**

```
pyarrow          # parquet read/write
```

**Optional (for faster clustering):**

```
rust + maturin   # install via src/installations/install_rust_backend.sh
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