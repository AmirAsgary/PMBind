#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================================================================
# CONFIGURATION & FLAG PARSING
# ==============================================================================

# Base defaults
INPUT_FASTA="peptides.fasta"
OUTPUT_DIR="mmseqs"
MASKED_FASTA=""
TMP_DIR=""
THREADS="10"

# Parse flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) INPUT_FASTA="$2"; shift ;;
        -m|--masked) MASKED_FASTA="$2"; shift ;;
        -o|--outdir) OUTPUT_DIR="$2"; shift ;;
        -t|--tmpdir) TMP_DIR="$2"; shift ;;
        -n|--threads) THREADS="$2"; shift ;;
        -h|--help) 
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --input     Path to the input FASTA file (default: peptides.fasta)"
            echo "  -o, --outdir    Ultimate output directory (default: mmseqs)"
            echo "  -m, --masked    Path to save masked FASTA (default: <outdir>/masked.fa)"
            echo "  -t, --tmpdir    Temporary directory (default: <outdir>/tmp)"
            echo "  -n, --threads   Number of CPU threads for MMseqs2 (default: 10)"
            echo "  -h, --help      Show this help message and exit"
            echo ""
            exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set dependent defaults if flags weren't provided
if [ -z "$MASKED_FASTA" ]; then MASKED_FASTA="${OUTPUT_DIR}/masked.fa"; fi
if [ -z "$TMP_DIR" ]; then TMP_DIR="${OUTPUT_DIR}/tmp"; fi

PYTHON_REPORT_SCRIPT="src/generate_report.py"

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

# Create directories right away so output files have a place to go
mkdir -p "$OUTPUT_DIR" "$TMP_DIR"

echo "=============================================================================="
echo " STARTING PEPTIDE CLUSTERING PIPELINE (3-First / 3-Last Strategy)"
echo "=============================================================================="
echo "Configuration:"
echo "  Input FASTA:  $INPUT_FASTA"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Masked FASTA: $MASKED_FASTA"
echo "  Temp Dir:     $TMP_DIR"
echo "  Threads:      $THREADS"
echo "=============================================================================="

# ------------------------------------------------------------------------------
echo -e "\n[Step 1/3] Masking sequences (keeping first 3 and last 3)..."
# ------------------------------------------------------------------------------
time awk '/^>/ {print $0; next} { if(length($0)>=6) print substr($0,1,3) substr($0,length($0)-2,3); else print $0 }' "$INPUT_FASTA" > "$MASKED_FASTA"

echo " -> Masking complete. Output saved to: $MASKED_FASTA"

# ------------------------------------------------------------------------------
echo -e "\n[Step 2/3] Running MMseqs2 clustering (Exact match mode)..."
# ------------------------------------------------------------------------------
MMSEQS_FLAGS=(
    "$MASKED_FASTA"
    "${OUTPUT_DIR}/cluster_final"
    "$TMP_DIR"
    --min-seq-id 0.95
    -c 1.0
    --cov-mode 0
    --similarity-type 1
    --mask 0
    --ignore-multi-kmer 0
    -k 6
    --gap-open 16
    --gap-extend 2
    --threads "$THREADS"
    -s 7.0
    --target-search-mode 1
    --seed-sub-mat VTML40.out
    --spaced-kmer-mode 0  # 0 = consecutive k-mers (better for short sequences)
    --alignment-mode 4    # 4 = ungapped alignment (fast and strict)
    --cluster-mode 0      
)

# Run the command using the array
time mmseqs easy-cluster "${MMSEQS_FLAGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/cluster_final.log"

echo " -> Clustering complete. Results saved in: $OUTPUT_DIR/"

# ------------------------------------------------------------------------------
echo -e "\n[Step 3/3] Generating statistical report..."
# ------------------------------------------------------------------------------
if [ ! -f "$PYTHON_REPORT_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_REPORT_SCRIPT!"
    exit 1
fi

# Pass the dynamic input and output variables to Python
time python3 "$PYTHON_REPORT_SCRIPT" "$INPUT_FASTA" "$OUTPUT_DIR"

# ------------------------------------------------------------------------------
echo -e "\n=============================================================================="
echo " PIPELINE FINISHED SUCCESSFULLY!"
echo "=============================================================================="