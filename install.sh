#!/bin/bash
set -e

ENV_NAME="pmbind_peptide"

echo "=============================================="
echo " PMBind — Automatic Installation"
echo "=============================================="

# ── Step 1: Create Conda/Mamba Environment ──────────────────────────
if command -v mamba &> /dev/null; then
    CONDA_EXE="mamba"
else
    CONDA_EXE="conda"
fi

echo "[1/4] Creating $ENV_NAME environment using $CONDA_EXE..."
$CONDA_EXE env create -f environment.yml || $CONDA_EXE env update -f environment.yml

# --- Robust Shell Activation ---
# This part finds your conda installation and sources the required script
# to make 'conda activate' work inside this bash script.
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# ── Step 2: Check for Rust ──────────────────────────────────────────
if ! command -v rustc &> /dev/null; then
    echo ""
    echo "[2/4] Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[2/4] Rust already installed: $(rustc --version)"
fi

# ── Step 3: Build Rust Backend ──────────────────────────────────────
echo "[3/4] Building Rust extension (anchor_cluster_rs)..."
# Navigate to the Rust project directory relative to the script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/src/anchor_cluster_rs"

# Build and install the extension into the active conda environment
maturin develop --release
cd "$SCRIPT_DIR"

# ── Step 4: Verification ────────────────────────────────────────────
echo "[4/4] Verifying installation..."
python -c 'from anchor_cluster_rs import cluster_anchors_rs; print("✅ Rust backend OK")'


echo ""
echo "=============================================="
echo " Setup Complete!"
echo " To start, run: conda activate $ENV_NAME"
echo "=============================================="