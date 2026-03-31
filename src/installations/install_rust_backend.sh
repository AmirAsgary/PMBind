#!/bin/bash
set -e

echo "=============================================="
echo " anchor_cluster_rs — Installation"
echo "=============================================="

# ── Step 1: Check for Rust ──────────────────────────────────────────
if ! command -v rustc &> /dev/null; then
    echo ""
    echo "[1/3] Rust not found. Installing via rustup …"
    echo "      (This is a one-time setup, ~1 minute)"
    echo ""
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[1/3] Rust already installed: $(rustc --version)"
fi

# ── Step 2: Install maturin ─────────────────────────────────────────
if ! command -v maturin &> /dev/null; then
    echo "[2/3] Installing maturin (Rust→Python build tool) …"
    pip install maturin
else
    echo "[2/3] maturin already installed: $(maturin --version)"
fi

# ── Step 3: Build and install the extension ─────────────────────────
echo "[3/3] Building Rust extension (first build takes ~30s) …"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../anchor_cluster_rs"
maturin develop --release
cd "$SCRIPT_DIR"

echo ""
echo "=============================================="
echo " Done! Verify with:"
echo "   python -c 'from anchor_cluster_rs import cluster_anchors_rs; print(\"Rust backend OK\")'"
echo ""
echo " Run clustering:"
echo "   python anchor_cluster.py -i peptides.fasta -o results -t 0.6"
echo "=============================================="