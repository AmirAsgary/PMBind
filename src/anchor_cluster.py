#!/usr/bin/env python3
"""
anchor_cluster.py — BLOSUM-aware anchor clustering for immunopeptides

Clusters peptides by the similarity of their anchor residues (first 3 + last 3 aa)
using a BLOSUM62-normalized similarity metric with position-specific weighting
(P2 and PΩ weighted 2× for MHC-I relevance).

Speed strategy:
  1. Deduplicate peptides by their 6-mer anchor
  2. Block unique anchors by reduced-alphabet at P2 and PΩ
  3. Greedy centroid clustering within blocks

Outputs:
  <outdir>/clusters.tsv             cluster assignment for every peptide
  <outdir>/cluster_summary.tsv      cluster stats sorted by size
  <outdir>/fasta/cluster_*.fasta    per-cluster FASTA ready for MSA
  <outdir>/summary.txt              run statistics
"""

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

# Try to load the Rust-accelerated backend
try:
    from anchor_cluster_rs import cluster_anchors_rs
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# ============================================================================
# BLOSUM62 — standard 20×20
# ============================================================================
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

BLOSUM62_FLAT = [
#    A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
     4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0,
    -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3,
    -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,
    -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,
     0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1,
    -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,
    -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,
     0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3,
    -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,
    -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3,
    -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1,
    -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,
    -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1,
    -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1,
    -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2,
     1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,
     0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0,
    -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3,
    -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1,
     0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4,
]

N_AA = len(AA_ORDER)
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Precompute normalized similarity: sim(a,b) = B(a,b) / sqrt(B(a,a) * B(b,b))
_self = {aa: BLOSUM62_FLAT[i * N_AA + i] for i, aa in enumerate(AA_ORDER)}
SIM = {}
for i, a in enumerate(AA_ORDER):
    for j, b in enumerate(AA_ORDER):
        denom = math.sqrt(_self[a] * _self[b])
        SIM[(a, b)] = BLOSUM62_FLAT[i * N_AA + j] / denom if denom > 0 else 0.0

# ============================================================================
# Reduced alphabet for blocking (10 groups)
# ============================================================================
COARSE = {}
for gid, aas in enumerate(["AST", "VILM", "FYW", "DE", "KR", "NQ",
                            "G", "H", "C", "P"]):
    for aa in aas:
        COARSE[aa] = gid

# ============================================================================
# Position weights: [P1, P2, P3, PΩ-2, PΩ-1, PΩ]
#   P2 and PΩ get 2× weight (primary MHC-I anchors)
# ============================================================================
_RAW_W = [1.0, 2.0, 1.0, 1.0, 1.0, 2.0]
_WSUM  = sum(_RAW_W)
WEIGHTS = [w / _WSUM for w in _RAW_W]


# ============================================================================
# Precompute fast lookup arrays
# ============================================================================
# 6 flat arrays of size 128×128, one per position, with weight baked in.
# Indexed by ord(aa1)*128 + ord(aa2) → weighted normalized similarity.
WSIM = []
for _k in range(6):
    _arr = [0.0] * (128 * 128)
    for (_a, _b), _v in SIM.items():
        _arr[ord(_a) * 128 + ord(_b)] = WEIGHTS[_k] * _v
    WSIM.append(_arr)

# Precompute max possible score from remaining positions (for early exit).
# _MAX_REMAINING[k] = sum of WEIGHTS[k+1:] (max sim at each pos is 1.0)
_MAX_REMAINING = [0.0] * 7  # index 6 = 0.0 sentinel
for _k in range(5, -1, -1):
    _MAX_REMAINING[_k] = _MAX_REMAINING[_k + 1] + WEIGHTS[_k]

# Check order for positions: check high-weight positions (1, 5) first
# so early termination kicks in sooner.
_CHECK_ORDER = [1, 5, 0, 2, 3, 4]
_REMAINING_AFTER = [0.0] * 7
for _i in range(5, -1, -1):
    _REMAINING_AFTER[_i] = _REMAINING_AFTER[_i + 1] + WEIGHTS[_CHECK_ORDER[_i]]

# Reorder WSIM to match _CHECK_ORDER
_WSIM_ORD = [WSIM[_CHECK_ORDER[i]] for i in range(6)]

# Coarse-group lookup array (indexed by ord)
_COARSE_ORD = [-1] * 128
for _aa, _gid in COARSE.items():
    _COARSE_ORD[ord(_aa)] = _gid


# ============================================================================
# Core logic
# ============================================================================

def extract_anchor(seq, nf=3, nb=3):
    """Return first nf + last nb residues, or None if too short."""
    if len(seq) < nf + nb:
        return None
    return seq[:nf] + seq[-nb:]


def _to_ords(anchor):
    """Convert 6-char anchor to tuple of ord values."""
    return (ord(anchor[0]), ord(anchor[1]), ord(anchor[2]),
            ord(anchor[3]), ord(anchor[4]), ord(anchor[5]))


def anchor_sim_fast(a, b, threshold):
    """
    Weighted BLOSUM62-normalized similarity with early termination.
    Checks high-weight positions first; bails if remaining positions
    can't push score above threshold. Returns score or -1.0 on early exit.
    """
    s = 0.0
    # Unrolled loop over _CHECK_ORDER = [1, 5, 0, 2, 3, 4]
    # Position index 1 (P2, weight 2×)
    s += _WSIM_ORD[0][a[1] * 128 + b[1]]
    if s + _REMAINING_AFTER[1] < threshold:
        return -1.0
    # Position index 5 (PΩ, weight 2×)
    s += _WSIM_ORD[1][a[5] * 128 + b[5]]
    if s + _REMAINING_AFTER[2] < threshold:
        return -1.0
    # Position index 0 (P1)
    s += _WSIM_ORD[2][a[0] * 128 + b[0]]
    if s + _REMAINING_AFTER[3] < threshold:
        return -1.0
    # Position index 2 (P3)
    s += _WSIM_ORD[3][a[2] * 128 + b[2]]
    # Position index 3 (PΩ-2)
    s += _WSIM_ORD[4][a[3] * 128 + b[3]]
    # Position index 4 (PΩ-1)
    s += _WSIM_ORD[5][a[4] * 128 + b[4]]
    return s


def block_key_fast(ords):
    """Reduced-alphabet key at P2 (idx 1) and PΩ (idx 5)."""
    return (_COARSE_ORD[ords[1]], _COARSE_ORD[ords[5]])


def parse_fasta(path):
    """Yield (header_id, full_sequence) from a FASTA file."""
    hdr = None
    parts = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(parts)
                hdr = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())
    if hdr is not None:
        yield hdr, "".join(parts)


def cluster_unique_anchors(anchor_counts, threshold):
    """
    Greedy centroid clustering on unique anchors.

    1. Convert anchors to ordinal tuples
    2. Partition into blocks by coarse(P2)+coarse(PΩ)
    3. Within each block, process anchors most-frequent-first
    4. Assign to first centroid above threshold, or become new centroid

    Returns: (dict  anchor_str → centroid_str, int comparison_count)
    """
    # Convert to ordinals; sort by frequency descending
    items = sorted(anchor_counts.items(), key=lambda x: -x[1])
    str_to_ords = {}
    blocks = defaultdict(list)
    for anchor_str, cnt in items:
        ords = _to_ords(anchor_str)
        str_to_ords[anchor_str] = ords
        blocks[block_key_fast(ords)].append(anchor_str)

    mapping = {}
    n_cmp = 0
    n_early = 0

    for bk, anchors in blocks.items():
        centroids_ords = []    # list of ord-tuples for centroids
        centroids_str  = []    # matching string anchors
        for anchor_str in anchors:
            a_ords = str_to_ords[anchor_str]
            matched = False
            for ci in range(len(centroids_ords)):
                n_cmp += 1
                score = anchor_sim_fast(a_ords, centroids_ords[ci], threshold)
                if score < 0:
                    n_early += 1
                    continue
                if score >= threshold:
                    mapping[anchor_str] = centroids_str[ci]
                    matched = True
                    break
            if not matched:
                centroids_ords.append(a_ords)
                centroids_str.append(anchor_str)
                mapping[anchor_str] = anchor_str

    return mapping, n_cmp, n_early


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="BLOSUM-aware anchor clustering for immunopeptides")
    ap.add_argument("-i", "--input",  required=True, help="Input FASTA")
    ap.add_argument("-o", "--outdir", default="anchor_clusters", help="Output dir")
    ap.add_argument("-t", "--threshold", type=float, default=0.6,
                    help="Similarity threshold (default: 0.6)")
    ap.add_argument("--min-cluster-size", type=int, default=2,
                    help="Min members for per-cluster FASTA (default: 2)")
    ap.add_argument("--n-front", type=int, default=3,
                    help="N-terminal anchor length (default: 3)")
    ap.add_argument("--n-back",  type=int, default=3,
                    help="C-terminal anchor length (default: 3)")
    args = ap.parse_args()

    outdir    = Path(args.outdir)
    fastadir  = outdir / "fasta"
    outdir.mkdir(parents=True, exist_ok=True)
    fastadir.mkdir(exist_ok=True)

    t0 = time.time()

    # ── Step 1: read & extract ────────────────────────────────────────
    print("[1/4] Reading FASTA and extracting anchors …", flush=True)
    peptides = []
    short    = []
    acounts  = defaultdict(int)

    for hdr, seq in parse_fasta(args.input):
        seq = seq.upper()
        anc = extract_anchor(seq, args.n_front, args.n_back)
        if anc is None:
            short.append((hdr, seq))
        else:
            peptides.append((hdr, seq, anc))
            acounts[anc] += 1

    n_total  = len(peptides) + len(short)
    n_unique = len(acounts)
    print(f"      {n_total:>12,}  total peptides")
    print(f"      {len(peptides):>12,}  valid (>={args.n_front + args.n_back} aa)")
    print(f"      {len(short):>12,}  too short")
    print(f"      {n_unique:>12,}  unique anchors")

    # ── Step 2: cluster unique anchors ────────────────────────────────
    backend = "Rust" if _HAS_RUST else "Python"
    print(f"\n[2/4] Clustering unique anchors (threshold {args.threshold}, "
          f"backend: {backend}) …", flush=True)

    if _HAS_RUST:
        mapping, n_cmp, n_early = cluster_anchors_rs(dict(acounts), args.threshold)
    else:
        mapping, n_cmp, n_early = cluster_unique_anchors(acounts, args.threshold)
    n_clusters = len(set(mapping.values()))
    print(f"      {n_clusters:>12,}  clusters")
    print(f"      {n_cmp:>12,}  pairwise comparisons (within blocks)")
    early_pct = 100.0 * n_early / n_cmp if n_cmp else 0
    print(f"      {n_early:>12,}  early-terminated ({early_pct:.1f}%)")

    # ── Step 3: assign peptides → clusters ────────────────────────────
    print("\n[3/4] Assigning peptides to clusters …", flush=True)
    clusters = defaultdict(list)
    for hdr, seq, anc in peptides:
        clusters[mapping[anc]].append((hdr, seq, anc))

    ranked = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    # ── Step 4: write outputs ─────────────────────────────────────────
    print("[4/4] Writing outputs …", flush=True)

    # 4a. cluster_summary.tsv
    with open(outdir / "cluster_summary.tsv", "w") as f:
        f.write("cluster_id\trepresentative_anchor\tsize\n")
        for idx, (ctr, members) in enumerate(ranked):
            f.write(f"cluster_{idx}\t{ctr}\t{len(members)}\n")

    # 4b. clusters.tsv  (full mapping)
    with open(outdir / "clusters.tsv", "w") as f:
        f.write("cluster_id\trepresentative_anchor\tpeptide_header"
                "\tsequence\tanchor\n")
        for idx, (ctr, members) in enumerate(ranked):
            cname = f"cluster_{idx}"
            for hdr, seq, anc in members:
                f.write(f"{cname}\t{ctr}\t{hdr}\t{seq}\t{anc}\n")

    # 4c. per-cluster FASTA
    n_fasta = 0
    for idx, (ctr, members) in enumerate(ranked):
        if len(members) < args.min_cluster_size:
            continue
        with open(fastadir / f"cluster_{idx}.fasta", "w") as f:
            for hdr, seq, _ in members:
                f.write(f">{hdr}\n{seq}\n")
        n_fasta += 1

    # 4d. short peptides
    if short:
        with open(fastadir / "SHORT_peptides.fasta", "w") as f:
            for hdr, seq in short:
                f.write(f">{hdr}\n{seq}\n")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    sizes   = [len(m) for _, m in ranked]

    report = [
        "=" * 62,
        "  ANCHOR CLUSTERING SUMMARY",
        "=" * 62,
        f"  Input:               {args.input}",
        f"  Total peptides:      {n_total:,}",
        f"  Valid peptides:      {len(peptides):,}",
        f"  Too short:           {len(short):,}",
        f"  Unique anchors:      {n_unique:,}",
        f"  Clusters:            {n_clusters:,}",
        f"  FASTA files:         {n_fasta:,}  (>={args.min_cluster_size} members)",
        "",
        f"  Threshold:           {args.threshold}",
        f"  Backend:             {backend}",
        f"  Matrix:              BLOSUM62 (normalized)",
        f"  Weights:             P1={_RAW_W[0]:.0f}  P2={_RAW_W[1]:.0f}  "
        f"P3={_RAW_W[2]:.0f}  PO-2={_RAW_W[3]:.0f}  PO-1={_RAW_W[4]:.0f}  "
        f"PO={_RAW_W[5]:.0f}",
        f"  Blocking:            coarse alphabet at P2 + PO  (10x10 = 100 bins)",
        f"  Comparisons:         {n_cmp:,}",
        f"  Early-terminated:    {n_early:,}  ({early_pct:.1f}%)",
        "",
        "  Cluster size distribution:",
        f"    singletons:        {sum(1 for s in sizes if s == 1):,}",
        f"    2-10:              {sum(1 for s in sizes if 2 <= s <= 10):,}",
        f"    11-100:            {sum(1 for s in sizes if 11 <= s <= 100):,}",
        f"    101-1000:          {sum(1 for s in sizes if 101 <= s <= 1000):,}",
        f"    >1000:             {sum(1 for s in sizes if s > 1000):,}",
        f"    largest:           {max(sizes):,}" if sizes else "",
        "",
        f"  Elapsed:             {elapsed:.1f} s",
        "=" * 62,
    ]

    with open(outdir / "summary.txt", "w") as f:
        for line in report:
            f.write(line + "\n")

    print()
    for line in report:
        print(line)


if __name__ == "__main__":
    main()