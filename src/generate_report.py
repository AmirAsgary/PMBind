import sys
from collections import defaultdict
import statistics
import os

# Grab the paths passed from the Bash script
if len(sys.argv) < 3:
    print("Usage: python3 generate_report.py <input_fasta> <output_dir>")
    sys.exit(1)

FASTA_FILE = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# Dynamically construct your cluster file paths
# FIXED: Renamed CLUSTER_FILE to TSV_FILE
TSV_FILE = os.path.join(OUTPUT_DIR, "cluster_final_cluster.tsv")
REP_SEQ_FILE = os.path.join(OUTPUT_DIR, "cluster_final_rep_seq.fasta")

def main():
    print("1. Loading original sequences into memory...")
    header_to_seq = {}
    with open(FASTA_FILE, "r") as f:
        current_header = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_header = line[1:].split()[0] # Remove '>' and take first word
            elif current_header:
                header_to_seq[current_header] = line
                current_header = "" # Assumes 1 sequence line per header (standard for peptides)

    print("2. Parsing MMseqs2 cluster TSV...")
    # Dictionary to hold: cluster_representative_header -> list of member headers
    clusters = defaultdict(list)
    with open(TSV_FILE, "r") as f:
        for line in f:
            rep, member = line.strip().split('\t')
            clusters[rep].append(member)

    print("3. Calculating statistics...\n")
    
    # Calculate sizes
    cluster_sizes = [len(members) for members in clusters.values()]
    total_clusters = len(cluster_sizes)
    singletons = sum(1 for size in cluster_sizes if size == 1)
    
    # Sort clusters by size (descending)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    # --- GENERATE REPORT ---
    print("="*50)
    print(" CLUSTERING REPORT (3-First / 3-Last Strategy)")
    print("="*50)
    print(f"Total Peptides Processed : {sum(cluster_sizes):,}")
    print(f"Total Unique Clusters    : {total_clusters:,}")
    print(f"Number of Singletons     : {singletons:,} ({(singletons/total_clusters)*100:.1f}% of clusters)")
    print("-" * 50)
    print("SIZE DISTRIBUTION:")
    print(f"  Maximum Size : {max(cluster_sizes):,}")
    print(f"  Mean Size    : {statistics.mean(cluster_sizes):.1f}")
    print(f"  Median Size  : {statistics.median(cluster_sizes):.1f}")
    
    print("\n" + "="*50)
    print(" TOP 5 LARGEST CLUSTERS (Representatives)")
    print("="*50)
    for i in range(min(5, total_clusters)):
        rep_header, members = sorted_clusters[i]
        true_seq = header_to_seq.get(rep_header, "[Sequence Not Found]")
        print(f"{i+1}. Size: {len(members):<8} | Original Seq: {true_seq} | Header: {rep_header}")

    print("\n" + "="*50)
    print(" BOTTOM 5 SMALLEST CLUSTERS (Representatives)")
    print("="*50)
    # Get the last 5 elements, reversed so it counts up from the absolute smallest
    for i, (rep_header, members) in enumerate(reversed(sorted_clusters[-5:])):
        true_seq = header_to_seq.get(rep_header, "[Sequence Not Found]")
        print(f"{5-i}. Size: {len(members):<8} | Original Seq: {true_seq} | Header: {rep_header}")
    print("="*50)

if __name__ == "__main__":
    main()