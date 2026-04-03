import re
import csv
import json
import argparse
import os


def parse_log(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    chunks = re.split(r"--- Gibbs Iteration (\d+)/\d+ ---", text)

    results = []

    for i in range(1, len(chunks), 2):
        iteration = int(chunks[i])
        block = chunks[i + 1]

        data = {"iteration": iteration}

        def extract(pattern, name, cast=float, group=1):
            m = re.search(pattern, block)
            if m:
                try:
                    data[name] = cast(m.group(group).replace(",", ""))
                except:
                    data[name] = None
            else:
                data[name] = None

        # θ
        extract(r"Sampled theta: ([\d,]+)/", "theta_binders", int)
        extract(r"\(([\d.]+)%\)", "theta_pct")

        # alpha / beta
        extract(r"alpha: mean=([\d.]+)", "alpha_mean")
        extract(r"beta: mean=([\d.]+)", "beta_mean")

        # p_h
        extract(r"p_h: mean=([\d.]+)", "ph_mean")
        extract(r"p_h: mean=[\d.]+, median=([\d.]+)", "ph_median")

        # gamma
        extract(r"gamma: mean=([\d.]+)", "gamma_mean")
        extract(r"gamma: mean=[\d.]+, median=([\d.]+)", "gamma_median")
        extract(r">0.5: ([\d,]+)/", "gamma_gt_half", int)

        # tau_h
        extract(r"tau_h: mean=([\d.]+)", "tau_mean")
        extract(r"tau_h: mean=[\d.]+, median=([\d.]+)", "tau_median")

        # tau_max cap
        extract(r"tau_max cap applied to (\d+)/", "tau_cap_count", int)

        # positive calls
        extract(r"Positive calls: ([\d,]+)/", "positive_calls", int)
        extract(r"Positive calls: [\d,]+/[\d,]+ \(([\d.]+)%\)", "positive_pct")

        # delta gamma
        extract(r"max\|delta gamma\| = ([\d.]+)", "delta_gamma_max")
        extract(r"mean\|delta\| = ([\d.]+)", "delta_gamma_mean")

        # L2
        extract(r"Significant pairs: ([\d,]+) /", "l2_significant_pairs", int)
        extract(r"Non-zero entries: ([\d,]+)", "l2_nonzero_entries", int)

        # L3
        extract(r"Propagated pairs: ([\d,]+)", "l3_propagated", int)
        extract(r"Positive-leaning \(p_tilde > 0.5\): ([\d,]+)", "l3_positive", int)
        extract(r"Mean lambda: ([\d.]+)", "l3_lambda_mean")

        # iteration time
        extract(r"Iteration \d+ done \(([\d.]+)s\)", "iteration_time")

        results.append(data)

    return results


def save_csv(data, output_path):
    if not data:
        print("No data extracted.")
        return []

    keys = sorted({k for d in data for k in d.keys()})

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    return keys


def save_column_mapping(columns, output_csv_path):
    mapping = {
        "iteration": "Gibbs iteration number",

        "theta_binders": "Number of sampled binders (theta)",
        "theta_pct": "Percentage of binders (theta)",

        "alpha_mean": "Mean of alpha parameter (M level)",
        "beta_mean": "Mean of beta parameter (M level)",

        "ph_mean": "Mean of p_h parameter",
        "ph_median": "Median of p_h parameter",

        "gamma_mean": "Mean posterior binding probability (gamma)",
        "gamma_median": "Median posterior binding probability (gamma)",
        "gamma_gt_half": "Number of entries where gamma > 0.5",

        "tau_mean": "Mean tau_h threshold",
        "tau_median": "Median tau_h threshold",

        "tau_cap_count": "Number of alleles where tau_max cap was applied",

        "positive_calls": "Number of positive binding calls",
        "positive_pct": "Percentage of positive binding calls",

        "delta_gamma_max": "Maximum absolute change in gamma",
        "delta_gamma_mean": "Mean absolute change in gamma",

        "l2_significant_pairs": "Number of significant HLA pairs after FDR correction",
        "l2_nonzero_entries": "Number of non-zero entries in similarity matrix",

        "l3_propagated": "Number of propagated pairs in label propagation",
        "l3_positive": "Number of positive-leaning propagated pairs",
        "l3_lambda_mean": "Mean lambda in label propagation",

        "iteration_time": "Total time per iteration in seconds",
    }

    # Keep only columns that actually exist
    filtered_mapping = {col: mapping.get(col, col) for col in columns}

    output_dir = os.path.dirname(os.path.abspath(output_csv_path))
    json_path = os.path.join(output_dir, "csv_column_full.json")

    with open(json_path, "w") as f:
        json.dump(filtered_mapping, f, indent=4)

    print(f"Column mapping saved → {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse Gibbs log to CSV + JSON mapping")
    parser.add_argument("input_log", help="Path to log file")
    parser.add_argument("output_csv", help="Path to output CSV")

    args = parser.parse_args()

    data = parse_log(args.input_log)
    columns = save_csv(data, args.output_csv)

    save_column_mapping(columns, args.output_csv)

    print(f"Parsed {len(data)} iterations → {args.output_csv}")


if __name__ == "__main__":
    main()