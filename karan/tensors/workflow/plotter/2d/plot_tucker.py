"""Plot Tucker/ALS experiment metrics from JSON sweep files with configurable labels."""

import json
import matplotlib.pyplot as plt
import os
import re

# Metrics to plot
SELECTED_METRICS = [
    "final_train_rmse",
    "l2_norm_error_train",
    "rmse_eval_grid",
    "rmse_test_points",
    "maxe_test_points",
    "l2_coeff_error",
    "rel_l2_coeff_error",
]

# Files to compare (edit as needed)
FILES = {
    # "adam_tucker_rank_sweep_results.json": "Adam Tucker",
    # "bfgs_tucker_rank_sweep_results.json": "BFGS Tucker",
    # "als_tucker_tensorly_sin.json": "ALS Tucker Cheb (fcn2)",
    # "als_tucker_tensorly_cheb.json": "ALS Tucker Cheb",
    # "als_tucker_classic_results.json": "ALS Tucker Cheb (Ill conditioned)",
    # "als_tucker_tensorly_random.json": "ALS Tucker Random",
    # "als_tucker3d_tensorly_cheb.json": "ALS Tucker 3D Cheb",
    # "tucker_fcn2.json": "Tucker Baseline (fcn2)",
    # "als_tucker3d_tensorly_cheb.json": "Tucker Baseline 3D (Cheb)",
    # "als_tucker_zulip_results.json": "Chebyshev ALS Method",
    # "tucker_rank_sweep_cheb_16.json": "Tucker Baseline (Cheb 16)",
    # "tucker_rank_sweep_cheb_32.json": "Tucker Baseline (Cheb 32)",
    # "tucker_rank_sweep_cheb_64.json": "Tucker Baseline (Cheb 64)",
    # "ace_tucker_zulip_results.json": "ACE-Tucker (Cheb 32)",
    # "ace_tucker_64_zulip_results.json": "ACE-Tucker (Cheb 64)",
    # "ace_tucker_3d_zulip_results.json": "ACE-Tucker (Cheb 32 3D)",
    # "tucker_rank_sweep_random_320.json": "Tucker Baseline (Random 320)",
    # "tucker_rank_sweep_random_640.json": "Tucker Baseline (Random 640)",
    # "tucker_rank_sweep_random_960.json": "Tucker Baseline (Random 960)",
    # "tucker_rank_sweep_random_1280.json": "Tucker Baseline (Random 1280)",
    # "als_tucker_zulip_uniform320_results.json": "ALS Tucker Uniform 320",
    # "als_tucker_zulip_uniform448_results.json": "ALS Tucker Uniform 448",
    # "als_tucker_zulip_uniform_results.json": "ALS Tucker Uniform 640",
    # "als_tucker_zulip_random320_results.json": "ALS Tucker Random 320",
    # "als_tucker_zulip_random640_results.json": "ALS Tucker Random 640",
    # "als_tucker_zulip_random960_results.json": "ALS Tucker Random 960",
    # "als_tucker_zulip_random_results.json": "ALS Tucker Random 1280",
    # "als_tucker_zulip_chebyshev16_results.json": "ALS Tucker Cheb 16",
    # "als_tucker_zulip_chebyshev32_results.json": "ALS Tucker Cheb 32",
    # "als_tucker_zulip_results.json": "ALS Tucker Cheb 64",
    "ace_tucker_3d_zulip_results_noo.json": "ACE-Tucker (Cheb 32 3D)",
    "ace_sym_tucker_3d_zulip_results_noo64.json": "ACE-Tucker (Cheb 64 3D)",
}

def extract_rank(key):
    """
    Extract integer rank from keys like:
    - 'rank_3'       -> 3
    - 'rank_3x3'     -> 3
    - 'rank_10x10'   -> 10
    - fallback: None
    """
    match = re.search(r"rank_(\d+)(?:x(\d+))?", key)
    if not match:
        return None
    R1 = int(match.group(1))
    R2 = int(match.group(2)) if match.group(2) else R1
    return (R1 + R2) / 2  # same if R1 == R2, otherwise midpoint

def plot_all_methods(files=FILES, outdir="combined_plots_tucker"):
    os.makedirs(outdir, exist_ok=True)

    # Load all JSON results
    results = {}
    for fname, label in files.items():
        if not os.path.exists(fname):
            print(f"⚠️ File not found: {fname}")
            continue
        with open(fname, "r") as f:
            results[label] = json.load(f)

    if not results:
        print("❌ No result files found. Check file paths.")
        return

    # Extract all case names (e.g., 'C_case_0', 'C_case_1', etc.)
    case_names = list(next(iter(results.values())).keys())

    for case_name in case_names:
        for metric in SELECTED_METRICS:
            plt.figure(figsize=(8, 6))

            for label, res in results.items():
                ranks_data = res.get(case_name, {})
                rank_values = []
                metric_values = []

                for rkey, entry in ranks_data.items():
                    rank_val = extract_rank(rkey)
                    if rank_val is None or rank_val > 20:
                        continue
                    metrics = entry.get("metrics", {})
                    if metric not in metrics:
                        continue
                    rank_values.append(rank_val)
                    metric_values.append(metrics[metric])

                if not rank_values:
                    continue

                sorted_pairs = sorted(zip(rank_values, metric_values))
                ranks_sorted, vals_sorted = zip(*sorted_pairs)
                plt.plot(ranks_sorted, vals_sorted, marker="o", label=label)

            plt.yscale("log")
            plt.xlabel("Tucker Rank (Rₓ = Rᵧ)")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Rank ({case_name})")
            plt.grid(True, which="both", ls="--", alpha=0.7)
            plt.legend()

            # plt.ylim(bottom=1e-10)

            fname = f"{case_name}_{metric}_all_methods_tucker.png"
            plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"✅ Plots saved in {outdir}/")

if __name__ == "__main__":
    plot_all_methods()
