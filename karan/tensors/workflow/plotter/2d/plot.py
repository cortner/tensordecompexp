"""Plot selected metrics from saved JSON rank-sweep results (Adam/BFGS/ALS Tucker)."""

import json
import matplotlib.pyplot as plt
import os

# Only these metrics will be plotted
SELECTED_METRICS = [
    "final_train_rmse",
    "l2_norm_error_train",
    "rmse_eval_grid",
    "rmse_test_points",
    "maxe_test_points",
    "l2_coeff_error",
    "rel_l2_coeff_error",
    # "final_loss_mse",
]

FILES = {
    # "bfgs_true_rank_sweep_results.json": "BFGS (true)",
    # "bfgs_false_rank_sweep_results.json": "BFGS (false)",
    # "adam_true_rank_sweep_results.json": "Adam (true)",
    # "adam_false_rank_sweep_results.json": "Adam (false)",
    # "als_true_rank_sweep_results.json": "ALS (true)",
    # "als_false_rank_sweep_results.json": "ALS (false)",
    "adam_tucker_rank_sweep_results.json": "Adam Tucker",
    "bfgs_tucker_rank_sweep_results.json": "BFGS Tucker",
}

def plot_all_methods(files=FILES, outdir="combined_plots"):
    os.makedirs(outdir, exist_ok=True)

    # Load all results
    results = {}
    for fname, label in files.items():
        with open(fname, "r") as f:
            results[label] = json.load(f)

    # All case names (assume same across files)
    case_names = list(next(iter(results.values())).keys())

    for case_name in case_names:
        for metric in SELECTED_METRICS:
            plt.figure(figsize=(8,6))

            for label, res in results.items():
                ranks_data = res[case_name]
                ranks = sorted(int(r.split("_")[1]) for r in ranks_data.keys())
                values = [ranks_data[f"rank_{R}"]["metrics"][metric] for R in ranks]
                plt.plot(ranks, values, marker="o", label=label)

            plt.yscale("log")  # Log scale for readability
            plt.xlabel("Rank R")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Rank ({case_name})")
            plt.grid(True, which="both", ls="--", alpha=0.7)
            plt.legend()

            fname = f"{case_name}_{metric}_all_methods.png"
            plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"Plots saved in {outdir}/")

if __name__ == "__main__":
    plot_all_methods()
