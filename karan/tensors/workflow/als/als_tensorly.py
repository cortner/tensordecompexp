"""2D one-shot TensorLy Tucker on the value tensor, then LS fit of Chebyshev factors (no ALS loop)."""

# now we would talk about same spaces here 
# value space and coeficient space
# by including the weights so the system is orthogonal
# which makes minimising the function equivalent to minimising the tensor

#  ======================================
#  SIMPLE TUCKER + CHEBYSHEV CONVERSION
#  ======================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
import json
import os
from datetime import datetime

import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker
tl.set_backend("numpy")


# ============================================================
# Function to Approximate (Matrix C)
# ============================================================
def f(x, y, C=None):
    """Generalized 2D test function with anisotropy via matrix C."""
    if C is None:
        C = np.eye(2) * 5.0
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    trans = coords @ C.T
    vals = 1.0 / (1.0 + np.sum(trans**2, axis=1))
    return vals.reshape(x.shape)


# ============================================================
# Node Generators
# ============================================================
def generate_nodes(num_points, mode="chebyshev"):
    """Generate 1D nodes in [-1,1]."""
    if mode == "chebyshev":
        return np.cos(np.pi * np.arange(num_points) / (num_points - 1))
    elif mode == "uniform":
        return np.linspace(-1, 1, num_points)
    elif mode == "random":
        return np.random.uniform(-1, 1, num_points)
    else:
        raise ValueError(f"Unknown mode {mode}")


# ============================================================
# True Chebyshev coefficient tensor (interpolation)
# ============================================================
def compute_reference_coeffs(d_x, d_y, C):
    """
    Compute the true Chebyshev coefficients of f(x,y) by interpolation.
    Uses (d_x+1) × (d_y+1) Chebyshev grid.
    """
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)

    X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F = f(X, Y, C=C)
    F_flat = F.ravel(order="F")

    Tx = chebvander(nodes_x, d_x)
    Ty = chebvander(nodes_y, d_y)

    # Build Kronecker system
    A = np.kron(Ty.T, Tx.T)

    coeffs_flat, *_ = np.linalg.lstsq(A, F_flat, rcond=None)
    coeffs = coeffs_flat.reshape((d_x + 1, d_y + 1), order="F")
    return coeffs


# ============================================================
# UPDATED SIMPLE TUCKER EXPERIMENT (NO ALS)
# ============================================================
def run_als_tucker_experiment(
    N=64, M=64, d_x=63, d_y=63,
    R_x=10, R_y=10, n_iter=500,
    C=np.eye(2) * 5.0, resolution=100,
    num_test_points=2048, epsilon=1e-6, random_seed=42,
    outdir="als_tucker_results", train_points="chebyshev",
    test_points="uniform",
):
    """
    Runs Tucker decomposition on 2D function f(x,y).
    Replaces ALS with: TensorLy Tucker + LS fit to Chebyshev basis.

    The outer structure (JSON saving, plotting, configs) remains unchanged.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    np.random.seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"tucker_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # -------------------- Training Setup --------------------
    x_nodes = generate_nodes(N, train_points)
    y_nodes = generate_nodes(M, train_points)

    X, Y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    F = f(X, Y, C=C)  # training tensor (N × M)

    # -------------------- TensorLy Tucker --------------------
    print("Running TensorLy Tucker...")
    tucker_tensor, rec_errors = tl_tucker(
        F,
        rank=[R_x, R_y],
        n_iter_max=n_iter,
        init="svd",
        tol=1e-8,
        random_state=random_seed,
        return_errors=True,
    )

    core_val, (Ux, Uy) = tucker_tensor   # Ux: N×R_x, Uy: M×R_y
    F_tucker_val = tl.tucker_to_tensor(tucker_tensor)
    tucker_rmse = np.sqrt(np.mean((F - F_tucker_val)**2))

    print(f"Value-space Tucker RMSE: {tucker_rmse:.3e}")

    # -------------------- Fit Tucker factors in Chebyshev basis --------------------
    Tx = chebvander(x_nodes, d_x)  # N×(d_x+1)
    Ty = chebvander(y_nodes, d_y)  # M×(d_y+1)

    # LS solves (no weights, no normalization)
    A, *_ = np.linalg.lstsq(Tx, Ux, rcond=None)  # (d_x+1) × R_x
    B, *_ = np.linalg.lstsq(Ty, Uy, rcond=None)  # (d_y+1) × R_y

    # -------------------- Build coefficient tensor --------------------
    C_tucker = A @ core_val @ B.T  # Chebyshev coefficient tensor

    # Reconstruction using Chebyshev representation
    F_reconstructed = (Tx @ A) @ core_val @ (Ty @ B).T

    diff_final = F - F_reconstructed
    rmse_eval = np.sqrt(np.mean(diff_final**2))
    l2_norm_error = np.linalg.norm(diff_final)

    # -------------------- Compare with true Chebyshev coeffs --------------------
    C_interp = compute_reference_coeffs(d_x, d_y, C)

    l2_coeff_error = np.linalg.norm(C_interp - C_tucker)
    rel_l2_coeff_error = l2_coeff_error / np.linalg.norm(C_interp)

    metrics = {
        "final_train_weighted_rmse": float(rmse_eval),
        "rmse_eval_grid": float(rmse_eval),
        "l2_norm_error_train": float(l2_norm_error),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error),
    }

    # -------------------- Save Results --------------------
    with open(os.path.join(outdir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    np.savez(
        os.path.join(outdir, "coeffs.npz"),
        A=A, B=B, G=core_val, C_tucker=C_tucker,
        Ux=Ux, Uy=Uy
    )

    config = {
        "N": N, "M": M, "d_x": d_x, "d_y": d_y,
        "R_x": R_x, "R_y": R_y, "n_iter": n_iter,
        "C": C.tolist(), "resolution": resolution,
        "num_test_points": num_test_points,
        "epsilon": epsilon, "random_seed": random_seed,
        "timestamp": timestamp,
        "train_points": train_points,
        "test_points": test_points,
    }

    with open(os.path.join(outdir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # -------------------- Plot --------------------
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, F, cmap=cm.viridis)
    ax1.set_title(f'Original f(x,y) ({train_points} training)')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, F_reconstructed, cmap=cm.viridis)
    ax2.set_title(f'Tucker Reconstructed (R_x={R_x}, R_y={R_y})')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reconstruction.png"))
    plt.close(fig)

    return metrics, {"A": A, "B": B, "G": core_val}, config, outdir


# ============================================================
# Rank-Sweep Example
# ============================================================
if __name__ == "__main__":
    def rotation_matrix(theta_deg):
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    C_matrices = [
        np.array([[5, 0], [0, 1]]),
        np.array([[1, 0], [0, 5]]),
        np.array([[1, 0], [0, 5]]) @ rotation_matrix(30),
        5 * np.eye(2),
        0.2 * np.eye(2),
    ]

    all_results = {}

    for idx, C in enumerate(C_matrices):
        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        for R in range(1, 41):
            print(f"\n=== Running Tucker experiment for C={idx}, Rank={R} ===")
            metrics, coeffs, config, savedir = run_als_tucker_experiment(
                N=64, M=64,
                d_x=63, d_y=63,
                R_x=R, R_y=R,
                n_iter=100,
                C=C,
                train_points="chebyshev",  # choose "uniform" or "random"
                test_points="uniform",
                num_test_points=2048,
                resolution=120,
                random_seed=42,
                outdir="als_tucker_cheb_results"
            )

            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config
            }

    with open("als_tucker_tensorly_cheb.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll Tucker runs complete. Results saved to als_tucker_tensorly_cheb.json")