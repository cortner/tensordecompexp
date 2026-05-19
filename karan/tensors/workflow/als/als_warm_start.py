"""2D ALS CP with optional warm_start for factor matrices and lambda weights."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
import json
import os
from datetime import datetime


# ============================================================
# Function to Approximate (Matrix C)
# ============================================================
def f(x, y, C=None):
    """Generalized 2D test function with anisotropy via matrix C."""
    if C is None:
        C = np.eye(2) * 5.0

    coords = np.stack([x.ravel(), y.ravel()], axis=1)  # (N*M, 2)
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
# Reference Chebyshev coefficient tensor
# ============================================================
def compute_reference_coeffs(d_x, d_y, C):
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F = f(X, Y, C=C)

    Tx = chebvander(nodes_x, d_x)
    Ty = chebvander(nodes_y, d_y)

    A = np.kron(Ty, Tx)
    F_flat = F.ravel(order="F")

    coeffs_flat, *_ = np.linalg.lstsq(A, F_flat, rcond=None)
    C_interp = coeffs_flat.reshape((d_x + 1, d_y + 1), order="F")
    return C_interp


# ============================================================
# ALS Training Function
# ============================================================
def run_als_experiment(
    N=64, M=64, d_x=63, d_y=63, R=10, n_iter=500,
    C=np.eye(2) * 5.0, resolution=100, num_test_points=2048,
    epsilon=1e-2, random_seed=42, outdir="als_results",
    train_points="chebyshev", test_points="uniform",
    use_lambda=True,
    warm_start=None,
):
    """
    Runs ALS-based CP decomposition on 2D function f(x,y) with anisotropy matrix C.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)

    np.random.seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"als_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # -------------------- Training Setup --------------------
    x_nodes = generate_nodes(N, train_points)
    y_nodes = generate_nodes(M, train_points)
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    F = f(X, Y, C=C)

    Tx = chebvander(x_nodes, d_x)
    Ty = chebvander(y_nodes, d_y)

    # Warm-start logic
    if warm_start is not None:
        prev_A, prev_B, prev_lambdas = warm_start
        old_R = prev_A.shape[1]

        A_coeffs = np.zeros((d_x+1, R))
        B_coeffs = np.zeros((d_y+1, R))
        lambdas  = np.ones(R)

        # copy over old factors
        A_coeffs[:, :old_R] = prev_A
        B_coeffs[:, :old_R] = prev_B
        lambdas[:old_R] = prev_lambdas

        # random init for new column(s)
        if R > old_R:
            A_coeffs[:, old_R:] = np.random.randn(d_x+1, R-old_R)
            B_coeffs[:, old_R:] = np.random.randn(d_y+1, R-old_R)

    else:
        A_coeffs = np.random.randn(d_x + 1, R)
        B_coeffs = np.random.randn(d_y + 1, R)
        lambdas  = np.ones(R) if use_lambda else np.ones(R)

    # -------------------- ALS Iterations --------------------
    final_train_rmse = None
    for it in range(n_iter):
        A_eval = Tx @ A_coeffs
        B_eval = Ty @ B_coeffs
        F_hat = sum(lambdas[r] * np.outer(A_eval[:, r], B_eval[:, r]) for r in range(R))

        err = np.sqrt(np.mean((F - F_hat) ** 2))
        final_train_rmse = err
        if (it + 1) % 50 == 0:
            print(f"Iter {it+1}, RMSE: {err:.2e}")

        if np.isnan(err) or err > 1e5:
            print("Stopping due to divergence.")
            break

        # Update A_coeffs
        for r in range(R):
            Fr = F - sum(
                lambdas[k] * np.outer(Tx @ A_coeffs[:, k], Ty @ B_coeffs[:, k])
                for k in range(R) if k != r
            )
            Z = Ty @ B_coeffs[:, r]
            rhs = Fr @ Z
            if use_lambda and lambdas[r] != 0:
                rhs /= lambdas[r]
            lhs = Tx.T @ Tx + epsilon * np.eye(Tx.shape[1])
            A_coeffs[:, r] = np.linalg.solve(lhs, Tx.T @ rhs)
            A_coeffs[:, r] /= np.linalg.norm(A_coeffs[:, r])

        # Update B_coeffs
        for r in range(R):
            Fr = F - sum(
                lambdas[k] * np.outer(Tx @ A_coeffs[:, k], Ty @ B_coeffs[:, k])
                for k in range(R) if k != r
            )
            Z = Tx @ A_coeffs[:, r]
            rhs = Fr.T @ Z
            if use_lambda and lambdas[r] != 0:
                rhs /= lambdas[r]
            lhs = Ty.T @ Ty + epsilon * np.eye(Ty.shape[1])
            B_coeffs[:, r] = np.linalg.solve(lhs, Ty.T @ rhs)
            B_coeffs[:, r] /= np.linalg.norm(B_coeffs[:, r])

        # Update lambdas if enabled
        if use_lambda:
            for r in range(R):
                A_r = Tx @ A_coeffs[:, r]
                B_r = Ty @ B_coeffs[:, r]
                num = np.sum(F * np.outer(A_r, B_r))
                denom = np.sum((np.outer(A_r, B_r)) ** 2)
                lambdas[r] = num / (denom if denom > 1e-12 else 1e-12)

    # -------------------- Evaluation --------------------
    F_reconstructed = sum(
        lambdas[r] * np.outer(Tx @ A_coeffs[:, r], Ty @ B_coeffs[:, r])
        for r in range(R)
    )
    l2_norm_error = np.linalg.norm(F - F_reconstructed)

    # Dense evaluation grid
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    F_true_eval = f(X_eval, Y_eval, C=C)
    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    A_eval_pred = Tx_eval @ A_coeffs
    B_eval_pred = Ty_eval @ B_coeffs
    F_pred_eval = sum(
        lambdas[r] * np.outer(A_eval_pred[:, r], B_eval_pred[:, r]) for r in range(R)
    )
    rmse_eval = np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2))

    # Test points
    x_rand = generate_nodes(num_test_points, test_points)
    y_rand = generate_nodes(num_test_points, test_points)
    Tx_rand = chebvander(x_rand, d_x)
    Ty_rand = chebvander(y_rand, d_y)
    F_pred_rand = np.zeros(num_test_points)
    for r in range(R):
        F_pred_rand += lambdas[r] * (Tx_rand @ A_coeffs[:, r]) * (Ty_rand @ B_coeffs[:, r])
    F_true_rand = f(x_rand, y_rand, C=C)
    rmse_rand = np.sqrt(np.mean((F_true_rand - F_pred_rand) ** 2))
    maxe_rand = np.max(np.abs(F_true_rand - F_pred_rand))

    # -------------------- Coefficient Tensor Comparison --------------------
    C_als = np.zeros((d_x+1, d_y+1))
    for r in range(R):
        C_als += lambdas[r] * np.outer(A_coeffs[:, r], B_coeffs[:, r])

    C_interp = compute_reference_coeffs(d_x, d_y, C)

    l2_coeff_error = np.linalg.norm(C_interp - C_als)
    rel_l2_coeff_error = l2_coeff_error / np.linalg.norm(C_interp)

    # -------------------- Save Results --------------------
    metrics = {
        "final_train_rmse": float(final_train_rmse),
        "l2_norm_error_train": float(l2_norm_error),
        "rmse_eval_grid": float(rmse_eval),
        "rmse_test_points": float(rmse_rand),
        "maxe_test_points": float(maxe_rand),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    coeffs_data = {
        "A_coeffs": A_coeffs.tolist(),
        "B_coeffs": B_coeffs.tolist(),
        "lambdas": lambdas.tolist(),
        "C_als": C_als.tolist(),
        "C_interp": C_interp.tolist(),
    }
    with open(os.path.join(outdir, "coeffs.json"), "w") as f_out:
        json.dump(coeffs_data, f_out, indent=4)

    config = {
        "N": N, "M": M,
        "d_x": d_x, "d_y": d_y,
        "R": R, "n_iter": n_iter,
        "C": C.tolist(),
        "resolution": resolution,
        "num_test_points": num_test_points,
        "epsilon": epsilon,
        "random_seed": random_seed,
        "timestamp": timestamp,
        "train_points": train_points,
        "test_points": test_points,
        "use_lambda": use_lambda,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # -------------------- Plots --------------------
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, F, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_title(f'Original f(x,y) ({train_points} training)')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, F_reconstructed, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_title(f'Reconstructed (Rank {R}, use_lambda={use_lambda})')

    plt.tight_layout()
    plot_path = os.path.join(outdir, "reconstruction.png")
    plt.savefig(plot_path)
    plt.close(fig)

    return metrics, coeffs_data, config, outdir, (A_coeffs, B_coeffs, lambdas)


# ============================================================
# Run Example
# ============================================================
if __name__ == "__main__":
    def rotation_matrix(theta_deg):
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    C_matrices = [
        np.array([[5, 0], [0, 1]]),        # vertical
        np.array([[1, 0], [0, 5]]),        # horizontal
        np.array([[1, 0], [0, 5]]) @ rotation_matrix(30),  # tilted ellipse
        5 * np.eye(2),                     # steeper dome
        0.2 * np.eye(2),                   # flatter dome
    ]

    # Collect everything
    all_results = {}

    for idx, C in enumerate(C_matrices):
        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        prev_factors = None

        for R in range(1, 21):  # sweep ranks 1 → 20
            print(f"\n=== Running ALS experiment for C={idx}, Rank={R} ===")

            metrics, coeffs, config, savedir, prev_factors = run_als_experiment(
                N=64, M=64,
                d_x=63, d_y=63,
                R=R, n_iter=1000,
                C=C,
                train_points="chebyshev",
                test_points="uniform",
                num_test_points=2048,
                resolution=120,
                random_seed=42,
                outdir="als_results",
                use_lambda=True,
                warm_start=prev_factors,
            )

            # Save minimal info for plotting
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config
            }

    # Save as one JSON file
    with open("als_true_warm_rank_sweep_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll ALS runs complete. Results saved to als_true_warm_rank_sweep_results.json")