"""Rank-1 ALS CP decomposition on a 3D Chebyshev tensor grid for a radial test function."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
import json
import os
from datetime import datetime


# ============================================================
# Function to Approximate (3D)
# ============================================================
def f(x, y, z, c=5):
    return 1 / (1 + c**2 * (x**2 + y**2 + z**2))


# ============================================================
# Node Generators
# ============================================================
def generate_nodes(num_points, mode="chebyshev"):
    if mode == "chebyshev":
        return np.cos(np.pi * np.arange(num_points) / (num_points - 1))
    elif mode == "uniform":
        return np.linspace(-1, 1, num_points)
    elif mode == "random":
        return np.random.uniform(-1, 1, num_points)
    else:
        raise ValueError(f"Unknown mode {mode}")


# ============================================================
# ALS Training Function (3D)
# ============================================================
def run_als_experiment_3d(
    N=16, M=16, L=16,
    d_x=15, d_y=15, d_z=15,
    R=5, n_iter=200,
    c=5, resolution=20, num_test_points=512,
    epsilon=1e-8, random_seed=42, outdir="als3d_results",
    train_points="chebyshev", test_points="uniform"
):
    """
    Runs ALS-based CP decomposition on 3D function f(x,y,z).
    Training and testing nodes can be "uniform", "chebyshev", or "random".
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)

    np.random.seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"als3d_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # -------------------- Training Setup --------------------
    x_nodes = generate_nodes(N, train_points)
    y_nodes = generate_nodes(M, train_points)
    z_nodes = generate_nodes(L, train_points)
    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
    F = f(X, Y, Z, c=c)

    Tx = chebvander(x_nodes, d_x)
    Ty = chebvander(y_nodes, d_y)
    Tz = chebvander(z_nodes, d_z)

    A_coeffs = np.random.randn(d_x + 1, R)
    B_coeffs = np.random.randn(d_y + 1, R)
    C_coeffs = np.random.randn(d_z + 1, R)
    lambdas = np.ones(R)

    # -------------------- ALS Iterations --------------------
    final_train_rmse = None
    for it in range(n_iter):
        A_eval = Tx @ A_coeffs
        B_eval = Ty @ B_coeffs
        C_eval = Tz @ C_coeffs
        F_hat = np.zeros_like(F)
        for r in range(R):
            F_hat += lambdas[r] * np.einsum('i,j,k->ijk', A_eval[:, r], B_eval[:, r], C_eval[:, r])

        err = np.sqrt(np.mean((F - F_hat) ** 2))
        final_train_rmse = err
        print(f"Iter {it+1}, RMSE: {err:.2e}")

        if np.isnan(err) or err > 1e5:
            print("Stopping due to divergence.")
            break

        # Update A_coeffs
        for r in range(R):
            Fr = F - sum(
                lambdas[k] * np.einsum('i,j,k->ijk', Tx @ A_coeffs[:, k], Ty @ B_coeffs[:, k], Tz @ C_coeffs[:, k])
                for k in range(R) if k != r
            )
            ZB = np.outer(Ty @ B_coeffs[:, r], Tz @ C_coeffs[:, r])  # (M, L)
            rhs = np.tensordot(Fr, ZB, axes=([1, 2], [0, 1]))       # (N,)
            lhs = Tx.T @ Tx + epsilon * np.eye(Tx.shape[1])
            A_coeffs[:, r] = np.linalg.solve(lhs, Tx.T @ rhs)
            A_coeffs[:, r] /= np.linalg.norm(A_coeffs[:, r])

        # Update B_coeffs
        for r in range(R):
            Fr = F - sum(
                lambdas[k] * np.einsum('i,j,k->ijk', Tx @ A_coeffs[:, k], Ty @ B_coeffs[:, k], Tz @ C_coeffs[:, k])
                for k in range(R) if k != r
            )
            ZA = np.outer(Tx @ A_coeffs[:, r], Tz @ C_coeffs[:, r])  # (N, L)
            rhs = np.tensordot(Fr, ZA, axes=([0, 2], [0, 1]))        # (M,)
            lhs = Ty.T @ Ty + epsilon * np.eye(Ty.shape[1])
            B_coeffs[:, r] = np.linalg.solve(lhs, Ty.T @ rhs)
            B_coeffs[:, r] /= np.linalg.norm(B_coeffs[:, r])

        # Update C_coeffs
        for r in range(R):
            Fr = F - sum(
                lambdas[k] * np.einsum('i,j,k->ijk', Tx @ A_coeffs[:, k], Ty @ B_coeffs[:, k], Tz @ C_coeffs[:, k])
                for k in range(R) if k != r
            )
            ZAB = np.outer(Tx @ A_coeffs[:, r], Ty @ B_coeffs[:, r])  # (N, M)
            rhs = np.tensordot(Fr, ZAB, axes=([0, 1], [0, 1]))        # (L,)
            lhs = Tz.T @ Tz + epsilon * np.eye(Tz.shape[1])
            C_coeffs[:, r] = np.linalg.solve(lhs, Tz.T @ rhs)
            C_coeffs[:, r] /= np.linalg.norm(C_coeffs[:, r])

        # Update lambdas
        for r in range(R):
            A_r = Tx @ A_coeffs[:, r]
            B_r = Ty @ B_coeffs[:, r]
            C_r = Tz @ C_coeffs[:, r]
            num = np.sum(F * np.einsum('i,j,k->ijk', A_r, B_r, C_r))
            denom = np.sum(np.einsum('i,j,k->ijk', A_r, B_r, C_r) ** 2)
            lambdas[r] = num / denom if denom > 1e-12 else 0.0

        lambdas = np.clip(lambdas, 1e-3, 1e3)

    # -------------------- Evaluation --------------------
    # Training reconstruction error
    F_reconstructed = np.zeros_like(F)
    for r in range(R):
        F_reconstructed += lambdas[r] * np.einsum('i,j,k->ijk',
                                                 Tx @ A_coeffs[:, r],
                                                 Ty @ B_coeffs[:, r],
                                                 Tz @ C_coeffs[:, r])
    l2_norm_error = np.linalg.norm(F - F_reconstructed)

    # Dense evaluation grid
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    z_eval = np.linspace(-1, 1, resolution)
    X_eval, Y_eval, Z_eval = np.meshgrid(x_eval, y_eval, z_eval, indexing='ij')
    F_true_eval = f(X_eval, Y_eval, Z_eval, c=c)
    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    Tz_eval = chebvander(z_eval, d_z)
    A_eval_pred = Tx_eval @ A_coeffs
    B_eval_pred = Ty_eval @ B_coeffs
    C_eval_pred = Tz_eval @ C_coeffs
    F_pred_eval = np.zeros_like(F_true_eval)
    for r in range(R):
        F_pred_eval += lambdas[r] * np.einsum('i,j,k->ijk',
                                             A_eval_pred[:, r],
                                             B_eval_pred[:, r],
                                             C_eval_pred[:, r])
    rmse_eval = np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2))

    # Random test points
    x_rand = generate_nodes(num_test_points, test_points)
    y_rand = generate_nodes(num_test_points, test_points)
    z_rand = generate_nodes(num_test_points, test_points)
    Tx_rand = chebvander(x_rand, d_x)
    Ty_rand = chebvander(y_rand, d_y)
    Tz_rand = chebvander(z_rand, d_z)
    F_pred_rand = np.zeros(num_test_points)
    for r in range(R):
        F_pred_rand += lambdas[r] * (Tx_rand @ A_coeffs[:, r]) * \
                       (Ty_rand @ B_coeffs[:, r]) * \
                       (Tz_rand @ C_coeffs[:, r])
    F_true_rand = f(x_rand, y_rand, z_rand, c=c)
    rmse_rand = np.sqrt(np.mean((F_true_rand - F_pred_rand) ** 2))
    maxe_rand = np.max(np.abs(F_true_rand - F_pred_rand))

    # -------------------- Save Results --------------------
    metrics = {
        "final_train_rmse": float(final_train_rmse),
        "l2_norm_error_train": float(l2_norm_error),
        "rmse_eval_grid": float(rmse_eval),
        "rmse_test_points": float(rmse_rand),
        "maxe_test_points": float(maxe_rand),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    coeffs_data = {
        "A_coeffs": A_coeffs.tolist(),
        "B_coeffs": B_coeffs.tolist(),
        "C_coeffs": C_coeffs.tolist(),
        "lambdas": lambdas.tolist(),
    }
    with open(os.path.join(outdir, "coeffs.json"), "w") as f_out:
        json.dump(coeffs_data, f_out, indent=4)

    config = {
        "N": N, "M": M, "L": L,
        "d_x": d_x, "d_y": d_y, "d_z": d_z,
        "R": R, "n_iter": n_iter,
        "c": c, "resolution": resolution,
        "num_test_points": num_test_points,
        "epsilon": epsilon,
        "random_seed": random_seed,
        "timestamp": timestamp,
        "train_points": train_points,
        "test_points": test_points,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # -------------------- Simple Visualization --------------------
    mid = L // 2
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax1.imshow(F[:, :, mid], extent=[-1, 1, -1, 1], origin='lower', cmap="viridis")
    ax1.set_title("Original (z=0 slice)")

    ax2 = fig.add_subplot(122)
    ax2.imshow(F_reconstructed[:, :, mid], extent=[-1, 1, -1, 1], origin='lower', cmap="viridis")
    ax2.set_title("Reconstructed (z=0 slice)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reconstruction_slice.png"))
    plt.close(fig)

    return metrics, coeffs_data, config, outdir


# ============================================================
# Run Example
# ============================================================
if __name__ == "__main__":
    results, coeffs, config, savedir = run_als_experiment_3d(
        R=10, n_iter=1000, train_points="chebyshev", test_points="uniform"
    )
    print("\n3D Experiment complete. Results stored in:", savedir)
