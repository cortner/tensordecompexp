"""2D ALS CP with SVD-based ridge subproblems for stable rank updates (avoids normal equations)."""

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
    if mode == "chebyshev":
        return np.cos(np.pi * np.arange(num_points) / (num_points - 1))
    elif mode == "uniform":
        return np.linspace(-1, 1, num_points)
    elif mode == "random":
        return np.random.uniform(-1, 1, num_points)
    else:
        raise ValueError(f"Unknown mode {mode}")


# (Optional helper, not used below; kept for reference)
def chebyshev_polys(x, deg):
    T = np.zeros((deg+1, len(x)))
    T[0] = 1
    if deg > 0:
        T[1] = x
    for k in range(2, deg+1):
        T[k] = 2 * x * T[k-1] - T[k-2]
    return T


# ============================================================
# Reference Chebyshev coefficient tensor (fixed mapping)
# vec(F) = (Ty ⊗ Tx) vec(C)
# ============================================================
def compute_reference_coeffs(d_x, d_y, C):
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F = f(X, Y, C=C)
    F_flat = F.ravel(order="F")
    Tx = chebvander(nodes_x, d_x)
    Ty = chebvander(nodes_y, d_y)
    E = np.kron(Ty, Tx)
    coeffs_flat, *_ = np.linalg.lstsq(E, F_flat, rcond=None)
    coeffs = coeffs_flat.reshape((d_x+1, d_y+1), order="F")
    return coeffs


# ============================================================
# Stable ridge solver via precomputed SVD
# ============================================================
def make_ridge_solver(T, eps):
    # T is (m, p). Precompute SVD once, then solve:
    # argmin_a ||T a - y||^2 + eps ||a||^2
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    S_filt = S / (S**2 + eps)

    def solve(y):
        return Vt.T @ (S_filt * (U.T @ y))

    return solve


# ============================================================
# ALS Training Function (SVD-based ridge + residual-free matvecs)
# ============================================================
def run_als_experiment(
    N=64, M=64, d_x=63, d_y=63, R=10, n_iter=500,
    C=np.eye(2) * 5.0, resolution=100, num_test_points=2048,
    epsilon=1e-6, random_seed=42, outdir="als_results",
    train_points="chebyshev", test_points="uniform",
    use_lambda=True,
):
    """
    ALS CP on Chebyshev coefficient grid with stable ridge subproblems:
      a_r := argmin ||T_x a - y||^2 + ε ||a||^2, where y = (F - Σ_{k≠r} ... ) (T_y b_r) / λ_r
      b_r analogous. Solved by SVD-based ridge solvers. No normal equations are formed.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    results_root = os.path.join(base_dir, outdir)

    np.random.seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"als_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # -------------------- Training Setup --------------------
    x_nodes = generate_nodes(N, train_points)
    y_nodes = generate_nodes(M, train_points)
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    F = f(X, Y, C=C)                    # (N, M)

    Tx = chebvander(x_nodes, d_x)       # (N, d_x+1)
    Ty = chebvander(y_nodes, d_y)       # (M, d_y+1)

    # Precompute stable ridge solvers once
    solve_Tx = make_ridge_solver(Tx, epsilon)
    solve_Ty = make_ridge_solver(Ty, epsilon)

    # ---------- Random initialization (normalize eval-columns) ----------
    A_coeffs = np.random.randn(d_x + 1, R)
    B_coeffs = np.random.randn(d_y + 1, R)
    lambdas = np.ones(R)

    A_eval = Tx @ A_coeffs               # (N, R)
    B_eval = Ty @ B_coeffs               # (M, R)
    for r in range(R):
        nA = np.linalg.norm(A_eval[:, r]) + 1e-12
        nB = np.linalg.norm(B_eval[:, r]) + 1e-12
        A_coeffs[:, r] /= nA
        B_coeffs[:, r] /= nB
        lambdas[r] *= nA * nB
    A_eval = Tx @ A_coeffs
    B_eval = Ty @ B_coeffs

    tiny = 1e-12

    # -------------------- ALS Iterations --------------------
    final_train_rmse = None
    for it in range(n_iter):
        # Current reconstruction and loss
        F_hat = sum(lambdas[r] * np.outer(A_eval[:, r], B_eval[:, r]) for r in range(R))
        err = np.sqrt(np.mean((F - F_hat) ** 2))
        final_train_rmse = err
        if (it + 1) % 50 == 0:
            print(f"Iter {it+1}, RMSE: {err:.2e}")
        if not np.isfinite(err) or err > 1e5:
            print("Stopping due to divergence.")
            break

        # ----- Update A_coeffs with SVD ridge, no explicit Fr -----
        # Keep B_eval fixed while sweeping A
        for r in range(R):
            v = B_eval[:, r]                     # v = (T_y b_r)
            # y = Fr v = F v - sum_{k≠r} λ_k <B_k, v> A_k
            Fv = F @ v                           # (N,)
            proj = np.zeros_like(Fv)
            # use current A_eval and B_eval
            dots = B_eval.T @ v                  # (R,)
            for k in range(R):
                if k == r:
                    continue
                proj += lambdas[k] * dots[k] * A_eval[:, k]
            y = Fv - proj                        # (N,)

            scale = lambdas[r] if use_lambda else 1.0
            y_scaled = y / max(abs(scale), tiny)
            a = solve_Tx(y_scaled)               # ridge solve

            # Normalize in evaluation space and fold into lambda
            Acol = Tx @ a
            nA = np.linalg.norm(Acol)
            if nA > tiny:
                a /= nA
                if use_lambda:
                    lambdas[r] *= nA
            A_coeffs[:, r] = a
            A_eval[:, r] = Acol / max(nA, tiny)

        # ----- Update B_coeffs with SVD ridge, no explicit Fr -----
        # Keep A_eval fixed while sweeping B
        for r in range(R):
            u = A_eval[:, r]                     # u = (T_x a_r)
            # y = Fr^T u = F^T u - sum_{k≠r} λ_k <A_k, u> B_k
            Fu = F.T @ u                         # (M,)
            proj = np.zeros_like(Fu)
            dots = A_eval.T @ u                  # (R,)
            for k in range(R):
                if k == r:
                    continue
                proj += lambdas[k] * dots[k] * B_eval[:, k]
            y = Fu - proj                        # (M,)

            scale = lambdas[r] if use_lambda else 1.0
            y_scaled = y / max(abs(scale), tiny)
            b = solve_Ty(y_scaled)               # ridge solve

            Bcol = Ty @ b
            nB = np.linalg.norm(Bcol)
            if nB > tiny:
                b /= nB
                if use_lambda:
                    lambdas[r] *= nB
            B_coeffs[:, r] = b
            B_eval[:, r] = Bcol / max(nB, tiny)

        # ----- Update lambdas (small ridge for safety) -----
        if use_lambda:
            U = [np.outer(A_eval[:, i], B_eval[:, i]) for i in range(R)]
            G = np.array([[np.sum(U[i] * U[j]) for j in range(R)] for i in range(R)])
            bb = np.array([np.sum(F * U[i]) for i in range(R)])
            lambdas = np.linalg.lstsq(G + 1e-12 * np.eye(R), bb, rcond=None)[0]
            # mild clipping to avoid runaway scaling
            lambdas = np.clip(lambdas, -1e12, 1e12)

    # -------------------- Evaluation --------------------
    F_reconstructed = sum(
        lambdas[r] * np.outer(A_eval[:, r], B_eval[:, r]) for r in range(R)
    )
    l2_norm_error = np.linalg.norm(F - F_reconstructed)

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
    rel_l2_coeff_error = l2_coeff_error / (np.linalg.norm(C_interp) + 1e-12)

    # -------------------- Save Results --------------------
    metrics = {
        "final_train_rmse": float(final_train_rmse),
        "l2_norm_error_train": float(l2_norm_error),
        "rmse_eval_grid": float(rmse_eval),
        "rmse_test_points": float(rmse_rand),
        "maxe_test_points": float(maxe_rand),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error)
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
        "init_mode": "random",
        "solver": "svd_ridge"
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

    return metrics, coeffs_data, config, outdir

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
        np.array([[5, 0], [0, 1]]),                          # vertical
        np.array([[1, 0], [0, 5]]),                          # horizontal
        np.array([[1, 0], [0, 5]]) @ rotation_matrix(30),    # tilted ellipse
        5 * np.eye(2),                                       # steeper dome
        0.2 * np.eye(2),                                     # flatter dome
    ]

    # Collect everything
    all_results = {}

    for idx, C in enumerate(C_matrices):
        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        for R in range(1, 21):  # sweep ranks 1 → 20
            print(f"\n=== Running ALS experiment for C={idx}, Rank={R} ===")

            metrics, coeffs, config, savedir = run_als_experiment(
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
            )

            # Save minimal info for plotting
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config
            }

    # Save as one JSON file
    with open("als_true_eps_test_rank_sweep_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll ALS runs complete. Results saved to als_true_eps_test_rank_sweep_results.json")
