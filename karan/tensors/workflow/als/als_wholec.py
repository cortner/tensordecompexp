"""Block ALS CP like als_whole.py; rank sweep runs with use_lambda=False."""

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

    coords = np.stack([x.ravel(), y.ravel()], axis=1)  # (N*M, 2) if grids; or (N,2) if vectors
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

    Tx = chebvander(nodes_x, d_x)  # (d_x+1, d_x+1)
    Ty = chebvander(nodes_y, d_y)  # (d_y+1, d_y+1)

    E = np.kron(Ty, Tx)            # ((d_x+1)(d_y+1), (d_x+1)(d_y+1))
    coeffs_flat, *_ = np.linalg.lstsq(E, F_flat, rcond=None)
    coeffs = coeffs_flat.reshape((d_x+1, d_y+1), order="F")
    return coeffs


# ============================================================
# Helpers for block ALS updates
# ============================================================
def solve_two_sided_least_squares(G_left, G_right, RHS):
    """
    Solve (G_left) X (G_right) = RHS for X by Kronecker linear system:
    vec(X) solves (G_right^T ⊗ G_left) vec(X) = vec(RHS).
    Uses column-major ('F') vec/reshape to match math identities.
    """
    K = np.kron(G_right.T, G_left)
    rhs = RHS.reshape(-1, order="F")
    try:
        x = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    X = x.reshape(RHS.shape, order="F")
    return X


def update_A_block(F, Tx, Ty, B_coeffs, lambdas, epsilon, use_lambda):
    """
    Fix B, lambdas. Solve for full A:
    Let S = (Ty @ B) diag(l) if use_lambda else (Ty @ B).
    Normal eq: (Tx^T Tx) A (S^T S) = Tx^T F S.
    """
    S = Ty @ B_coeffs  # (M, R)
    if use_lambda:
        S = S * lambdas[np.newaxis, :]  # scale columns

    Gx = Tx.T @ Tx + epsilon * np.eye(Tx.shape[1])
    Gs = S.T @ S + epsilon * np.eye(S.shape[1])
    RHS = Tx.T @ F @ S
    A_coeffs = solve_two_sided_least_squares(Gx, Gs, RHS)
    return A_coeffs


def update_B_block(F, Tx, Ty, A_coeffs, lambdas, epsilon, use_lambda):
    """
    Fix A, lambdas. Solve for full B:
    Work on transposed form Y^T ≈ (Ty B) diag(l) (Tx A)^T.
    Let T = (Tx @ A) diag(l) if use_lambda else (Tx @ A).
    Normal eq: (Ty^T Ty) B (T^T T) = Ty^T F^T T.
    """
    T = Tx @ A_coeffs  # (N, R)
    if use_lambda:
        T = T * lambdas[np.newaxis, :]

    Gy = Ty.T @ Ty + epsilon * np.eye(Ty.shape[1])
    Gt = T.T @ T + epsilon * np.eye(T.shape[1])
    RHS = Ty.T @ F.T @ T
    B_coeffs = solve_two_sided_least_squares(Gy, Gt, RHS)
    return B_coeffs


def update_lambdas(F, Tx, Ty, A_coeffs, B_coeffs, epsilon):
    """
    With A, B fixed, solve small R-by-R system for lambdas:
    G_ij = <u_i v_i^T, u_j v_j^T> = (u_i^T u_j) * (v_i^T v_j).
    b_i = <F, u_i v_i^T> = u_i^T F v_i.
    """
    U = Tx @ A_coeffs  # (N, R)
    V = Ty @ B_coeffs  # (M, R)

    GA = U.T @ U
    GB = V.T @ V
    G = GA * GB + epsilon * np.eye(GA.shape[0])  # Hadamard plus damping

    b = np.array([U[:, i].T @ F @ V[:, i] for i in range(U.shape[1])])
    try:
        lambdas = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        lambdas, *_ = np.linalg.lstsq(G, b, rcond=None)
    return lambdas


def normalize_columns(Tx, Ty, A_coeffs, B_coeffs, lambdas):
    """
    Normalize columns in evaluation space to keep scales reasonable:
    Make ||Tx A[:,r]|| = 1 and ||Ty B[:,r]|| = 1, fold magnitudes into lambda.
    No-op if R == 0. Only use when lambdas are enabled.
    """
    U = Tx @ A_coeffs
    V = Ty @ B_coeffs
    tiny = 1e-12
    for r in range(A_coeffs.shape[1]):
        na = np.linalg.norm(U[:, r]) + tiny
        nb = np.linalg.norm(V[:, r]) + tiny
        A_coeffs[:, r] /= na
        B_coeffs[:, r] /= nb
        lambdas[r] *= (na * nb)
    return A_coeffs, B_coeffs, lambdas


# ============================================================
# ALS Training Function (block updates for A, B)
# ============================================================
def run_als_experiment(
    N=64, M=64, d_x=63, d_y=63, R=10, n_iter=500,
    C=np.eye(2) * 5.0, resolution=100, num_test_points=2048,
    epsilon=1e-4, random_seed=42, outdir="als_results",
    train_points="chebyshev", test_points="uniform",
    use_lambda=True, tol=1e-7, patience=5
):
    """
    Runs ALS-based CP decomposition on 2D function f(x,y) with anisotropy matrix C.

    Block ALS:
      1) Fix B, lambda, solve full A with a two-sided least-squares system.
      2) Fix A, lambda, solve full B likewise.
      3) Solve small R-by-R for lambda.
      4) Optional normalization to keep scales stable.

    If use_lambda=True:
      F_hat = (Tx A) diag(lambdas) (Ty B)^T
    If use_lambda=False:
      lambdas are kept at 1 and are not updated or used in scaling.
    """
    # Base dir for outputs
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
    F = f(X, Y, C=C)

    Tx = chebvander(x_nodes, d_x)  # (N, d_x+1)
    Ty = chebvander(y_nodes, d_y)  # (M, d_y+1)

    # ---------- Random initialization ----------
    A_coeffs = np.random.randn(d_x + 1, R)
    B_coeffs = np.random.randn(d_y + 1, R)
    lambdas = np.ones(R)

    # -------------------- ALS Iterations --------------------
    best_rmse = np.inf
    final_train_rmse = None
    no_improve = 0

    for it in range(n_iter):
        # Update A in one shot
        A_coeffs = update_A_block(F, Tx, Ty, B_coeffs, lambdas, epsilon, use_lambda)

        # Update B in one shot
        B_coeffs = update_B_block(F, Tx, Ty, A_coeffs, lambdas, epsilon, use_lambda)

        # Update lambdas
        if use_lambda:
            lambdas = update_lambdas(F, Tx, Ty, A_coeffs, B_coeffs, epsilon)
            # Optional normalization to keep columns well-scaled
            A_coeffs, B_coeffs, lambdas = normalize_columns(Tx, Ty, A_coeffs, B_coeffs, lambdas)

        # Compute current reconstruction and loss
        U = Tx @ A_coeffs
        V = Ty @ B_coeffs
        if use_lambda:
            F_hat = (U * lambdas[np.newaxis, :]) @ V.T
        else:
            # lambdas are all 1
            F_hat = U @ V.T
        err = np.sqrt(np.mean((F - F_hat) ** 2))
        final_train_rmse = err

        if (it + 1) % 25 == 0:
            print(f"Iter {it+1}, RMSE: {err:.3e}")

        # Early stopping
        # if err + tol < best_rmse:
        #     best_rmse = err
        #     no_improve = 0
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         print(f"Early stopping at iter {it+1} with RMSE {err:.3e}")
        #         break

        if np.isnan(err) or err > 1e5:
            print("Stopping due to divergence.")
            break

    # -------------------- Evaluation --------------------
    # Reconstruction on training grid
    U = Tx @ A_coeffs
    V = Ty @ B_coeffs
    if use_lambda:
        F_reconstructed = (U * lambdas[np.newaxis, :]) @ V.T
    else:
        F_reconstructed = U @ V.T

    l2_norm_error = np.linalg.norm(F - F_reconstructed)

    # Dense evaluation grid
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    F_true_eval = f(X_eval, Y_eval, C=C)
    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    U_pred = Tx_eval @ A_coeffs
    V_pred = Ty_eval @ B_coeffs
    if use_lambda:
        F_pred_eval = (U_pred * lambdas[np.newaxis, :]) @ V_pred.T
    else:
        F_pred_eval = U_pred @ V_pred.T
    rmse_eval = np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2))

    # Test points (paired samples)
    x_rand = generate_nodes(num_test_points, test_points)
    y_rand = generate_nodes(num_test_points, test_points)
    Tx_rand = chebvander(x_rand, d_x)
    Ty_rand = chebvander(y_rand, d_y)
    if use_lambda:
        F_pred_rand = np.sum(
            (Tx_rand @ A_coeffs) * lambdas[np.newaxis, :] * (Ty_rand @ B_coeffs),
            axis=1
        )
    else:
        F_pred_rand = np.sum((Tx_rand @ A_coeffs) * (Ty_rand @ B_coeffs), axis=1)
    F_true_rand = f(x_rand, y_rand, C=C)
    rmse_rand = np.sqrt(np.mean((F_true_rand - F_pred_rand) ** 2))
    maxe_rand = np.max(np.abs(F_true_rand - F_pred_rand))

    # -------------------- Coefficient Tensor Comparison --------------------
    C_als = A_coeffs @ np.diag(lambdas) @ B_coeffs.T if use_lambda else A_coeffs @ B_coeffs.T
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
        "tol": tol,
        "patience": patience,
        "update_mode": "block"
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

        for R in range(1, 21):  # sweep ranks 1 -> 20
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
                use_lambda=False,      # set True to enable lambda updates and normalization
                epsilon=1e-4,          # damping inside Gram systems
                tol=1e-7,
                patience=10
            )

            # Save minimal info for plotting
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config
            }

    # Save as one JSON file
    with open("als_true_whole_rank_sweep_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll ALS runs complete. Results saved to als_true_whole_rank_sweep_results.json")
