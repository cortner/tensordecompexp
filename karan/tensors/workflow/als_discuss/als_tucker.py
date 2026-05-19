"""2D ALS Tucker with Chebyshev-metric orthonormalized factors and closed-form core G updates."""

# this is classical algorithm that i talked about in recent reserach meeting based on ill conditioned normal equations


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


def chebyshev_polys(x, deg):
    T = np.zeros((deg + 1, len(x)))
    T[0] = 1
    if deg > 0:
        T[1] = x
    for k in range(2, deg + 1):
        T[k] = 2 * x * T[k - 1] - T[k - 2]
    return T


# ============================================================
# Reference Chebyshev coefficient tensor
# ============================================================
def compute_reference_coeffs(d_x, d_y, C):
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F = f(X, Y, C=C)
    F_flat = F.ravel(order="F")

    Tx = chebyshev_polys(nodes_x, d_x)
    Ty = chebyshev_polys(nodes_y, d_y)
    A = np.kron(Ty.T, Tx.T)
    Q, R = np.linalg.qr(A)
    coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)
    coeffs = coeffs_flat.reshape((d_x + 1, d_y + 1), order="F")
    return coeffs


# ============================================================
# Orthonormalization in a metric: A^T G A = I
# ============================================================
def orthonormalize_in_metric(A, G):
    """
    Return A_tilde with columns G-orthonormal:
        A_tilde^T G A_tilde = I
    using Cholesky factorization G = L L^T and QR on L^T A.
    """
    try:
        L = np.linalg.cholesky(G)          # G = L L^T
        Y = L.T @ A                        # shape (n, r)
        Q, _ = np.linalg.qr(Y)             # Q^T Q = I
        # solve L^T X = Q for X
        A_tilde = np.linalg.solve(L.T, Q)  # A_tilde^T G A_tilde = I
        return A_tilde
    except np.linalg.LinAlgError:
        # Fallback: plain Euclidean QR if G is too ill-conditioned
        Q, _ = np.linalg.qr(A)
        return Q


# ============================================================
# ALS–Tucker Experiment
# ============================================================
def run_als_tucker_experiment(
    N=64, M=64, d_x=63, d_y=63,
    R_x=10, R_y=10, n_iter=500,
    C=np.eye(2) * 5.0, resolution=100, num_test_points=2048,
    epsilon=None,  # kept for compatibility, not used
    random_seed=42, outdir="als_tucker_results",
    train_points="chebyshev", test_points="uniform",
):
    """
    Runs ALS-based Tucker decomposition on 2D function f(x,y).

    Model:
        F ≈ (T_x A) G (T_y B)^T

    Updates:
        - A and B via normal-equation ALS (no regularization).
        - After each A, B update, they are orthonormalized in
          the Chebyshev metric:
              A^T (T_x^T T_x) A = I
              B^T (T_y^T T_y) B = I
        - G via classical closed-form:
              G = (A^T Gx A)^-1 A^T T_x^T F T_y B (B^T Gy B)^-1
          where Gx = T_x^T T_x, Gy = T_y^T T_y.

    No clipping, no explicit damping.
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
    F = f(X, Y, C=C)

    Tx = chebvander(x_nodes, d_x)  # shape (N, d_x+1)
    Ty = chebvander(y_nodes, d_y)  # shape (M, d_y+1)

    # Gram matrices in value space
    Gx = Tx.T @ Tx
    Gy = Ty.T @ Ty

    print("\n=== Initial Gram condition numbers ===")
    print("cond(Tx^T Tx) =", np.linalg.cond(Gx))
    print("cond(Ty^T Ty) =", np.linalg.cond(Gy))

    # -------------------- Initialization --------------------
    A = np.random.randn(d_x + 1, R_x)
    B = np.random.randn(d_y + 1, R_y)

    # Orthonormalize initial A, B in Chebyshev metric
    A = orthonormalize_in_metric(A, Gx)
    B = orthonormalize_in_metric(B, Gy)

    # Initial core using classical formula
    At_Gx_A = A.T @ Gx @ A
    Bt_Gy_B = B.T @ Gy @ B

    print("\n=== Initial metric-orthonormality check ===")
    print("cond(A^T Gx A) =", np.linalg.cond(At_Gx_A))
    print("cond(B^T Gy B) =", np.linalg.cond(Bt_Gy_B))

    G = (
        np.linalg.inv(At_Gx_A)
        @ (A.T @ Tx.T @ F @ Ty @ B)
        @ np.linalg.inv(Bt_Gy_B)
    )

    # -------------------- ALS Loop --------------------
    for it in range(n_iter):
        # Reconstruction using current A, B, G
        F_hat = (Tx @ A) @ G @ (Ty @ B).T
        rmse = np.sqrt(np.mean((F - F_hat) ** 2))

        print(f"\nIter {it+1}")
        print(f"RMSE (before updates): {rmse:.3e}")

        if np.isnan(rmse) or rmse > 1e6:
            print("Stopping early due to divergence.")
            break

        # === Update A ===
        # middle_A = G @ (B^T Gy B) @ G^T
        BtB_metric = B.T @ Gy @ B
        middle_A = G @ BtB_metric @ G.T
        right_A = Tx.T @ F @ Ty @ B @ G.T

        print("cond(middle_A) =", np.linalg.cond(middle_A))

        # Solve (Gx) A (middle_A) = right_A
        A_new = np.linalg.solve(Gx, right_A @ np.linalg.pinv(middle_A))

        # Orthonormalize A in Chebyshev metric
        A = orthonormalize_in_metric(A_new, Gx)
        At_Gx_A = A.T @ Gx @ A

        # === Update B ===
        AtA_metric = A.T @ Gx @ A
        middle_B = G.T @ AtA_metric @ G
        right_B = Ty.T @ F.T @ Tx @ A @ G

        print("cond(middle_B) =", np.linalg.cond(middle_B))

        B_new = np.linalg.solve(Gy, right_B @ np.linalg.pinv(middle_B))

        # Orthonormalize B in Chebyshev metric
        B = orthonormalize_in_metric(B_new, Gy)
        Bt_Gy_B = B.T @ Gy @ B

        # === Classical closed-form G update ===
        At_Gx_A = A.T @ Gx @ A
        Bt_Gy_B = B.T @ Gy @ B

        print("cond(A^T Gx A) =", np.linalg.cond(At_Gx_A))
        print("cond(B^T Gy B) =", np.linalg.cond(Bt_Gy_B))

        G = (
            np.linalg.inv(At_Gx_A)
            @ (A.T @ Tx.T @ F @ Ty @ B)
            @ np.linalg.inv(Bt_Gy_B)
        )

    # -------------------- Final Evaluation --------------------
    F_reconstructed = (Tx @ A) @ G @ (Ty @ B).T
    l2_norm_error = np.linalg.norm(F - F_reconstructed)
    rmse_eval = np.sqrt(np.mean((F - F_reconstructed) ** 2))

    # Dense evaluation grid
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    F_true_eval = f(X_eval, Y_eval, C=C)
    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    F_pred_eval = (Tx_eval @ A) @ G @ (Ty_eval @ B).T
    rmse_grid = np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2))

    # Coefficient comparison
    C_tucker = A @ G @ B.T
    C_interp = compute_reference_coeffs(d_x, d_y, C)
    l2_coeff_error = np.linalg.norm(C_interp - C_tucker)
    rel_l2_coeff_error = l2_coeff_error / np.linalg.norm(C_interp)

    metrics = {
        "final_train_rmse": float(rmse_eval),
        "rmse_eval_grid": float(rmse_grid),
        "l2_norm_error_train": float(l2_norm_error),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error),
    }

    # Save results
    with open(os.path.join(outdir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)
    np.savez(os.path.join(outdir, "coeffs.npz"), A=A, B=B, G=G)

    config = {
        "N": N, "M": M, "d_x": d_x, "d_y": d_y,
        "R_x": R_x, "R_y": R_y, "n_iter": n_iter,
        "C": C.tolist(), "resolution": resolution,
        "num_test_points": num_test_points,
        "epsilon": epsilon,
        "random_seed": random_seed,
        "timestamp": timestamp,
        "train_points": train_points,
        "test_points": test_points,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Plot
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

    return metrics, {"A": A, "B": B, "G": G}, config, outdir


# ============================================================
# Run Rank-Sweep Example
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

        for R in range(1, 21):  # sweep Tucker ranks 1–20
            print(f"\n=== Running Tucker ALS experiment for C={idx}, Rank={R} ===")
            metrics, coeffs, config, savedir = run_als_tucker_experiment(
                N=64, M=64,
                d_x=63, d_y=63,
                R_x=R, R_y=R,
                n_iter=500,
                C=C,
                train_points="chebyshev",
                test_points="uniform",
                num_test_points=2048,
                resolution=120,
                random_seed=42,
                outdir="als_tucker_classic_results"
            )
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config
            }

    with open("als_tucker_classic_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll ALS-Tucker runs completed.")