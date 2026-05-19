"""Compares ALS Tucker fitting against full Chebyshev interpolation plus Tucker factorization."""

# this code is written after giving a thought on
# why the two problems should be equivalent
# the als using tucker and the chebyshev interpolation and tucker decomposition


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
        # Chebyshev–Gauss–Lobatto nodes
        return np.cos(np.pi * np.arange(num_points) / (num_points - 1))
    elif mode == "uniform":
        return np.linspace(-1, 1, num_points)
    elif mode == "random":
        return np.random.uniform(-1, 1, num_points)
    else:
        raise ValueError(f"Unknown mode {mode}")


def chebyshev_polys(x, deg):
    """Compute Chebyshev polynomials T_0,...,T_deg at points x."""
    T = np.zeros((deg + 1, len(x)))
    T[0] = 1
    if deg > 0:
        T[1] = x
    for k in range(2, deg + 1):
        T[k] = 2 * x * T[k - 1] - T[k - 2]
    return T


# ============================================================
# Full Chebyshev coefficient tensor via interpolation
# (Method 0: "gold standard" full polynomial)
# ============================================================
def compute_reference_coeffs(d_x, d_y, C):
    """
    Compute full Chebyshev coefficients C_interp (degree d_x,d_y)
    s.t. f(x_i,y_j) ≈ sum_{p,q} C_interp[p,q] T_p(x_i) T_q(y_j)
    on the Chebyshev grid.
    """
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F = f(X, Y, C=C)
    F_flat = F.ravel(order="F")  # Fortran order to match kron construction

    Tx = chebyshev_polys(nodes_x, d_x)  # (d_x+1) x (d_x+1)
    Ty = chebyshev_polys(nodes_y, d_y)  # (d_y+1) x (d_y+1)

    # Design matrix for coefficients: F_flat ≈ (Ty^T ⊗ Tx^T) c_flat
    A = np.kron(Ty.T, Tx.T)
    Q, R = np.linalg.qr(A)
    coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)
    coeffs = coeffs_flat.reshape((d_x + 1, d_y + 1), order="F")
    return coeffs, nodes_x, nodes_y


# ============================================================
# Method 1: Tucker on full Chebyshev coefficients (SVD)
# ============================================================
def tucker_via_svd_on_coeffs(C_interp, R_x, R_y):
    """
    Method 1 (prof's first description):
    - Start from full Chebyshev coefficient tensor C_interp.
    - Compute its best rank-(R_x,R_y) Tucker approximation.
    For 2D, Tucker = truncated SVD.
    """
    U, S, Vt = np.linalg.svd(C_interp, full_matrices=False)
    R = min(R_x, R_y, len(S))
    U_r = U[:, :R]                  # (d_x+1) x R
    V_r = Vt[:R, :].T               # (d_y+1) x R
    G = np.diag(S[:R])              # R x R core (diagonal in 2D case)
    C_tucker = U_r @ G @ V_r.T      # best rank-R approximation in Frobenius norm
    return U_r, G, V_r, C_tucker


# ============================================================
# Method 2: ALS–Tucker directly on function values
# ============================================================
def als_tucker_from_samples(
    C_mat,
    d_x=63, d_y=63,
    R_x=10, R_y=10,
    n_iter=500,
    epsilon=1e-6,
    random_seed=42,
):
    """
    Method 2 (prof's second description):
    - Work directly with function values on Chebyshev nodes.
    - Represent approximation as F_hat = T_x C_tucker T_y^T
      where C_tucker = A G B^T is a rank-(R_x,R_y) Tucker structure.
    - ALS updates A,B,G to minimize ||F - F_hat||_F over the grid.

    Here:
    - C_mat is the 2x2 anisotropy matrix used in f(x,y).
    """
    np.random.seed(random_seed)

    # Chebyshev nodes (must match compute_reference_coeffs)
    N = d_x + 1
    M = d_y + 1
    x_nodes = generate_nodes(N, "chebyshev")
    y_nodes = generate_nodes(M, "chebyshev")
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    F = f(X, Y, C=C_mat)

    # Chebyshev Vandermonde matrices (values of T_k at nodes)
    Tx = chebvander(x_nodes, d_x)  # N x (d_x+1)
    Ty = chebvander(y_nodes, d_y)  # M x (d_y+1)

    # Initialize factors in coefficient space
    A = np.random.randn(d_x + 1, R_x)
    B = np.random.randn(d_y + 1, R_y)
    G = np.random.randn(R_x, R_y)

    # Precompute normal matrices
    Gx = Tx.T @ Tx + epsilon * np.eye(Tx.shape[1])
    Gy = Ty.T @ Ty + epsilon * np.eye(Ty.shape[1])

    for it in range(n_iter):
        # Current reconstruction on the grid
        C_current = A @ G @ B.T                # (d_x+1) x (d_y+1)
        F_hat = Tx @ C_current @ Ty.T          # N x M
        rmse = np.sqrt(np.mean((F - F_hat) ** 2))

        if (it + 1) % 50 == 0:
            print(f"[ALS] Iter {it+1}, RMSE on grid: {rmse:.2e}")

        if np.isnan(rmse) or rmse > 1e5:
            print("Stopping ALS due to divergence.")
            break

        # --- Update A ---
        BtB = B.T @ (Ty.T @ Ty) @ B           # R_y x R_y
        right_A = Tx.T @ F @ Ty @ B @ G.T     # (d_x+1) x R_x
        middle_A = G @ BtB @ G.T              # R_x x R_x

        try:
            A = np.linalg.solve(Gx, right_A @ np.linalg.pinv(middle_A))
        except np.linalg.LinAlgError:
            print(f"[ALS] Iter {it+1}: damping A update.")
            middle_A += 1e-4 * np.eye(middle_A.shape[0])
            A = np.linalg.solve(Gx, right_A @ np.linalg.pinv(middle_A))

        # Normalize columns of A
        norms_A = np.linalg.norm(A, axis=0, keepdims=True)
        norms_A = np.clip(norms_A, 1e-8, None)
        A /= norms_A
        G *= norms_A.T

        # --- Update B ---
        AtA = A.T @ (Tx.T @ Tx) @ A           # R_x x R_x
        right_B = Ty.T @ F.T @ Tx @ A @ G     # (d_y+1) x R_y
        middle_B = G.T @ AtA @ G              # R_y x R_y

        try:
            B = np.linalg.solve(Gy, right_B @ np.linalg.pinv(middle_B))
        except np.linalg.LinAlgError:
            print(f"[ALS] Iter {it+1}: damping B update.")
            middle_B += 1e-4 * np.eye(middle_B.shape[0])
            B = np.linalg.solve(Gy, right_B @ np.linalg.pinv(middle_B))

        # Normalize columns of B
        norms_B = np.linalg.norm(B, axis=0, keepdims=True)
        norms_B = np.clip(norms_B, 1e-8, None)
        B /= norms_B
        G *= norms_B

    # Final tensors
    C_tucker_als = A @ G @ B.T
    F_hat_final = Tx @ C_tucker_als @ Ty.T
    final_rmse = np.sqrt(np.mean((F - F_hat_final) ** 2))

    metrics = {
        "final_train_rmse": float(final_rmse),
    }

    return metrics, {"A": A, "B": B, "G": G}, {
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "Tx": Tx,
        "Ty": Ty,
        "F": F,
    }


# ============================================================
# Combined experiment: Method 1 vs Method 2
# ============================================================
def run_tucker_equivalence_experiment(
    d_x=63, d_y=63,
    R=10,
    C=np.eye(2) * 5.0,
    n_iter_als=500,
    resolution=120,
    epsilon=1e-6,
    random_seed=42,
    outdir="als_tucker_results",
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    np.random.seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"tucker_equiv_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # ---------- Step 0: Full Chebyshev interpolation ----------
    C_interp, nodes_x, nodes_y = compute_reference_coeffs(d_x, d_y, C)

    # Ground truth on Chebyshev grid via interpolation
    Tx = chebvander(nodes_x, d_x)
    Ty = chebvander(nodes_y, d_y)
    F_full = Tx @ C_interp @ Ty.T

    # True function values on same grid (for sanity)
    Xg, Yg = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F_true = f(Xg, Yg, C=C)
    print("Sanity check: ||F_full - F_true||_F =", np.linalg.norm(F_full - F_true))

    # ---------- Method 1: Tucker via SVD on C_interp ----------
    U1, G1, V1, C_tucker_svd = tucker_via_svd_on_coeffs(C_interp, R, R)
    F_svd = Tx @ C_tucker_svd @ Ty.T

    # ---------- Method 2: ALS Tucker from samples ----------
    als_metrics, coeffs_als, ctx = als_tucker_from_samples(
        C_mat=C,
        d_x=d_x,
        d_y=d_y,
        R_x=R,
        R_y=R,
        n_iter=n_iter_als,
        epsilon=epsilon,
        random_seed=random_seed,
    )
    A_als, B_als, G_als = coeffs_als["A"], coeffs_als["B"], coeffs_als["G"]
    C_tucker_als = A_als @ G_als @ B_als.T
    F_als = ctx["Tx"] @ C_tucker_als @ ctx["Ty"].T

    # ---------- Metrics ----------
    # Errors vs full polynomial coefficients
    l2_coeff_error_svd = np.linalg.norm(C_interp - C_tucker_svd)
    rel_l2_coeff_error_svd = l2_coeff_error_svd / np.linalg.norm(C_interp)

    l2_coeff_error_als = np.linalg.norm(C_interp - C_tucker_als)
    rel_l2_coeff_error_als = l2_coeff_error_als / np.linalg.norm(C_interp)

    # Difference between the two rank-R coefficient tensors
    l2_coeff_diff_between_methods = np.linalg.norm(C_tucker_svd - C_tucker_als)
    rel_l2_coeff_diff_between_methods = (
        l2_coeff_diff_between_methods / np.linalg.norm(C_interp)
    )

    # Errors in value space on Chebyshev grid
    rmse_svd_grid = np.sqrt(np.mean((F_true - F_svd) ** 2))
    rmse_als_grid = np.sqrt(np.mean((F_true - F_als) ** 2))

    # Dense evaluation grid
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing="ij")
    F_true_eval = f(X_eval, Y_eval, C=C)
    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    F_svd_eval = Tx_eval @ C_tucker_svd @ Ty_eval.T
    F_als_eval = Tx_eval @ C_tucker_als @ Ty_eval.T

    rmse_svd_dense = np.sqrt(np.mean((F_true_eval - F_svd_eval) ** 2))
    rmse_als_dense = np.sqrt(np.mean((F_true_eval - F_als_eval) ** 2))

    metrics = {
        "rmse_method1_svd_grid": float(rmse_svd_grid),
        "rmse_method2_als_grid": float(rmse_als_grid),
        "rmse_method1_svd_dense": float(rmse_svd_dense),
        "rmse_method2_als_dense": float(rmse_als_dense),
        "l2_coeff_error_method1_svd": float(l2_coeff_error_svd),
        "rel_l2_coeff_error_method1_svd": float(rel_l2_coeff_error_svd),
        "l2_coeff_error_method2_als": float(l2_coeff_error_als),
        "rel_l2_coeff_error_method2_als": float(rel_l2_coeff_error_als),
        "l2_coeff_diff_between_methods": float(l2_coeff_diff_between_methods),
        "rel_l2_coeff_diff_between_methods": float(rel_l2_coeff_diff_between_methods),
        "als_final_train_rmse": als_metrics["final_train_rmse"],
    }

    # Save results
    with open(os.path.join(outdir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    np.savez(
        os.path.join(outdir, "coeffs_method1_vs_method2.npz"),
        C_interp=C_interp,
        C_tucker_svd=C_tucker_svd,
        C_tucker_als=C_tucker_als,
        U1=U1,
        V1=V1,
        G1=G1,
        A_als=A_als,
        B_als=B_als,
        G_als=G_als,
    )

    config = {
        "d_x": d_x,
        "d_y": d_y,
        "R": R,
        "C": C.tolist(),
        "resolution": resolution,
        "epsilon": epsilon,
        "random_seed": random_seed,
        "timestamp": timestamp,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Simple plot: true vs both approximations on Chebyshev grid
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(Xg, Yg, F_true, cmap=cm.viridis)
    ax1.set_title('True f(x,y) on Chebyshev grid')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(Xg, Yg, F_svd, cmap=cm.viridis)
    ax2.set_title(f'Method 1: SVD on coeffs (R={R})')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(Xg, Yg, F_als, cmap=cm.viridis)
    ax3.set_title(f'Method 2: ALS from samples (R={R})')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_chebyshev_grid.png"))
    plt.close(fig)

    return metrics, config, outdir


# ============================================================
# Main: run a single equivalence test
# ============================================================
if __name__ == "__main__":
    def rotation_matrix(theta_deg):
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    # Example: a few different anisotropy matrices C
    C_matrices = [
        np.array([[5, 0], [0, 1]]),
        np.array([[1, 0], [0, 5]]),
        np.array([[1, 0], [0, 5]]) @ rotation_matrix(30),
        5 * np.eye(2),
        0.2 * np.eye(2),
    ]

    all_results = {}

    for idx, C in enumerate(C_matrices):
        print(f"\n=== Equivalence experiment for C case {idx} ===")
        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        # You can sweep ranks if you like; here I just do a few
        for R in [2, 5, 10, 15]:
            print(f"\n--- Rank R = {R} ---")
            metrics, config, savedir = run_tucker_equivalence_experiment(
                d_x=63,
                d_y=63,
                R=R,
                C=C,
                n_iter_als=800,
                resolution=120,
                random_seed=42,
                outdir="als_tucker_results_equiv"
            )
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config,
                "savedir": savedir,
            }

    with open("als_tucker_equivalence_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll equivalence experiments complete. Results saved to als_tucker_equivalence_results.json")
