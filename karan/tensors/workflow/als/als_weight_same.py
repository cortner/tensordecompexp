"""2D TensorLy Tucker on the unweighted value tensor, then LS lift to Chebyshev coefficients."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
import json
import os
from datetime import datetime

# Optional: TensorLy for Tucker
try:
    import tensorly as tl
    from tensorly.decomposition import tucker
    tl.set_backend("numpy")
    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False
    print("Warning: tensorly not found. Tucker-based decomposition will not work.")


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
    """Generate 1D nodes in [-1, 1]."""
    if mode == "chebyshev":
        return np.cos(np.pi * np.arange(num_points) / (num_points - 1))
    elif mode == "uniform":
        return np.linspace(-1, 1, num_points)
    elif mode == "random":
        return np.random.uniform(-1, 1, num_points)
    else:
        raise ValueError(f"Unknown mode {mode}")


def chebyshev_polys(x, deg):
    """Explicit Chebyshev polynomials T_0,...,T_deg evaluated at x."""
    T = np.zeros((deg + 1, len(x)))
    T[0] = 1
    if deg > 0:
        T[1] = x
    for k in range(2, deg + 1):
        T[k] = 2 * x * T[k - 1] - T[k - 2]
    return T


# ============================================================
# Chebyshev quadrature weights (Clenshaw-Curtis style)
# ============================================================
def chebyshev_weights(N):
    """
    Chebyshev-Gauss-Lobatto-like quadrature weights on N nodes.
    These make the discrete inner product approximate the Chebyshev
    weighted inner product.
    """
    w = np.zeros(N)
    if N == 1:
        w[0] = np.pi
        return w
    w[0] = w[-1] = np.pi / (N - 1)
    w[1:-1] = 2.0 * np.pi / (N - 1)
    return w


# ============================================================
# Reference Chebyshev coefficient tensor
# ============================================================
def compute_reference_coeffs(d_x, d_y, C):
    """
    Compute Chebyshev coefficients of f(x,y) on [-1,1]^2
    up to degrees d_x and d_y using a stable QR solve.
    """
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F = f(X, Y, C=C)
    F_flat = F.ravel(order="F")

    Tx = chebyshev_polys(nodes_x, d_x)   # (d_x+1) x (d_x+1)
    Ty = chebyshev_polys(nodes_y, d_y)   # (d_y+1) x (d_y+1)

    # Build Kronecker system for coefficients
    A = np.kron(Ty.T, Tx.T)              # ( (d_x+1)(d_y+1) x (d_x+1)(d_y+1) )
    Q, R = np.linalg.qr(A)
    coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)
    coeffs = coeffs_flat.reshape((d_x + 1, d_y + 1), order="F")
    return coeffs


# ============================================================
# TensorLy-based Tucker approximation in Chebyshev basis
# ============================================================
def run_als_tucker_experiment(
    N=64, M=64, d_x=63, d_y=63,
    R_x=10, R_y=10, n_iter=500,   # n_iter kept only for config compatibility
    C=np.eye(2) * 5.0, resolution=100, num_test_points=2048,
    epsilon=1e-6, random_seed=42, outdir="als_tucker_results",
    train_points="chebyshev", test_points="uniform",
    lr=1e-3, init_mode="tucker",   # lr, init_mode kept for config compatibility
):
    """
    Use TensorLy Tucker (HOOI) on the value tensor F (unweighted),
    then lift the Tucker factors into the Chebyshev coefficient basis.

    TensorLy gives:
        F ≈ U_tl @ G @ V_tl.T

    We then solve:
        U_tl = Tx @ A
        V_tl = Ty @ B

    via least squares, and use C_tucker = A @ G @ B.T as the
    approximate coefficient tensor for comparison.
    """
    if not HAS_TENSORLY:
        raise RuntimeError("TensorLy is required for this version. Please install 'tensorly'.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)
    np.random.seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"tensorly_tucker_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # -------------------- Training Setup --------------------
    x_nodes = generate_nodes(N, train_points)
    y_nodes = generate_nodes(M, train_points)
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    F = f(X, Y, C=C)                       # N x M

    Tx = chebvander(x_nodes, d_x)          # N x (d_x+1)
    Ty = chebvander(y_nodes, d_y)          # M x (d_y+1)

    # Chebyshev quadrature weights and 2D weight matrix (for metrics only)
    wx = chebyshev_weights(N)              # length N
    wy = chebyshev_weights(M)              # length M
    W = np.outer(wx, wy)                   # N x M
    W_sqrt = np.sqrt(W)

    # Reference coefficient tensor (for diagnostics)
    C_interp = compute_reference_coeffs(d_x, d_y, C)

    # -------------------- TensorLy Tucker on F --------------------
    print("Running TensorLy Tucker decomposition on F (unweighted)...")
    F_tensor = tl.tensor(F)
    # tucker(...) returns a TuckerTensor object in the version you pasted
    tt = tucker(F_tensor, rank=[R_x, R_y])
    core = np.array(tt.core)
    factors = [np.array(f) for f in tt.factors]

    # For 2D tensor: factors[0] is U_tl (N x R_x), factors[1] is V_tl (M x R_y)
    U_tl = factors[0]
    V_tl = factors[1]
    G = core  # R_x x R_y

    # Solve for A and B in Chebyshev coefficient space:
    # U_tl = Tx @ A, V_tl = Ty @ B
    # Use SVD-based least squares
    A, *_ = np.linalg.lstsq(Tx, U_tl, rcond=None)   # (d_x+1) x R_x
    B, *_ = np.linalg.lstsq(Ty, V_tl, rcond=None)   # (d_y+1) x R_y

    print("TensorLy Tucker completed. Solved A, B via least squares in Chebyshev basis.")

    # For compatibility with old metrics, define initial coeff error as NaN
    init_coeff_error = float("nan")

    # -------------------- Evaluation on Training Grid --------------------
    U = Tx @ A
    V = Ty @ B
    F_reconstructed = U @ G @ V.T

    diff_final = F_reconstructed - F
    diff_final_w = W_sqrt * diff_final
    l2_norm_error = np.linalg.norm(diff_final)
    rmse_eval = np.sqrt(np.sum(diff_final_w**2))

    # Dense evaluation grid (unweighted RMSE on dense grid)
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing="ij")
    F_true_eval = f(X_eval, Y_eval, C=C)
    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    U_eval = Tx_eval @ A
    V_eval = Ty_eval @ B
    F_pred_eval = U_eval @ G @ V_eval.T
    rmse_grid = np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2))

    # Coefficient comparison
    C_tucker = A @ G @ B.T
    l2_coeff_error = np.linalg.norm(C_interp - C_tucker)
    rel_l2_coeff_error = l2_coeff_error / np.linalg.norm(C_interp)

    metrics = {
        "initial_l2_coeff_error": float(init_coeff_error),
        "final_train_weighted_rmse": float(rmse_eval),
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
        "epsilon": epsilon, "random_seed": random_seed,
        "timestamp": timestamp,
        "train_points": train_points,
        "test_points": test_points,
        "lr": lr,
        "init_mode": init_mode,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, F, cmap=cm.viridis)
    ax1.set_title(f"Original f(x,y) ({train_points} training)")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, F_reconstructed, cmap=cm.viridis)
    ax2.set_title(f"TensorLy-Tucker Reconstructed (R_x={R_x}, R_y={R_y})")
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

        for R in range(20, 21):  # change to range(1, 21) if you want a full sweep
            print(f"\n=== Running TensorLy Tucker experiment for C={idx}, Rank={R} ===")
            metrics, coeffs, config, savedir = run_als_tucker_experiment(
                N=64, M=64,
                d_x=63, d_y=63,
                R_x=R, R_y=R,
                n_iter=300,        # stored in config only
                lr=1e-3,           # stored in config only
                C=C,
                train_points="chebyshev",
                test_points="uniform",
                num_test_points=2048,
                resolution=120,∏
                random_seed=42,
                outdir="tensorly_tucker_results",
                init_mode="tucker",
            )
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config
            }

    with open("tensorly_tucker_rank_sweep_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll TensorLy Tucker runs complete. Results saved to tensorly_tucker_rank_sweep_results.json")
