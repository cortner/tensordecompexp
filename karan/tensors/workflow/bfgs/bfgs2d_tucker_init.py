"""BFGS Tucker 2D with TensorLy Tucker or random initialization."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import minimize
from datetime import datetime
from tensorly.decomposition import tucker

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
def generate_nodes(num_points, mode="chebyshev", rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if mode == "chebyshev":
        N = num_points
        return np.cos(np.pi * np.arange(N) / (N - 1))
    elif mode == "uniform":
        return np.linspace(-1.0, 1.0, num_points)
    elif mode == "random":
        return rng.uniform(-1.0, 1.0, size=num_points)
    elif mode == "normal":
        x = rng.normal(loc=0.0, scale=0.5, size=num_points)
        return np.clip(x, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown mode '{mode}'")


# ============================================================
# Pack / Unpack for Tucker
# ============================================================
def pack_params(A, B, G):
    return np.concatenate([A.ravel(), B.ravel(), G.ravel()])


def unpack_params(params, d_x, d_y, R_x, R_y):
    A_size = (d_x + 1) * R_x
    B_size = (d_y + 1) * R_y
    G_size = R_x * R_y
    A = params[:A_size].reshape((d_x + 1, R_x))
    B = params[A_size:A_size + B_size].reshape((d_y + 1, R_y))
    G = params[A_size + B_size:A_size + B_size + G_size].reshape((R_x, R_y))
    return A, B, G


# ============================================================
# Compute reference Chebyshev coefficient tensor
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
# Main experiment
# ============================================================
def run_bfgs_experiment(
    N=64,
    M=64,
    d_x=63,
    d_y=63,
    R_x=10,
    R_y=10,
    C=np.eye(2)*5.0,
    maxiter=10000,
    train_points="chebyshev",
    test_points="uniform",
    num_test_points=2048,
    resolution=100,
    random_seed=42,
    outdir="bfgs_tucker_results",
    init_mode="tucker"  # new flag: "tucker" or "random"
):
    """
    BFGS optimization for a 2D Tucker model with Chebyshev polynomial factors.
    F_hat = A G B^T
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"bfgs_tucker_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # Training grid
    x_nodes = generate_nodes(N, train_points, rng=rng)
    y_nodes = generate_nodes(M, train_points, rng=rng)
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    F = f(X, Y, C=C)

    Tx = chebvander(x_nodes, d_x)
    Ty = chebvander(y_nodes, d_y)

    # Objective function
    def loss(params):
        A, B, G = unpack_params(params, d_x, d_y, R_x, R_y)
        A_eval = Tx @ A
        B_eval = Ty @ B
        F_hat = A_eval @ G @ B_eval.T
        return np.mean((F - F_hat) ** 2)

    # Initialization strategy
    if init_mode == "tucker":
        print("Using Tucker-based initialization...")
        C_interp = compute_reference_coeffs(d_x, d_y, C)
        core, factors = tucker(C_interp, rank=[R_x, R_y])
        A_init, B_init = factors
        G_init = core
        # small perturbations
        A_init += 0.01 * rng.standard_normal(A_init.shape)
        B_init += 0.01 * rng.standard_normal(B_init.shape)
        G_init += 0.01 * rng.standard_normal(G_init.shape)
    else:
        print("Using random Gaussian initialization...")
        A_init = 0.01 * rng.standard_normal((d_x + 1, R_x))
        B_init = 0.01 * rng.standard_normal((d_y + 1, R_y))
        G_init = 0.01 * rng.standard_normal((R_x, R_y))

    params_init = pack_params(A_init, B_init, G_init)

    # Optimization
    result = minimize(
        loss,
        params_init,
        method="BFGS",
        options={"maxiter": maxiter, "disp": True}
    )

    # Unpack optimized
    A_opt, B_opt, G_opt = unpack_params(result.x, d_x, d_y, R_x, R_y)

    # Reconstruct coefficients and compare
    C_interp = compute_reference_coeffs(d_x, d_y, C)
    C_recon = np.array(A_opt) @ np.array(G_opt) @ np.array(B_opt).T

    # Metrics
    l2_coeff_error = float(np.linalg.norm(C_interp - C_recon))
    rel_l2_coeff_error = l2_coeff_error / float(np.linalg.norm(C_interp))

    A_eval = Tx @ A_opt
    B_eval = Ty @ B_opt
    F_hat = A_eval @ G_opt @ B_eval.T

    train_mse = np.mean((F - F_hat) ** 2)
    train_rmse = float(np.sqrt(train_mse))
    l2_norm_error_train = float(np.linalg.norm(F - F_hat))

    # Dense eval grid
    x_eval = np.linspace(-1.0, 1.0, resolution)
    y_eval = np.linspace(-1.0, 1.0, resolution)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing="ij")
    F_true_eval = f(X_eval, Y_eval, C=C)
    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    A_eval_grid = Tx_eval @ A_opt
    B_eval_grid = Ty_eval @ B_opt
    F_pred_eval = A_eval_grid @ G_opt @ B_eval_grid.T
    rmse_eval_grid = float(np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2)))

    # Test points
    x_test = generate_nodes(num_test_points, test_points, rng=rng)
    y_test = generate_nodes(num_test_points, test_points, rng=rng)
    Tx_test = chebvander(x_test, d_x)
    Ty_test = chebvander(y_test, d_y)
    F_pred_test = np.diag(Tx_test @ A_opt @ G_opt @ (Ty_test @ B_opt).T)
    F_true_test = f(x_test, y_test, C=C)
    rmse_test_points = float(np.sqrt(np.mean((F_true_test - F_pred_test) ** 2)))
    maxe_test_points = float(np.max(np.abs(F_true_test - F_pred_test)))

    # Save metrics, coeffs, config
    metrics = {
        "final_train_rmse": train_rmse,
        "l2_norm_error_train": l2_norm_error_train,
        "rmse_eval_grid": rmse_eval_grid,
        "rmse_test_points": rmse_test_points,
        "maxe_test_points": maxe_test_points,
        "l2_coeff_error": l2_coeff_error,
        "rel_l2_coeff_error": rel_l2_coeff_error,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nit": int(result.nit),
        "optimizer_nfev": int(result.nfev),
        "final_loss_mse": float(result.fun),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    coeffs = {
        "A_coeffs": A_opt.tolist(),
        "B_coeffs": B_opt.tolist(),
        "G_core": G_opt.tolist(),
        "C_recon": C_recon.tolist(),
        "C_interp": C_interp.tolist(),
    }
    with open(os.path.join(run_dir, "coeffs.json"), "w") as f_out:
        json.dump(coeffs, f_out, indent=4)

    config = {
        "N": N, "M": M,
        "d_x": d_x, "d_y": d_y,
        "R_x": R_x, "R_y": R_y,
        "C": C.tolist(),
        "maxiter": maxiter,
        "train_points": train_points,
        "test_points": test_points,
        "num_test_points": num_test_points,
        "resolution": resolution,
        "random_seed": random_seed,
        "timestamp": timestamp,
        "init_mode": init_mode,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Plot reconstruction
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, F, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_title(f"Original f(x,y) [{train_points} grid]")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, F_hat, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_title(f"Tucker Reconstruction R_x={R_x}, R_y={R_y}")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "reconstruction.png"), dpi=150)
    plt.close(fig)

    return metrics, coeffs, config, run_dir


# ============================================================
# Example Run
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
        for R in range(1, 21):
            R_x = R_y = R
            print(f"\n=== Running BFGS Tucker experiment for C={idx}, R={R} ===")
            metrics, coeffs, config, savedir = run_bfgs_experiment(
                R_x=R_x, R_y=R_y, C=C,
                init_mode="tucker",  # use tucker initialization
            )
            all_results[C_key][f"rank_{R}x{R}"] = {
                "metrics": metrics,
                "config": config
            }

    with open("bfgs_tucker_init_rank_sweep_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)