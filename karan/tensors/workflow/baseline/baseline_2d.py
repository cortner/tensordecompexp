"""Baseline 2D Chebyshev interpolation (no low-rank decomposition) across grid sizes and C matrices."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================
# Core function (matrix C instead of scalar c)
# ============================================================
def fcn(coords, C):
    """
    Evaluate f(x,y) = 1 / (1 + ||C [x,y]^T||^2).
    coords : array of shape (n, 2)
    C : 2x2 numpy array
    """
    y = coords @ C.T   # shape (n,2)
    sqnorm = np.sum(y**2, axis=1)
    return 1.0 / (1.0 + sqnorm)


# ============================================================
# Helper: Chebyshev polynomials
# ============================================================
def chebyshev_polys(x, deg):
    T = np.zeros((deg+1, len(x)))
    T[0] = 1
    if deg > 0:
        T[1] = x
    for k in range(2, deg+1):
        T[k] = 2 * x * T[k-1] - T[k-2]
    return T


# ============================================================
# Training step
# ============================================================
def train_interpolant(N, C, mode="chebyshev"):
    # Select training nodes
    if mode == "chebyshev":
        k = np.arange(N+1)
        nodes = np.cos((2*k + 1) * np.pi / (2*(N+1)))
    elif mode == "uniform":
        nodes = np.linspace(-1, 1, N+1)
    else:
        raise ValueError("Unknown training mode, choose 'chebyshev' or 'uniform'")

    X, Y = np.meshgrid(nodes, nodes, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    F = fcn(coords, C).reshape((N+1, N+1))

    # Build polynomial basis
    Tx = chebyshev_polys(nodes, N)
    Ty = chebyshev_polys(nodes, N)

    # Solve coefficients
    F_flat = F.ravel()
    A = np.kron(Ty.T, Tx.T)
    Q, R = np.linalg.qr(A)
    coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)
    coeffs = coeffs_flat.reshape((N+1, N+1))

    # Training predictions
    Z_train = np.zeros_like(F)
    for i in range(N+1):
        for j in range(N+1):
            Z_train += coeffs[i, j] * np.outer(Tx[i], Ty[j])

    diff_train = F - Z_train
    rmse_train = float(np.sqrt(np.mean(diff_train**2)))
    maxe_train = float(np.max(np.abs(diff_train)))

    # Relative errors
    rel_rmse_train = float(rmse_train / np.sqrt(np.mean(F**2)))
    rel_maxe_train = float(maxe_train / np.max(np.abs(F)))

    return coeffs, rmse_train, maxe_train, rel_rmse_train, rel_maxe_train, nodes


# ============================================================
# Testing step (grid + fixed test points)
# ============================================================
def test_interpolant(coeffs, N, C, mode="uniform", resolution=100,
                     test_points="uniform", num_test_points=2048, seed=42):
    rng = np.random.default_rng(seed)

    # ---- Grid evaluation (for visualization) ----
    if mode == "uniform":
        xx = np.linspace(-1, 1, resolution)
        yy = np.linspace(-1, 1, resolution)
    elif mode == "chebyshev":
        k = np.arange(resolution)
        xx = np.cos((2*k + 1) * np.pi / (2*resolution))
        yy = np.cos((2*k + 1) * np.pi / (2*resolution))
    else:
        raise ValueError("Unknown testing mode, choose 'uniform' or 'chebyshev'")

    XX, YY = np.meshgrid(xx, yy, indexing="ij")
    coords = np.stack([XX.ravel(), YY.ravel()], axis=1)

    F_exact = fcn(coords, C).reshape((resolution, resolution))

    Tx_eval = chebyshev_polys(xx, N)
    Ty_eval = chebyshev_polys(yy, N)
    Z = np.zeros_like(F_exact)
    for i in range(N+1):
        for j in range(N+1):
            Z += coeffs[i, j] * np.outer(Tx_eval[i], Ty_eval[j])

    diff = F_exact - Z
    rmse_test_grid = float(np.sqrt(np.mean(diff**2)))
    maxe_test_grid = float(np.max(np.abs(diff)))
    rel_rmse_test_grid = float(rmse_test_grid / np.sqrt(np.mean(F_exact**2)))
    rel_maxe_test_grid = float(maxe_test_grid / np.max(np.abs(F_exact)))

    # ---- Fixed test points (like ALS/BFGS/Adam) ----
    if test_points == "uniform":
        x_test = np.linspace(-1, 1, num_test_points)
        y_test = np.linspace(-1, 1, num_test_points)
    elif test_points == "chebyshev":
        k = np.arange(num_test_points)
        x_test = np.cos((2*k + 1) * np.pi / (2*num_test_points))
        y_test = np.cos((2*k + 1) * np.pi / (2*num_test_points))
    elif test_points == "random":
        x_test = rng.uniform(-1, 1, size=num_test_points)
        y_test = rng.uniform(-1, 1, size=num_test_points)
    else:
        raise ValueError("Unknown test_points mode")

    Tx_test = chebyshev_polys(x_test, N)
    Ty_test = chebyshev_polys(y_test, N)

    F_pred_test = np.zeros(num_test_points)
    for i in range(N+1):
        for j in range(N+1):
            F_pred_test += coeffs[i, j] * (Tx_test[i] * Ty_test[j])

    F_true_test = fcn(np.stack([x_test, y_test], axis=1), C)

    diff_test = F_true_test - F_pred_test
    rmse_test_points = float(np.sqrt(np.mean(diff_test**2)))
    maxe_test_points = float(np.max(np.abs(diff_test)))
    rel_rmse_test_points = float(rmse_test_points / np.sqrt(np.mean(F_true_test**2)))
    rel_maxe_test_points = float(maxe_test_points / np.max(np.abs(F_true_test)))

    return {
        "grid": {
            "rmse": rmse_test_grid,
            "maxe": maxe_test_grid,
            "rel_rmse": rel_rmse_test_grid,
            "rel_maxe": rel_maxe_test_grid,
            "F_exact": F_exact,
            "Z": Z,
            "XX": XX,
            "YY": YY,
        },
        "points": {
            "rmse": rmse_test_points,
            "maxe": maxe_test_points,
            "rel_rmse": rel_rmse_test_points,
            "rel_maxe": rel_maxe_test_points,
        }
    }


# ============================================================
# Experiment runner
# ============================================================
def run_experiment(
    Ns=[64],
    C_matrices=None,
    resolution=100,
    train_mode="chebyshev",
    test_mode="uniform",
    test_points="uniform",
    num_test_points=2048,
    outdir="results"
):
    if C_matrices is None:
        C_matrices = [
            np.array([[5, 0], [0, 1]]),
            np.array([[1, 0], [0, 5]]),
            np.array([[2, 1], [1, 3]]),
            5 * np.eye(2),
            0.2 * np.eye(2)
        ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"experiment_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    all_results = []

    for idx, C in enumerate(C_matrices):
        for N in Ns:
            coeffs, train_rmse, train_maxe, train_rel_rmse, train_rel_maxe, _ = train_interpolant(N, C, mode=train_mode)
            test_results = test_interpolant(
                coeffs, N, C, mode=test_mode, resolution=resolution,
                test_points=test_points, num_test_points=num_test_points
            )

            grid_res = test_results["grid"]
            point_res = test_results["points"]

            # Save side-by-side comparison (grid only for plots)
            fig = plt.figure(figsize=(18, 5))
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            ax1.plot_surface(grid_res["XX"], grid_res["YY"], grid_res["F_exact"], cmap="plasma", edgecolor="none")
            ax1.set_title(f"True f(x), C{idx}, N={N}")

            ax2 = fig.add_subplot(1, 3, 2, projection="3d")
            ax2.plot_surface(grid_res["XX"], grid_res["YY"], grid_res["Z"], cmap="viridis", edgecolor="none")
            ax2.set_title(f"Interpolant p(x), C{idx}, N={N}")

            ax3 = fig.add_subplot(1, 3, 3, projection="3d")
            ax3.plot_surface(grid_res["XX"], grid_res["YY"], np.abs(grid_res["F_exact"] - grid_res["Z"]),
                             cmap="inferno", edgecolor="none")
            ax3.set_title(f"Abs Error |f - p|, C{idx}, N={N}")

            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"compare_C{idx}_N{N}.png"))
            plt.close(fig)

            result_entry = {
                "N": N,
                "C": C.tolist(),
                "train_mode": train_mode,
                "test_mode": test_mode,
                "train_RMSE": train_rmse,
                "train_MaxE": train_maxe,
                "train_RelRMSE": train_rel_rmse,
                "train_RelMaxE": train_rel_maxe,
                "test_grid_RMSE": grid_res["rmse"],
                "test_grid_MaxE": grid_res["maxe"],
                "test_grid_RelRMSE": grid_res["rel_rmse"],
                "test_grid_RelMaxE": grid_res["rel_maxe"],
                "test_points_RMSE": point_res["rmse"],
                "test_points_MaxE": point_res["maxe"],
                "test_points_RelRMSE": point_res["rel_rmse"],
                "test_points_RelMaxE": point_res["rel_maxe"],
                "timestamp": timestamp
            }
            all_results.append(result_entry)

            with open(os.path.join(outdir, f"results_C{idx}_N{N}.json"), "w") as f:
                json.dump(result_entry, f, indent=4)

    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    return outdir, all_results


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    def rotation_matrix(theta_deg):
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    C_matrices = []
    C_matrices.append(np.array([[5, 0],[0, 1]]))  # vertical
    C_matrices.append(np.array([[1, 0],[0, 5]]))  # horizontal
    D = np.array([[1, 0],[0, 5]])
    R = rotation_matrix(30)
    C_matrices.append(D @ R)  # tilted ellipse
    C_matrices.append(5 * np.eye(2))  # steeper dome
    C_matrices.append(0.2 * np.eye(2))  # flatter dome

    outdir, results = run_experiment(
        Ns=[64],
        C_matrices=C_matrices,
        resolution=100,
        train_mode="chebyshev",
        test_mode="uniform",
        test_points="uniform",
        num_test_points=2048,
        outdir="results"
    )

    print(f"Experiment complete. Results in {outdir}")