"""Baseline 3D Chebyshev interpolation without tensor decomposition."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================
# Core function (3D)
# ============================================================
def fcn(x, c):
    # x shape: (M, 3) for (x,y,z)
    return 1 / (1 + c**2 * np.sum(x**2, axis=1))


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
# Training step (3D tensor grid)
# ============================================================
def train_interpolant(N, c, mode="chebyshev"):
    # Select training nodes
    if mode == "chebyshev":
        k = np.arange(N+1)
        nodes = np.cos((2*k + 1) * np.pi / (2*(N+1)))
    elif mode == "uniform":
        nodes = np.linspace(-1, 1, N+1)
    else:
        raise ValueError("Unknown training mode, choose 'chebyshev' or 'uniform'")

    X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    F = fcn(coords, c).reshape((N+1, N+1, N+1))

    # Build polynomial basis
    Tx = chebyshev_polys(nodes, N)
    Ty = chebyshev_polys(nodes, N)
    Tz = chebyshev_polys(nodes, N)

    # Kronecker product structure
    F_flat = F.ravel()
    A = np.kron(np.kron(Tz.T, Ty.T), Tx.T)   # size ((N+1)^3, (N+1)^3)
    Q, R = np.linalg.qr(A)
    coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)
    coeffs = coeffs_flat.reshape((N+1, N+1, N+1))

    # Training predictions
    Z_train = np.zeros_like(F)
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                Z_train += coeffs[i, j, k] * np.einsum('p,q,r->pqr', Tx[i], Ty[j], Tz[k])

    diff_train = F - Z_train
    rmse_train = float(np.sqrt(np.mean(diff_train**2)))
    maxe_train = float(np.max(np.abs(diff_train)))

    return coeffs, rmse_train, maxe_train, nodes


# ============================================================
# Testing step (3D)
# ============================================================
def test_interpolant(coeffs, N, c, mode="uniform", resolution=30):
    if mode == "uniform":
        xx = np.linspace(-1, 1, resolution)
        yy = np.linspace(-1, 1, resolution)
        zz = np.linspace(-1, 1, resolution)
    elif mode == "chebyshev":
        k = np.arange(resolution)
        xx = np.cos((2*k + 1) * np.pi / (2*resolution))
        yy = np.cos((2*k + 1) * np.pi / (2*resolution))
        zz = np.cos((2*k + 1) * np.pi / (2*resolution))
    else:
        raise ValueError("Unknown testing mode, choose 'uniform' or 'chebyshev'")

    XX, YY, ZZ = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    F_exact = fcn(coords, c).reshape((resolution, resolution, resolution))

    Tx_eval = chebyshev_polys(xx, N)
    Ty_eval = chebyshev_polys(yy, N)
    Tz_eval = chebyshev_polys(zz, N)

    Z = np.zeros_like(F_exact)
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                Z += coeffs[i, j, k] * np.einsum('p,q,r->pqr', Tx_eval[i], Ty_eval[j], Tz_eval[k])

    diff = F_exact - Z
    rmse_test = float(np.sqrt(np.mean(diff**2)))
    maxe_test = float(np.max(np.abs(diff)))

    return rmse_test, maxe_test, F_exact, Z, XX, YY, ZZ


# ============================================================
# Experiment runner
# ============================================================
def run_experiment(
    Ns=[2, 4, 8, 16, 32],
    c_values=[1, 3, 5, 10],
    resolution=100,
    train_mode="chebyshev",
    test_mode="uniform",
    outdir="results_3d"
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"experiment_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    all_results = []

    for c in c_values:
        for N in Ns:
            coeffs, train_rmse, train_maxe, _ = train_interpolant(N, c, mode=train_mode)
            test_rmse, test_maxe, F_exact, Z, XX, YY, ZZ = test_interpolant(
                coeffs, N, c, mode=test_mode, resolution=resolution
            )

            # Example visualization: central slice z=0
            mid = resolution // 2
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            im0 = ax[0].imshow(F_exact[:, :, mid], extent=[-1, 1, -1, 1], origin="lower", cmap="plasma")
            ax[0].set_title(f"True slice z=0, c={c}, N={N}")
            fig.colorbar(im0, ax=ax[0])
            im1 = ax[1].imshow(Z[:, :, mid], extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
            ax[1].set_title(f"Interp slice z=0, c={c}, N={N}")
            fig.colorbar(im1, ax=ax[1])
            fig.savefig(os.path.join(outdir, f"slice_c{c}_N{N}.png"))
            plt.close(fig)

            result_entry = {
                "N": N,
                "c": c,
                "train_mode": train_mode,
                "test_mode": test_mode,
                "train_RMSE": train_rmse,
                "train_MaxE": train_maxe,
                "test_RMSE": test_rmse,
                "test_MaxE": test_maxe,
                "timestamp": timestamp
            }
            all_results.append(result_entry)

            with open(os.path.join(outdir, f"results_c{c}_N{N}.json"), "w") as f:
                json.dump(result_entry, f, indent=4)

    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    return outdir, all_results


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    outdir, results = run_experiment()
    print(f"Experiment complete. Results in {outdir}")
