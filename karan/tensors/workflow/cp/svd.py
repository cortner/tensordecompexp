"""2D Chebyshev interp then truncated SVD rank sweep vs exact function (baseline for CP quality)."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================
# Core function
# ============================================================
def fcn(x, c):
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
# Training step
# ============================================================
def train_interpolant(N, c, mode="chebyshev"):
    if mode == "chebyshev":
        k = np.arange(N+1)
        nodes = np.cos((2*k + 1) * np.pi / (2*(N+1)))
    elif mode == "uniform":
        nodes = np.linspace(-1, 1, N+1)
    else:
        raise ValueError("Unknown training mode, choose 'chebyshev' or 'uniform'")

    X, Y = np.meshgrid(nodes, nodes, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    F = fcn(coords, c).reshape((N+1, N+1))

    Tx = chebyshev_polys(nodes, N)
    Ty = chebyshev_polys(nodes, N)

    F_flat = F.T.ravel()
    A = np.kron(Ty.T, Tx.T)
    Q, R = np.linalg.qr(A)
    coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)
    coeffs = coeffs_flat.reshape((N+1, N+1))

    Z_train = np.zeros_like(F)
    for i in range(N+1):
        for j in range(N+1):
            Z_train += coeffs[i, j] * np.outer(Tx[i], Ty[j])

    diff_train = F - Z_train
    rmse_train = float(np.sqrt(np.mean(diff_train**2)))
    maxe_train = float(np.max(np.abs(diff_train)))

    return coeffs, rmse_train, maxe_train, nodes


# ============================================================
# Testing step
# ============================================================
def xtest_interpolant(coeffs, N, c, mode="uniform", resolution=100):
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

    F_exact = fcn(coords, c).reshape((resolution, resolution))

    Tx_eval = chebyshev_polys(xx, N)
    Ty_eval = chebyshev_polys(yy, N)
    Z = np.zeros_like(F_exact)
    for i in range(N+1):
        for j in range(N+1):
            Z += coeffs[i, j] * np.outer(Tx_eval[i], Ty_eval[j])

    diff = F_exact - Z
    rmse_test = float(np.sqrt(np.mean(diff**2)))
    maxe_test = float(np.max(np.abs(diff)))

    return rmse_test, maxe_test, F_exact, Z, XX, YY


# ============================================================
# SVD error computation
# ============================================================
def compute_svd_rmse_maxe(Z, F, r):
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    Uk = U[:, :r]
    Sk = np.diag(S[:r])
    Vk = Vt[:r, :]
    Zr = Uk @ Sk @ Vk

    # Reconstruction error (Z vs Zr)
    diff_recon = Z - Zr
    l2_error = float(np.linalg.norm(diff_recon))
    maxe_recon = float(np.max(np.abs(diff_recon)))

    # Function error (F vs Zr)
    diff_func = F - Zr
    rmse_func = float(np.sqrt(np.mean(diff_func**2)))
    maxe_func = float(np.max(np.abs(diff_func)))

    return l2_error, maxe_recon, rmse_func, maxe_func


# ============================================================
# SVD experiment runner
# ============================================================
def run_svd_experiment(
    Ns=[2, 4, 8, 16, 32, 64],
    r_values=list(range(1, 11)),
    c_values=[1, 3, 5, 10],
    resolution=100,
    outdir="results_svd"
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"experiment_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    all_results = []

    for N in Ns:
        for c in c_values:
            coeffs, _, _, _ = train_interpolant(N, c, mode="chebyshev")
            _, _, F_exact, Z, XX, YY = xtest_interpolant(coeffs, N, c, mode="uniform", resolution=resolution)


            rmse_func_list = []
            maxe_func_list = []

            for r in r_values:
                l2_error, maxe_recon, rmse_func, maxe_func = compute_svd_rmse_maxe(Z, F_exact, r)

                result_entry = {
                    "N": N,
                    "c": c,
                    "r": r,
                    "l2_error_ZZr": l2_error,
                    "maxe_ZZr": maxe_recon,
                    "rmse_func": rmse_func,
                    "maxe_func": maxe_func,
                    "timestamp": timestamp
                }
                all_results.append(result_entry)

                fname = os.path.join(outdir, f"results_c{c}_N{N}_r{r}.json")
                with open(fname, "w") as f:
                    json.dump(result_entry, f, indent=4)

                rmse_func_list.append(rmse_func)
                maxe_func_list.append(maxe_func)

            # Plot for this (N, c)
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"SVD Approximation Errors (N={N}, c={c})")

            axs[0].plot(r_values, rmse_func_list, marker="o")
            axs[0].set_xlabel("Rank r")
            axs[0].set_ylabel("RMSE vs true function")
            axs[0].set_yscale("log")
            axs[0].grid(True, which="both", linestyle="--")

            axs[1].plot(r_values, maxe_func_list, marker="s")
            axs[1].set_xlabel("Rank r")
            axs[1].set_ylabel("MaxE vs true function")
            axs[1].set_yscale("log")
            axs[1].grid(True, which="both", linestyle="--")

            plt.tight_layout()
            fig.savefig(os.path.join(outdir, f"errors_c{c}_N{N}.png"))
            plt.close(fig)

    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    return outdir, all_results


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    outdir, results = run_svd_experiment()
    print(f"SVD experiment complete. Results in {outdir}")
