"""3D TensorLy PARAFAC on Chebyshev coefficient tensor built by projection; rank and error sweeps."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
from numpy.polynomial.chebyshev import chebval

# Use NumPy backend
tl.set_backend("numpy")


# ============================================================
# Core 3D radial function
# ============================================================
def fcn(xyz, c):
    """xyz: (M, 3) array of points in [-1,1]^3. Returns shape (M,)."""
    return 1.0 / (1.0 + (c**2) * np.sum(xyz**2, axis=1))


# ============================================================
# Helper: Chebyshev polynomials T_0..T_deg evaluated at x
# ============================================================
def chebyshev_polys(x, deg):
    """
    Returns array of shape (deg+1, len(x)) where row k = T_k(x).
    Uses numpy.polynomial.chebyshev.chebval for numerical stability.
    """
    coeffs = np.eye(deg + 1)
    return np.array([chebval(x, coeff) for coeff in coeffs])  # (deg+1, len(x))


# ============================================================
# Build Chebyshev coefficient tensor C via projection on Chebyshev nodes
# ============================================================
def generate_coeff_tensor_projection(N, c):
    """
    N: degree in each dim. Basis size is (N+1)^3.
    Returns:
      C  : (N+1, N+1, N+1) Chebyshev coefficient tensor
      nodes: Chebyshev nodes used
    """
    k = np.arange(N + 1)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * (N + 1)))  # Gauss-Chebyshev nodes of first kind

    X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    F = fcn(coords, c).reshape((N + 1, N + 1, N + 1))

    Tx = chebyshev_polys(nodes, N)  # (N+1, N+1)
    Ty = Tx.copy()
    Tz = Tx.copy()

    # Discrete projection on Chebyshev grid
    # Note: for exact orthogonality one would include quadrature weights.
    # For our smooth test function this simple projection works well.
    C = np.einsum("il,jm,kn,ijk->lmn", Tx, Ty, Tz, F)  # (N+1, N+1, N+1)

    return C, nodes


# ============================================================
# Evaluate Chebyshev interpolant directly from C on a uniform grid
# ============================================================
def evaluate_direct_interp(C, N, resolution=50):
    xx = np.linspace(-1, 1, resolution)
    yy = np.linspace(-1, 1, resolution)
    zz = np.linspace(-1, 1, resolution)

    Tx = chebyshev_polys(xx, N)  # (N+1, res)
    Ty = chebyshev_polys(yy, N)
    Tz = chebyshev_polys(zz, N)

    # Correct einsum: (res,a)(res,b)(res,c)(a,b,c) -> (res,res,res)
    F = np.einsum("ia,jb,kc,abc->ijk", Tx.T, Ty.T, Tz.T, C)
    return F


# ============================================================
# Evaluate Chebyshev interpolant from CP factors on a uniform grid
# ============================================================
def evaluate_cp_interp(weights, factors, N, resolution=50):
    xx = np.linspace(-1, 1, resolution)
    yy = np.linspace(-1, 1, resolution)
    zz = np.linspace(-1, 1, resolution)

    Tx = chebyshev_polys(xx, N)  # (N+1, res)
    Ty = chebyshev_polys(yy, N)
    Tz = chebyshev_polys(zz, N)

    # For each rank component r:
    # a = sum_a T_a(xx) * A[a, r]  -> shape (res,)
    # b = sum_b T_b(yy) * B[b, r]  -> shape (res,)
    # c = sum_c T_c(zz) * C[c, r]  -> shape (res,)
    # Contribution = w[r] * outer(a, b, c)
    F_cp = np.zeros((resolution, resolution, resolution))
    A, B, C = factors  # each (N+1, R)
    R = A.shape[1]

    for r in range(R):
        ax = Tx.T @ A[:, r]
        by = Ty.T @ B[:, r]
        cz = Tz.T @ C[:, r]
        F_cp += weights[r] * np.einsum("i,j,k->ijk", ax, by, cz)

    return F_cp


# ============================================================
# Ground truth on a uniform grid
# ============================================================
def compute_exact_function_grid(c, resolution=50):
    xx = np.linspace(-1, 1, resolution)
    yy = np.linspace(-1, 1, resolution)
    zz = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return fcn(coords, c).reshape((resolution, resolution, resolution))


# ============================================================
# Error helpers
# ============================================================
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def maxe(a, b):
    return float(np.max(np.abs(a - b)))


# ============================================================
# CP experiment runner
# ============================================================
def run_cp_experiment(
    Ns=(8, 16, 32),       # choose N values
    c_values=(1, 3, 5),   # test smoothness parameter
    ranks=tuple(range(1, 11)),
    resolution=50,
    outdir="results_cp3d",
    random_state=0,
):
    base_dir = os.getcwd()
    results_root = os.path.join(base_dir, outdir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"experiment_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    all_results = []

    for N in Ns:
        for c in c_values:
            # Build coefficients
            C, nodes = generate_coeff_tensor_projection(N, c)

            # Baseline direct interpolation errors
            F_true = compute_exact_function_grid(c, resolution=resolution)
            F_direct = evaluate_direct_interp(C, N, resolution=resolution)
            rmse_baseline = rmse(F_true, F_direct)
            maxe_baseline = maxe(F_true, F_direct)

            # Save a small summary for baseline
            base_entry = {
                "N": N,
                "c": c,
                "rank": 0,
                "rmse_vs_true": rmse_baseline,
                "maxe_vs_true": maxe_baseline,
                "coef_rel_frob_err": 0.0,
                "coef_max_abs_err": 0.0,
                "timestamp": timestamp,
            }
            all_results.append(base_entry)

            # Per-rank CP fits
            rmse_list = []
            maxe_list = []

            # Normalize C Frobenius once for relative errors
            fro_C = float(np.linalg.norm(C))

            for r in ranks:
                # CP decomp of coefficient tensor C
                # init='svd' usually converges faster for well-behaved tensors
                cp = parafac(
                    C,
                    rank=r,
                    init="svd",
                    n_iter_max=1000,
                    random_state=random_state,
                )

                # Coefficient-space reconstruction error
                C_hat = cp_to_tensor(cp)
                coef_frob = float(np.linalg.norm(C - C_hat))
                coef_rel_frob = float(coef_frob / (fro_C + 1e-30))
                coef_max_abs = float(np.max(np.abs(C - C_hat)))

                # Evaluate interpolant from factors on uniform grid
                weights, factors = cp
                F_cp = evaluate_cp_interp(weights, factors, N, resolution=resolution)

                r_entry = {
                    "N": N,
                    "c": c,
                    "rank": r,
                    "rmse_vs_true": rmse(F_true, F_cp),
                    "maxe_vs_true": maxe(F_true, F_cp),
                    "coef_rel_frob_err": coef_rel_frob,
                    "coef_max_abs_err": coef_max_abs,
                    "timestamp": timestamp,
                }

                all_results.append(r_entry)

                # Save per-rank JSON for quick scans
                fname = os.path.join(outdir, f"results_c{c}_N{N}_r{r}.json")
                with open(fname, "w") as f:
                    json.dump(r_entry, f, indent=4)

                rmse_list.append(r_entry["rmse_vs_true"])
                maxe_list.append(r_entry["maxe_vs_true"])

            # Plot errors vs rank
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"3D CP Interpolant Errors (N={N}, c={c})")

            axs[0].plot(ranks, rmse_list, marker="o", label="CP vs true")
            axs[0].axhline(rmse_baseline, linestyle="--", label="Direct Cheb baseline")
            axs[0].set_xlabel("CP rank r")
            axs[0].set_ylabel("RMSE vs true")
            axs[0].set_yscale("log")
            axs[0].grid(True, which="both")
            axs[0].legend()

            axs[1].plot(ranks, maxe_list, marker="s", label="CP vs true")
            axs[1].axhline(maxe_baseline, linestyle="--", label="Direct Cheb baseline")
            axs[1].set_xlabel("CP rank r")
            axs[1].set_ylabel("MaxE vs true")
            axs[1].set_yscale("log")
            axs[1].grid(True, which="both")
            axs[1].legend()

            plt.tight_layout()
            fig.savefig(os.path.join(outdir, f"errors_c{c}_N{N}.png"), dpi=150)
            plt.close(fig)

    # Save all results
    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    return outdir, all_results


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Tweak as needed. N=32 or 64 with resolution=75..100 is heavier.
    out_dir, results = run_cp_experiment(
        Ns=(8, 16, 32),
        c_values=(1, 3, 5),
        ranks=tuple(range(1, 25)),
        resolution=100,
        outdir="results_cp3d",
        random_state=0,
    )
    print(f"CP 3D experiment complete. Results in {out_dir}")
