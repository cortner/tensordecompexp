"""3D Tucker rank sweep on interpolated Chebyshev coefficient tensors."""

# tucker_rank_sweep_3d.py

import os
import json
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from datetime import datetime
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
import tensorly as tl


# ============================================================
# 3D Function to Approximate (Matrix C)
# ============================================================
def fcn3(coords, C=None):
    """
    Direct 3D extension of the original fcn:

        f(x,y,z) = 1 / (1 + || C [x,y,z]^T ||^2 )

    coords : (N,3)
    C      : 3x3 anisotropy matrix (defaults to 5I)
    """
    if C is None:
        C = np.eye(3) * 5.0

    transformed = coords @ C.T
    sqnorm = np.sum(transformed**2, axis=1)

    return 1.0 / (1.0 + sqnorm)


# ============================================================
# Chebyshev Polynomials
# ============================================================
def chebyshev_polys(x, deg):
    T = np.zeros((deg + 1, len(x)))
    T[0] = 1
    if deg > 0:
        T[1] = x
    for k in range(2, deg + 1):
        T[k] = 2 * x * T[k - 1] - T[k - 2]
    return T


# ============================================================
# Compute 3D Chebyshev Interpolation Coefficients
# ============================================================
def compute_interpolation_coeffs_3d(N, C):
    """
    Compute Chebyshev interpolation coefficients for f(x,y,z).
    Produces a (N+1)x(N+1)x(N+1) tensor.
    """
    nodes = np.cos(np.pi * np.arange(N + 1) / N)

    X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    F = fcn3(coords, C).reshape((N + 1, N + 1, N + 1))

    Tx = chebvander(nodes, N)
    Ty = chebvander(nodes, N)
    Tz = chebvander(nodes, N)

    F_flat = F.ravel(order="F")

    # Tensor-product Chebyshev transform:  Tz ⊗ Ty ⊗ Tx
    A = np.kron(Tz.T, np.kron(Ty.T, Tx.T))

    Q, R = np.linalg.qr(A)
    coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)

    coeffs = coeffs_flat.reshape((N + 1, N + 1, N + 1), order="F")
    return coeffs


# ============================================================
# Tucker Rank Metrics (3D)
# ============================================================
def tucker_rank_metrics_3d(coeffs, ranks):
    """
    Compute Tucker decompositions for given ranks on a 3D tensor.
    """
    tl.set_backend("numpy")
    results = {}

    for r in ranks:
        core, factors = tucker(coeffs, rank=(r, r, r))
        coeffs_r = tucker_to_tensor((core, factors))
        diff = coeffs - coeffs_r

        l2 = float(np.linalg.norm(diff))
        rl2 = float(l2 / np.linalg.norm(coeffs))
        mse = float(np.mean(diff**2))
        maxe = float(np.max(np.abs(diff)))

        results[f"rank_{r}x{r}x{r}"] = {
            "metrics": {
                "l2_coeff_error": l2,
                "rel_l2_coeff_error": rl2,
                "final_loss_mse": mse,
                "maxe_coeff_error": maxe
            },
            "core_shape": list(core.shape),
            "factor_shapes": [list(f.shape) for f in factors],
            "core_tensor": core.tolist(),
            "factors": [f.tolist() for f in factors]
        }

    return results


# ============================================================
# Rotation Matrix (Z-axis)
# ============================================================
def rotation_matrix_3d_z(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])


# ============================================================
# Main Experiment — 3D Tucker Rank Sweep
# ============================================================
def run_tucker_rank_sweep_3d(
    N=31,
    C_matrices=None,
    ranks=list(range(1, 21)),
    outdir="tucker_rank_sweep_results_3d"
):
    if C_matrices is None:
        C_matrices = [
            np.diag([5.0, 1.0, 1.0]),
            np.diag([1.0, 5.0, 1.0]),
            np.diag([1.0, 1.0, 5.0]),
            rotation_matrix_3d_z(30) @ np.diag([5.0, 1.0, 1.0]),
            5.0 * np.eye(3),
        ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for idx, C in enumerate(C_matrices):
        print(f"[3D] Running Tucker rank sweep for C_case_{idx}")
        coeffs = compute_interpolation_coeffs_3d(N, C)
        rank_results = tucker_rank_metrics_3d(coeffs, ranks)
        all_results[f"C_case_{idx}"] = rank_results

    outpath = os.path.join(results_root, f"tucker_rank_sweep_3d_{timestamp}.json")
    with open(outpath, "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print(f"[3D] All results saved to {outpath}")
    return all_results


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    run_tucker_rank_sweep_3d()
