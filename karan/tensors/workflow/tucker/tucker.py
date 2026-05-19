"""2D Tucker rank sweep on Chebyshev coefficient tensors (TensorLy) for multiple C cases."""

# tucker_rank_sweep.py

import os
import json
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from datetime import datetime
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
import tensorly as tl

# ============================================================
# Function to Approximate (Matrix C)
# ============================================================
def fcn(coords, C=None):
    """Generalized 2D anisotropic test function."""
    if C is None:
        C = np.eye(2) * 5.0
    transformed = coords @ C.T
    sqnorm = np.sum(transformed**2, axis=1)
    return 1.0 / (1.0 + sqnorm)


def fcn2(coords, C=None, gamma=4.0):
    """
    Vectorized 2D anisotropic radial × angular test function.

    coords : array (N, 2)
        Each row is (x, y).
    C : 2x2 matrix (anisotropy / rotation / scaling)
        Defaults to identity.
    gamma : float
        Radial sharpness parameter.

    Computes:
        u, v = (C @ [x, y]^T)
        r = sqrt(u^2 + v^2)
        theta = atan2(v, u)
        f = exp(-gamma * r^2) * (1 + cos(theta))
    """
    if C is None:
        C = np.eye(2)

    # Apply anisotropic transform
    transformed = coords @ C.T
    u = transformed[:, 0]
    v = transformed[:, 1]

    # Polar coordinates
    r = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)

    # Radial × angular function
    vals = np.exp(-gamma * r * r) * (1.0 + np.cos(theta))

    return vals

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
# Compute Interpolation Coefficients
# ============================================================
def compute_interpolation_coeffs(N, M, d_x, d_y, C, node_type="chebyshev"):
    """Compute Chebyshev interpolation coefficients for f(x,y).
    
    Args:
        N: number of sampling points in x direction
        M: number of sampling points in y direction
        d_x: degree of Chebyshev polynomials in x
        d_y: degree of Chebyshev polynomials in y
        C: anisotropy matrix
        node_type: 'chebyshev', 'uniform', or 'random'
    """
    x_nodes = generate_nodes(N, mode=node_type)
    y_nodes = generate_nodes(M, mode=node_type)
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    F = fcn(coords, C).reshape((N, M))

    Tx = chebvander(x_nodes, d_x)  # shape (N, d_x+1)
    Ty = chebvander(y_nodes, d_y)  # shape (M, d_y+1)

    F_flat = F.ravel(order="F")
    A = np.kron(Ty.T, Tx.T)  # shape ((d_x+1)*(d_y+1), N*M)
    
    # Check if system is square (exact interpolation) or overdetermined
    if N == d_x + 1 and M == d_y + 1:
        # Square system: use QR method like als_zulip.py
        Q, R = np.linalg.qr(A)
        coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)
    else:
        # Overdetermined: solve A^T @ x = F_flat using least squares
        # Since A is ((d_x+1)*(d_y+1), N*M) and we need x of size (d_x+1)*(d_y+1)
        coeffs_flat, residuals, rank, s = np.linalg.lstsq(A.T, F_flat, rcond=None)
    
    coeffs = coeffs_flat.reshape((d_x + 1, d_y + 1), order="F")
    return coeffs


# ============================================================
# Tucker Rank Metrics
# ============================================================
def tucker_rank_metrics(coeffs, R_x, R_y):
    """
    Compute Tucker decomposition for given ranks and calculate reconstruction errors.
    
    Args:
        coeffs: coefficient matrix (d_x+1, d_y+1)
        R_x: Tucker rank in x direction
        R_y: Tucker rank in y direction
    """
    tl.set_backend("numpy")  # Ensure we're using NumPy backend
    
    rank_tuple = (R_x, R_y)
    core, factors = tucker(coeffs, rank=rank_tuple)

    coeffs_r = tucker_to_tensor((core, factors))
    diff = coeffs - coeffs_r

    l2_error = float(np.linalg.norm(diff))
    rel_l2_error = float(l2_error / np.linalg.norm(coeffs))
    mse = float(np.mean(diff**2))
    maxe = float(np.max(np.abs(diff)))

    result = {
        "metrics": {
            "l2_coeff_error": l2_error,
            "rel_l2_coeff_error": rel_l2_error,
            "final_loss_mse": mse,
            "maxe_coeff_error": maxe
        },
        "core_shape": list(core.shape),
        "factor_shapes": [list(f.shape) for f in factors],
        "core_tensor": core.tolist(),
        "factors": [f.tolist() for f in factors]
    }

    return result


# ============================================================
# Main Experiment
# ============================================================
def run_tucker_rank_sweep(
    N=64,
    M=64,
    d_x=63,
    d_y=63,
    max_rank=20,
    C_matrices=None,
    outdir="tucker_rank_sweep_results",
    node_type="chebyshev",
    random_seed=42
):
    """Run Tucker decomposition rank sweep experiment.
    
    Args:
        N: number of sampling points in x direction
        M: number of sampling points in y direction
        d_x: degree of Chebyshev polynomials in x
        d_y: degree of Chebyshev polynomials in y
        max_rank: maximum Tucker rank to sweep (will test 1 to max_rank)
        C_matrices: list of anisotropy matrices to test
        outdir: output directory name
        node_type: 'chebyshev', 'uniform', or 'random'
        random_seed: random seed for reproducibility
    """
    if C_matrices is None:
        C_matrices = [
            np.array([[5, 0], [0, 1]]),
            np.array([[1, 0], [0, 5]]),
            np.array([[1, 0], [0, 5]]) @ np.array([
                [np.cos(np.pi / 6), -np.sin(np.pi / 6)],
                [np.sin(np.pi / 6),  np.cos(np.pi / 6)]
            ]),
            5 * np.eye(2),
            0.2 * np.eye(2),
        ]

    np.random.seed(random_seed)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for idx, C in enumerate(C_matrices):
        print(f"\nRunning Tucker rank sweep for C_case_{idx} with {node_type} nodes")
        print(f"Sampling: N={N}, M={M} | Polynomial degrees: d_x={d_x}, d_y={d_y}")
        
        coeffs = compute_interpolation_coeffs(N, M, d_x, d_y, C, node_type=node_type)
        all_results[f"C_case_{idx}"] = {}
        
        for r in range(1, max_rank + 1):
            print(f"  Rank {r}x{r}...", end=" ")
            rank_result = tucker_rank_metrics(coeffs, R_x=r, R_y=r)
            all_results[f"C_case_{idx}"][f"rank_{r}x{r}"] = rank_result
            print(f"rel_error={rank_result['metrics']['rel_l2_coeff_error']:.3e}")

    combined_path = os.path.join(results_root, f"tucker_rank_sweep_all_{timestamp}.json")
    with open(combined_path, "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print(f"\nAll results saved to {combined_path}")
    return all_results


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    # Example: Run with different node types and grid sizes
    # Uncomment the configuration you want to use:
    
    # Default: Chebyshev nodes, 64x64 sampling, degree 63
    # run_tucker_rank_sweep(
    #     N=64, M=64, d_x=63, d_y=63, max_rank=20,
    #     node_type="chebyshev", 
    #     outdir="tucker_chebyshev_results"
    # )
    
    # Random nodes, various grid sizes:
    # run_tucker_rank_sweep(
    #     N=320, M=320, d_x=63, d_y=63, max_rank=20,
    #     node_type="random", 
    #     outdir="tucker_random_320_results"
    # )
    # run_tucker_rank_sweep(
    #     N=640, M=640, d_x=63, d_y=63, max_rank=20,
    #     node_type="random", 
    #     outdir="tucker_random_640_results"
    # )
    # run_tucker_rank_sweep(
    #     N=960, M=960, d_x=63, d_y=63, max_rank=20,
    #     node_type="random", 
    #     outdir="tucker_random_960_results"
    # )
    run_tucker_rank_sweep(
        N=1280, M=1280, d_x=63, d_y=63, max_rank=20,
        node_type="random", 
        outdir="tucker_random_1280_results"
    )
    
    # Uniform nodes:
    # run_tucker_rank_sweep(
    #     N=64, M=64, d_x=63, d_y=63, max_rank=20,
    #     node_type="uniform", 
    #     outdir="tucker_uniform_results"
    # )
