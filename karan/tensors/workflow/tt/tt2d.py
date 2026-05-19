"""2D Tensor Train decomposition of the Chebyshev coefficient tensor via TensorLy."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
import json
import os
from datetime import datetime

# Set backend
tl.set_backend('numpy')

# ============================================================
# Function to Approximate (2D)
# ============================================================
def f(xy, c=5):
    return 1 / (1 + c**2 * np.sum(xy**2, axis=1))

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
# Generate coefficient tensor (2D)
# ============================================================
def generate_coeff_tensor(N, c):
    k = np.arange(N+1)
    nodes = np.cos((2*k + 1) * np.pi / (2*(N+1)))
    X, Y = np.meshgrid(nodes, nodes, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    F = f(coords, c).reshape((N+1, N+1))

    Tx = chebyshev_polys(nodes, N)
    Ty = Tx

    # Least squares solve for coefficients
    A = np.kron(Ty.T, Tx.T)
    F_flat = F.ravel()
    c_flat, *_ = np.linalg.lstsq(A, F_flat, rcond=None)
    C = c_flat.reshape((N+1, N+1))
    return C, nodes

# ============================================================
# Evaluate interpolated function using Chebyshev coefficients
# ============================================================
def evaluate_interp(C, nodes, N, resolution=100):
    xx = np.linspace(-1, 1, resolution)
    yy = np.linspace(-1, 1, resolution)
    Tx = chebyshev_polys(xx, N)
    Ty = chebyshev_polys(yy, N)

    F = np.zeros((resolution, resolution))
    for i in range(N+1):
        for j in range(N+1):
            F += C[i, j] * np.outer(Tx[i], Ty[j])
    return xx, yy, F

# ============================================================
# Exact ground truth
# ============================================================
def compute_exact_function_grid(f, c, resolution=100):
    xx = np.linspace(-1, 1, resolution)
    yy = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    return xx, yy, f(coords, c).reshape((resolution, resolution))

# ============================================================
# Run Tensor Train Experiment (2D)
# ============================================================
def run_tt_experiment_2d(
    N=63, c=5, resolution=100, max_rank=20,
    tensorization_shape=None, random_seed=42, outdir="tt_results_2d"
):
    """
    Run a Tensor Train decomposition on the 2D Chebyshev coefficient tensor.

    Args:
        N (int): Degree (N=63 → tensor shape (64,64)).
        c (float): Function parameter.
        resolution (int): Grid resolution for evaluation.
        max_rank (int): TT-rank upper bound.
        tensorization_shape (list/tuple or None): Factorization of (N+1,N+1).
            Example: [2]*12 for (64,64).
            If None, defaults to [2]*12 when N=63.
        random_seed (int): RNG seed.
        outdir (str): Root directory to store results.
    """
    np.random.seed(random_seed)

    base_dir = os.getcwd()
    results_root = os.path.join(base_dir, outdir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"tt_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # Generate coefficients
    C, nodes = generate_coeff_tensor(N, c)

    # Tensorize
    if tensorization_shape is None:
        if (N+1) == 64:
            tensorization_shape = [2] * 12  # 64*64 = 2^12
        else:
            raise ValueError("Please provide a tensorization_shape for this N.")
    C_tensorized = C.reshape(tensorization_shape)

    # TT decomposition
    tt_cores = tensor_train(C_tensorized, rank=max_rank)
    C_reconstructed_tensor = tt_to_tensor(tt_cores)
    C_reconstructed = C_reconstructed_tensor.reshape(C.shape)

    # Metrics on coefficients
    diff_coeff = C - C_reconstructed
    l2_norm_diff = np.linalg.norm(diff_coeff)
    rel_l2_error = l2_norm_diff / np.linalg.norm(C)

    # Evaluate reconstructed function
    xx, yy, F_tt = evaluate_interp(C_reconstructed, nodes, N, resolution)
    xx, yy, F_true = compute_exact_function_grid(f, c, resolution)

    diff_func = F_true - F_tt
    rmse = np.sqrt(np.mean(diff_func**2))
    maxe = np.max(np.abs(diff_func))

    # Save metrics
    metrics = {
        "l2_norm_diff_coeffs": float(l2_norm_diff),
        "rel_l2_error_coeffs": float(rel_l2_error),
        "rmse_function": float(rmse),
        "maxe_function": float(maxe),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    # Save config
    config = {
        "N": N,
        "c": c,
        "resolution": resolution,
        "max_rank": max_rank,
        "tensorization_shape": tensorization_shape,
        "random_seed": random_seed,
        "timestamp": timestamp,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Save 3D surface comparison
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, F_true, cmap='viridis', linewidth=0, antialiased=False)
    ax1.set_title("True Function")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, F_tt, cmap='viridis', linewidth=0, antialiased=False)
    ax2.set_title("TT Reconstruction")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_3d.png"))
    plt.close(fig)

    return metrics, config, outdir

# ============================================================
# Run Example
# ============================================================
if __name__ == "__main__":
    # Example: N=63, factorize (64,64) as [2]*12
    results, config, savedir = run_tt_experiment_2d(
        N=63, resolution=100, max_rank=10, tensorization_shape=[2]*12
    )
    print("\n2D Experiment complete. Results stored in:", savedir)
