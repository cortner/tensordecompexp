"""3D TensorLy Tucker on sample values, lift to Chebyshev coefficients; reuses a cached reference coeff tensor."""

# now we would talk about same spaces here
# value space and coefficient space
# by including the weights so the system is orthogonal
# which makes minimising the function equivalent to minimising the tensor

#  ======================================
#  3D SIMPLE TUCKER + CHEBYSHEV CONVERSION (with caching)
#  ======================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
import json
import os
from datetime import datetime

import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker
from tensorly.tenalg import multi_mode_dot

tl.set_backend("numpy")


# ============================================================
# 3D Function to Approximate (Matrix C: 3x3)
# ============================================================
def f3d(x, y, z, C=None):
    """
    Generalized 3D test function with anisotropy via a 3x3 matrix C:

        f(x,y,z) = 1 / (1 + || C [x, y, z]^T ||^2)
    """
    if C is None:
        C = np.eye(3) * 5.0

    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (Npts, 3)
    trans = coords @ C.T
    vals = 1.0 / (1.0 + np.sum(trans**2, axis=1))
    return vals.reshape(x.shape)


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
# 3D True Chebyshev coefficient tensor
# ============================================================
def compute_reference_coeffs_3d(d_x, d_y, d_z, C):
    """
    Compute (only ONCE) the true 3D Chebyshev coefficients of f3d(x,y,z).
    """
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    nodes_z = np.cos(np.pi * np.arange(d_z + 1) / d_z)

    X, Y, Z = np.meshgrid(nodes_x, nodes_y, nodes_z, indexing="ij")
    F = f3d(X, Y, Z, C=C)

    F_flat = F.ravel(order="F")

    Tx = chebvander(nodes_x, d_x)
    Ty = chebvander(nodes_y, d_y)
    Tz = chebvander(nodes_z, d_z)

    A = np.kron(Tz.T, np.kron(Ty.T, Tx.T))

    coeffs_flat, *_ = np.linalg.lstsq(A, F_flat, rcond=None)
    coeffs = coeffs_flat.reshape((d_x + 1, d_y + 1, d_z + 1), order="F")

    return coeffs


# ============================================================
# 3D TUCKER EXPERIMENT (NO ALS) — with precomputed coefficients
# ============================================================
def run_als_tucker_experiment_3d(
    N_x=64, N_y=64, N_z=64,
    d_x=31, d_y=31, d_z=31,
    R_x=5, R_y=5, R_z=5,
    n_iter=200,
    C=np.eye(3) * 5.0,
    resolution=50,
    C_interp_3d=None,   # Cached reference tensor
    num_test_points=2048,
    epsilon=1e-6,
    random_seed=42,
    outdir="als_tucker3d_results",
    train_points="chebyshev",
    test_points="uniform",
):
    """
    EXACT same metrics and structure as the 2D version,
    but updated to full 3D Tucker + Chebyshev lifting.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    np.random.seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"tucker3d_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # Compute reference coefficients ONCE if not provided
    if C_interp_3d is None:
        print("Computing reference 3D Chebyshev coefficients (once)...")
        C_interp_3d = compute_reference_coeffs_3d(d_x, d_y, d_z, C)

    # -------------------- Training Grid --------------------
    x_nodes = generate_nodes(N_x, train_points)
    y_nodes = generate_nodes(N_y, train_points)
    z_nodes = generate_nodes(N_z, train_points)

    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    F = f3d(X, Y, Z, C=C)

    # -------------------- TensorLy Tucker --------------------
    print("Running 3D TensorLy Tucker...")
    tucker_tensor, rec_errors = tl_tucker(
        F,
        rank=[R_x, R_y, R_z],
        n_iter_max=n_iter,
        init="svd",
        tol=1e-8,
        random_state=random_seed,
        return_errors=True,
    )

    core_val, (Ux, Uy, Uz) = tucker_tensor
    F_tucker_val = tl.tucker_to_tensor(tucker_tensor)
    print(f"3D value-space Tucker RMSE: {np.sqrt(np.mean((F - F_tucker_val)**2)):.3e}")

    # -------------------- Chebyshev Lifting --------------------
    Tx = chebvander(x_nodes, d_x)
    Ty = chebvander(y_nodes, d_y)
    Tz = chebvander(z_nodes, d_z)

    A_x, *_ = np.linalg.lstsq(Tx, Ux, rcond=None)
    A_y, *_ = np.linalg.lstsq(Ty, Uy, rcond=None)
    A_z, *_ = np.linalg.lstsq(Tz, Uz, rcond=None)

    C_tucker = multi_mode_dot(core_val, [A_x, A_y, A_z], modes=[0, 1, 2])

    # Reconstruct in value space
    Ux_val = Tx @ A_x
    Uy_val = Ty @ A_y
    Uz_val = Tz @ A_z

    F_reconstructed = tl.tucker_to_tensor((core_val, [Ux_val, Uy_val, Uz_val]))
    diff_final = F - F_reconstructed

    rmse_train = np.sqrt(np.mean(diff_final**2))
    l2_norm_error = np.linalg.norm(diff_final)

    # -------------------- Dense Grid Evaluation --------------------
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    z_eval = np.linspace(-1, 1, resolution)

    X_eval, Y_eval, Z_eval = np.meshgrid(x_eval, y_eval, z_eval, indexing="ij")
    F_true_eval = f3d(X_eval, Y_eval, Z_eval, C=C)

    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    Tz_eval = chebvander(z_eval, d_z)

    Ux_eval = Tx_eval @ A_x
    Uy_eval = Ty_eval @ A_y
    Uz_eval = Tz_eval @ A_z

    F_pred_eval = tl.tucker_to_tensor((core_val, [Ux_eval, Uy_eval, Uz_eval]))
    rmse_grid = np.sqrt(np.mean((F_true_eval - F_pred_eval)**2))

    # -------------------- Coefficient Errors --------------------
    l2_coeff_error = np.linalg.norm(C_interp_3d - C_tucker)
    rel_l2_coeff_error = l2_coeff_error / np.linalg.norm(C_interp_3d)

    # -------------------- Metrics (MATCH 2D) --------------------
    metrics = {
        "final_train_weighted_rmse": float(rmse_train),
        "rmse_eval_grid": float(rmse_grid),
        "l2_norm_error_train": float(l2_norm_error),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error),
    }

    # -------------------- Save --------------------
    with open(os.path.join(outdir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    np.savez(
        os.path.join(outdir, "coeffs3d.npz"),
        A_x=A_x, A_y=A_y, A_z=A_z,
        G=core_val,
        C_tucker=C_tucker,
        Ux=Ux, Uy=Uy, Uz=Uz,
        C_interp_3d=C_interp_3d,
    )

    config = {
        "N_x": N_x, "N_y": N_y, "N_z": N_z,
        "d_x": d_x, "d_y": d_y, "d_z": d_z,
        "R_x": R_x, "R_y": R_y, "R_z": R_z,
        "n_iter": n_iter,
        "C": C.tolist(),
        "resolution": resolution,
        "train_points": train_points,
        "random_seed": random_seed,
        "timestamp": timestamp,
    }

    with open(os.path.join(outdir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # -------------------- Plot mid-z slice --------------------
    mid_k = N_z // 2
    X_slice = X[:, :, mid_k]
    Y_slice = Y[:, :, mid_k]
    F_slice = F[:, :, mid_k]
    F_rec_slice = F_reconstructed[:, :, mid_k]

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X_slice, Y_slice, F_slice, cmap=cm.viridis)
    ax1.set_title("Original f3d slice")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X_slice, Y_slice, F_rec_slice, cmap=cm.viridis)
    ax2.set_title("Reconstructed slice")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reconstruction_mid_slice.png"))
    plt.close(fig)

    return metrics, {
        "A_x": A_x, "A_y": A_y, "A_z": A_z, "G": core_val,
        "C_interp_3d": C_interp_3d,
    }, config, outdir


# ============================================================
# 3D Rank Sweep With Caching
# ============================================================
if __name__ == "__main__":

    def rotation_matrix_3d_z(theta_deg):
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,          1.0],
        ])

    C_matrices = [
        np.diag([5.0, 1.0, 1.0]),
        np.diag([1.0, 5.0, 1.0]),
        np.diag([1.0, 1.0, 5.0]),
        rotation_matrix_3d_z(30) @ np.diag([5.0, 1.0, 1.0]),
        5.0 * np.eye(3),
    ]

    all_results = {}

    # ---------- GLOBAL CACHING ----------
    d_x = d_y = d_z = 31
    print("Precomputing 3D reference Chebyshev coefficients...")
    # Only depends on C, so cached per C
    C_interp_cache = {}

    for idx, C in enumerate(C_matrices):
        print(f"Computing reference for C_case={idx} ...")
        C_interp_cache[idx] = compute_reference_coeffs_3d(d_x, d_y, d_z, C)

        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        for R in range(1, 21):
            print(f"\n=== 3D Tucker experiment for C_case={idx}, Rank={R} ===")

            metrics, coeffs, config, savedir = run_als_tucker_experiment_3d(
                N_x=32, N_y=32, N_z=32,
                d_x=d_x, d_y=d_y, d_z=d_z,
                R_x=R, R_y=R, R_z=R,
                n_iter=100,
                C=C,
                C_interp_3d=C_interp_cache[idx],   # <--- cached
                train_points="chebyshev",
                resolution=40,
                random_seed=42,
                outdir="als_tucker3d_cheb_results",
            )

            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config,
                "savedir": savedir,
            }

    with open("als_tucker3d_tensorly_cheb.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll 3D Tucker runs complete. Results saved.")
