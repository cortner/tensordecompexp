"""3D Christoph-style Tucker ALS on an anisotropic test function with optional cached reference coeffs."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
import json
import os
from datetime import datetime
from scipy.sparse.linalg import LinearOperator, lsqr


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
    Compute (only ONCE) the 3D Chebyshev coefficients of f3d(x,y,z).
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

    # Design matrix mapping coeffs -> values on Chebyshev grid
    A = np.kron(Tz.T, np.kron(Ty.T, Tx.T))  # shape ((d_x+1)(d_y+1)(d_z+1), same)

    try:
        coeffs_flat, *_ = np.linalg.lstsq(A, F_flat, rcond=1e-12)
    except np.linalg.LinAlgError:
        print("Stopping early due to LinAlgError in lstsq.")
        return None

    coeffs = coeffs_flat.reshape((d_x + 1, d_y + 1, d_z + 1), order="F")

    return coeffs


# ============================================================
# Orthonormalization in a metric: A^T G A = I
# ============================================================
def orthonormalize_in_metric(A, G):
    """
    Return A_tilde with columns G-orthonormal:
        A_tilde^T G A_tilde = I
    using Cholesky factorization G = L L^T and QR on L^T A.
    """
    try:
        L = np.linalg.cholesky(G)          # G = L L^T
        Y = L.T @ A                        # shape (n, r)
        Q, _ = np.linalg.qr(Y)             # Q^T Q = I
        # solve L^T X = Q for X
        A_tilde = np.linalg.solve(L.T, Q)  # A_tilde^T G A_tilde = I
        return A_tilde
    except np.linalg.LinAlgError:
        # Fallback: plain Euclidean QR if G is too ill-conditioned
        Q, _ = np.linalg.qr(A)
        return Q


# ============================================================
# Helper: reconstruct F and C from Tucker factors in 3D
# ============================================================
def reconstruct_value_tensor(Tx, Ty, Tz, A_x, A_y, A_z, G):
    """
    F_hat(x,y,z) from Chebyshev Vandermonde matrices, coefficient factors, and core.

    Tx: (N_x, d_x+1)
    A_x: (d_x+1, R_x)
    G: (R_x, R_y, R_z)
    """
    Ux = Tx @ A_x  # (N_x, R_x)
    Uy = Ty @ A_y  # (N_y, R_y)
    Uz = Tz @ A_z  # (N_z, R_z)

    # F_hat[i,j,k] = sum_{a,b,c} Ux[i,a] Uy[j,b] Uz[k,c] G[a,b,c]
    F_hat = np.einsum("ia,jb,kc,abc->ijk", Ux, Uy, Uz, G)
    return F_hat, Ux, Uy, Uz


def reconstruct_coeff_tensor(A_x, A_y, A_z, G):
    """
    C_tucker(i,j,k) = sum_{a,b,c} A_x[i,a] A_y[j,b] A_z[k,c] G[a,b,c]
    """
    C_tucker = np.einsum("ia,jb,kc,abc->ijk", A_x, A_y, A_z, G)
    return C_tucker


# ============================================================
# 3D TUCKER EXPERIMENT (Christoph-style ALS)
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
    3D Christoph-style ALS Tucker on f3d, mirroring the 2D algorithm:

        F ≈ (T_x A_x, T_y A_y, T_z A_z, core G)

    Updates:
        - Freeze A_y, A_z and update A_x, G via LS on F ≈ T_x tildeG_x (U_y ⊗ U_z)^T.
        - Freeze A_x, A_z and update A_y, G via LS on F ≈ T_y tildeG_y (U_x ⊗ U_z)^T.
        - Freeze A_x, A_y and update A_z, G via LS on F ≈ T_z tildeG_z (U_x ⊗ U_y)^T.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    np.random.seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(results_root, f"tucker3d_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # -------------------- Reference coeffs --------------------
    if C_interp_3d is None:
        print("Computing reference 3D Chebyshev coefficients (once)...")
        C_interp_3d = compute_reference_coeffs_3d(d_x, d_y, d_z, C)

    # -------------------- Training grid --------------------
    x_nodes = generate_nodes(N_x, train_points)
    y_nodes = generate_nodes(N_y, train_points)
    z_nodes = generate_nodes(N_z, train_points)

    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    F = f3d(X, Y, Z, C=C)

    # Chebyshev Vandermonde on training nodes
    Tx = chebvander(x_nodes, d_x)  # (N_x, d_x+1)
    Ty = chebvander(y_nodes, d_y)  # (N_y, d_y+1)
    Tz = chebvander(z_nodes, d_z)  # (N_z, d_z+1)

    # Gram matrices in value space (diagnostics)
    Gx = Tx.T @ Tx
    Gy = Ty.T @ Ty
    Gz = Tz.T @ Tz

    print("\n=== Initial Gram condition numbers (3D) ===")
    print("cond(Tx^T Tx) =", np.linalg.cond(Gx))
    print("cond(Ty^T Ty) =", np.linalg.cond(Gy))
    print("cond(Tz^T Tz) =", np.linalg.cond(Gz))

    # -------------------- Initialization --------------------
    A_x = np.random.randn(d_x + 1, R_x)
    A_y = np.random.randn(d_y + 1, R_y)
    A_z = np.random.randn(d_z + 1, R_z)

    A_x = orthonormalize_in_metric(A_x, Gx)
    A_y = orthonormalize_in_metric(A_y, Gy)
    A_z = orthonormalize_in_metric(A_z, Gz)

    # Random initial core
    G_core = np.random.randn(R_x, R_y, R_z)

    N_xyz = N_x * N_y * N_z

    # -------------------- ALS Loop --------------------
    for it in range(n_iter):
        # Reconstruction and RMSE before updates
        F_hat, Ux, Uy, Uz = reconstruct_value_tensor(Tx, Ty, Tz, A_x, A_y, A_z, G_core)
        rmse = np.sqrt(np.mean((F - F_hat) ** 2))

        print(f"\nIter {it+1}")
        print(f"RMSE (before updates): {rmse:.3e}")

        if np.isnan(rmse) or rmse > 1e6:
            print("Stopping early due to divergence.")
            break

        # =====================================================
        # Mode-1 update: A_x and G_core
        # F_flat_x ≈ Tx tildeG_x (YZ_feat)^T
        # =====================================================
        Uy = Ty @ A_y  # (N_y, R_y)
        Uz = Tz @ A_z  # (N_z, R_z)

        # Combined feature matrix for (y,z): shape (N_y*N_z, R_y*R_z)
        YZ_feat = np.kron(Uz, Uy)

        # Flatten F with x as rows, (y,z) as columns
        F_flat_x = F.reshape(N_x, N_y * N_z, order="F")
        target_A = F_flat_x.reshape(-1, order="F")  # vec(F_flat_x)

        # Design matrix is D_A = kron(YZ_feat, Tx) but we do it implicitly.
        n_rows_A = N_xyz
        n_cols_A = (d_x + 1) * (R_y * R_z)
        R_yz = R_y * R_z

        def A_matvec(v):
            # v: ((d_x+1)*R_yz,) -> X: (d_x+1, R_yz)
            X = v.reshape(d_x + 1, R_yz, order="F")
            # Y: (N_x, N_yz) with N_yz = N_y*N_z
            Y = Tx @ X @ YZ_feat.T
            return Y.ravel(order="F")

        def A_rmatvec(w):
            # w: (N_xyz,) -> Y: (N_x, N_y*N_z)
            Y = w.reshape(N_x, N_y * N_z, order="F")
            X = Tx.T @ Y @ YZ_feat
            return X.ravel(order="F")

        A_op = LinearOperator(
            shape=(n_rows_A, n_cols_A),
            matvec=A_matvec,
            rmatvec=A_rmatvec,
            dtype=Tx.dtype,
        )

        try:
            tildeGx_vec = lsqr(A_op, target_A, atol=1e-12, btol=1e-12)[0]
        except Exception as e:
            print("Stopping early due to LSQR failure in mode-1:", e)
            break

        tildeGx = tildeGx_vec.reshape((d_x + 1, R_yz), order="F")

        # SVD: tildeGx ≈ U_xc diag(S_x) V_x^T
        U_xc, S_x, Vt_x = np.linalg.svd(tildeGx, full_matrices=False)

        U_xc_r = U_xc[:, :R_x]          # (d_x+1, R_x)
        S_x_r = S_x[:R_x]               # (R_x,)
        Vt_x_r = Vt_x[:R_x, :]          # (R_x, R_y*R_z)

        # New A_x and core G with this as mode-1 unfolding
        A_x = U_xc_r
        core1_unfold = np.diag(S_x_r) @ Vt_x_r   # (R_x, R_y*R_z)
        G_core = core1_unfold.reshape(R_x, R_y, R_z, order="F")

        At_Gx_Ax = A_x.T @ Gx @ A_x
        print("cond(A_x^T Gx A_x) after mode-1 update =", np.linalg.cond(At_Gx_Ax))

        # =====================================================
        # Mode-2 update: A_y and G_core
        # F_flat_y ≈ Ty tildeG_y (XZ_feat)^T
        # =====================================================
        Ux = Tx @ A_x
        Uz = Tz @ A_z

        # Combined feature matrix for (x,z): shape (N_x*N_z, R_x*R_z)
        XZ_feat = np.kron(Uz, Ux)

        # Permute F to (y,x,z) and flatten (x,z) into columns
        F_perm_y = np.transpose(F, (1, 0, 2))               # (N_y, N_x, N_z)
        F_flat_y = F_perm_y.reshape(N_y, N_x * N_z, order="F")
        target_B = F_flat_y.reshape(-1, order="F")

        n_rows_B = N_xyz
        n_cols_B = (d_y + 1) * (R_x * R_z)
        R_xz = R_x * R_z

        def B_matvec(v):
            # v: ((d_y+1)*R_xz,) -> X: (d_y+1, R_xz)
            X = v.reshape(d_y + 1, R_xz, order="F")
            Y = Ty @ X @ XZ_feat.T  # (N_y, N_x*N_z)
            return Y.ravel(order="F")

        def B_rmatvec(w):
            # w: (N_xyz,) -> Y: (N_y, N_x*N_z)
            Y = w.reshape(N_y, N_x * N_z, order="F")
            X = Ty.T @ Y @ XZ_feat
            return X.ravel(order="F")

        B_op = LinearOperator(
            shape=(n_rows_B, n_cols_B),
            matvec=B_matvec,
            rmatvec=B_rmatvec,
            dtype=Ty.dtype,
        )

        try:
            tildeGy_vec = lsqr(B_op, target_B, atol=1e-12, btol=1e-12)[0]
        except Exception as e:
            print("Stopping early due to LSQR failure in mode-2:", e)
            break

        tildeGy = tildeGy_vec.reshape((d_y + 1, R_xz), order="F")

        # SVD: tildeGy ≈ U_yc diag(S_y) V_y^T
        U_yc, S_y, Vt_y = np.linalg.svd(tildeGy, full_matrices=False)

        U_yc_r = U_yc[:, :R_y]          # (d_y+1, R_y)
        S_y_r = S_y[:R_y]               # (R_y,)
        Vt_y_r = Vt_y[:R_y, :]          # (R_y, R_x*R_z)

        A_y = U_yc_r

        # Mode-2 unfolding G_(2) should be diag(S_y) @ Vt_y_r
        core2_unfold = np.diag(S_y_r) @ Vt_y_r   # (R_y, R_x*R_z)
        core2_temp = core2_unfold.reshape(R_y, R_x, R_z, order="F")
        G_core = np.transpose(core2_temp, (1, 0, 2))        # (R_x, R_y, R_z)

        Bt_Gy_B = A_y.T @ Gy @ A_y
        print("cond(A_y^T Gy A_y) after mode-2 update =", np.linalg.cond(Bt_Gy_B))

        # =====================================================
        # Mode-3 update: A_z and G_core
        # F_flat_z ≈ Tz tildeG_z (XY_feat)^T
        # =====================================================
        Ux = Tx @ A_x
        Uy = Ty @ A_y

        # Combined feature matrix for (x,y): shape (N_x*N_y, R_x*R_y)
        XY_feat = np.kron(Uy, Ux)

        # Permute F to (z,x,y) and flatten (x,y) into columns
        F_perm_z = np.transpose(F, (2, 0, 1))               # (N_z, N_x, N_y)
        F_flat_z = F_perm_z.reshape(N_z, N_x * N_y, order="F")
        target_C = F_flat_z.reshape(-1, order="F")

        n_rows_C = N_xyz
        n_cols_C = (d_z + 1) * (R_x * R_y)
        R_xy = R_x * R_y

        def C_matvec(v):
            # v: ((d_z+1)*R_xy,) -> X: (d_z+1, R_xy)
            X = v.reshape(d_z + 1, R_xy, order="F")
            Y = Tz @ X @ XY_feat.T  # (N_z, N_x*N_y)
            return Y.ravel(order="F")

        def C_rmatvec(w):
            # w: (N_xyz,) -> Y: (N_z, N_x*N_y)
            Y = w.reshape(N_z, N_x * N_y, order="F")
            X = Tz.T @ Y @ XY_feat
            return X.ravel(order="F")

        C_op = LinearOperator(
            shape=(n_rows_C, n_cols_C),
            matvec=C_matvec,
            rmatvec=C_rmatvec,
            dtype=Tz.dtype,
        )

        try:
            tildeGz_vec = lsqr(C_op, target_C, atol=1e-12, btol=1e-12)[0]
        except Exception as e:
            print("Stopping early due to LSQR failure in mode-3:", e)
            break

        tildeGz = tildeGz_vec.reshape((d_z + 1, R_xy), order="F")

        # SVD: tildeGz ≈ U_zc diag(S_z) V_z^T
        U_zc, S_z, Vt_z = np.linalg.svd(tildeGz, full_matrices=False)

        U_zc_r = U_zc[:, :R_z]          # (d_z+1, R_z)
        S_z_r = S_z[:R_z]               # (R_z,)
        Vt_z_r = Vt_z[:R_z, :]          # (R_z, R_x*R_y)

        A_z = U_zc_r

        # Mode-3 unfolding G_(3) should be diag(S_z) @ Vt_z_r
        core3_unfold = np.diag(S_z_r) @ Vt_z_r   # (R_z, R_x*R_y)
        core3_temp = core3_unfold.reshape(R_z, R_x, R_y, order="F")
        G_core = np.transpose(core3_temp, (1, 2, 0))        # (R_x, R_y, R_z)

        Ct_Gz_C = A_z.T @ Gz @ A_z
        print("cond(A_z^T Gz A_z) after mode-3 update =", np.linalg.cond(Ct_Gz_C))

    # -------------------- Final Evaluation --------------------
    F_reconstructed, Ux, Uy, Uz = reconstruct_value_tensor(Tx, Ty, Tz, A_x, A_y, A_z, G_core)
    diff_final = F - F_reconstructed
    l2_norm_error = np.linalg.norm(diff_final)
    rmse_train = np.sqrt(np.mean(diff_final**2))

    # Dense grid evaluation
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    z_eval = np.linspace(-1, 1, resolution)

    X_eval, Y_eval, Z_eval = np.meshgrid(x_eval, y_eval, z_eval, indexing="ij")
    F_true_eval = f3d(X_eval, Y_eval, Z_eval, C=C)

    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    Tz_eval = chebvander(z_eval, d_z)

    F_pred_eval, _, _, _ = reconstruct_value_tensor(Tx_eval, Ty_eval, Tz_eval, A_x, A_y, A_z, G_core)
    rmse_grid = np.sqrt(np.mean((F_true_eval - F_pred_eval)**2))

    # Coefficient tensor from Tucker representation
    C_tucker = reconstruct_coeff_tensor(A_x, A_y, A_z, G_core)

    l2_coeff_error = np.linalg.norm(C_interp_3d - C_tucker)
    rel_l2_coeff_error = l2_coeff_error / np.linalg.norm(C_interp_3d)

    metrics = {
        "final_train_weighted_rmse": float(rmse_train),
        "rmse_eval_grid": float(rmse_grid),
        "l2_norm_error_train": float(l2_norm_error),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error),
    }

    # Save results
    with open(os.path.join(outdir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    np.savez(
        os.path.join(outdir, "coeffs3d.npz"),
        A_x=A_x, A_y=A_y, A_z=A_z,
        G=G_core,
        C_tucker=C_tucker,
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

    # Plot mid-z slice
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
    ax2.set_title("Reconstructed slice (3D Christoph ALS)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reconstruction_mid_slice.png"))
    plt.close(fig)

    return metrics, {
        "A_x": A_x, "A_y": A_y, "A_z": A_z,
        "G": G_core,
        "C_interp_3d": C_interp_3d,
    }, config, outdir


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

    d_x = d_y = d_z = 31
    print("Precomputing 3D reference Chebyshev coefficients...")
    C_interp_cache = {}

    for idx, C in enumerate(C_matrices):
        print(f"Computing reference for C_case={idx} ...")
        C_interp_cache[idx] = compute_reference_coeffs_3d(d_x, d_y, d_z, C)

        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        for R in range(1, 21):
            print(f"\n=== 3D Tucker experiment for C_case={idx}, Rank={R} ===")

            metrics, coeffs, config, savedir = run_als_tucker_experiment_3d(
                N_x=10*32, N_y=10*32, N_z=10*32,
                d_x=d_x, d_y=d_y, d_z=d_z,
                R_x=R, R_y=R, R_z=R,
                n_iter=20,
                C=C,
                C_interp_3d=C_interp_cache[idx],
                train_points="uniform",
                resolution=40,
                random_seed=42,
                outdir="als_tucker3d_zulip_uniform_results",
            )

            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config,
                "savedir": savedir,
            }

    with open("als_tucker3d_zulip_uniform_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll 3D Tucker runs complete. Results saved.")
