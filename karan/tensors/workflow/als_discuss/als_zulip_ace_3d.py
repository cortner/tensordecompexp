"""3D ACE energy data with symmetric Tucker ALS on pooled Chebyshev features."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
import json
import os
from datetime import datetime


# ============================================================
# Underlying 3D function (used only to synthesize training targets)
# ============================================================
def f3(x, y, z, C=None):
    """
    Generalized 3D test function with anisotropy via matrix C (3x3):
        f(x,y,z) = 1 / (1 + || C [x,y,z]^T ||^2 )
    """
    if C is None:
        C = np.eye(3) * 5.0

    # C = 0.5 * (C + C.T)  # ensure symmetry

    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (N, 3)
    trans = coords @ C.T                                          # (N, 3)
    vals = 1.0 / (1.0 + np.sum(trans**2, axis=1))
    return vals.reshape(x.shape)


# ============================================================
# Chebyshev helpers for reference coefficient tensor (3D)
# ============================================================
def chebyshev_polys(x, deg):
    """
    Return T_k(x) for k=0..deg as array shape (deg+1, len(x)).
    (Chebyshev polynomials of the first kind.)
    """
    T = np.zeros((deg + 1, len(x)))
    T[0] = 1.0
    if deg > 0:
        T[1] = x
    for k in range(2, deg + 1):
        T[k] = 2 * x * T[k - 1] - T[k - 2]
    return T


def compute_reference_coeffs_3d(d_x, d_y, d_z, C):
    """
    Compute reference Chebyshev coefficient tensor C_true for f3(x,y,z)
    on a tensor Chebyshev grid, avoiding an enormous Kronecker matrix by
    solving three sequential square systems (since nodes count = degree+1).

    We want coefficients so that on the grid:
        F(i,j,k) = sum_{a,b,c} C_true[a,b,c] * T_a(x_i)*T_b(y_j)*T_c(z_k)

    Returns:
        C_true of shape ((d_x+1), (d_y+1), (d_z+1))
    """
    # Chebyshev nodes of the first kind
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    nodes_z = np.cos(np.pi * np.arange(d_z + 1) / d_z)

    X, Y, Z = np.meshgrid(nodes_x, nodes_y, nodes_z, indexing="ij")
    F = f3(X, Y, Z, C=C)  # shape (nx, ny, nz)

    Tx = chebyshev_polys(nodes_x, d_x)  # (dx+1, nx) square
    Ty = chebyshev_polys(nodes_y, d_y)  # (dy+1, ny) square
    Tz = chebyshev_polys(nodes_z, d_z)  # (dz+1, nz) square

    # Step 1: solve along x  -> U[a, j, k]  such that Tx^T U(:,j,k) = F(:,j,k)
    nx, ny, nz = F.shape
    U = np.linalg.solve(Tx.T, F.reshape(nx, -1)).reshape(nx, ny, nz)

    # Step 2: solve along y  -> V[a, b, k]  such that Ty^T V(a,:,k) = U(a,:,k)
    # Do this by viewing each (a,k) slice as a RHS for Ty^T
    U_t = np.transpose(U, (1, 0, 2))               # (ny, nx, nz)
    V_t = np.linalg.solve(Ty.T, U_t.reshape(ny, -1)).reshape(ny, nx, nz)
    V = np.transpose(V_t, (1, 0, 2))               # (nx, ny, nz)

    # Step 3: solve along z  -> C_true[a, b, c] such that Tz^T C_true(a,b,:) = V(a,b,:)
    V_t2 = np.transpose(V, (2, 0, 1))              # (nz, nx, ny)
    C_t2 = np.linalg.solve(Tz.T, V_t2.reshape(nz, -1)).reshape(nz, nx, ny)
    C_true = np.transpose(C_t2, (1, 2, 0))         # (nx, ny, nz) = (dx+1, dy+1, dz+1)

    return C_true


# ============================================================
# ACE feature construction (1D pooled sums)
# ============================================================
def ace_sum_features(points_1d, deg):
    """
    ACE pooled invariant features for 1D points:
        a_k = sum_i T_k(x_i), k=0..deg
    points_1d: shape (n_atoms,)
    return: shape (deg+1,)
    """
    T = chebvander(points_1d, deg)  # (n_atoms, deg+1)
    return T.sum(axis=0)            # (deg+1,)


def generate_configs_3d(S, n_atoms=25, mode="random", seed=42):
    """
    Generate S configurations, each is a single array X of shape (n_atoms, 3).
    Each row is one point/atom r_i = (x_i, y_i, z_i) in [-1, 1]^3.
    """
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(S):
        if mode == "random":
            X = rng.uniform(-1.0, 1.0, size=(n_atoms, 3))
        elif mode == "uniform":
            t = np.linspace(-1.0, 1.0, n_atoms)
            X = np.stack([t, t, t], axis=1)
        else:
            raise ValueError(f"Unknown mode {mode}")
        configs.append(X)
    return configs


# ---------------- CHANGED: energy now matches LaTeX (mean/sum over POINTS, not a 3D meshgrid) ----------------
def energy_from_f3_points(X, C, agg="mean"):
    """
    Synthesize scalar target E_s from f3 on the POINTS in the configuration.

    agg:
      - "sum":  sum_i f3(x_i, y_i, z_i)
      - "mean": (1/n) sum_i f3(x_i, y_i, z_i)
    """
    vals = f3(X[:, 0], X[:, 1], X[:, 2], C=C).ravel()
    if agg == "sum":
        return float(np.sum(vals))
    if agg == "mean":
        return float(np.mean(vals))
    raise ValueError(f"Unknown agg {agg}")


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
        L = np.linalg.cholesky(G)
        Y = L.T @ A
        Q, _ = np.linalg.qr(Y)
        A_tilde = np.linalg.solve(L.T, Q)
        return A_tilde
    except np.linalg.LinAlgError:
        Q, _ = np.linalg.qr(A)
        return Q


def symmetrize_core_3d(G):
    """
    Make a 3D core supersymmetric: invariant under any permutation of axes.
    """
    return (
        G
        + np.transpose(G, (0, 2, 1))
        + np.transpose(G, (1, 0, 2))
        + np.transpose(G, (1, 2, 0))
        + np.transpose(G, (2, 0, 1))
        + np.transpose(G, (2, 1, 0))
    ) / 6.0


# ============================================================
# ACE + Tucker ALS Experiment (3D)
# ============================================================
def run_ace_tucker_experiment_3d(
    S=2000,
    n_atoms=25,  # ---------------- CHANGED: single n_atoms ----------------
    d_x=15,
    d_y=15,
    d_z=15,
    R_x=8,
    R_y=8,
    R_z=8,
    n_iter=50,
    C=np.eye(3) * 5.0,
    config_mode="random",
    energy_agg="mean",
    random_seed=42,
    outdir="ace_tucker_3d_results",
):
    """
    Same as your original function, updated to match the LaTeX:
      - configs are X^(s) with shape (n_atoms, 3)
      - E^(s) is mean/sum over points in X^(s)
      - pooled ACE features use columns of X^(s)
    """

    # --- Case 3 requires equal shapes across modes ---
    if not (d_x == d_y == d_z):
        raise ValueError("Case 3 symmetry requires d_x == d_y == d_z (same polynomial degree per mode).")
    if not (R_x == R_y == R_z):
        raise ValueError("Case 3 symmetry requires R_x == R_y == R_z (same rank per mode).")

    # Output directory handling
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"ace_tucker_3d_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # -------------------- Build ACE training data --------------------
    configs = generate_configs_3d(
        S=S,
        n_atoms=n_atoms,  # ---------------- CHANGED ----------------
        mode=config_mode,
        seed=random_seed,
    )

    # ---------------- CHANGED: pooled features from columns of X ----------------
    Ax_sum = np.vstack([ace_sum_features(X[:, 0], d_x) for X in configs])  # (S, d_x+1)
    Ay_sum = np.vstack([ace_sum_features(X[:, 1], d_y) for X in configs])  # (S, d_y+1)
    Az_sum = np.vstack([ace_sum_features(X[:, 2], d_z) for X in configs])  # (S, d_z+1)

    # ---------------- CHANGED: energy from points (no meshgrid) ----------------
    E = np.array([energy_from_f3_points(X, C=C, agg=energy_agg) for X in configs])  # (S,)

    # Gram matrices in ACE feature space
    Gx = Ax_sum.T @ Ax_sum
    Gy = Ay_sum.T @ Ay_sum
    Gz = Az_sum.T @ Az_sum

    print("\n=== Initial Gram condition numbers (ACE feature space, 3D) ===")
    print("cond(Ax_sum^T Ax_sum) =", np.linalg.cond(Gx))
    print("cond(Ay_sum^T Ay_sum) =", np.linalg.cond(Gy))
    print("cond(Az_sum^T Az_sum) =", np.linalg.cond(Gz))

    # -------------------- Initialization --------------------
    A = rng.standard_normal((d_x + 1, R_x))
    B = rng.standard_normal((d_y + 1, R_y))
    D = rng.standard_normal((d_z + 1, R_z))

    A = orthonormalize_in_metric(A, Gx)
    B = orthonormalize_in_metric(B, Gy)
    D = orthonormalize_in_metric(D, Gz)

    print("\n=== Initial metric-orthonormality check ===")
    print("cond(A^T Gx A) =", np.linalg.cond(A.T @ Gx @ A))
    print("cond(B^T Gy B) =", np.linalg.cond(B.T @ Gy @ B))
    print("cond(D^T Gz D) =", np.linalg.cond(D.T @ Gz @ D))

    # Initialize core by LS on reduced features
    X0 = Ax_sum @ A  # (S, R_x)
    Y0 = Ay_sum @ B  # (S, R_y)
    Z0 = Az_sum @ D  # (S, R_z)

    D0 = np.einsum("sp,sq,sr->spqr", X0, Y0, Z0).reshape(S, R_x * R_y * R_z)
    G_vec, *_ = np.linalg.lstsq(D0, E, rcond=1e-12)
    Gcore = G_vec.reshape((R_x, R_y, R_z), order="F")

    # Initial Case 3 projection (fine to keep)
    U0 = (A + B + D) / 3.0
    Gavg0 = (Gx + Gy + Gz) / 3.0
    U0 = orthonormalize_in_metric(U0, Gavg0)
    A = U0
    B = U0
    D = U0
    Gcore = symmetrize_core_3d(Gcore)

    # -------------------- ALS Loop --------------------
    history = []
    for it in range(n_iter):
        X_feat = Ax_sum @ A
        Y_feat = Ay_sum @ B
        Z_feat = Az_sum @ D

        E_hat = np.einsum("sp,sq,sr,pqr->s", X_feat, Y_feat, Z_feat, Gcore)
        rmse = np.sqrt(np.mean((E - E_hat) ** 2))

        print(f"\nIter {it+1}")
        print(f"RMSE (before updates): {rmse:.3e}")
        history.append({"iter": it + 1, "rmse": float(rmse)})

        if np.isnan(rmse) or rmse > 1e12:
            print("Stopping early due to divergence.")
            break

        # -------------------- Update A and core (mode-1) --------------------
        Y_feat = Ay_sum @ B
        Z_feat = Az_sum @ D

        D_A = np.einsum("sr,sq,sk->skqr", Z_feat, Y_feat, Ax_sum).reshape(
            S, (d_x + 1) * R_y * R_z, order="F"
        )

        try:
            tildeG_vec, *_ = np.linalg.lstsq(D_A, E, rcond=1e-12)
        except np.linalg.LinAlgError:
            print("Stopping early due to LinAlgError in lstsq (A update).")
            break

        tildeG = tildeG_vec.reshape((d_x + 1, R_y, R_z), order="F")
        tildeG_mat = tildeG.reshape((d_x + 1, R_y * R_z), order="F")

        U_svd, Svals, Vt = np.linalg.svd(tildeG_mat, full_matrices=False)
        A = U_svd[:, :R_x]

        A_orth = orthonormalize_in_metric(A, Gx)
        core_unfold1 = A_orth.T @ Gx @ tildeG_mat
        A = A_orth
        Gcore = core_unfold1.reshape((R_x, R_y, R_z), order="F")

        print("cond(A^T Gx A) after A update =", np.linalg.cond(A.T @ Gx @ A))

        # -------------------- Update B and core (mode-2) --------------------
        X_feat = Ax_sum @ A
        Z_feat = Az_sum @ D

        D_B = np.einsum("sr,sp,sl->slpr", Z_feat, X_feat, Ay_sum).reshape(
            S, (d_y + 1) * R_x * R_z, order="F"
        )

        try:
            tildeH_vec, *_ = np.linalg.lstsq(D_B, E, rcond=1e-12)
        except np.linalg.LinAlgError:
            print("Stopping early due to LinAlgError in lstsq (B update).")
            break

        tildeH = tildeH_vec.reshape((d_y + 1, R_x, R_z), order="F")
        tildeH_mat = tildeH.reshape((d_y + 1, R_x * R_z), order="F")

        U_svd, Svals, Vt = np.linalg.svd(tildeH_mat, full_matrices=False)
        B = U_svd[:, :R_y]

        B_orth = orthonormalize_in_metric(B, Gy)
        core_unfold2 = B_orth.T @ Gy @ tildeH_mat
        B = B_orth

        core_tmp = core_unfold2.reshape((R_y, R_x, R_z), order="F")
        Gcore = np.transpose(core_tmp, (1, 0, 2))

        print("cond(B^T Gy B) after B update =", np.linalg.cond(B.T @ Gy @ B))

        # -------------------- Update D and core (mode-3) --------------------
        X_feat = Ax_sum @ A
        Y_feat = Ay_sum @ B

        D_D = np.einsum("sp,sq,sm->smpq", X_feat, Y_feat, Az_sum).reshape(
            S, (d_z + 1) * R_x * R_y, order="F"
        )

        try:
            tildeK_vec, *_ = np.linalg.lstsq(D_D, E, rcond=1e-12)
        except np.linalg.LinAlgError:
            print("Stopping early due to LinAlgError in lstsq (D update).")
            break

        tildeK = tildeK_vec.reshape((d_z + 1, R_x, R_y), order="F")
        tildeK_mat = tildeK.reshape((d_z + 1, R_x * R_y), order="F")

        U_svd, Svals, Vt = np.linalg.svd(tildeK_mat, full_matrices=False)
        D = U_svd[:, :R_z]

        D_orth = orthonormalize_in_metric(D, Gz)
        core_unfold3 = D_orth.T @ Gz @ tildeK_mat
        D = D_orth

        core_tmp = core_unfold3.reshape((R_z, R_x, R_y), order="F")
        Gcore = np.transpose(core_tmp, (1, 2, 0))

        print("cond(D^T Gz D) after D update =", np.linalg.cond(D.T @ Gz @ D))

        # =====================================================
        # THE FIX (Case 3 projection) - enforce EVERY iteration
        # =====================================================
        U = (A + B + D) / 3.0
        Gavg = (Gx + Gy + Gz) / 3.0
        U = orthonormalize_in_metric(U, Gavg)

        A = U
        B = U
        D = U

        Gcore = symmetrize_core_3d(Gcore)

    # -------------------- Final Evaluation --------------------
    X_feat = Ax_sum @ A
    Y_feat = Ay_sum @ B
    Z_feat = Az_sum @ D
    E_hat = np.einsum("sp,sq,sr,pqr->s", X_feat, Y_feat, Z_feat, Gcore)

    rmse_final = np.sqrt(np.mean((E - E_hat) ** 2))
    l2_err = np.linalg.norm(E - E_hat)

    # Learned coefficient tensor in ACE feature space
    C_coef = np.einsum("kp,lq,mr,pqr->klm", A, B, D, Gcore)

    # Reference coefficient tensor from f3(x,y,z) Chebyshev expansion
    C_true = compute_reference_coeffs_3d(d_x, d_y, d_z, C)

    # ---------------- CHANGED: with pointwise mean/sum, do NOT divide by n_atoms_x*n_atoms_y*n_atoms_z ----------------
    C_ref = C_true

    l2_coeff_error = np.linalg.norm(C_ref - C_coef)
    rel_l2_coeff_error = l2_coeff_error / np.linalg.norm(C_ref)

    metrics = {
        "final_train_rmse": float(rmse_final),
        "l2_norm_error_train": float(l2_err),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error),
        "timestamp": timestamp,
        "S": int(S),
        "n_atoms": int(n_atoms),  # ---------------- CHANGED ----------------
        "d_x": int(d_x),
        "d_y": int(d_y),
        "d_z": int(d_z),
        "R_x": int(R_x),
        "R_y": int(R_y),
        "R_z": int(R_z),
        "n_iter": int(n_iter),
        "config_mode": config_mode,
        "energy_agg": energy_agg,
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    with open(os.path.join(run_dir, "history.json"), "w") as f_out:
        json.dump(history, f_out, indent=4)

    np.savez(
        os.path.join(run_dir, "coeffs.npz"),
        A=A, B=B, D=D, Gcore=Gcore,
        C_coef=C_coef, C_ref=C_ref, C_true=C_true
    )

    config = {
        "S": S,
        "n_atoms": n_atoms,  # ---------------- CHANGED ----------------
        "d_x": d_x,
        "d_y": d_y,
        "d_z": d_z,
        "R_x": R_x,
        "R_y": R_y,
        "R_z": R_z,
        "n_iter": n_iter,
        "C": C.tolist(),
        "config_mode": config_mode,
        "energy_agg": energy_agg,
        "random_seed": random_seed,
        "timestamp": timestamp,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Plot: predicted vs true energies
    plt.figure(figsize=(7, 6))
    plt.scatter(E, E_hat, s=10)
    plt.xlabel("True E")
    plt.ylabel("Predicted E")
    plt.title(f"ACE-Tucker 3D fit (R=({R_x},{R_y},{R_z}))")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pred_vs_true.png"))
    plt.close()

    # Plot: show a central slice of C_ref and C_coef along z-mode
    mid = (d_z + 1) // 2
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(C_ref[:, :, mid], aspect="auto", cmap=cm.viridis)
    plt.colorbar()
    plt.title(f"Reference C_ref[:,:,{mid}]")

    plt.subplot(1, 2, 2)
    plt.imshow(C_coef[:, :, mid], aspect="auto", cmap=cm.viridis)
    plt.colorbar()
    plt.title(f"Learned C_coef[:,:,{mid}]")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "C_ref_vs_C_coef_slice.png"))
    plt.close()

    return metrics, {"A": A, "B": B, "D": D, "Gcore": Gcore, "C_coef": C_coef, "C_ref": C_ref}, config, run_dir


# ============================================================
# Run Rank-Sweep Example (3D)
# ============================================================
if __name__ == "__main__":
    def rotation_matrix_3d(theta_deg, axis="z"):
        theta = np.deg2rad(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        if axis == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        raise ValueError("axis must be 'x', 'y', or 'z'")

    C_matrices = [
        5.0 * np.eye(3),
        0.2 * np.eye(3),
    ]

    all_results = {}
    for idx, Cmat in enumerate(C_matrices):
        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        for R in range(1, 11):
            print(f"\n=== Running ACE-Tucker 3D ALS for C={idx}, Rank={R} ===")
            metrics, coeffs, config, savedir = run_ace_tucker_experiment_3d(
                S=3200,
                n_atoms=3,  # ---------------- CHANGED: single n_atoms ----------------
                d_x=31,
                d_y=31,
                d_z=31,
                R_x=R,
                R_y=R,
                R_z=R,
                n_iter=30,
                C=Cmat,
                config_mode="random",
                energy_agg="mean",
                random_seed=42,
                outdir="ace_tucker_3d_zulip_results",
            )
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config,
            }

    with open("ace_tucker_3d_zulip_results_noo.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll ACE-Tucker 3D runs completed.")