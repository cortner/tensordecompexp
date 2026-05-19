"""ACE standard (untied) Tucker fit for separable f(X) in professor notation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import os
from datetime import datetime


# ============================================================
# 3-body kernel U(x, y, z)
# ============================================================
def U_kernel(x, y, z, M=None):
    """
    3-body scalar kernel:
        U(x, y, z) = 1 / (1 + || M [x, y, z]^T ||^2 )

    x, y, z can be arrays of the same shape.
    M is a 3x3 anisotropy matrix.
    """
    if M is None:
        M = np.eye(3) * 5.0

    coords = np.stack([np.ravel(x), np.ravel(y), np.ravel(z)], axis=1)  # (N, 3)
    transformed = coords @ M.T                                           # (N, 3)
    vals = 1.0 / (1.0 + np.sum(transformed**2, axis=1))
    return vals.reshape(np.shape(x))


# ============================================================
# Chebyshev helpers
# ============================================================
def chebyshev_polys(x, deg):
    """
    Return T_k(x) for k=0..deg as array shape (deg+1, len(x)).
    """
    T = np.zeros((deg + 1, len(x)))
    T[0] = 1.0
    if deg > 0:
        T[1] = x
    for k in range(2, deg + 1):
        T[k] = 2.0 * x * T[k - 1] - T[k - 2]
    return T


def pooled_A(X, deg):
    """
    A_k(X) = sum_i T_k(x_i),  k=0,...,deg

    X: shape (n_atoms,)
    returns: shape (deg+1,)
    """
    T = chebyshev_polys(X, deg)   # (deg+1, n_atoms)
    return np.sum(T, axis=1)      # (deg+1,)


def compute_reference_coeffs_C(d, M):
    """
    Compute the Chebyshev coefficient tensor C for the 3-body kernel U(x,y,z)
    using tensor-product Chebyshev interpolation on (d+1)^3 nodes.

    Returns:
        C of shape (d+1, d+1, d+1)
    """
    nodes = np.cos(np.pi * np.arange(d + 1) / d)

    Xg, Yg, Zg = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    F = U_kernel(Xg, Yg, Zg, M=M)  # (d+1, d+1, d+1)

    T = chebyshev_polys(nodes, d)   # (d+1, d+1)

    n = d + 1

    # Solve along mode 1
    U1 = np.linalg.solve(T.T, F.reshape(n, -1)).reshape(n, n, n)

    # Solve along mode 2
    U1_t = np.transpose(U1, (1, 0, 2))
    U2_t = np.linalg.solve(T.T, U1_t.reshape(n, -1)).reshape(n, n, n)
    U2 = np.transpose(U2_t, (1, 0, 2))

    # Solve along mode 3
    U2_t2 = np.transpose(U2, (2, 0, 1))
    C_t = np.linalg.solve(T.T, U2_t2.reshape(n, -1)).reshape(n, n, n)
    C = np.transpose(C_t, (1, 2, 0))

    return C


def evaluate_reference_coeff_error(C, d, M, n_eval=64):
    """
    Sanity-check C_ref by comparing Chebyshev reconstruction of U on a dense grid.
    Returns (rmse, max_abs_error).
    """
    x_eval = np.linspace(-1.0, 1.0, n_eval)
    Xg, Yg, Zg = np.meshgrid(x_eval, x_eval, x_eval, indexing="ij")
    U_true = U_kernel(Xg, Yg, Zg, M=M)

    Tx = chebyshev_polys(x_eval, d)  # (d+1, n_eval)
    U_pred = np.einsum("klm,ki,lj,mn->ijn", C, Tx, Tx, Tx)

    err = U_true - U_pred
    rmse = np.sqrt(np.mean(err**2))
    max_abs = np.max(np.abs(err))
    return float(rmse), float(max_abs)


# ============================================================
# Configuration generation
# ============================================================
def generate_configs(S, n_atoms=25, mode="random", seed=42):
    """
    Generate S configurations X = [x_1, ..., x_n], each X in [-1,1]^n.
    """
    rng = np.random.default_rng(seed)
    configs = []

    for _ in range(S):
        if mode == "random":
            X = rng.uniform(-1.0, 1.0, size=n_atoms)
        elif mode == "uniform":
            X = np.linspace(-1.0, 1.0, n_atoms)
        else:
            raise ValueError(f"Unknown mode {mode}")
        configs.append(X)

    return configs


# ============================================================
# Exact energy from professor's formula
# ============================================================
def energy_from_C_and_A(C, A):
    """
    Given coefficient tensor C and pooled feature vector A,
    compute

        f(X) = sum_{k1,k2,k3} C_{k1,k2,k3} A_{k1} A_{k2} A_{k3}
    """
    return float(np.einsum("klm,k,l,m->", C, A, A, A))


# ============================================================
# Orthonormalization in metric
# ============================================================
def orthonormalize_in_metric(U_factor, G):
    """
    Return U_factor_tilde with columns G-orthonormal:
        U_factor_tilde^T G U_factor_tilde = I
    """
    try:
        L = np.linalg.cholesky(G)
        Y = L.T @ U_factor
        Q, _ = np.linalg.qr(Y)
        U_factor_tilde = np.linalg.solve(L.T, Q)
        return U_factor_tilde
    except np.linalg.LinAlgError:
        Q, _ = np.linalg.qr(U_factor)
        return Q


def symmetrize_core_3d(G_core):
    """
    Make a 3D core fully symmetric under permutations of axes.
    """
    return (
        G_core
        + np.transpose(G_core, (0, 2, 1))
        + np.transpose(G_core, (1, 0, 2))
        + np.transpose(G_core, (1, 2, 0))
        + np.transpose(G_core, (2, 0, 1))
        + np.transpose(G_core, (2, 1, 0))
    ) / 6.0


# ============================================================
# ACE + standard Tucker fit for
#   f(X) = sum_{k1,k2,k3} C_{k1k2k3} A_{k1} A_{k2} A_{k3}
# with
#   C_{k1k2k3} approx sum_{p,q,r} G_{pqr} U1_{k1p} U2_{k2q} U3_{k3r}
# ============================================================
def run_ace_standard_tucker_experiment(
    S=2000,
    n_atoms=25,
    d=15,
    R=8,
    n_iter=50,
    M=np.eye(3) * 5.0,
    config_mode="random",
    random_seed=42,
    outdir="ace_std_tucker_results",
):
    """
    This matches the professor's notation:

        X = [x_1, ..., x_n]
        A_k(X) = sum_i T_k(x_i)

        f(X) = sum_{k1,k2,k3} C_{k1k2k3} A_{k1} A_{k2} A_{k3}

    We fit a standard (untied) Tucker approximation
        C approx [[G_core; U1, U2, U3]]

    so that
        f(X) approx sum_{p,q,r} G_core[p,q,r] B1_p(X) B2_q(X) B3_r(X),
    where
        B1(X) = A(X) U1, B2(X) = A(X) U2, B3(X) = A(X) U3.
    """

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"ace_std_tucker_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # --------------------------------------------------------
    # Build exact reference coefficient tensor C
    # --------------------------------------------------------
    C = compute_reference_coeffs_C(d=d, M=M)  # (d+1, d+1, d+1)
    c_ref_rmse, c_ref_max_abs = evaluate_reference_coeff_error(C, d=d, M=M, n_eval=64)
    print("\n=== C_ref interpolation sanity check ===")
    print(f"C_ref grid RMSE: {c_ref_rmse:.3e}")
    print(f"C_ref grid max abs error: {c_ref_max_abs:.3e}")

    # --------------------------------------------------------
    # Build training configurations and pooled features A
    # --------------------------------------------------------
    configs = generate_configs(
        S=S,
        n_atoms=n_atoms,
        mode=config_mode,
        seed=random_seed,
    )

    A = np.vstack([pooled_A(X, d) for X in configs])  # (S, d+1)

    # Exact energies from the professor's formula
    E = np.einsum("klm,sk,sl,sm->s", C, A, A, A)      # (S,)

    # Metric in pooled feature space
    G_A = A.T @ A

    print("\n=== Initial Gram condition number ===")
    print("cond(A^T A) =", np.linalg.cond(G_A))

    # --------------------------------------------------------
    # Initialize factors U1, U2, U3
    # --------------------------------------------------------
    U1 = orthonormalize_in_metric(rng.standard_normal((d + 1, R)), G_A)
    U2 = orthonormalize_in_metric(rng.standard_normal((d + 1, R)), G_A)
    U3 = orthonormalize_in_metric(rng.standard_normal((d + 1, R)), G_A)

    print("\n=== Initial metric-orthonormality check ===")
    print("cond(U1^T (A^T A) U1) =", np.linalg.cond(U1.T @ G_A @ U1))
    print("cond(U2^T (A^T A) U2) =", np.linalg.cond(U2.T @ G_A @ U2))
    print("cond(U3^T (A^T A) U3) =", np.linalg.cond(U3.T @ G_A @ U3))

    # --------------------------------------------------------
    # Initial reduced features and initial core
    # --------------------------------------------------------
    B1 = A @ U1
    B2 = A @ U2
    B3 = A @ U3
    D0 = np.einsum("sp,sq,sr->spqr", B1, B2, B3).reshape(S, R * R * R)
    g_vec, *_ = np.linalg.lstsq(D0, E, rcond=1e-12)
    G_core = g_vec.reshape((R, R, R), order="F")

    history = []

    # --------------------------------------------------------
    # ALS loop
    # --------------------------------------------------------
    for it in range(n_iter):
        # Current prediction
        B1 = A @ U1
        B2 = A @ U2
        B3 = A @ U3
        E_hat = np.einsum("sp,sq,sr,pqr->s", B1, B2, B3, G_core)

        rmse = np.sqrt(np.mean((E - E_hat) ** 2))
        rel_rmse = rmse / (np.sqrt(np.mean(E**2)) + 1e-16)

        print(f"\nIter {it + 1}")
        print(f"RMSE (before update): {rmse:.3e}")
        print(f"Relative RMSE: {rel_rmse:.3e}")

        history.append(
            {
                "iter": it + 1,
                "rmse": float(rmse),
                "rel_rmse": float(rel_rmse),
            }
        )

        if np.isnan(rmse) or rmse > 1e12:
            print("Stopping early due to divergence.")
            break

        # ----------------------------------------------------
        # Step 1: update G_core by least squares for fixed U1,U2,U3
        # ----------------------------------------------------
        D_core = np.einsum("sp,sq,sr->spqr", B1, B2, B3).reshape(S, R * R * R)
        g_vec, *_ = np.linalg.lstsq(D_core, E, rcond=1e-12)
        G_core = g_vec.reshape((R, R, R), order="F")

        # ----------------------------------------------------
        # Step 2: mode-1 update (U1 and core)
        # For fixed B2,B3:
        #   E_s approx sum_{k,q,r} tilde1_{k,q,r} A_{s,k} B2_{s,q} B3_{s,r}
        # ----------------------------------------------------
        D_tilde1 = np.einsum("sk,sq,sr->skqr", A, B2, B3).reshape(
            S, (d + 1) * R * R, order="F"
        )
        tilde1_vec, *_ = np.linalg.lstsq(D_tilde1, E, rcond=1e-12)
        tilde1 = tilde1_vec.reshape((d + 1, R * R), order="F")
        U1_svd, S1_svd, Vt1_svd = np.linalg.svd(tilde1, full_matrices=False)
        U1 = orthonormalize_in_metric(U1_svd[:, :R], G_A)
        core1_unfold = np.diag(S1_svd[:R]) @ Vt1_svd[:R, :]
        G_core = core1_unfold.reshape((R, R, R), order="F")
        B1 = A @ U1

        # ----------------------------------------------------
        # Step 3: mode-2 update (U2 and core)
        # For fixed B1,B3:
        #   E_s approx sum_{l,p,r} tilde2_{l,p,r} A_{s,l} B1_{s,p} B3_{s,r}
        # ----------------------------------------------------
        D_tilde2 = np.einsum("sl,sp,sr->slpr", A, B1, B3).reshape(
            S, (d + 1) * R * R, order="F"
        )
        tilde2_vec, *_ = np.linalg.lstsq(D_tilde2, E, rcond=1e-12)
        tilde2 = tilde2_vec.reshape((d + 1, R * R), order="F")
        U2_svd, S2_svd, Vt2_svd = np.linalg.svd(tilde2, full_matrices=False)
        U2 = orthonormalize_in_metric(U2_svd[:, :R], G_A)
        core2_unfold = np.diag(S2_svd[:R]) @ Vt2_svd[:R, :]   # (R, R*R)
        core2_temp = core2_unfold.reshape((R, R, R), order="F")
        G_core = np.transpose(core2_temp, (1, 0, 2))
        B2 = A @ U2

        # ----------------------------------------------------
        # Step 4: mode-3 update (U3 and core)
        # For fixed B1,B2:
        #   E_s approx sum_{m,p,q} tilde3_{m,p,q} A_{s,m} B1_{s,p} B2_{s,q}
        # ----------------------------------------------------
        D_tilde3 = np.einsum("sm,sp,sq->smpq", A, B1, B2).reshape(
            S, (d + 1) * R * R, order="F"
        )
        tilde3_vec, *_ = np.linalg.lstsq(D_tilde3, E, rcond=1e-12)
        tilde3 = tilde3_vec.reshape((d + 1, R * R), order="F")
        U3_svd, S3_svd, Vt3_svd = np.linalg.svd(tilde3, full_matrices=False)
        U3 = orthonormalize_in_metric(U3_svd[:, :R], G_A)
        core3_unfold = np.diag(S3_svd[:R]) @ Vt3_svd[:R, :]   # (R, R*R)
        core3_temp = core3_unfold.reshape((R, R, R), order="F")
        G_core = np.transpose(core3_temp, (1, 2, 0))
        B3 = A @ U3

        # Recompute core cleanly after all factor updates
        D_core = np.einsum("sp,sq,sr->spqr", B1, B2, B3).reshape(S, R * R * R)
        g_vec, *_ = np.linalg.lstsq(D_core, E, rcond=1e-12)
        G_core = g_vec.reshape((R, R, R), order="F")

        print("cond(U1^T (A^T A) U1) after update =",
              np.linalg.cond(U1.T @ G_A @ U1))
        print("cond(U2^T (A^T A) U2) after update =",
              np.linalg.cond(U2.T @ G_A @ U2))
        print("cond(U3^T (A^T A) U3) after update =",
              np.linalg.cond(U3.T @ G_A @ U3))

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------
    B1 = A @ U1
    B2 = A @ U2
    B3 = A @ U3
    E_hat = np.einsum("sp,sq,sr,pqr->s", B1, B2, B3, G_core)

    rmse_final = np.sqrt(np.mean((E - E_hat) ** 2))
    rel_rmse_final = rmse_final / (np.sqrt(np.mean(E**2)) + 1e-16)
    l2_error_train = np.linalg.norm(E - E_hat)

    C_fit = np.einsum("kp,lq,mr,pqr->klm", U1, U2, U3, G_core)

    l2_coeff_error = np.linalg.norm(C - C_fit)
    rel_l2_coeff_error = l2_coeff_error / (np.linalg.norm(C) + 1e-16)

    metrics = {
        "final_train_rmse": float(rmse_final),
        "final_train_rel_rmse": float(rel_rmse_final),
        "l2_norm_error_train": float(l2_error_train),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error),
        "timestamp": timestamp,
        "S": int(S),
        "n_atoms": int(n_atoms),
        "d": int(d),
        "R": int(R),
        "n_iter": int(n_iter),
        "config_mode": config_mode,
        "model_type": "standard_tucker",
        "c_ref_grid_rmse": float(c_ref_rmse),
        "c_ref_grid_max_abs_error": float(c_ref_max_abs),
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    with open(os.path.join(run_dir, "history.json"), "w") as f_out:
        json.dump(history, f_out, indent=4)

    np.savez(
        os.path.join(run_dir, "coeffs.npz"),
        U1=U1,
        U2=U2,
        U3=U3,
        G_core=G_core,
        C_fit=C_fit,
        C_ref=C,
    )

    config = {
        "S": S,
        "n_atoms": n_atoms,
        "d": d,
        "R": R,
        "n_iter": n_iter,
        "M": M.tolist(),
        "config_mode": config_mode,
        "random_seed": random_seed,
        "timestamp": timestamp,
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(E, E_hat, s=10)
    plt.xlabel("True f(X)")
    plt.ylabel("Predicted f(X)")
    plt.title(f"ACE standard Tucker fit (R={R})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pred_vs_true.png"))
    plt.close()

    mid = (d + 1) // 2
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(C[:, :, mid], aspect="auto", cmap=cm.viridis)
    plt.colorbar()
    plt.title(f"Reference C[:, :, {mid}]")

    plt.subplot(1, 2, 2)
    plt.imshow(C_fit[:, :, mid], aspect="auto", cmap=cm.viridis)
    plt.colorbar()
    plt.title(f"Learned C_fit[:, :, {mid}]")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "C_ref_vs_C_fit_slice.png"))
    plt.close()

    return metrics, {"U1": U1, "U2": U2, "U3": U3, "G_core": G_core, "C_fit": C_fit, "C_ref": C}, config, run_dir


# ============================================================
# Rank sweep
# ============================================================
if __name__ == "__main__":
    M_matrices = [
        5.0 * np.eye(3),
        0.2 * np.eye(3),
    ]

    all_results = {}

    for idx, M in enumerate(M_matrices):
        M_key = f"M_case_{idx}"
        all_results[M_key] = {}

        for R in range(1, 21):
            print(f"\n=== Running ACE standard Tucker for M={idx}, Rank={R} ===")

            metrics, coeffs, config, savedir = run_ace_standard_tucker_experiment(
                S=2000,
                n_atoms=25,
                d=63,
                R=R,
                n_iter=30,
                M=M,
                config_mode="random",
                random_seed=42,
                outdir="ace_std_tucker_prof_naming_results",
            )

            all_results[M_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config,
            }
            print(
                f"Rank {R} summary: "
                f"final_train_rmse={metrics['final_train_rmse']:.3e}, "
                f"l2_coeff_error={metrics['l2_coeff_error']:.3e}, "
                f"rel_l2_coeff_error={metrics['rel_l2_coeff_error']:.3e}"
            )

    with open("ace_std_tucker_prof_naming_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll ACE standard Tucker runs completed.")