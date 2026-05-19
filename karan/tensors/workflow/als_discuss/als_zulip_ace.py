"""ACE synthetic energy configs plus Tucker ALS on a low-rank Chebyshev coefficient tensor."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
import json
import os
from datetime import datetime


# ============================================================
# Underlying 2D function (used only to synthesize training targets)
# ============================================================
def f(x, y, C=None):
    """Generalized 2D test function with anisotropy via matrix C."""
    if C is None:
        C = np.eye(2) * 5.0
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    trans = coords @ C.T
    vals = 1.0 / (1.0 + np.sum(trans**2, axis=1))
    return vals.reshape(x.shape)


# ============================================================
# Chebyshev helpers for reference coefficient tensor
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
        T[k] = 2 * x * T[k - 1] - T[k - 2]
    return T


def compute_reference_coeffs(d_x, d_y, C):
    """
    Compute reference Chebyshev coefficient matrix C_true for f(x,y)
    on a tensor Chebyshev grid, via LS in the tensor-product basis.

    Returns: C_true of shape ((d_x+1), (d_y+1))
    """
    # Chebyshev nodes of the first kind
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)

    X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F = f(X, Y, C=C)
    F_flat = F.ravel(order="F")

    Tx = chebyshev_polys(nodes_x, d_x)  # (d_x+1, d_x+1)
    Ty = chebyshev_polys(nodes_y, d_y)  # (d_y+1, d_y+1)

    # Design matrix for vec(F) = (Ty^T ⊗ Tx^T) vec(C_true)
    A = np.kron(Ty.T, Tx.T)

    Q, R = np.linalg.qr(A)
    coeffs_flat = np.linalg.solve(R, Q.T @ F_flat)
    coeffs = coeffs_flat.reshape((d_x + 1, d_y + 1), order="F")
    return coeffs


# ============================================================
# ACE feature construction
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


def generate_configs(S, n_atoms_x=25, n_atoms_y=25, mode="random", seed=42):
    """
    Generate S configurations, each is a tuple (xs, ys).
    xs and ys are sets of points in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(S):
        if mode == "random":
            xs = rng.uniform(-1.0, 1.0, size=n_atoms_x)
            ys = rng.uniform(-1.0, 1.0, size=n_atoms_y)
        elif mode == "uniform":
            xs = np.linspace(-1.0, 1.0, n_atoms_x)
            ys = np.linspace(-1.0, 1.0, n_atoms_y)
        else:
            raise ValueError(f"Unknown mode {mode}")
        configs.append((xs, ys))
    return configs


def energy_from_f(xs, ys, C, agg="mean"):
    """
    Synthesize scalar target E_s from the underlying f on the cross product grid.

    agg:
      - "sum": sum_{i,j} f(x_i, y_j)
      - "mean": mean_{i,j} f(x_i, y_j)
    """
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    vals = f(X, Y, C=C)
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
        L = np.linalg.cholesky(G)          # G = L L^T
        Y = L.T @ A                        # (n, r)
        Q, _ = np.linalg.qr(Y)             # Q^T Q = I
        A_tilde = np.linalg.solve(L.T, Q)  # A_tilde^T G A_tilde = I
        return A_tilde
    except np.linalg.LinAlgError:
        Q, _ = np.linalg.qr(A)
        return Q


# ============================================================
# ACE + Tucker ALS Experiment
# ============================================================
def run_ace_tucker_experiment(
    S=2000,
    n_atoms_x=25,
    n_atoms_y=25,
    d_x=15,
    d_y=15,
    R_x=8,
    R_y=8,
    n_iter=50,
    C=np.eye(2) * 5.0,
    config_mode="random",
    energy_agg="mean",
    random_seed=42,
    outdir="ace_tucker_results",
):
    """
    ACE data:
      For each config s:
        a_x[s,k] = sum_i T_k(x_i^(s))
        a_y[s,l] = sum_j T_l(y_j^(s))
      Target:
        E[s] = synthesized energy

    Model:
      C_coef ≈ A G B^T
      E[s] ≈ a_x[s]^T C_coef a_y[s]
    """

    # Output directory handling
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"ace_tucker_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # -------------------- Build ACE training data --------------------
    configs = generate_configs(
        S=S,
        n_atoms_x=n_atoms_x,
        n_atoms_y=n_atoms_y,
        mode=config_mode,
        seed=random_seed,
    )

    Ax_sum = np.vstack([ace_sum_features(xs, d_x) for xs, _ in configs])  # (S, d_x+1)
    Ay_sum = np.vstack([ace_sum_features(ys, d_y) for _, ys in configs])  # (S, d_y+1)

    E = np.array([energy_from_f(xs, ys, C=C, agg=energy_agg) for xs, ys in configs])  # (S,)

    # Gram matrices in ACE feature space
    Gx = Ax_sum.T @ Ax_sum
    Gy = Ay_sum.T @ Ay_sum

    print("\n=== Initial Gram condition numbers (ACE feature space) ===")
    print("cond(Ax_sum^T Ax_sum) =", np.linalg.cond(Gx))
    print("cond(Ay_sum^T Ay_sum) =", np.linalg.cond(Gy))

    # -------------------- Initialization --------------------
    A = rng.standard_normal((d_x + 1, R_x))
    B = rng.standard_normal((d_y + 1, R_y))

    # Orthonormalize in ACE metric
    A = orthonormalize_in_metric(A, Gx)
    B = orthonormalize_in_metric(B, Gy)

    At_Gx_A = A.T @ Gx @ A
    Bt_Gy_B = B.T @ Gy @ B

    print("\n=== Initial metric-orthonormality check ===")
    print("cond(A^T Gx A) =", np.linalg.cond(At_Gx_A))
    print("cond(B^T Gy B) =", np.linalg.cond(Bt_Gy_B))

    # Initialize G by LS on reduced features
    X0 = Ax_sum @ A  # (S, R_x)
    Y0 = Ay_sum @ B  # (S, R_y)
    D0 = np.einsum("sp,sq->spq", X0, Y0).reshape(S, R_x * R_y)
    G_vec, *_ = np.linalg.lstsq(D0, E, rcond=1e-12)
    G = G_vec.reshape(R_x, R_y)

    # -------------------- ALS Loop --------------------
    history = []
    for it in range(n_iter):
        # Prediction and RMSE
        X_feat = Ax_sum @ A  # (S, R_x)
        Y_feat = Ay_sum @ B  # (S, R_y)
        E_hat = np.einsum("sp,pq,sq->s", X_feat, G, Y_feat)  # (S,)
        rmse = np.sqrt(np.mean((E - E_hat) ** 2))

        print(f"\nIter {it+1}")
        print(f"RMSE (before updates): {rmse:.3e}")

        history.append({"iter": it + 1, "rmse": float(rmse)})

        if np.isnan(rmse) or rmse > 1e12:
            print("Stopping early due to divergence.")
            break

        # =====================================================
        # Update A and G
        # E[s] ≈ a_x[s]^T tildeG (a_y[s]^T B)^T
        # =====================================================
        Y_feat = Ay_sum @ B  # (S, R_y)

        # Row s corresponds to kron(Y_feat[s], Ax_sum[s])
        D_A = np.einsum("sr,sk->skr", Y_feat, Ax_sum).reshape(
            S, (d_x + 1) * R_y, order="F"
        )

        try:
            tildeG_vec, *_ = np.linalg.lstsq(D_A, E, rcond=1e-12)
        except np.linalg.LinAlgError:
            print("Stopping early due to LinAlgError in lstsq (A update).")
            break

        tildeG = tildeG_vec.reshape((d_x + 1, R_y), order="F")

        U_A, S_A, Vt_A = np.linalg.svd(tildeG, full_matrices=False)

        U_A_r = U_A[:, :R_x]
        S_A_r = S_A[:R_x]
        Vt_A_r = Vt_A[:R_x, :]

        A = U_A_r
        G = np.diag(S_A_r) @ Vt_A_r

        At_Gx_A = A.T @ Gx @ A
        print("cond(A^T Gx A) after A update =", np.linalg.cond(At_Gx_A))

        # =====================================================
        # Update B and G
        # E[s] ≈ a_y[s]^T tildeH (a_x[s]^T A)^T
        # =====================================================
        X_feat = Ax_sum @ A  # (S, R_x)

        # Row s corresponds to kron(X_feat[s], Ay_sum[s])
        D_B = np.einsum("sr,sl->slr", X_feat, Ay_sum).reshape(
            S, (d_y + 1) * R_x, order="F"
        )

        try:
            tildeH_vec, *_ = np.linalg.lstsq(D_B, E, rcond=1e-12)
        except np.linalg.LinAlgError:
            print("Stopping early due to LinAlgError in lstsq (B update).")
            break

        tildeH = tildeH_vec.reshape((d_y + 1, R_x), order="F")

        U_B, S_B, Vt_B = np.linalg.svd(tildeH, full_matrices=False)

        U_B_r = U_B[:, :R_y]
        S_B_r = S_B[:R_y]
        Vt_B_r = Vt_B[:R_y, :]          # (R_y, R_x)
        V_B_r = Vt_B_r.T                # (R_x, R_y)

        B = U_B_r
        G = V_B_r @ np.diag(S_B_r)

        Bt_Gy_B = B.T @ Gy @ B
        print("cond(B^T Gy B) after B update =", np.linalg.cond(Bt_Gy_B))

    # -------------------- Final Evaluation --------------------
    X_feat = Ax_sum @ A
    Y_feat = Ay_sum @ B
    E_hat = np.einsum("sp,pq,sq->s", X_feat, G, Y_feat)

    rmse_final = np.sqrt(np.mean((E - E_hat) ** 2))
    l2_err = np.linalg.norm(E - E_hat)

    # Learned coefficient matrix in ACE feature space
    C_coef = A @ G @ B.T

    # Reference coefficient tensor from f(x,y) Chebyshev expansion
    C_true = compute_reference_coeffs(d_x, d_y, C)

    # Match scaling depending on how E was constructed
    if energy_agg == "mean":
        C_ref = C_true / (n_atoms_x * n_atoms_y)
    elif energy_agg == "sum":
        C_ref = C_true
    else:
        raise ValueError(f"Unknown energy_agg {energy_agg}")

    l2_coeff_error = np.linalg.norm(C_ref - C_coef)
    rel_l2_coeff_error = l2_coeff_error / np.linalg.norm(C_ref)

    metrics = {
        "final_train_rmse": float(rmse_final),
        "l2_norm_error_train": float(l2_err),
        "l2_coeff_error": float(l2_coeff_error),
        "rel_l2_coeff_error": float(rel_l2_coeff_error),
        "timestamp": timestamp,
        "S": int(S),
        "n_atoms_x": int(n_atoms_x),
        "n_atoms_y": int(n_atoms_y),
        "d_x": int(d_x),
        "d_y": int(d_y),
        "R_x": int(R_x),
        "R_y": int(R_y),
        "n_iter": int(n_iter),
        "config_mode": config_mode,
        "energy_agg": energy_agg,
    }

    # Save results
    with open(os.path.join(run_dir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    with open(os.path.join(run_dir, "history.json"), "w") as f_out:
        json.dump(history, f_out, indent=4)

    np.savez(
        os.path.join(run_dir, "coeffs.npz"),
        A=A,
        B=B,
        G=G,
        C_coef=C_coef,
        C_ref=C_ref,
        C_true=C_true,
    )

    config = {
        "S": S,
        "n_atoms_x": n_atoms_x,
        "n_atoms_y": n_atoms_y,
        "d_x": d_x,
        "d_y": d_y,
        "R_x": R_x,
        "R_y": R_y,
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
    plt.title(f"ACE-Tucker fit (R_x={R_x}, R_y={R_y})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pred_vs_true.png"))
    plt.close()

    # Plot: heatmaps of learned and reference coefficients
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(C_ref, aspect="auto", cmap=cm.viridis)
    plt.colorbar()
    plt.title("Reference C_ref")

    plt.subplot(1, 2, 2)
    plt.imshow(C_coef, aspect="auto", cmap=cm.viridis)
    plt.colorbar()
    plt.title("Learned C_coef = A G B^T")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "C_ref_vs_C_coef.png"))
    plt.close()

    return metrics, {"A": A, "B": B, "G": G, "C_coef": C_coef, "C_ref": C_ref}, config, run_dir


# ============================================================
# Run Rank-Sweep Example
# ============================================================
if __name__ == "__main__":
    def rotation_matrix(theta_deg):
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    C_matrices = [
        np.array([[5, 0], [0, 1]]),
        np.array([[1, 0], [0, 5]]),
        np.array([[1, 0], [0, 5]]) @ rotation_matrix(30),
        5 * np.eye(2),
        0.2 * np.eye(2),
    ]

    all_results = {}
    for idx, Cmat in enumerate(C_matrices):
        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        for R in range(1, 21):
            print(f"\n=== Running ACE-Tucker ALS for C={idx}, Rank={R} ===")
            metrics, coeffs, config, savedir = run_ace_tucker_experiment(
                S=2000,
                n_atoms_x=25,
                n_atoms_y=25,
                d_x=63,
                d_y=63,
                R_x=R,
                R_y=R,
                n_iter=50,
                C=Cmat,
                config_mode="random",
                energy_agg="mean",
                random_seed=42,
                outdir="ace_tucker_64_zulip_results",
            )
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config,
            }

    with open("ace_tucker_64_zulip_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll ACE-Tucker runs completed.")
