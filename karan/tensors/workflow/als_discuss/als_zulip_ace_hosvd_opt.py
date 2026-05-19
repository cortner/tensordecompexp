"""3D ACE: HOEVD init on coeff tensor, then gradient descent on coefficient reconstruction loss."""

import json
import os
from datetime import datetime

import numpy as np


def U_kernel(x, y, z, M=None):
    if M is None:
        M = np.eye(3) * 5.0
    coords = np.stack([np.ravel(x), np.ravel(y), np.ravel(z)], axis=1)
    transformed = coords @ M.T
    vals = 1.0 / (1.0 + np.sum(transformed**2, axis=1))
    return vals.reshape(np.shape(x))


def chebyshev_polys(x, deg):
    T = np.zeros((deg + 1, len(x)))
    T[0] = 1.0
    if deg > 0:
        T[1] = x
    for k in range(2, deg + 1):
        T[k] = 2.0 * x * T[k - 1] - T[k - 2]
    return T


def pooled_A(X, deg):
    T = chebyshev_polys(X, deg)
    return np.sum(T, axis=1)


def compute_reference_coeffs_C(d, M):
    nodes = np.cos(np.pi * np.arange(d + 1) / d)
    Xg, Yg, Zg = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    F = U_kernel(Xg, Yg, Zg, M=M)
    T = chebyshev_polys(nodes, d)
    n = d + 1

    U1 = np.linalg.solve(T.T, F.reshape(n, -1)).reshape(n, n, n)
    U1_t = np.transpose(U1, (1, 0, 2))
    U2_t = np.linalg.solve(T.T, U1_t.reshape(n, -1)).reshape(n, n, n)
    U2 = np.transpose(U2_t, (1, 0, 2))
    U2_t2 = np.transpose(U2, (2, 0, 1))
    C_t = np.linalg.solve(T.T, U2_t2.reshape(n, -1)).reshape(n, n, n)
    C = np.transpose(C_t, (1, 2, 0))
    return C


def generate_configs(S, n_atoms=25, seed=42):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.0, 1.0, size=n_atoms) for _ in range(S)]


def reconstruct_C_from_GQ(G, Q):
    return np.einsum("abc,ia,jb,kc->ijk", G, Q, Q, Q)


def energy_from_GQ(A, G, Q):
    B = A @ Q
    return np.einsum("abc,sa,sb,sc->s", G, B, B, B)


def hoevd_symmetric_init(C, R):
    n = C.shape[0]
    C_mode1 = C.reshape(n, n * n, order="F")
    # Higher-order eigenvalue decomposition (mode-1):
    # eigenvectors of C_(1) C_(1)^T give the Tucker subspace.
    G1 = C_mode1 @ C_mode1.T
    evals, evecs = np.linalg.eigh(G1)
    order = np.argsort(evals)[::-1]
    Q0 = evecs[:, order[:R]]
    G0 = np.einsum("ijk,ia,jb,kc->abc", C, Q0, Q0, Q0)
    return G0, Q0


def coeff_loss_and_grads(C_ref, G, Q):
    C_hat = reconstruct_C_from_GQ(G, Q)
    R = C_hat - C_ref
    loss = 0.5 * float(np.sum(R**2))

    grad_G = np.einsum("ijk,ia,jb,kc->abc", R, Q, Q, Q)

    R2 = np.transpose(R, (1, 0, 2))
    R3 = np.transpose(R, (2, 0, 1))
    G2 = np.transpose(G, (1, 0, 2))
    G3 = np.transpose(G, (2, 0, 1))

    grad_Q_1 = np.einsum("ijk,aqr,jq,kr->ia", R, G, Q, Q)
    grad_Q_2 = np.einsum("ijk,aqr,jq,kr->ia", R2, G2, Q, Q)
    grad_Q_3 = np.einsum("ijk,aqr,jq,kr->ia", R3, G3, Q, Q)
    grad_Q = grad_Q_1 + grad_Q_2 + grad_Q_3

    return loss, grad_G, grad_Q


def optimize_GQ_from_hoevd(
    C_ref,
    R,
    n_steps=300,
    lr_G=3e-2,
    lr_Q=3e-3,
):
    G, Q = hoevd_symmetric_init(C_ref, R)

    best = {
        "loss": np.inf,
        "iter": 0,
        "G": G.copy(),
        "Q": Q.copy(),
    }
    history = []

    for it in range(n_steps):
        loss, grad_G, grad_Q = coeff_loss_and_grads(C_ref, G, Q)

        # Track best parameters at the point where the loss was evaluated.
        if loss < best["loss"]:
            best["loss"] = float(loss)
            best["iter"] = it + 1
            best["G"] = G.copy()
            best["Q"] = Q.copy()

        G -= lr_G * grad_G
        Q -= lr_Q * grad_Q
        Q, _ = np.linalg.qr(Q)

        if (it + 1) % 20 == 0 or it == 0:
            history.append({"iter": it + 1, "coeff_loss": float(loss)})
            print(f"  GD iter {it + 1:4d} | coeff_loss={loss:.3e}")

    return best["G"], best["Q"], history, best["iter"], best["loss"]


def run_hoevd_then_opt_experiment(
    S=2000,
    n_atoms=25,
    d=31,
    R=8,
    M=np.eye(3) * 5.0,
    random_seed=42,
    gd_steps=300,
    outdir="ace_hoevd_opt_results",
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"ace_hoevd_opt_{timestamp}_R{R}")
    os.makedirs(run_dir, exist_ok=True)

    # Strategy 1.1: full tensor C_ref from LS/interpolation
    C_ref = compute_reference_coeffs_C(d=d, M=M)

    # Data for model-loss sanity checks
    configs = generate_configs(S=S, n_atoms=n_atoms, seed=random_seed)
    A = np.vstack([pooled_A(X, d) for X in configs])
    E_ref = np.einsum("klm,sk,sl,sm->s", C_ref, A, A, A)

    # Strategy 1.2 + 1.3: HOEVD init, then optimize (G,Q)
    G_hoevd, Q_hoevd = hoevd_symmetric_init(C_ref, R)
    C_hoevd = reconstruct_C_from_GQ(G_hoevd, Q_hoevd)
    E_hoevd = energy_from_GQ(A, G_hoevd, Q_hoevd)

    l2_coeff_hoevd = np.linalg.norm(C_ref - C_hoevd)
    rel_l2_coeff_hoevd = l2_coeff_hoevd / (np.linalg.norm(C_ref) + 1e-16)
    rmse_E_hoevd = np.sqrt(np.mean((E_ref - E_hoevd) ** 2))

    print("\n=== HOEVD init metrics ===")
    print(f"coeff l2 error: {l2_coeff_hoevd:.3e}")
    print(f"coeff rel l2 error: {rel_l2_coeff_hoevd:.3e}")
    print(f"energy RMSE vs ref: {rmse_E_hoevd:.3e}")

    G_opt, Q_opt, gd_history, best_iter, best_coeff_loss = optimize_GQ_from_hoevd(
        C_ref=C_ref,
        R=R,
        n_steps=gd_steps,
    )
    C_opt = reconstruct_C_from_GQ(G_opt, Q_opt)
    E_opt = energy_from_GQ(A, G_opt, Q_opt)

    l2_coeff_opt = np.linalg.norm(C_ref - C_opt)
    rel_l2_coeff_opt = l2_coeff_opt / (np.linalg.norm(C_ref) + 1e-16)
    rmse_E_opt = np.sqrt(np.mean((E_ref - E_opt) ** 2))

    print("\n=== Optimized metrics ===")
    print(f"coeff l2 error: {l2_coeff_opt:.3e}")
    print(f"coeff rel l2 error: {rel_l2_coeff_opt:.3e}")
    print(f"energy RMSE vs ref: {rmse_E_opt:.3e}")
    print(f"best GD iteration (lowest coeff loss): {best_iter}")
    print(f"best coeff loss: {best_coeff_loss:.3e}")

    metrics = {
        "S": int(S),
        "n_atoms": int(n_atoms),
        "d": int(d),
        "R": int(R),
        "gd_steps": int(gd_steps),
        "M": M.tolist(),
        "hoevd_l2_coeff_error": float(l2_coeff_hoevd),
        "hoevd_rel_l2_coeff_error": float(rel_l2_coeff_hoevd),
        "hoevd_energy_rmse_vs_ref": float(rmse_E_hoevd),
        "opt_l2_coeff_error": float(l2_coeff_opt),
        "opt_rel_l2_coeff_error": float(rel_l2_coeff_opt),
        "opt_energy_rmse_vs_ref": float(rmse_E_opt),
        "best_gd_iteration": int(best_iter),
        "best_coeff_loss": float(best_coeff_loss),
        "timestamp": timestamp,
    }
    # Backward-compatible aliases for earlier JSON consumers.
    metrics["hosvd_l2_coeff_error"] = metrics["hoevd_l2_coeff_error"]
    metrics["hosvd_rel_l2_coeff_error"] = metrics["hoevd_rel_l2_coeff_error"]
    metrics["hosvd_energy_rmse_vs_ref"] = metrics["hoevd_energy_rmse_vs_ref"]

    with open(os.path.join(run_dir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)
    with open(os.path.join(run_dir, "gd_history.json"), "w") as f_out:
        json.dump(gd_history, f_out, indent=4)

    np.savez(
        os.path.join(run_dir, "coeffs.npz"),
        C_ref=C_ref,
        C_hoevd=C_hoevd,
        C_opt=C_opt,
        G_hoevd=G_hoevd,
        Q_hoevd=Q_hoevd,
        G_opt=G_opt,
        Q_opt=Q_opt,
    )

    return metrics, run_dir


if __name__ == "__main__":
    M_matrices = [5.0 * np.eye(3), 0.2 * np.eye(3)]
    all_results = {}

    for idx, M in enumerate(M_matrices):
        M_key = f"M_case_{idx}"
        all_results[M_key] = {}
        print(f"\n=== HOEVD+OPT run for M_case={idx} ===")
        for R in range(1, 21):
            print(f"\n-- Rank {R} --")
            metrics, savedir = run_hoevd_then_opt_experiment(
                S=1000,
                n_atoms=25,
                d=31,
                R=R,
                M=M,
                random_seed=42,
                gd_steps=5,
                outdir="ace_hoevd_opt_results",
            )
            all_results[M_key][f"rank_{R}"] = {
                "metrics": metrics,
                "savedir": savedir,
            }
            print(
                f"Rank {R} summary: "
                f"HOEVD rel_l2={metrics['hoevd_rel_l2_coeff_error']:.3e}, "
                f"OPT rel_l2={metrics['opt_rel_l2_coeff_error']:.3e}, "
                f"OPT energy_rmse={metrics['opt_energy_rmse_vs_ref']:.3e}"
            )
            print(f"savedir: {savedir}")

    with open("ace_hoevd_opt_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll HOEVD+OPT runs complete. Aggregate results saved.")
