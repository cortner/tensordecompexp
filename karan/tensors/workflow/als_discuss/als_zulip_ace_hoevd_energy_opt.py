"""3D ACE: HOEVD init on coeff tensor, then gradient descent on energy prediction loss."""

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
    E_hat = np.einsum("abc,sa,sb,sc->s", G, B, B, B)
    return E_hat, B


def hoevd_symmetric_init(C, R):
    n = C.shape[0]
    C_mode1 = C.reshape(n, n * n, order="F")
    G1 = C_mode1 @ C_mode1.T
    evals, evecs = np.linalg.eigh(G1)
    order = np.argsort(evals)[::-1]
    Q0 = evecs[:, order[:R]]
    G0 = np.einsum("ijk,ia,jb,kc->abc", C, Q0, Q0, Q0)
    return G0, Q0


def energy_loss_and_grads(A, E_ref, G, Q):
    S = A.shape[0]
    E_hat, B = energy_from_GQ(A, G, Q)
    delta = E_hat - E_ref
    loss = 0.5 * float(np.mean(delta**2))

    # dL/dG
    grad_G = np.einsum("s,sa,sb,sc->abc", delta / S, B, B, B)

    # dL/dB from cubic contraction
    grad_B = np.einsum("s,abc,sb,sc->sa", delta / S, G, B, B)
    grad_B += np.einsum("s,bac,sb,sc->sa", delta / S, G, B, B)
    grad_B += np.einsum("s,bca,sb,sc->sa", delta / S, G, B, B)

    # B = A Q
    grad_Q_from_B = A.T @ grad_B

    # Q also appears inside B in 3 slots; symmetric tied factor.
    # grad_Q_from_B already includes all slots via grad_B expression above.
    grad_Q = grad_Q_from_B
    return loss, grad_G, grad_Q, E_hat


def optimize_GQ_on_energy(
    A,
    E_ref,
    G0,
    Q0,
    n_steps=300,
    lr_G=1e-6,
    lr_Q=1e-8,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    clip_norm=1e3,
):
    G = G0.copy()
    Q = Q0.copy()

    best = {
        "loss": np.inf,
        "iter": 0,
        "G": G.copy(),
        "Q": Q.copy(),
    }
    history = []
    mG = np.zeros_like(G)
    vG = np.zeros_like(G)
    mQ = np.zeros_like(Q)
    vQ = np.zeros_like(Q)

    for it in range(n_steps):
        loss, grad_G, grad_Q, _ = energy_loss_and_grads(A, E_ref, G, Q)

        if loss < best["loss"]:
            best["loss"] = float(loss)
            best["iter"] = it + 1
            best["G"] = G.copy()
            best["Q"] = Q.copy()

        # Global gradient clipping for stability.
        gnorm_sq = float(np.sum(grad_G**2) + np.sum(grad_Q**2))
        if np.isfinite(gnorm_sq) and gnorm_sq > 0.0:
            gnorm = np.sqrt(gnorm_sq)
            if gnorm > clip_norm:
                scale = clip_norm / (gnorm + 1e-16)
                grad_G *= scale
                grad_Q *= scale

        if not np.isfinite(loss) or not np.all(np.isfinite(grad_G)) or not np.all(np.isfinite(grad_Q)):
            print(f"  Stopping early at iter {it + 1}: non-finite loss/grad.")
            break

        t = it + 1
        mG = beta1 * mG + (1.0 - beta1) * grad_G
        vG = beta2 * vG + (1.0 - beta2) * (grad_G * grad_G)
        mQ = beta1 * mQ + (1.0 - beta1) * grad_Q
        vQ = beta2 * vQ + (1.0 - beta2) * (grad_Q * grad_Q)

        mG_hat = mG / (1.0 - beta1**t)
        vG_hat = vG / (1.0 - beta2**t)
        mQ_hat = mQ / (1.0 - beta1**t)
        vQ_hat = vQ / (1.0 - beta2**t)

        G -= lr_G * mG_hat / (np.sqrt(vG_hat) + eps)
        Q -= lr_Q * mQ_hat / (np.sqrt(vQ_hat) + eps)
        Q, _ = np.linalg.qr(Q)

        if (it + 1) % 20 == 0 or it == 0:
            history.append({"iter": it + 1, "energy_loss": float(loss)})
            print(f"  GD iter {it + 1:4d} | energy_loss={loss:.3e}")

    return best["G"], best["Q"], history, best["iter"], best["loss"]


def run_hoevd_energy_opt_experiment(
    S=2000,
    n_atoms=25,
    d=31,
    R=8,
    M=np.eye(3) * 5.0,
    random_seed=42,
    gd_steps=300,
    outdir="ace_hoevd_energy_opt_results",
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"ace_hoevd_energy_opt_{timestamp}_R{R}")
    os.makedirs(run_dir, exist_ok=True)

    C_ref = compute_reference_coeffs_C(d=d, M=M)
    configs = generate_configs(S=S, n_atoms=n_atoms, seed=random_seed)
    A = np.vstack([pooled_A(X, d) for X in configs])
    E_ref = np.einsum("klm,sk,sl,sm->s", C_ref, A, A, A)

    G_hoevd, Q_hoevd = hoevd_symmetric_init(C_ref, R)
    C_hoevd = reconstruct_C_from_GQ(G_hoevd, Q_hoevd)
    E_hoevd, _ = energy_from_GQ(A, G_hoevd, Q_hoevd)

    hoevd_l2_coeff_error = np.linalg.norm(C_ref - C_hoevd)
    hoevd_rel_l2_coeff_error = hoevd_l2_coeff_error / (np.linalg.norm(C_ref) + 1e-16)
    hoevd_energy_rmse_vs_ref = np.sqrt(np.mean((E_ref - E_hoevd) ** 2))

    print("\n=== HOEVD init metrics ===")
    print(f"coeff l2 error: {hoevd_l2_coeff_error:.3e}")
    print(f"coeff rel l2 error: {hoevd_rel_l2_coeff_error:.3e}")
    print(f"energy RMSE vs ref: {hoevd_energy_rmse_vs_ref:.3e}")

    G_opt, Q_opt, gd_history, best_iter, best_energy_loss = optimize_GQ_on_energy(
        A=A,
        E_ref=E_ref,
        G0=G_hoevd,
        Q0=Q_hoevd,
        n_steps=gd_steps,
    )
    C_opt = reconstruct_C_from_GQ(G_opt, Q_opt)
    E_opt, _ = energy_from_GQ(A, G_opt, Q_opt)

    opt_l2_coeff_error = np.linalg.norm(C_ref - C_opt)
    opt_rel_l2_coeff_error = opt_l2_coeff_error / (np.linalg.norm(C_ref) + 1e-16)
    opt_energy_rmse_vs_ref = np.sqrt(np.mean((E_ref - E_opt) ** 2))

    print("\n=== Energy-optimized metrics ===")
    print(f"coeff l2 error: {opt_l2_coeff_error:.3e}")
    print(f"coeff rel l2 error: {opt_rel_l2_coeff_error:.3e}")
    print(f"energy RMSE vs ref: {opt_energy_rmse_vs_ref:.3e}")
    print(f"best GD iteration (lowest energy loss): {best_iter}")
    print(f"best energy loss: {best_energy_loss:.3e}")

    metrics = {
        "S": int(S),
        "n_atoms": int(n_atoms),
        "d": int(d),
        "R": int(R),
        "gd_steps": int(gd_steps),
        "M": M.tolist(),
        "hoevd_l2_coeff_error": float(hoevd_l2_coeff_error),
        "hoevd_rel_l2_coeff_error": float(hoevd_rel_l2_coeff_error),
        "hoevd_energy_rmse_vs_ref": float(hoevd_energy_rmse_vs_ref),
        "opt_l2_coeff_error": float(opt_l2_coeff_error),
        "opt_rel_l2_coeff_error": float(opt_rel_l2_coeff_error),
        "opt_energy_rmse_vs_ref": float(opt_energy_rmse_vs_ref),
        "best_gd_iteration": int(best_iter),
        "best_energy_loss": float(best_energy_loss),
        "timestamp": timestamp,
    }

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
        print(f"\n=== HOEVD+ENERGY-OPT run for M_case={idx} ===")
        for R in range(1, 21):
            print(f"\n-- Rank {R} --")
            metrics, savedir = run_hoevd_energy_opt_experiment(
                S=1000,
                n_atoms=25,
                d=31,
                R=R,
                M=M,
                random_seed=42,
                gd_steps=200,
                outdir="ace_hoevd_energy_opt_results",
            )
            all_results[M_key][f"rank_{R}"] = {
                "metrics": metrics,
                "savedir": savedir,
            }
            print(
                f"Rank {R} summary: "
                f"HOEVD energy_rmse={metrics['hoevd_energy_rmse_vs_ref']:.3e}, "
                f"OPT energy_rmse={metrics['opt_energy_rmse_vs_ref']:.3e}, "
                f"best_iter={metrics['best_gd_iteration']}"
            )
            print(f"savedir: {savedir}")

    with open("ace_hoevd_energy_opt_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll HOEVD+ENERGY-OPT runs complete. Aggregate results saved.")
