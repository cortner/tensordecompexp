"""2D Chebyshev CP decomposition via Adam; fits an anisotropic rational test function with rank sweeps, metrics, and plots."""

# adam_cp_chebyshev_2d_matrixC.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.polynomial.chebyshev import chebvander
from datetime import datetime

# Autograd
import autograd.numpy as anp
from autograd import grad

# ============================================================
# Function to Approximate (Matrix C)
# ============================================================
def f(x, y, C=None):
    """Generalized 2D test function with anisotropy via matrix C."""
    if C is None:
        C = np.eye(2) * 5.0

    coords = anp.stack([x.ravel(), y.ravel()], axis=1)  # (N*M, 2)
    trans = coords @ C.T
    vals = 1.0 / (1.0 + anp.sum(trans**2, axis=1))
    return vals.reshape(x.shape)


# ============================================================
# Node Generators
# ============================================================
def generate_nodes(num_points, mode="chebyshev", rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if mode == "chebyshev":
        N = num_points
        return np.cos(np.pi * np.arange(N) / (N - 1))
    elif mode == "uniform":
        return np.linspace(-1.0, 1.0, num_points)
    elif mode == "random":
        return rng.uniform(-1.0, 1.0, size=num_points)
    elif mode == "normal":
        x = rng.normal(loc=0.0, scale=0.5, size=num_points)
        return np.clip(x, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown mode '{mode}'")


# ============================================================
# Pack / Unpack helpers
# ============================================================
def pack_params(A, B, lambdas=None):
    if lambdas is None:
        return anp.concatenate([A.ravel(), B.ravel()])
    else:
        return anp.concatenate([A.ravel(), B.ravel(), lambdas])

def unpack_params(params, d_x, d_y, R, use_lambda=True):
    A_size = (d_x + 1) * R
    B_size = (d_y + 1) * R
    A = params[:A_size].reshape((d_x + 1, R))
    B = params[A_size:A_size + B_size].reshape((d_y + 1, R))
    if use_lambda:
        lambdas = params[A_size + B_size:A_size + B_size + R]
        return A, B, lambdas
    else:
        return A, B


# ============================================================
# Reference Chebyshev coefficient tensor
# ============================================================
def compute_reference_coeffs(d_x, d_y, C):
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    X, Y = np.meshgrid(nodes_x, nodes_y, indexing="ij")
    F = f(X, Y, C=C)

    Tx = chebvander(nodes_x, d_x)
    Ty = chebvander(nodes_y, d_y)

    A = np.kron(Ty, Tx)  # ((dx+1)(dy+1), (dx+1)(dy+1))
    F_flat = F.ravel(order="F")

    coeffs_flat, *_ = np.linalg.lstsq(A, F_flat, rcond=None)
    C_interp = coeffs_flat.reshape((d_x + 1, d_y + 1), order="F")
    return C_interp


# ============================================================
# Adam optimizer
# ============================================================
def adam_update(params, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * (grads ** 2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    params = params - lr * m_hat / (anp.sqrt(v_hat) + eps)
    return params, m, v


# ============================================================
# Main experiment
# ============================================================
def run_adam_experiment(
    N=64,
    M=64,
    d_x=63,
    d_y=63,
    R=10,
    C=np.eye(2)*5.0,
    train_points="chebyshev",
    test_points="uniform",
    num_test_points=2048,
    resolution=120,
    random_seed=42,
    outdir="adam_results",
    use_lambda=True,       # toggle scaling weights
    lr=1e-2,
    num_iters=2000,
    print_every=100,
):
    """
    Adam optimization for a 2D CP model with Chebyshev polynomial factors.

    If use_lambda=True:
      F_hat = sum_r lambda_r * A[:, r] ⊗ B[:, r]
    If use_lambda=False:
      lambdas are fixed to 1.
    """

    # Output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"adam_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # -------------------- Training grid --------------------
    x_nodes = generate_nodes(N, train_points, rng=rng)
    y_nodes = generate_nodes(M, train_points, rng=rng)
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    F = f(X, Y, C=C)
    F_a = anp.array(F)

    # Chebyshev Vandermonde
    Tx = chebvander(x_nodes, d_x)
    Ty = chebvander(y_nodes, d_y)
    Tx_a = anp.array(Tx)
    Ty_a = anp.array(Ty)

    # -------------------- Loss for autograd --------------------
    def loss(params):
        if use_lambda:
            A, B, lambdas = unpack_params(params, d_x, d_y, R, use_lambda=True)
        else:
            A, B = unpack_params(params, d_x, d_y, R, use_lambda=False)
            lambdas = anp.ones(R)

        A_eval = Tx_a @ A
        B_eval = Ty_a @ B

        F_hat = anp.zeros_like(F_a)
        for r in range(R):
            F_hat = F_hat + lambdas[r] * anp.outer(A_eval[:, r], B_eval[:, r])

        return anp.mean((F_a - F_hat) ** 2)

    grad_loss = grad(loss)

    # -------------------- Init --------------------
    A_init = 0.01 * rng.standard_normal((d_x + 1, R))
    B_init = 0.01 * rng.standard_normal((d_y + 1, R))
    if use_lambda:
        lambdas_init = np.ones(R)
        params = pack_params(anp.array(A_init), anp.array(B_init), anp.array(lambdas_init))
    else:
        params = pack_params(anp.array(A_init), anp.array(B_init))

    # Adam state
    m = anp.zeros_like(params)
    v = anp.zeros_like(params)

    # -------------------- Optimize --------------------
    history = []
    for t in range(1, num_iters + 1):
        g = grad_loss(params)
        params, m, v = adam_update(params, g, m, v, t, lr=lr)
        if (t % print_every == 0) or (t == 1):
            L = float(loss(params))
            history.append((t, L))
            print(f"Iter {t:5d}  Loss {L:.6e}")

    # -------------------- Unpack and evaluate --------------------
    if use_lambda:
        A_opt, B_opt, lambdas_opt = unpack_params(params, d_x, d_y, R, use_lambda=True)
    else:
        A_opt, B_opt = unpack_params(params, d_x, d_y, R, use_lambda=False)
        lambdas_opt = anp.ones(R)

    # Build coefficient tensor from CP
    C_recon = np.zeros((d_x+1, d_y+1))
    for r in range(R):
        C_recon += float(lambdas_opt[r]) * np.outer(np.array(A_opt)[:, r], np.array(B_opt)[:, r])

    # Reference coefficient tensor
    C_interp = compute_reference_coeffs(d_x, d_y, C)

    # -------------------- Metrics --------------------
    A_eval = Tx @ np.array(A_opt)
    B_eval = Ty @ np.array(B_opt)
    F_hat = np.zeros_like(F)
    for r in range(R):
        F_hat += float(lambdas_opt[r]) * np.outer(A_eval[:, r], B_eval[:, r])

    train_mse = np.mean((F - F_hat) ** 2)
    train_rmse = float(np.sqrt(train_mse))
    l2_norm_error_train = float(np.linalg.norm(F - F_hat))

    l2_coeff_error = float(np.linalg.norm(C_interp - C_recon))
    rel_l2_coeff_error = l2_coeff_error / float(np.linalg.norm(C_interp))

    # Dense eval grid
    x_eval = np.linspace(-1.0, 1.0, resolution)
    y_eval = np.linspace(-1.0, 1.0, resolution)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing="ij")
    F_true_eval = f(X_eval, Y_eval, C=C)
    Tx_eval = chebvander(x_eval, d_x)
    Ty_eval = chebvander(y_eval, d_y)
    A_eval_grid = Tx_eval @ np.array(A_opt)
    B_eval_grid = Ty_eval @ np.array(B_opt)
    F_pred_eval = np.zeros_like(F_true_eval)
    for r in range(R):
        F_pred_eval += float(lambdas_opt[r]) * np.outer(A_eval_grid[:, r], B_eval_grid[:, r])
    rmse_eval_grid = float(np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2)))

    # Test points
    x_test = generate_nodes(num_test_points, test_points, rng=rng)
    y_test = generate_nodes(num_test_points, test_points, rng=rng)
    Tx_test = chebvander(x_test, d_x)
    Ty_test = chebvander(y_test, d_y)
    F_pred_test = np.zeros(num_test_points)
    for r in range(R):
        F_pred_test += float(lambdas_opt[r]) * (Tx_test @ np.array(A_opt)[:, r]) * (Ty_test @ np.array(B_opt)[:, r])
    F_true_test = f(x_test, y_test, C=C)
    rmse_test_points = float(np.sqrt(np.mean((F_true_test - F_pred_test) ** 2)))
    maxe_test_points = float(np.max(np.abs(F_true_test - F_pred_test)))

    # -------------------- Save --------------------
    metrics = {
        "final_train_rmse": train_rmse,
        "l2_norm_error_train": l2_norm_error_train,
        "rmse_eval_grid": rmse_eval_grid,
        "rmse_test_points": rmse_test_points,
        "maxe_test_points": maxe_test_points,
        "l2_coeff_error": l2_coeff_error,
        "rel_l2_coeff_error": rel_l2_coeff_error,
        "final_loss_mse": float(loss(params)),
        "history": [{"iter": it, "loss": float(L)} for it, L in history],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    coeffs = {
        "A_coeffs": np.array(A_opt).tolist(),
        "B_coeffs": np.array(B_opt).tolist(),
        "lambdas": np.array(lambdas_opt).tolist(),
        "C_recon": C_recon.tolist(),
        "C_interp": C_interp.tolist(),
    }
    with open(os.path.join(run_dir, "coeffs.json"), "w") as f_out:
        json.dump(coeffs, f_out, indent=4)

    config = {
        "N": N, "M": M,
        "d_x": d_x, "d_y": d_y,
        "R": R, "C": C.tolist(),
        "train_points": train_points,
        "test_points": test_points,
        "num_test_points": num_test_points,
        "resolution": resolution,
        "random_seed": random_seed,
        "timestamp": timestamp,
        "use_lambda": use_lambda,
        "optimizer": "Adam",
        "lr": lr,
        "num_iters": num_iters,
        "print_every": print_every,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # -------------------- Plot --------------------
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, F, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_title(f"Original f(x,y)  [{train_points} train grid]")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, F_hat, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_title(f"Reconstructed  rank={R}, use_lambda={use_lambda}")

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "reconstruction.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return metrics, coeffs, config, run_dir


# ============================================================
# Run Example
# ============================================================
if __name__ == "__main__":
    def rotation_matrix(theta_deg):
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    C_matrices = [
        np.array([[5, 0], [0, 1]]),        # vertical
        np.array([[1, 0], [0, 5]]),        # horizontal
        np.array([[1, 0], [0, 5]]) @ rotation_matrix(30),  # tilted ellipse
        5 * np.eye(2),                     # steeper dome
        0.2 * np.eye(2),                   # flatter dome
    ]

    # Results dictionary to store everything
    all_results = {}

    for idx, C in enumerate(C_matrices):
        C_key = f"C_case_{idx}"
        all_results[C_key] = {}

        for R in range(1, 21):  # Rank sweep 1 → 20
            print(f"\n=== Running experiment for C={idx}, Rank={R} ===")

            metrics, coeffs, config, savedir = run_adam_experiment(
                N=64,
                M=64,
                d_x=63,
                d_y=63,
                R=R,
                C=C,
                train_points="chebyshev",
                test_points="uniform",
                num_test_points=2048,
                resolution=120,
                random_seed=42,
                outdir="adam_results",
                use_lambda=False,
                lr=1e-2,
                num_iters=2000,
                print_every=200,
            )

            # Store just the key parts you’ll want for plotting
            all_results[C_key][f"rank_{R}"] = {
                "metrics": metrics,
                "config": config
            }

    # Save everything into one JSON
    with open("adam_false_rank_sweep_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll runs complete. Results saved to adam_false_rank_sweep_results.json")