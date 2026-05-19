"""2D Chebyshev Tucker model (A G B^T) optimized with Adam against a reference coefficient tensor."""

# adam2d_tucker.py
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
    coords = anp.stack([x.ravel(), y.ravel()], axis=1)
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
# Pack / Unpack for Tucker
# ============================================================
def pack_params(A, B, G):
    return anp.concatenate([A.ravel(), B.ravel(), G.ravel()])


def unpack_params(params, d_x, d_y, R_x, R_y):
    A_size = (d_x + 1) * R_x
    B_size = (d_y + 1) * R_y
    G_size = R_x * R_y
    A = params[:A_size].reshape((d_x + 1, R_x))
    B = params[A_size:A_size + B_size].reshape((d_y + 1, R_y))
    G = params[A_size + B_size:A_size + B_size + G_size].reshape((R_x, R_y))
    return A, B, G


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
    A = np.kron(Ty, Tx)
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
# Main experiment (Tucker)
# ============================================================
def run_adam_experiment(
    N=64,
    M=64,
    d_x=63,
    d_y=63,
    R_x=10,
    R_y=10,
    C=np.eye(2)*5.0,
    train_points="chebyshev",
    test_points="uniform",
    num_test_points=2048,
    resolution=120,
    random_seed=42,
    outdir="adam_tucker_results",
    lr=1e-2,
    num_iters=2000,
    print_every=100,
):
    """
    Adam optimization for a 2D Tucker model with Chebyshev polynomial factors.
    F_hat = A G B^T
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"adam_tucker_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # Training grid
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

    # Loss for autograd
    def loss(params):
        A, B, G = unpack_params(params, d_x, d_y, R_x, R_y)
        A_eval = Tx_a @ A
        B_eval = Ty_a @ B
        F_hat = A_eval @ G @ B_eval.T
        return anp.mean((F_a - F_hat) ** 2)

    grad_loss = grad(loss)

    # Init
    A_init = 0.01 * rng.standard_normal((d_x + 1, R_x))
    B_init = 0.01 * rng.standard_normal((d_y + 1, R_y))
    G_init = 0.01 * rng.standard_normal((R_x, R_y))
    params = pack_params(anp.array(A_init), anp.array(B_init), anp.array(G_init))

    m = anp.zeros_like(params)
    v = anp.zeros_like(params)

    # Optimize
    history = []
    for t in range(1, num_iters + 1):
        g = grad_loss(params)
        params, m, v = adam_update(params, g, m, v, t, lr=lr)
        if (t % print_every == 0) or (t == 1):
            L = float(loss(params))
            history.append((t, L))
            print(f"Iter {t:5d}  Loss {L:.6e}")

    # Unpack and evaluate
    A_opt, B_opt, G_opt = unpack_params(params, d_x, d_y, R_x, R_y)

    # Reference coeff tensor
    C_interp = compute_reference_coeffs(d_x, d_y, C)

    # Reconstruction in coefficient space
    C_recon = np.array(A_opt) @ np.array(G_opt) @ np.array(B_opt).T

    # Evaluate reconstruction
    A_eval = Tx @ np.array(A_opt)
    B_eval = Ty @ np.array(B_opt)
    F_hat = A_eval @ np.array(G_opt) @ B_eval.T

    # Metrics
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
    F_pred_eval = A_eval_grid @ np.array(G_opt) @ B_eval_grid.T
    rmse_eval_grid = float(np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2)))

    # Test points
    x_test = generate_nodes(num_test_points, test_points, rng=rng)
    y_test = generate_nodes(num_test_points, test_points, rng=rng)
    Tx_test = chebvander(x_test, d_x)
    Ty_test = chebvander(y_test, d_y)
    F_pred_test = np.diag(Tx_test @ np.array(A_opt) @ np.array(G_opt) @ (Ty_test @ np.array(B_opt)).T)
    F_true_test = f(x_test, y_test, C=C)
    rmse_test_points = float(np.sqrt(np.mean((F_true_test - F_pred_test) ** 2)))
    maxe_test_points = float(np.max(np.abs(F_true_test - F_pred_test)))

    # Save results
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
        "G_core": np.array(G_opt).tolist(),
        "C_recon": C_recon.tolist(),
        "C_interp": C_interp.tolist(),
    }
    with open(os.path.join(run_dir, "coeffs.json"), "w") as f_out:
        json.dump(coeffs, f_out, indent=4)

    config = {
        "N": N, "M": M,
        "d_x": d_x, "d_y": d_y,
        "R_x": R_x, "R_y": R_y,
        "C": C.tolist(),
        "train_points": train_points,
        "test_points": test_points,
        "num_test_points": num_test_points,
        "resolution": resolution,
        "random_seed": random_seed,
        "timestamp": timestamp,
        "optimizer": "Adam",
        "lr": lr,
        "num_iters": num_iters,
        "print_every": print_every,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, F, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_title(f"Original f(x,y) [{train_points} grid]")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, F_hat, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_title(f"Tucker Reconstruction R_x={R_x}, R_y={R_y}")

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
        np.array([[5, 0], [0, 1]]),
        np.array([[1, 0], [0, 5]]),
        np.array([[1, 0], [0, 5]]) @ rotation_matrix(30),
        5 * np.eye(2),
        0.2 * np.eye(2),
    ]

    all_results = {}
    for idx, C in enumerate(C_matrices):
        C_key = f"C_case_{idx}"
        all_results[C_key] = {}
        for R in range(1, 21):  # Rank sweep 1 → 20
            R_x = R_y = R
            print(f"\n=== Running Tucker experiment for C={idx}, R_x={R_x}, R_y={R_y} ===")
            metrics, coeffs, config, savedir = run_adam_experiment(
                N=64,
                M=64,
                d_x=63,
                d_y=63,
                R_x=R_x,
                R_y=R_y,
                C=C,
                train_points="chebyshev",
                test_points="uniform",
                num_test_points=2048,
                resolution=120,
                random_seed=42,
                outdir="adam_tucker_results",
                lr=1e-2,
                num_iters=2000,
                print_every=200,
            )
            all_results[C_key][f"rank_{R_x}x{R_y}"] = {
                "metrics": metrics,
                "config": config
            }

    with open("adam_tucker_rank_sweep_results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=4)

    print("\nAll Tucker runs complete. Results saved to adam_tucker_rank_sweep_results.json")
