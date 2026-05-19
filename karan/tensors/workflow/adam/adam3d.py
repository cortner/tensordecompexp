"""3D Chebyshev CP decomposition via Adam on a radial 1/(1+c^2 r^2) test function."""

# adam_cp_chebyshev_3d.py
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
# Function to Approximate
# ============================================================
def f(x, y, z, c=5.0):
    """Radial test function (3D)."""
    return 1.0 / (1.0 + (c**2) * (x**2 + y**2 + z**2))


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
def pack_params(A, B, C, lambdas=None):
    if lambdas is None:
        return anp.concatenate([A.ravel(), B.ravel(), C.ravel()])
    else:
        return anp.concatenate([A.ravel(), B.ravel(), C.ravel(), lambdas])

def unpack_params(params, d_x, d_y, d_z, R, use_lambda=True):
    A_size = (d_x + 1) * R
    B_size = (d_y + 1) * R
    C_size = (d_z + 1) * R
    A = params[:A_size].reshape((d_x + 1, R))
    B = params[A_size:A_size + B_size].reshape((d_y + 1, R))
    C = params[A_size + B_size:A_size + B_size + C_size].reshape((d_z + 1, R))
    if use_lambda:
        lambdas = params[A_size + B_size + C_size:A_size + B_size + C_size + R]
        return A, B, C, lambdas
    else:
        return A, B, C


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
def run_adam_experiment_3d(
    N=32, M=32, K=32,
    d_x=15, d_y=15, d_z=15,
    R=5,
    c=5.0,
    train_points="chebyshev",
    test_points="uniform",
    num_test_points=512,
    resolution=40,
    random_seed=42,
    outdir="adam_results_3d",
    use_lambda=True,
    lr=1e-2,
    num_iters=2000,
    print_every=100,
):
    """
    Adam optimization for a 3D CP model with Chebyshev polynomial factors.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"adam3d_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # Training grid
    x_nodes = generate_nodes(N, train_points, rng=rng)
    y_nodes = generate_nodes(M, train_points, rng=rng)
    z_nodes = generate_nodes(K, train_points, rng=rng)
    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    F = f(X, Y, Z, c=c)
    F_a = anp.array(F)

    # Chebyshev Vandermonde
    Tx = chebvander(x_nodes, d_x)
    Ty = chebvander(y_nodes, d_y)
    Tz = chebvander(z_nodes, d_z)
    Tx_a, Ty_a, Tz_a = anp.array(Tx), anp.array(Ty), anp.array(Tz)

    # Loss
    def loss(params):
        if use_lambda:
            A, B, C, lambdas = unpack_params(params, d_x, d_y, d_z, R, use_lambda=True)
        else:
            A, B, C = unpack_params(params, d_x, d_y, d_z, R, use_lambda=False)
            lambdas = anp.ones(R)

        A_eval = Tx_a @ A  # (N,R)
        B_eval = Ty_a @ B  # (M,R)
        C_eval = Tz_a @ C  # (K,R)

        F_hat = anp.zeros_like(F_a)
        for r in range(R):
            F_hat = F_hat + lambdas[r] * anp.einsum("i,j,k->ijk", A_eval[:, r], B_eval[:, r], C_eval[:, r])
        return anp.mean((F_a - F_hat) ** 2)

    grad_loss = grad(loss)

    # Init
    A_init = 0.01 * rng.standard_normal((d_x + 1, R))
    B_init = 0.01 * rng.standard_normal((d_y + 1, R))
    C_init = 0.01 * rng.standard_normal((d_z + 1, R))
    if use_lambda:
        lambdas_init = np.ones(R)
        params = pack_params(anp.array(A_init), anp.array(B_init), anp.array(C_init), anp.array(lambdas_init))
    else:
        params = pack_params(anp.array(A_init), anp.array(B_init), anp.array(C_init))

    m = anp.zeros_like(params)
    v = anp.zeros_like(params)

    # Optimize
    history = []
    for t in range(1, num_iters + 1):
        g = grad_loss(params)
        params, m, v = adam_update(params, g, m, v, t, lr=lr)
        if t % print_every == 0 or t == 1:
            L = float(loss(params))
            history.append((t, L))
            print(f"Iter {t:5d}  Loss {L:.6e}")

    # Unpack final
    if use_lambda:
        A_opt, B_opt, C_opt, lambdas_opt = unpack_params(params, d_x, d_y, d_z, R, use_lambda=True)
    else:
        A_opt, B_opt, C_opt = unpack_params(params, d_x, d_y, d_z, R, use_lambda=False)
        lambdas_opt = anp.ones(R)

    # Reconstruct
    A_eval = Tx @ np.array(A_opt)
    B_eval = Ty @ np.array(B_opt)
    C_eval = Tz @ np.array(C_opt)
    F_hat = np.zeros_like(F)
    for r in range(R):
        F_hat += float(lambdas_opt[r]) * np.einsum("i,j,k->ijk", A_eval[:, r], B_eval[:, r], C_eval[:, r])

    train_rmse = float(np.sqrt(np.mean((F - F_hat) ** 2)))
    l2_norm_error_train = float(np.linalg.norm(F - F_hat))

    # Metrics (dense eval grid)
    x_eval = np.linspace(-1, 1, resolution)
    y_eval = np.linspace(-1, 1, resolution)
    z_eval = np.linspace(-1, 1, resolution)
    X_eval, Y_eval, Z_eval = np.meshgrid(x_eval, y_eval, z_eval, indexing="ij")
    F_true_eval = f(X_eval, Y_eval, Z_eval, c=c)
    Tx_eval, Ty_eval, Tz_eval = chebvander(x_eval, d_x), chebvander(y_eval, d_y), chebvander(z_eval, d_z)
    A_eval_g, B_eval_g, C_eval_g = Tx_eval @ np.array(A_opt), Ty_eval @ np.array(B_opt), Tz_eval @ np.array(C_opt)
    F_pred_eval = np.zeros_like(F_true_eval)
    for r in range(R):
        F_pred_eval += float(lambdas_opt[r]) * np.einsum("i,j,k->ijk", A_eval_g[:, r], B_eval_g[:, r], C_eval_g[:, r])
    rmse_eval_grid = float(np.sqrt(np.mean((F_true_eval - F_pred_eval) ** 2)))

    # Save metrics
    metrics = {
        "final_train_rmse": train_rmse,
        "l2_norm_error_train": l2_norm_error_train,
        "rmse_eval_grid": rmse_eval_grid,
        "final_loss_mse": float(loss(params)),
        "history": [{"iter": it, "loss": float(L)} for it, L in history],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    # Save coeffs
    coeffs = {
        "A": np.array(A_opt).tolist(),
        "B": np.array(B_opt).tolist(),
        "C": np.array(C_opt).tolist(),
        "lambdas": np.array(lambdas_opt).tolist(),
    }
    with open(os.path.join(run_dir, "coeffs.json"), "w") as f_out:
        json.dump(coeffs, f_out, indent=4)

    config = {
        "N": N, "M": M, "K": K,
        "d_x": d_x, "d_y": d_y, "d_z": d_z,
        "R": R, "c": c,
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
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # Visualization (show z=mid slice)
    mid = K // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(F[:, :, mid], extent=[-1,1,-1,1], origin="lower", cmap="viridis")
    axes[0].set_title("Original (z mid-slice)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(F_hat[:, :, mid], extent=[-1,1,-1,1], origin="lower", cmap="viridis")
    axes[1].set_title("Reconstruction (z mid-slice)")
    fig.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "slice_comparison.png"), dpi=150)
    plt.close(fig)

    return metrics, coeffs, config, run_dir


if __name__ == "__main__":
    results, coeffs, config, savedir = run_adam_experiment_3d(
        N=32, M=32, K=32,
        d_x=15, d_y=15, d_z=15,
        R=5,
        c=5.0,
        train_points="chebyshev",
        test_points="uniform",
        num_test_points=512,
        resolution=40,
        random_seed=42,
        outdir="adam_results_3d",
        use_lambda=True,
        lr=1e-2,
        num_iters=1000,
        print_every=100,
    )
    print(json.dumps(results, indent=2))
    print("Saved run at:", savedir)
