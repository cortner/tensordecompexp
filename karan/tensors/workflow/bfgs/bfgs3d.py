"""3D Chebyshev CP decomposition via L-BFGS-B (despite filename)."""

# lbfgsb_cp_chebyshev_3d.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import minimize
from datetime import datetime

# ============================================================
# Function to Approximate (3D)
# ============================================================
def f(x, y, z, c=5.0):
    """Radial 3D test function."""
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
        return np.concatenate([A.ravel(), B.ravel(), C.ravel()])
    else:
        return np.concatenate([A.ravel(), B.ravel(), C.ravel(), lambdas])

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
# Compute reference Chebyshev coefficient tensor
# ============================================================
def compute_reference_coeffs(d_x, d_y, d_z, c):
    nodes_x = np.cos(np.pi * np.arange(d_x + 1) / d_x)
    nodes_y = np.cos(np.pi * np.arange(d_y + 1) / d_y)
    nodes_z = np.cos(np.pi * np.arange(d_z + 1) / d_z)
    X, Y, Z = np.meshgrid(nodes_x, nodes_y, nodes_z, indexing="ij")
    F = f(X, Y, Z, c=c)

    Tx = chebvander(nodes_x, d_x)
    Ty = chebvander(nodes_y, d_y)
    Tz = chebvander(nodes_z, d_z)

    # Kronecker product Vandermonde
    A = np.kron(np.kron(Tz, Ty), Tx)  # shape: ((dx+1)(dy+1)(dz+1), (dx+1)(dy+1)(dz+1))
    F_flat = F.ravel(order="F")

    coeffs_flat, *_ = np.linalg.lstsq(A, F_flat, rcond=None)
    C_interp = coeffs_flat.reshape((d_x + 1, d_y + 1, d_z + 1), order="F")
    return C_interp


# ============================================================
# Main experiment
# ============================================================
def run_lbfgsb_experiment(
    N=32,
    M=32,
    L=32,
    d_x=15,
    d_y=15,
    d_z=15,
    R=5,
    c=5.0,
    maxiter=5000,
    train_points="chebyshev",
    test_points="uniform",
    num_test_points=1024,
    resolution=50,
    random_seed=42,
    outdir="lbfgsb_results_3d",
    use_lambda=True
):
    """
    L-BFGS-B optimization for a 3D CP model with Chebyshev polynomial factors.
    """

    # Output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, outdir)
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_root, f"lbfgsb_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # -------------------- Training grid --------------------
    x_nodes = generate_nodes(N, train_points, rng=rng)
    y_nodes = generate_nodes(M, train_points, rng=rng)
    z_nodes = generate_nodes(L, train_points, rng=rng)
    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    F = f(X, Y, Z, c=c)

    Tx = chebvander(x_nodes, d_x)
    Ty = chebvander(y_nodes, d_y)
    Tz = chebvander(z_nodes, d_z)

    # -------------------- Objective --------------------
    def loss(params):
        if use_lambda:
            A, B, C, lambdas = unpack_params(params, d_x, d_y, d_z, R, use_lambda=True)
        else:
            A, B, C = unpack_params(params, d_x, d_y, d_z, R, use_lambda=False)
            lambdas = np.ones(R)

        A_eval = Tx @ A
        B_eval = Ty @ B
        C_eval = Tz @ C

        F_hat = np.zeros_like(F)
        for r in range(R):
            F_hat += lambdas[r] * np.einsum("i,j,k->ijk", A_eval[:, r], B_eval[:, r], C_eval[:, r])
        return np.mean((F - F_hat) ** 2)

    # -------------------- Init --------------------
    A_init = rng.standard_normal((d_x + 1, R))
    B_init = rng.standard_normal((d_y + 1, R))
    C_init = rng.standard_normal((d_z + 1, R))
    if use_lambda:
        lambdas_init = np.ones(R)
        params_init = pack_params(A_init, B_init, C_init, lambdas_init)
    else:
        params_init = pack_params(A_init, B_init, C_init)

    # -------------------- Optimize --------------------
    result = minimize(
        loss,
        params_init,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": 100000, "disp": True}
    )

    # -------------------- Unpack and evaluate --------------------
    if use_lambda:
        A_opt, B_opt, C_opt, lambdas_opt = unpack_params(result.x, d_x, d_y, d_z, R, use_lambda=True)
    else:
        A_opt, B_opt, C_opt = unpack_params(result.x, d_x, d_y, d_z, R, use_lambda=False)
        lambdas_opt = np.ones(R)

    # Reconstruct coefficient tensor from CP
    C_recon = np.zeros((d_x+1, d_y+1, d_z+1))
    for r in range(R):
        C_recon += lambdas_opt[r] * np.einsum("i,j,k->ijk", A_opt[:, r], B_opt[:, r], C_opt[:, r])

    # Reference coefficient tensor
    C_interp = compute_reference_coeffs(d_x, d_y, d_z, c)

    # -------------------- Metrics --------------------
    l2_coeff_error = float(np.linalg.norm(C_interp - C_recon))
    rel_l2_coeff_error = l2_coeff_error / float(np.linalg.norm(C_interp))

    # Reconstruct values on training grid
    A_eval = Tx @ A_opt
    B_eval = Ty @ B_opt
    C_eval = Tz @ C_opt
    F_hat = np.zeros_like(F)
    for r in range(R):
        F_hat += lambdas_opt[r] * np.einsum("i,j,k->ijk", A_eval[:, r], B_eval[:, r], C_eval[:, r])

    train_rmse = float(np.sqrt(np.mean((F - F_hat) ** 2)))
    l2_norm_error_train = float(np.linalg.norm(F - F_hat))

    # Random test points
    x_test = generate_nodes(num_test_points, test_points, rng=rng)
    y_test = generate_nodes(num_test_points, test_points, rng=rng)
    z_test = generate_nodes(num_test_points, test_points, rng=rng)
    Tx_test = chebvander(x_test, d_x)
    Ty_test = chebvander(y_test, d_y)
    Tz_test = chebvander(z_test, d_z)
    F_pred_test = np.zeros(num_test_points)
    for r in range(R):
        F_pred_test += lambdas_opt[r] * (Tx_test @ A_opt[:, r]) * (Ty_test @ B_opt[:, r]) * (Tz_test @ C_opt[:, r])
    F_true_test = f(x_test, y_test, z_test, c=c)
    rmse_test_points = float(np.sqrt(np.mean((F_true_test - F_pred_test) ** 2)))
    maxe_test_points = float(np.max(np.abs(F_true_test - F_pred_test)))

    # Save results
    metrics = {
        "final_train_rmse": train_rmse,
        "l2_norm_error_train": l2_norm_error_train,
        "rmse_test_points": rmse_test_points,
        "maxe_test_points": maxe_test_points,
        "l2_coeff_error": l2_coeff_error,
        "rel_l2_coeff_error": rel_l2_coeff_error,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nit": int(result.nit),
        "optimizer_nfev": int(result.nfev),
        "final_loss_mse": float(result.fun),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    coeffs = {
        "A_coeffs": A_opt.tolist(),
        "B_coeffs": B_opt.tolist(),
        "C_coeffs": C_opt.tolist(),
        "lambdas": lambdas_opt.tolist(),
        "C_recon": C_recon.tolist(),
        "C_interp": C_interp.tolist(),
    }
    with open(os.path.join(run_dir, "coeffs.json"), "w") as f_out:
        json.dump(coeffs, f_out, indent=4)

    config = {
        "N": N, "M": M, "L": L,
        "d_x": d_x, "d_y": d_y, "d_z": d_z,
        "R": R, "c": c,
        "maxiter": maxiter,
        "train_points": train_points,
        "test_points": test_points,
        "num_test_points": num_test_points,
        "resolution": resolution,
        "random_seed": random_seed,
        "timestamp": timestamp,
        "use_lambda": use_lambda,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f_out:
        json.dump(config, f_out, indent=4)

    # -------------------- Plot slices --------------------
    mid = L // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(F[:, :, mid], extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
    axes[0].set_title("Original (z slice)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(F_hat[:, :, mid], extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
    axes[1].set_title("Reconstructed (z slice)")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "slice_plot.png"), dpi=150)
    plt.close(fig)

    return metrics, coeffs, config, run_dir


# ============================================================
# Run Example
# ============================================================
if __name__ == "__main__":
    results, coeffs, config, savedir = run_lbfgsb_experiment(
        N=32, M=32, L=32,
        d_x=15, d_y=15, d_z=15,
        R=5,
        c=5.0,
        maxiter=2000,
        train_points="chebyshev",
        test_points="uniform",
        num_test_points=512,
        resolution=50,
        random_seed=42,
        outdir="lbfgsb_results_3d",
        use_lambda=True
    )
    print(json.dumps(results, indent=2))
    print("Saved run at:", savedir)
