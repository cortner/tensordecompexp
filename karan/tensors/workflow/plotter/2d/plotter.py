"""Batch runner: sweep C matrices and call plotting helpers for method comparison figures."""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# Core function (generalized with matrix C)
# ============================================================
def fcn_matrix(x, C):
    """
    Evaluate f(x) = 1 / (1 + ||C x||^2)

    Parameters
    ----------
    x : ndarray, shape (n_points, 2)
        Input coordinates.
    C : ndarray, shape (2, 2)
        Transformation matrix.

    Returns
    -------
    values : ndarray, shape (n_points,)
        Function values.
    """
    x_trans = x @ C.T
    return 1 / (1 + np.sum(x_trans**2, axis=1))


# ============================================================
# Plotting helper
# ============================================================
def plot_variant(C, name, outdir="plots", resolution=200):
    """
    Generate 3D surface and 2D contour plots for a given matrix C.
    """
    os.makedirs(outdir, exist_ok=True)

    # Create grid
    xx = np.linspace(-1, 1, resolution)
    yy = np.linspace(-1, 1, resolution)
    XX, YY = np.meshgrid(xx, yy, indexing="ij")
    coords = np.stack([XX.ravel(), YY.ravel()], axis=1)

    # Evaluate function
    Z = fcn_matrix(coords, C).reshape((resolution, resolution))

    # 3D surface plot
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, Z, cmap="viridis", edgecolor="none")
    ax.set_title(f"3D Surface ({name})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x)")
    fig.savefig(os.path.join(outdir, f"{name}_3d.png"))
    plt.close(fig)

    # 2D contour plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(XX, YY, Z, levels=30, cmap="plasma")
    ax.set_title(f"Level Curves ({name})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cs, ax=ax, label="f(x)")
    fig.savefig(os.path.join(outdir, f"{name}_contour.png"))
    plt.close(fig)


# ============================================================
# Experiment runner
# ============================================================
def run_variants():
    # Define example C matrices
    variants = {
        "tilted": np.array([[1.0, 0.5],
                            [0.5, 1.0]]),
        "vertical": np.array([[0.2, 0.0],
                              [0.0, 2.0]]),
        "horizontal": np.array([[2.0, 0.0],
                                [0.0, 0.2]]),
        "constricted": np.array([[2.0, 0.0],
                                 [0.0, 2.0]]),
        "loose": np.array([[0.5, 0.0],
                           [0.0, 0.5]])
    }

    # Base "experiments" folder next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "experiments")
    os.makedirs(base_dir, exist_ok=True)

    # Timestamped subfolder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # Generate plots for each variant
    for name, C in variants.items():
        plot_variant(C, name, outdir)

    print(f"Plots saved to {outdir}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    run_variants()
