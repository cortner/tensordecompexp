"""Visualize 2D test function surfaces for several anisotropy matrices C."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# ============================================================
# Function Definition
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
# Define C matrices
# ============================================================
def rotation_matrix(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

C_case_1 = np.array([[5, 0], [0, 1]])  # Horizontal stretching
C_case_2 = np.array([[1, 0], [0, 5]]) @ rotation_matrix(30)  # Rotated vertical stretching

# ============================================================
# Plot Function
# ============================================================
def plot_function_case(C, case_name, outdir="function_plots", resolution=100):
    os.makedirs(outdir, exist_ok=True)
    
    # Create grid
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Evaluate function
    Z = f(X, Y, C=C)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(14, 6))
    
    # 3D Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title(f'{case_name} - 3D Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
    ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'{case_name} - Contour Plot')
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    
    # Save figure
    fname = f"{case_name.replace(' ', '_')}_function.png"
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {fname}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Plotting C_case_1 and C_case_2 functions...\n")
    
    plot_function_case(C_case_1, "C_case_1 (Horizontal Stretching)")
    plot_function_case(C_case_2, "C_case_2 (Rotated Vertical Stretching)")
    
    print("\n✅ All plots saved in function_plots/")
