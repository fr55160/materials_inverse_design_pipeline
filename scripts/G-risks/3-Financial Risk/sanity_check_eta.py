import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import norm
from matplotlib import rcParams

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent

# Font settings (same style as your script)
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 12
rcParams["axes.titlesize"] = 14
rcParams["mathtext.fontset"] = "cm"

# Load matrix
df = pd.read_csv(c_folder / "M_elast.csv", sep=";", index_col=0)
M = df.to_numpy()

# Initialization
n = M.shape[0]
eta_prior = np.ones(n)
I = np.eye(n)
A = I - M

# ----- Ridge solution (closed form, unconstrained) -----
def compute_eta_lambda(lamb: float) -> np.ndarray:
    # eta(lambda) = lambda * (A^T A + lambda I)^(-1) * eta_prior
    ATA_lambdaI = A.T @ A + lamb * I
    return lamb * np.linalg.solve(ATA_lambdaI, eta_prior)

def r_lambda(lamb: float) -> float:
    eta = compute_eta_lambda(lamb)
    return norm(A @ eta)

def s_lambda(lamb: float) -> float:
    eta = compute_eta_lambda(lamb)
    return norm(eta - eta_prior)

# ----- L-curve discretization -----
lambdas = np.logspace(-6, 2, 500)
log_r = np.array([np.log(r_lambda(lam)) for lam in lambdas])
log_s = np.array([np.log(s_lambda(lam)) for lam in lambdas])

def curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    return np.abs(dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** 1.5

curv = curvature(log_r, log_s)
idx_max = int(np.argmax(curv))
lambda_star = float(lambdas[idx_max])

eta_star = compute_eta_lambda(lambda_star)

# ----- Sensitivity around lambda* via omega -----
omegas = np.linspace(0.5, 2.0, 61)  # omega in [0.5, 2], 61 points
angles_deg = []
res_rel = []
corr = []

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    # Angle in degrees (°)
    # Guard against numerical issues near +/-1
    cosang = float(np.dot(u, v) / (norm(u) * norm(v)))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

for w in omegas:
    lam = w * lambda_star
    eta_w = compute_eta_lambda(lam)
    angles_deg.append(angle_between(eta_star, eta_w))
    res_rel.append(norm(A @ eta_w) / norm(eta_w))
    corr.append(np.corrcoef(eta_star, eta_w)[0, 1])

angles_deg = np.array(angles_deg)
res_rel = np.array(res_rel)
corr = np.array(corr)

# ----- Plot: angle vs omega -----
plt.figure(figsize=(8, 5))
plt.plot(omegas, angles_deg, marker='o', markersize=3, linewidth=1)
plt.axvline(1.0, linestyle="--", linewidth=1)
plt.xlabel(r"Scaling factor $\omega$ (with $\lambda=\omega\,\lambda^*$)")
plt.ylabel(r"Angle $\theta(\omega)$ between $\eta(\lambda^*)$ and $\eta(\omega\lambda^*)$ [degrees]")
plt.title(r"Sensitivity of $\eta(\lambda)$ around $\lambda^*$")
plt.grid(True)
plt.tight_layout()
plt.savefig(c_folder / "eta_sensitivity_angle_vs_omega.png", dpi=600)
plt.show()

# ----- Optional plot: relative residual and correlation vs omega -----
plt.figure(figsize=(8, 5))
plt.plot(omegas, res_rel, marker='o', markersize=3, linewidth=1, label=r"$\|(I-M)\eta\|/\|\eta\|$")
plt.axvline(1.0, linestyle="--", linewidth=1)
plt.xlabel(r"Scaling factor $\omega$ (with $\lambda=\omega\,\lambda^*$)")
plt.ylabel("Relative fixed-point residual (dimensionless)")
plt.title(r"Residual stability around $\lambda^*$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(c_folder / "eta_sensitivity_residual_vs_omega.png", dpi=600)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(omegas, corr, marker='o', markersize=3, linewidth=1)
plt.axvline(1.0, linestyle="--", linewidth=1)
plt.xlabel(r"Scaling factor $\omega$ (with $\lambda=\omega\,\lambda^*$)")
plt.ylabel(r"Pearson corr. between $\eta(\lambda^*)$ and $\eta(\omega\lambda^*)$ (dimensionless)")
plt.title(r"Pattern stability around $\lambda^*$")
plt.grid(True)
plt.tight_layout()
plt.savefig(c_folder / "eta_sensitivity_corr_vs_omega.png", dpi=600)
plt.show()

# Console output
print(f"lambda* (optimal): {lambda_star:.6f}")
print(f"Angle range over omega∈[0.5,2]: [{angles_deg.min():.3f}°, {angles_deg.max():.3f}°]")
print(f"Residual range: [{res_rel.min():.3f}, {res_rel.max():.3f}]")
print(f"Correlation range: [{corr.min():.5f}, {corr.max():.5f}]")
