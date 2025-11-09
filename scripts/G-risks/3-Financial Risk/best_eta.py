import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import norm
from matplotlib import rcParams
from scipy.interpolate import CubicSpline

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent

# Font settings
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 12
rcParams["axes.titlesize"] = 14
rcParams["mathtext.fontset"] = "cm"

# Load matrix
df = pd.read_csv(c_folder / "M_elast.csv", sep=";", index_col=0)
M = df.to_numpy()

# Initialization
eta_prior = np.ones(M.shape[0])
I = np.eye(M.shape[0])
A = I - M

# Define functions
def compute_eta_lambda(lamb):
    ATA_lambdaI_inv = np.linalg.inv(A.T @ A + lamb * I)
    return lamb * ATA_lambdaI_inv @ eta_prior

def r_lambda(lamb):
    eta = compute_eta_lambda(lamb)
    return norm(A @ eta)

def s_lambda(lamb):
    eta = compute_eta_lambda(lamb)
    return norm(eta - eta_prior)

def l_curve(lamb):
    return np.log(r_lambda(lamb)), np.log(s_lambda(lamb))

# Discretize lambda space
lambdas = np.logspace(-6, 2, 500)
log_r = []
log_s = []

for lam in lambdas:
    lr, ls = l_curve(lam)
    log_r.append(lr)
    log_s.append(ls)

log_r = np.array(log_r)
log_s = np.array(log_s)

# Compute curvature
def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    return np.abs(dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** 1.5

curv = curvature(log_r, log_s)
idx_max = np.argmax(curv)
lambda_opt = lambdas[idx_max]

# Evaluate optimal point
eta_opt = compute_eta_lambda(lambda_opt)
r_opt = r_lambda(lambda_opt)
s_opt = s_lambda(lambda_opt)
M_eta_opt = M @ eta_opt

# Plot
plt.figure(figsize=(8, 6))
plt.plot(log_r, log_s, label="L-curve")
plt.scatter(log_r[idx_max], log_s[idx_max], color='red', zorder=5, label="Maximum curvature point")
plt.axvline(log_r[idx_max], linestyle='--', color='gray')
plt.axhline(log_s[idx_max], linestyle='--', color='gray')

# Annotations
plt.text(log_r[idx_max], log_s[idx_max]+0.15, f"$\\lambda^* = {lambda_opt:.3f}$", ha='center', color='red')
plt.text(log_r[idx_max], plt.ylim()[0]-0.2, f"$r(\\lambda^*) = {r_opt:.2f}$", ha='center')
plt.text(plt.xlim()[0]+0.1, log_s[idx_max], f"$s(\\lambda^*) = {s_opt:.2f}$", va='center')

# Title and labels
plt.title(r"Parametric L-curve: $\log \|A \eta(\lambda)\|$ vs $\log \|\eta(\lambda) - \eta_0\|$")
plt.xlabel(r"$\log \|A \eta(\lambda)\|$")
plt.ylabel(r"$\log \|\eta(\lambda) - \eta_0\|$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(c_folder / "L_curve_lambda_star.png", dpi=900)
plt.show()

# Console output
print(f"lambda* (optimal): {lambda_opt:.6f}")
print(f"‖A η(λ*)‖ = {r_opt:.6f}")
print(f"‖η(λ*) - η_prior‖ = {s_opt:.6f}")

# Write output to CSV
n = df.shape[0]
df_eta = pd.Series(eta_opt, index=df.index)
df_M_eta = pd.Series(M_eta_opt, index=df.index)

df.loc["eta(lambda*)", :] = df_eta
df.loc["M_elast * eta(lambda*)", :] = df_M_eta
df.to_csv(c_folder / "M_elast_output.csv", sep=";")
print(f"Output written to {c_folder / 'M_elast_output.csv'}")
