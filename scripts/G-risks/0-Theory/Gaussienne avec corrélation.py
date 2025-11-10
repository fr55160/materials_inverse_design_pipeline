import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import multivariate_normal


from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# -------------------
# Paramètres
# -------------------
rng = np.random.default_rng(0)
n_points = 1000
Q1, Q2 = 1.8, 1.4        # seuils
xlim = (-5, 5)
ylim = (-5, 5)
levels = [0.0002, 0.002, 0.01, 0.02, 0.05, 0.1, 0.15, 0.3]

# -------------------
# Données
# -------------------
X_std = rng.standard_normal(n_points)
Y_std = rng.standard_normal(n_points)

Sigma = np.array([[1.0, 0.7],
                  [0.7, 1.0]])
L = np.linalg.cholesky(Sigma)
XY_corr = L @ np.vstack([X_std, Y_std])

# Grille pour les PDFs
xn, yn = np.mgrid[-4:4:.01, -4:4:.01]
pos = np.dstack((xn, yn))

# -------------------
# Figure (1 colonne x 2 lignes)
# -------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
#fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, hspace=0.3)

# --- helper: add counts in the red zones ---
def add_counts(ax, z1, z2, Q1, Q2, xlim, ylim):
    A = (z1 > Q1)              # vertical band  z1>Q1
    B = (z2 > Q2)              # horizontal band z2>Q2
    nA   = int(A.sum())        # count in z1>Q1  (includes intersection)
    nB   = int(B.sum())        # count in z2>Q2  (includes intersection)
    nAB  = int((A & B).sum())  # intersection z1>Q1 & z2>Q2
    xmin, xmax = xlim
    ymin, ymax = ylim
    # positions inside each red area (with readable white boxes)
    ax.text((Q1 + xmax)/2, ymin + 0.06*(ymax - ymin),
            rf'$n(z_1>Q_1)={nA}$',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))
    ax.text(xmin + 0.06*(xmax - xmin), (Q2 + ymax)/2,
            rf'$n(z_2>Q_2)={nB}$',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))
    ax.text((Q1 + xmax)/2, (Q2 + ymax)/2,
            rf'$n(\cap)={nAB}$',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))

def draw_panel(ax, Z1, Z2, title, contour_color):
    # Scatter
    ax.scatter(Z1, Z2, s=7, alpha=0.6, label='_nolegend_')

    # Axes & limites
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.set_aspect('equal')
    #ax.set_aspect('equal', adjustable='box')
    #ax.set_aspect('auto') 

    # Seuils Q1, Q2
    ax.axvline(Q1, color='red', lw=0.8, ls='--')
    ax.axhline(Q2, color='red', lw=0.8, ls='--')
    ax.annotate(r'$Q_1$', xy=(Q1, 0), xycoords='data',
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom')
    ax.annotate(r'$Q_2$', xy=(0, Q2), xycoords='data',
                xytext=(4, 0), textcoords='offset points',
                ha='left', va='center')

    # Zones de risque : z1>Q1 (bande verticale) et z2>Q2 (bande horizontale)
    ax.axvspan(Q1, xlim[1], ymin=0, ymax=1, facecolor='red', alpha=0.25)
    ax.axhspan(Q2, ylim[1], xmin=0, xmax=1, facecolor='red', alpha=0.25)

    # Iso-densités
    ax.set_title(title, fontsize=12)

# Panel 1 : indépendants ~ N(0, I)
rv_ind = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
Z_ind = rv_ind.pdf(pos)
cs1 = axes[0].contour(xn, yn, Z_ind, colors='tab:blue', levels=levels, linewidths=0.8)
draw_panel(axes[0], X_std, Y_std, 'A. Independent variables  ~  $\mathcal{N}(0, I)$', contour_color='tab:blue')
add_counts(axes[0], X_std, Y_std, Q1, Q2, xlim, ylim)

# Panel 2 : corrélés ~ N(0, Σ)
rv_cor = multivariate_normal(mean=[0, 0], cov=Sigma)
Z_cor = rv_cor.pdf(pos)
cs2 = axes[1].contour(xn, yn, Z_cor, colors='tab:purple', levels=levels, linewidths=0.8)
draw_panel(axes[1], XY_corr[0, :], XY_corr[1, :], r'B. Correlated variables  ~  $\mathcal{N}(0, \Sigma)$,  $\rho=0.7$', contour_color='tab:purple')
add_counts(axes[1], XY_corr[0, :], XY_corr[1, :], Q1, Q2, xlim, ylim)

# Étiquettes axes communs
axes[1].set_xlabel(r'$z_1$')
axes[0].set_ylabel(r'$z_2$')
axes[1].set_ylabel(r'$z_2$')

# -------------------
# Légende sous la figure
# -------------------
handles = [
    Patch(facecolor='red', alpha=0.25, label=r'Risk realization: $z>Q$'),
    #Patch(facecolor='red', alpha=0.25, label=r'Risk realization: $z_2>Q_2$'),
    Line2D([0], [0], color='tab:blue', lw=1, label=r'Isodensity contours $\mathcal{N}(0, I)$'),
    Line2D([0], [0], color='tab:purple', lw=1, label=r'Isodensity contours $\mathcal{N}(0, \Sigma)$')
]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.15),
           ncol=1, frameon=False)

plt.subplots_adjust(bottom=0.2, hspace=0.15)

# -------------------
# Sauvegarde 900 dpi
# -------------------
fig.savefig(c_folder / 'bivariate_risk_900dpi.png', dpi=900, bbox_inches='tight')
# (optionnel, export vectoriel net)
# fig.savefig('bivariate_risk.svg', bbox_inches='tight')

plt.show()
