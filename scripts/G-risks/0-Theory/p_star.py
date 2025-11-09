# pstar_contours_bands.py
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent

def p_star(kappa, theta):
    # p*(kappa, theta) = exp( - kappa / tan(theta)^2 )
    t = np.tan(theta)
    return np.exp(-kappa / (t * t))

if __name__ == "__main__":
    # Domaine
    theta_min, theta_max = 1e-4, np.pi/2 - 1e-4
    kappa_min, kappa_max = 0.0, 3.0

    n_theta, n_kappa = 600, 400
    theta_vals = np.linspace(theta_min, theta_max, n_theta)
    kappa_vals = np.linspace(kappa_min, kappa_max, n_kappa)
    TH, KA = np.meshgrid(theta_vals, kappa_vals)
    Z = p_star(KA, TH)

    fig, ax = plt.subplots(figsize=(8, 5.6))

    # Fond + lignes de niveau
    cf = ax.contourf(TH, KA, Z, levels=60, cmap="viridis")
    levels = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    cs = ax.contour(TH, KA, Z, levels=levels, colors="k", linewidths=1.0)

    # --- Étiquettes en 3 bandes d'ordonnée, alternées 1→2→3→… ---
    # bandes en fraction de la hauteur (0=bas, 1=haut)
    bands_frac = (0.52, 0.60, 0.68)          # ajuste au besoin
    bands_y = [kappa_min + f*(kappa_max-kappa_min) for f in bands_frac]

    manual_positions = []
    for i, plev in enumerate(levels):
        y = bands_y[i % 3]                   # bande choisie pour ce niveau
        L = -np.log(plev)                    # L = -ln(p*)
        # theta sur la courbe p*=const telle que kappa = y
        theta = np.arctan(np.sqrt(y / L))
        # clamp dans le domaine, puis recalcule kappa exact (reste sur la bonne courbe)
        theta = np.clip(theta, theta_min + 0.02, theta_max - 0.02)
        y = (np.tan(theta)**2) * L * 1.0     # = -tan^2(theta)*ln(p*), donc sur le bon niveau
        manual_positions.append((theta, y))

    # Une étiquette par niveau, texte noir; taille confortable
    ax.clabel(cs, levels=levels, manual=manual_positions,
              inline=True, fontsize=14, fmt="%.3g")

    # Colorbar + axes + titre (LaTeX simple)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$p^{*}(\kappa,\theta)$")

    ax.set_xlim(0.0, np.pi/2)
    ax.set_ylim(kappa_min, kappa_max)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(r"Threshold Probability $p^{*}(\kappa,\theta)=\exp(-\kappa\,\tan^{-2}\theta)$")

    # Export HD
    plt.savefig(c_folder / "pstar_contours.png", dpi=900, bbox_inches="tight")
    plt.show()
