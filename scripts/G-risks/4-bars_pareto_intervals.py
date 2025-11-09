#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import matplotlib as mpl

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

mpl.rcParams.update({
    "font.family": "Arial",     # tombera sur "sans-serif" si Arial n'est pas installé
    "font.size": 10,            # taille par défaut de tout le texte
    "axes.titlesize": 10,       # titres de sous-graphiques
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 12      # suptitle (titre global)
})

# --------------------------
# Données (means, std, min, max)
# --------------------------
metrics = [
    "Bulk Modulus (GPa)",
    "Shear Modulus (GPa)",
    "Density",
    "Melting Point (K)",
    "Creep (LMP)",
    "G/B (Pugh ratio)",
    r"HT Oxidation (Log $K_{A}$)",
    "Supply Risk",
    "Energy Footprint (MJ/mol)",
]

# ---------- Best Performance Pareto Front ----------
best_mean = np.array([151.752330226365, 63.0719041278296, 7.77193963604083,
                      1884.18995561474, 31.3168974700399, 0.41462494451842,
                      0.129720372836219, 6.35007123971499, 747.791216808188])
best_std  = np.array([31.3966023652672, 16.6215159964195, 0.766599670603333,
                      221.337158433586, 0.388295458290158, 0.0545529550205499,
                      0.474604595562627, 1.53268898598411, 2365.94544602588])
best_min  = np.array([63.95, 31.71, 4.52, 1169.81, 30.17, 0.29, -1.99, 1.64328405981867, 4.10034149132])
best_max  = np.array([242.04, 111.0, 9.90, 2544.43, 32.42, 0.63, 0.42, 12.4163248575177, 38289.6589385636])

# ---------- Sustainable Pareto Front ----------
sust_mean = np.array([191.407906976744, 92.113023255814, 8.15302325581395,
                      1733.86395348837, 30.7723255813953, 0.481395348837209,
                      -0.382558139534884, 4.42824390908752, 16.8984625802614])
sust_std  = np.array([12.4928207112915, 7.23819121541251, 0.653949666345168,
                      74.3773702188001, 0.354641584388563, 0.0281646008272151,
                      0.915147901195825, 0.77533522405395, 15.9717103139125])
sust_min  = np.array([166.6, 75.1, 7.29, 1588.68, 30.24, 0.42, -1.85, 3.11216382468495, 4.011559647451])
sust_max  = np.array([212.54, 104.43, 9.01, 1879.6, 31.22, 0.53, 0.42, 5.85807080467296, 68.85103773507])

# ---------- Legacy superalloys ----------
legacy_mean = np.array([190.653022873563, 96.421891183908, 8.60770924137931,
                        1665.00465517241, 30.9194485632184, 0.506395711149425,
                        0.197625790597701, 7.55657221772227, 51.8751351371443])
legacy_std  = np.array([8.54011357190764, 4.49071063039022, 0.323928669061694,
                        73.8395976184184, 0.729450685667993, 0.0268698002946411,
                        0.649924624727222, 1.58871266060218, 33.9341253027965])
legacy_min  = np.array([174.29443, 76.467995, 7.7425995, 1492.6608, 29.876247, 0.43872884,
                        -2.1865058, 2.22826474645125, 10.8063999208865])
legacy_max  = np.array([211.85733, 108.69456, 9.280688, 1798.505, 32.361553, 0.54947597,
                        1.4501953, 10.4022487840222, 155.762856480473])

# --------------------------
# Plot params
# --------------------------
COL_BEST   = "#1f77b4"   # bleu
COL_SUST   = "#2ca02c"   # vert
COL_LEGACY = "#7f7f7f"   # gris
EDGE = "black"
ALPHA = 0.95

bar_width = 0.04/2   # barres fines
gap = 0.04   /2      # faible espacement
marker_size = 18

fig, axes = plt.subplots(3, 3, figsize=(12.5, 14.5))
axes = axes.flatten()

def plot_metric(ax, title, i, log_scale=False, ef_ylim_min=None):
    """
    Bar = [mean − σ ; mean + σ] (68% interval).
    Whiskers = min–max (asymmetric). Dot = mean.
    """
    means = np.array([best_mean[i], sust_mean[i], legacy_mean[i]])
    stds  = np.array([best_std[i],  sust_std[i],  legacy_std[i]])
    mins  = np.array([best_min[i],  sust_min[i],  legacy_min[i]])
    maxs  = np.array([best_max[i],  sust_max[i],  legacy_max[i]])

    low68  = means - stds
    high68 = means + stds
    heights = high68 - low68
    bottoms = low68.copy()

    # positions
    x_center = np.arange(1)
    x_offsets = np.array([-bar_width - gap/2, 0.0, +bar_width + gap/2])
    x_pos = x_center + x_offsets
    colors = [COL_BEST, COL_SUST, COL_LEGACY]

    # log-scale clip for EF only
    if log_scale:
        eps = 4.0 if ef_ylim_min is None else ef_ylim_min
        bottoms = np.maximum(bottoms, eps)
        heights = np.maximum(high68 - bottoms, eps*0.001)
        mins = np.maximum(mins, eps)

    for k in range(3):
        ax.bar(x_pos[k], heights[k], bottom=bottoms[k], width=bar_width,
               color=colors[k], edgecolor=EDGE, linewidth=0.8, alpha=ALPHA)
        ax.scatter([x_pos[k]], [means[k]], c=[colors[k]], edgecolors=EDGE,
                   s=marker_size, zorder=3)
        lower = means[k] - mins[k]
        upper = maxs[k] - means[k]
        ax.errorbar(x_pos[k], means[k],
                    yerr=np.array([[lower], [upper]]),
                    fmt='none', ecolor=EDGE, elinewidth=1.1, capsize=5, capthick=1.1, zorder=4)

    ax.set_title(title, fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Best", "Sustain.", "Legacy"], fontsize=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if log_scale:
        ax.set_yscale("log")
        # y min = 4 ; y max = 1.2 × max des max/upper/hauts
        ymax = max(maxs.max(), high68.max()) * 1.2
        ax.set_ylim(ef_ylim_min if ef_ylim_min else 4.0, ymax)

# Tracer tous les panneaux
for idx, m in enumerate(metrics):
    is_ef = (m == "Energy Footprint (MJ/mol)")
    plot_metric(axes[idx], m, idx, log_scale=is_ef, ef_ylim_min=4.0 if is_ef else None)

# --- Pugh band (G/B panel) ---
gb_idx = metrics.index("G/B (Pugh ratio)")
ax_gb = axes[gb_idx]
# bande rouge translucide entre 0.5 et 0.6
ax_gb.axhspan(0.5, 0.6, color='red', alpha=0.25, zorder=0)
# lignes pointillées rouges aux bornes
ax_gb.axhline(0.5, color='red', linestyle='--', linewidth=1.1, alpha=0.9)
ax_gb.axhline(0.6, color='red', linestyle='--', linewidth=1.1, alpha=0.9)
# texte unique
xmid = np.mean(ax_gb.get_xlim())
ax_gb.text(xmid, 0.55, "Pugh criterion validation (ductility & brittleness)",
           color='red', fontsize=9.5, ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))

# Marges et espace pour la légende
plt.subplots_adjust(top=0.90, bottom=0.28, left=0.08, right=0.98, hspace=0.45, wspace=0.30)

# ---------- Legend (boxed, with visual keys + NOTE inside) ----------
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ds_handles = [
    Patch(facecolor=COL_BEST,   edgecolor=EDGE, label="Best Performance Pareto Front"),
    Patch(facecolor=COL_SUST,   edgecolor=EDGE, label="Sustainable Pareto Front"),
    Patch(facecolor=COL_LEGACY, edgecolor=EDGE, label="Legacy Superalloys"),
]
band_key   = Patch(facecolor="#cccccc", edgecolor=EDGE, label=r"bars: 68% interval [$\mu\pm\sigma$]")
dot_key    = Line2D([0], [0], marker='o', color='black', markersize=8, linestyle='',
                    markerfacecolor='white', label="Mean (dot)")
whisk_key  = Line2D([0,1], [0,0], color='black', linewidth=1.2, marker='|',
                    markersize=10, markeredgewidth=1.2, label="Whiskers (min–max)")

legend_handles = ds_handles + [band_key, dot_key, whisk_key]
legend_labels  = [h.get_label() for h in legend_handles]

leg = fig.legend(legend_handles, legend_labels,
                 loc="lower center", ncol=3, fontsize=10,
                 frameon=True, fancybox=True, framealpha=0.95,
                 bbox_to_anchor=(0.5, 0.18), title=(
                    r"Bars = [$\mu\pm\sigma$]; dots = mean ($\mu$); whiskers = min–max. "
                    r"$\sigma$ denotes the standard deviation."
                 ),
                 title_fontsize=10)

fig.suptitle(r"Properties Comparison across Alloy Sets (means ($\mu$), 68% intervals [$\mu\pm\sigma$], and min–max)",
             fontsize=16, y=0.965)

# Export 900 dpi
out_path = c_folder / "Global, sustainable & legacy_Barcharts.png"
fig.savefig(out_path, dpi=900, bbox_inches="tight")
print(f"Saved: {out_path}")

plt.show()
