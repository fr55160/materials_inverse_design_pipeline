# properties_by_cluster_ward.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import os

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

IN_CSV = c_folder / "Cluster_characterization.csv"
OUT_PNG = c_folder / "cluster_properties_small_multiples.png"

mean_cols = [
    "Melting_Temperature_mean","LMP_mean","Log_10(K_A)_mean",
    "Density_mean","Bulk Modulus (GPa)_mean","Shear Modulus (GPa)_mean",
    "GB ratio_mean"
]
std_cols = [
    "Melting_Temperature_std","LMP_std","Log_10(K_A)_std",
    "Density_std","Bulk Modulus (GPa)_std","Shear Modulus (GPa)_std",
    "GB ratio_std"
]
pretty_labels = {
    "Melting_Temperature":"Melting Point (K)",
    "LMP":"Creep (Larson–Miller parameter)",
    "Log_10(K_A)":r"High-T Oxidation (log $K_A$)",
    "Density":"Density",
    "Bulk Modulus (GPa)":"Bulk Modulus (GPa)",
    "Shear Modulus (GPa)":"Shear Modulus (GPa)",
    "GB ratio":"G/B ratio (Pugh)",
}

# Couleurs figées pour clusters 1..4 (si les labels sont 0..3, elles seront assignées par ordre)
palette_base = {1:"#1f77b4", 2:"#ff7f0e", 3:"#2ca02c", 4:"#d62728"}
mpl_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

df = pd.read_csv(IN_CSV, sep=";")
if "Cluster" not in df.columns: raise ValueError("Colonne 'Cluster' absente.")

missing = set(mean_cols+std_cols) - set(df.columns)
if missing: raise ValueError(f"Colonnes manquantes: {missing}")

df["Cluster"] = df["Cluster"].astype(int)
df2 = df.set_index("Cluster")

prop_names = [c.replace("_mean","").replace("_std","") for c in mean_cols]
means = df2[mean_cols].rename(columns=dict(zip(mean_cols, prop_names)))
stds  = df2[std_cols].rename(columns=dict(zip(std_cols,  prop_names)))
clusters = sorted(means.index.tolist())

# --- Colonnes de composition (tout ce qui est *_mean mais pas une propriété) ---
all_mean_cols = [c for c in df2.columns if c.endswith("_mean")]
comp_mean_cols = [c for c in all_mean_cols if c.replace("_mean","") not in prop_names]
comp_means = df2[comp_mean_cols].copy()
# Déterminer si compositions sont en % (~100) ou en fraction (~1)
row_sum = comp_means.iloc[0].sum() if len(comp_means) else 0
thr = 5.0 if row_sum > 2 else 0.05  # seuil >5% (ou >0.05)

# --- Légendes chimiques par cluster : éléments centraux >5% ---
def centroid_signature(row: pd.Series, threshold=thr, fallback_top=3):
    # row: série des colonnes *_mean pour un cluster
    vals = row.rename(index=lambda s: s.replace("_mean",""))
    # garder uniquement les éléments (valeurs numériques)
    vals = vals.astype(float)
    sel = vals[vals >= threshold]
    if sel.empty:
        sel = vals.sort_values(ascending=False).head(fallback_top)
    elems = sel.sort_values(ascending=False).index.tolist()
    # joindre avec un tiret demi-cadratin
    return " ~ " + "–".join(elems) if elems else ""

# Couleurs fixées (1..4) puis cycle pour le reste, en respectant l’ordre des clusters présents
cluster_colors = {}
for i, c in enumerate(clusters, start=1):
    if c in palette_base:
        cluster_colors[c] = palette_base[c]
    else:
        cluster_colors[c] = mpl_cycle[(i-1) % max(1, len(mpl_cycle))]

mpl.rcParams['font.family'] = 'Arial'

def plot_small_multiple_bars(
    means: pd.DataFrame,
    stds: pd.DataFrame,
    save_path: str,
    prop_labels: dict,
    cluster_colors: dict
):
    clusters = sorted(means.index)
    n_props  = len(means.columns)
    total_panels = n_props + 1  # +1 pour la légende
    n_cols = 3
    n_rows = int(np.ceil(total_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4*n_cols, 4.5*n_rows),
        constrained_layout=True
    )
    axes = np.atleast_1d(axes).flatten()

    # Panneaux propriétés
    for idx, prop in enumerate(means.columns):
        ax = axes[idx]
        for cl in clusters:
            m = means.at[cl, prop]; s = stds.at[cl, prop]
            ax.bar(cl, m, width=0.8, align="edge",
                   color=cluster_colors[cl], edgecolor="black", linewidth=0.6,
                   yerr=s, capsize=5, error_kw={"elinewidth":1})
        ax.set_xticks(clusters)
        ax.set_xticklabels([str(c) for c in clusters], fontsize=14, fontfamily="Arial")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(14); lbl.set_fontfamily("Arial")
        ax.set_title(prop_labels.get(prop, prop), fontsize=16, fontfamily="Arial")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    # Panneau légende + signatures chimiques
    legend_ax = axes[n_props]; legend_ax.axis("off")

    # Handles couleur
    handles = [Line2D([0],[0], marker='s', linestyle='none', markersize=8,
                      markerfacecolor=cluster_colors[c], markeredgecolor='black')
               for c in clusters]
    # Lignes de légende : "Cluster i : ~ El1–El2–..."
    chem_lines = []
    for c in clusters:
        # extraire la rangée *_mean pour ce cluster
        row = comp_means.loc[c] if c in comp_means.index else pd.Series(dtype=float)
        chem = centroid_signature(row)
        chem_lines.append(f"Cluster {c}:{chem}")

    labels = chem_lines + [r"Error bars = ±σ"]
    handles.append(Line2D([0],[0], color='black', lw=1, marker='_', markersize=10))

    legend_ax.legend(
        handles=handles, labels=labels,
        title="Legend (centroid elements ≥5%)",
        loc="center", frameon=False,
        fontsize=14, title_fontsize=16, handlelength=1.5, handletextpad=1.0
    )

    # Supprimer axes inutilisés
    for ax in axes[total_panels:]:
        ax.remove()

    fig.suptitle("Physical Characterization of Clusters (Mean Values & Standard Deviation)",
                 fontsize=18, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {save_path}")

plot_small_multiple_bars(means, stds, OUT_PNG, pretty_labels, cluster_colors)
