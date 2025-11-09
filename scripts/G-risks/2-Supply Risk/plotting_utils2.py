import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import colorsys
import numpy as np

# (vous devez toujours avoir ces deux constantes dans ce module)
_SHAPE_MAP = {
    "Brute Force HEA":        ("s", "Brute Force Generated HEA"),
    "Generative CVAE":        ("D", "Generative CVAE"),
    "Hamming & Gradient Augmentation": ("P", "Hamming & Gradient\nAugmentation"),
    "Annealing": ("*", "Annealing")
}
_CLASSIC_COLOR = (0.6, 0.6, 0.6, 1.0)
_CLASSIC_LABEL = "Classic superalloys"

def plot_ashby(df, x_col, y_col,
               cluster_labels, origin_labels,
               classic_mask, output_filename,
               log_x=False, log_y=False):

    mask_new      = ~classic_mask
    uniq_clusters = sorted(set(cluster_labels[mask_new]))

    # ─── 1) Construire les labels courts et fusionner ─────────────────────
    SHORT_THRESHOLD = 0.05

    # map original→list[(el,qty)]
    parsed = {
        lab: re.findall(r'([A-Z][a-z]?)(\d*\.?\d+)', lab)
        for lab in uniq_clusters
    }

    # map original→label court, p.ex. "~Fe-Cr-Al"
    short_map = {}
    for lab, toks in parsed.items():
        elems = [el for el, qty in toks if float(qty) >= SHORT_THRESHOLD]
        # si vous voulez trier alphabétiquement, remplacez par sorted(elems)
        short_map[lab] = "~" + "-".join(elems)

    # liste unique des labels courts, triée
    uniq_short = sorted(set(short_map.values()))

    # ─── 2) Générer une palette vive sur uniq_short ────────────────────────
    n = len(uniq_short)
    hues = np.linspace(0, 1, n, endpoint=False)
    cluster_to_color = {
        sl: colorsys.hsv_to_rgb(h, 1.0, 1.0)
        for sl, h in zip(uniq_short, hues)
    }

    # ─── 3) Préparer la figure ─────────────────────────────────────────────
    plt.figure(figsize=(8, 5))

    # Tracé des nouveaux points
    for origin, (marker, _) in _SHAPE_MAP.items():
        sel = mask_new & (origin_labels == origin)
        if not sel.any(): 
            continue
        # pour chaque point, on récupère d'abord son cluster original c,
        # puis son label court short_map[c], puis sa couleur:
        cols = [cluster_to_color[short_map[c]]
                for c in cluster_labels[sel]]
        plt.scatter(
            df.loc[sel, x_col], df.loc[sel, y_col],
            c=cols, marker=marker, s=50,
            edgecolor='k', linewidth=0.4
        )

    # Tracé des classiques
    if classic_mask.any():
        plt.scatter(
            df.loc[classic_mask, x_col], df.loc[classic_mask, y_col],
            c=[_CLASSIC_COLOR]*classic_mask.sum(),
            marker='o', s=60,
            edgecolor='k', linewidth=0.5,
            label="Legacy Superalloys"       
        )

    # Si x_col=="Log_10(K_A)", on affiche à la place du math-texte
    if x_col == "Log_10(K_A)":
        plt.xlabel(r"$\log(K_A)$")
    else:
        plt.xlabel(x_col)
    plt.ylabel(y_col)

    # ─── 4) Préparer les handles de légende ───────────────────────────────
    # a) Origins + Classic
    origin_handles = [
        Line2D([0], [0], marker=_SHAPE_MAP[o][0], linestyle='',
               markerfacecolor='k', markeredgecolor='k',
               markersize=8, label=_SHAPE_MAP[o][1])
        for o in _SHAPE_MAP
    ]
    classic_handle = Line2D([0], [0],
                            marker='o', linestyle='',
                            markerfacecolor=_CLASSIC_COLOR,
                            markeredgecolor='k',
                            markersize=8, label="Legacy Superalloys")

    # b) Clusters fusionnés
    cluster_handles = [
        Line2D([0], [0],
               marker='s', linestyle='',
               markerfacecolor=cluster_to_color[sl],
               markeredgecolor='k', markersize=8,
               label=sl)
        for sl in uniq_short
    ]

    # ─── 5) Deux légendes côte-à-côte ─────────────────────────────────────
    ax = plt.gca()

    # Légende 1 : Origins + Classic (colonne 1)
    leg1 = ax.legend(
        handles=origin_handles + [classic_handle],
        loc='upper left',
        bbox_to_anchor=(0.01, -0.12),
        ncol=1,
        fontsize=8,
        frameon=False
    )
    ax.add_artist(leg1)

    # Légende 2 : Clusters fusionnés (colonnes 2 & 3)
    leg2 = ax.legend(
        handles=cluster_handles,
        loc='upper left',
        bbox_to_anchor=(0.30, -0.12),
        ncol=2,
        fontsize=8,
        frameon=False
    )

    # ─── 6) Ajustements finaux ─────────────────────────────────────────────
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.25)

    if log_x: plt.xscale('log')
    if log_y: plt.yscale('log')

    plt.savefig(output_filename,
                dpi=900,
                bbox_inches='tight',   # ← recalcule les marges pour ne rien couper
                pad_inches=0.1,     # ← petit pad blanc autour
                bbox_extra_artists=(leg1, leg2))         
    plt.close()
