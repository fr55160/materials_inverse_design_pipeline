#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent

raw_companionality = c_folder / "companionality_matrix.csv"
global_companionality = c_folder / "M_E_over_HE.csv"
m_ehe_path = global_companionality
he_by_cty_path = c_folder / "HE_by_countries.csv"
out_m_ec = c_folder / "M_E_over_C.csv"
out_sigma_corr = c_folder / "Sigma_E.csv"
out_sigma_corr_reordered = c_folder / "Sigma_E_reordered.csv"
new_dir = Path(c_folder / "Figures")
new_dir.mkdir(parents=True, exist_ok=True)
out_heatmap = c_folder / "Figures" / "Sigma_E_heatmap.png"
out_heatmap_reordered = c_folder / "Figures" / "Sigma_E_heatmap_reordered.png"
out_dendro = c_folder / "Figures" / "Sigma_E_dendrogram.png"

def strip_star(s: str) -> str:
    return re.sub(r"\*$", "", str(s)).strip()

def load_numeric_csv(path, sep=";"):
    df = pd.read_csv(path, sep=sep, header=0, index_col=0)
    df = df.replace({",": "."}, regex=True).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def corr_from_XXT(XXT: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(XXT), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d = np.where(d > 0, 1.0 / d, 0.0)
        Dinv = np.diag(inv_d)
        Corr = Dinv @ XXT @ Dinv
    np.fill_diagonal(Corr, 1.0)
    zero_idx = np.where(d == 0)[0]
    if zero_idx.size:
        Corr[zero_idx, :] = 0.0
        Corr[:, zero_idx] = 0.0
        for i in zero_idx:
            Corr[i, i] = 1.0
    return Corr

def plot_heatmap(corr_df: pd.DataFrame, out_png="Sigma_E_heatmap.png", title="Correlation matrix Σ_E (elements)"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        cmap="inferno",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.7, "label": "Correlation"},
        xticklabels=True,
        yticklabels=True
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=900)
    plt.close()
    print(f"Saved heatmap to {out_png}")

def cluster_order_from_corr(corr_df: pd.DataFrame, method="average", eps=1e-12):
    """
    Retourne l'ordre des indices selon un dendrogramme average-linkage
    basé sur la distance d(i,j)=1-corr(i,j), avec diagnostic des valeurs
    hors bornes avant clipping.
    """
    elems = list(corr_df.index)

    # Copie float et symétrisation
    C = corr_df.values.astype(float, copy=True)
    C = 0.5 * (C + C.T)

    # --- Diagnostic corrélations hors [-1, 1]
    bad_corr = np.argwhere((C < -1 - eps) | (C > 1 + eps))
    if bad_corr.size > 0:
        print("⚠️ Corrélations hors [-1,1] détectées :")
        for (i, j) in bad_corr:
            if i < j:  # éviter doublons
                print(f"  {elems[i]} – {elems[j]} : corr = {C[i,j]:.6f}")

    # Clip corrélation dans [-1, 1]
    np.clip(C, -1.0, 1.0, out=C)

    # Distance = 1 - corr
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)

    # --- Diagnostic distances négatives ou > 2
    bad_dist = np.argwhere((D < -eps) | (D > 2 + eps))
    if bad_dist.size > 0:
        print("⚠️ Distances hors [0,2] détectées :")
        for (i, j) in bad_dist:
            if i < j:
                print(f"  {elems[i]} – {elems[j]} : dist = {D[i,j]:.6f}")

    # Clip distances dans [0, 2]
    np.clip(D, 0.0, 2.0, out=D)

    # Passage en forme condensée puis linkage
    dist_condensed = squareform(D, checks=False)
    Z = linkage(dist_condensed, method=method)

    # Ordre des feuilles
    ddata = dendrogram(Z, no_plot=True, labels=elems)
    leaf_order_labels = ddata["ivl"]
    order = [corr_df.index.get_loc(lbl) for lbl in leaf_order_labels]
    return order, Z

def plot_dendrogram(Z, labels, out_png="Sigma_E_dendrogram.png", method="average"):
    plt.figure(figsize=(10, 4))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(f"Hierarchical clustering (linkage: {method})", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=900)
    plt.close()
    print(f"Saved dendrogram to {out_png}")

def main(
    m_ehe_path="M_E_over_HE.csv",
    he_by_cty_path="HE_by_countries.csv",
    sep=";",
    out_m_ec="M_E_over_C.csv",
    out_sigma_corr="Sigma_E.csv",
    out_sigma_corr_reordered="Sigma_E_reordered.csv",
    out_heatmap="Sigma_E_heatmap.png",
    out_heatmap_reordered="Sigma_E_heatmap_reordered.png",
    out_dendro="Sigma_E_dendrogram.png",
):
    # 1) Charger M_(E/HE) et enlever les '*'
    M_host_elem = load_numeric_csv(m_ehe_path, sep=sep)
    M_elem_host_events = M_host_elem.T.copy()  # elements × host(events)
    M_elem_host_events.columns = [strip_star(c) for c in M_elem_host_events.columns]

    # Vérifier unicité des hosts après retrait du '*'
    if len(set(M_elem_host_events.columns)) != M_elem_host_events.shape[1]:
        dup = pd.Series(M_elem_host_events.columns).value_counts()
        dups = list(dup[dup > 1].index)
        raise ValueError(
            f"Duplicate host names after removing '*': {dups}\n"
            "If this can happen, aggregate homonym columns before multiplication."
        )

    # 2) Charger M_(HE/C) et aligner les hosts
    M_HE_C = load_numeric_csv(he_by_cty_path, sep=sep)       # countries × hosts
    common_hosts = list(M_elem_host_events.columns)
    M_HE_C_aligned = M_HE_C.reindex(columns=common_hosts, fill_value=0.0)

    # 3) X = elements × countries
    X = M_elem_host_events.values @ M_HE_C_aligned.values.T
    X_df = pd.DataFrame(X, index=M_elem_host_events.index, columns=M_HE_C_aligned.index)
    X_df.to_csv(out_m_ec, sep=sep, encoding="utf-8")

    # 4) Σ_cov = X X^T ; Σ_corr normalisée
    Sigma_cov = X @ X.T
    Sigma_corr = corr_from_XXT(Sigma_cov)

    Sigma_corr_df = pd.DataFrame(Sigma_corr, index=M_elem_host_events.index, columns=M_elem_host_events.index)
    Sigma_corr_df.to_csv(out_sigma_corr, sep=sep, encoding="utf-8")

    # 5) Clustering hiérarchique (distance = 1 - corr), average linkage
    order, Z = cluster_order_from_corr(Sigma_corr_df, method="average")
    ordered_labels = [Sigma_corr_df.index[i] for i in order]

    # 6) Réordonner la matrice de corrélation
    Sigma_corr_re = Sigma_corr_df.values[np.ix_(order, order)]
    Sigma_corr_re_df = pd.DataFrame(Sigma_corr_re, index=ordered_labels, columns=ordered_labels)
    Sigma_corr_re_df.to_csv(out_sigma_corr_reordered, sep=sep, encoding="utf-8")

    # 7) Tracer heatmaps (avant/après) et dendrogramme
    plot_heatmap(Sigma_corr_df, out_png=out_heatmap, title="Correlation matrix Σ_E (original order)")
    plot_heatmap(Sigma_corr_re_df, out_png=out_heatmap_reordered, title="Correlation matrix Σ_E (clustered order)")
    plot_dendrogram(Z, labels=list(Sigma_corr_df.index), out_png=out_dendro, method="average")

    # 8) Logs
    print("✅ Done.")
    print(f"- Elements: {list(Sigma_corr_df.index)}")
    print(f"- Hosts used: {common_hosts}")
    print(f"- Countries: {list(M_HE_C_aligned.index)}")
    print(f"Saved elements×countries matrix to: {out_m_ec}")
    print(f"Saved CORRELATION Sigma_E to: {out_sigma_corr}")
    print(f"Saved clustered CORRELATION Sigma_E to: {out_sigma_corr_reordered}")
    print(f"Saved heatmaps to: {out_heatmap} and {out_heatmap_reordered}")
    print(f"Saved dendrogram to: {out_dendro}")

if __name__ == "__main__":
    argv = sys.argv
    #m_ehe_path = argv[1] if len(argv) > 1 else "M_E_over_HE.csv"
    #he_by_cty_path = argv[2] if len(argv) > 2 else "HE_by_countries.csv"
    sep = ";"
    #out_m_ec = argv[4] if len(argv) > 4 else "M_E_over_C.csv"
    #out_sigma_corr = argv[5] if len(argv) > 5 else "Sigma_E.csv"
    #out_sigma_corr_reordered = argv[6] if len(argv) > 6 else "Sigma_E_reordered.csv"
    #out_heatmap = argv[7] if len(argv) > 7 else "Sigma_E_heatmap.png"
    #out_heatmap_reordered = argv[8] if len(argv) > 8 else "Sigma_E_heatmap_reordered.png"
    #out_dendro = argv[9] if len(argv) > 9 else "Sigma_E_dendrogram.png"
    main(m_ehe_path, he_by_cty_path, sep, out_m_ec, out_sigma_corr, out_sigma_corr_reordered,
         out_heatmap, out_heatmap_reordered, out_dendro)
