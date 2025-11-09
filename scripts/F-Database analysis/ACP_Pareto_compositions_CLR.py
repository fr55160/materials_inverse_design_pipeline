import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from networkx.algorithms.community import greedy_modularity_communities

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# ─── CONFIGURATION ────────────────────────────────────────────────────────
CSV_FILE     = PROJECT_FOLDER / "D-Beyond brute force" / "Genuine_Pareto_Clustered.csv"
ELEMENTS     = [
    'Nb','Sc','Ta','Ti','Zr','Mo','Cr','Mn','Fe','Hf','Co','Si','Mg','V','W',
    'Y','In','Ag','Ga','Cd','Sb','Te','Cu','Re','Ge','Hg','Ru','Al','As','Zn',
    'Ni','Ir','Bi','Au','Pt','Pb','Rh','Pd','Sn'
]
N_COMPONENTS = 6
THRESHOLD    = 0.3        # Seuil minimal de |corrélation| pour tracer une arête
SMALL_THRESHOLD = 0.25
LAYOUT       = 'kamada'   # 'spring', 'kamada', 'circular', 'shell'
SPRING_K     = 0.2        # pour spring_layout
SPRING_ITER  = 500        # pour spring_layout
SEED         = 42
MAX_AREA     = 1600.0     # aire maximale (points²) pour le nœud le plus gros
MIN_EDGE_LEN = 1.5  # distance minimale en “unités de figure” à ajuster
N_EDGE_AT = 3 # seuil pour noeud avec beaucoup d'arêtes fortes
EDGE_WIDTH_MULT = 4.0 # 

def clr_transform(X, eps=1e-9):
    """
    X : np.ndarray de forme (n_samples, n_features) avec compositions positives
    Retourne le même tableau CLR-transformé.
    """
    Xs = X + eps
    # 1) calcul du logarithme
    logX = np.log(Xs)
    # 2) centre log-géométrique sur chaque ligne (sample)
    gm = logX.mean(axis=1, keepdims=True)
    # 3) CLR = logX – gm
    return logX - gm


def find_slot(occupied, alpha):
    """
    occupied : list de (start,end) en [0,2π), non-chevauchants
    alpha    : longueur angulaire demandée
    → renvoie θ_center, ajoute l'intervalle à occupied sans jamais échouer.
    """
    TWO_PI = 2*np.pi

    # 1) éclater les intervalles qui wrap et normaliser
    intervals = []
    for (s,e) in occupied:
        s_, e_ = s % TWO_PI, e % TWO_PI
        if e_ < s_:
            intervals.append((s_, TWO_PI))
            intervals.append((0.0, e_))
        else:
            intervals.append((s_, e_))

    # 2) trier et fusionner
    if intervals:
        intervals.sort(key=lambda iv: iv[0])
        merged = [intervals[0]]
        for s,e in intervals[1:]:
            ms, me = merged[-1]
            if s <= me:
                merged[-1] = (ms, max(me, e))
            else:
                merged.append((s,e))
    else:
        merged = []

    # 3) construire la liste des gaps (y compris wrap)
    gaps = []
    if merged:
        # gaps entre intervalles consécutifs
        for (s1,e1),(s2,e2) in zip(merged, merged[1:]):
            gaps.append((e1, s2, s2 - e1))
        # gap wrap
        s_last, e_last = merged[-1]
        s0, _ = merged[0]
        gaps.append((e_last, s0 + TWO_PI, (s0 + TWO_PI) - e_last))
    else:
        # pas d'intervalles déjà occupés → on considère tout [0,2π[
        gaps = [(0.0, TWO_PI, TWO_PI)]

    # 4) rechercher un gap assez grand, sinon prendre le plus grand
    slot = next(((start, length) 
                 for start,_,length in gaps if length >= alpha),
                None)
    if slot is not None:
        start_new = slot[0]
    else:
        # fallback : le plus grand gap
        start_new, _, _ = max(gaps, key=lambda x: x[2])

    # 5) réserver et renvoyer le centre
    end_new = start_new + alpha
    s_mod, e_mod = start_new % TWO_PI, end_new % TWO_PI
    occupied.append((s_mod, e_mod))
    return (start_new + alpha/2) % TWO_PI



def enforce_min_edge_length(pos, edges, min_dist):
    """
    pos : dict node -> np.array([x,y])
    edges : iterable de (u,v)
    min_dist : distance minimale souhaitée
    """
    moved = True
    # Tant qu'on a déplacé au moins une fois
    while moved:
        moved = False
        for u, v in edges:
            pu = pos[u]
            pv = pos[v]
            delta = pv - pu
            d = np.linalg.norm(delta)
            if d < min_dist:
                # vecteur unitaire
                if d == 0:
                    # si superposés, aléatoire
                    theta = np.random.rand() * 2*np.pi
                    direction = np.array([np.cos(theta), np.sin(theta)])
                else:
                    direction = delta / d
                # écarter chacun de min_dist/2
                shift = (min_dist - d) / 1.5
                pos[u] = pu - direction * shift
                pos[v] = pv + direction * shift
                moved = True
    return pos

def plot_correlation_graph(corr: pd.DataFrame, df_raw: pd.DataFrame):
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ─── 1) Construire les graphes ───────────────────────────────────────
    G_plot   = nx.Graph()
    G_strong = nx.Graph()
    G_plot.add_nodes_from(ELEMENTS)
    G_strong.add_nodes_from(ELEMENTS)
    for i in range(len(ELEMENTS)):
        for j in range(i+1, len(ELEMENTS)):
            r = corr.iat[i, j]
            if abs(r) >= SMALL_THRESHOLD:
                G_plot.add_edge(ELEMENTS[i], ELEMENTS[j],
                                r_val=r, weight=abs(r))
                if abs(r) >= THRESHOLD:
                    G_strong.add_edge(ELEMENTS[i], ELEMENTS[j],
                                      r_val=r, weight=abs(r))

    # ─── 2) Calculer les “aires” des nœuds (points²) ────────────────────
    avg_pct = df_raw.mean() * 100.0
    k       = MAX_AREA / avg_pct.max()
    areas   = {el: pct * k for el, pct in avg_pct.items()}

    # ─── 3) Paramètres des ellipses ─────────────────────────────────────
    INNER_A, INNER_B = 0.6, 0.3   # demi-axes intérieurs
    OUTER_A, OUTER_B = 0.9, 0.6    # demi-axes extérieurs
    CX, CY          = 0.5, 0.5     # centre commun
    R_inner         = (INNER_A + INNER_B) / 2
    R_outer         = (OUTER_A + OUTER_B) / 2

    # ─── 4) Calcul des poids v_i ∝ √(mean composition) ─────────────────
    # Inner nodes = ceux qui ont ≥1 arête forte
    strong_deg   = dict(G_strong.degree())
    inner_nodes  = [n for n,d in strong_deg.items() if d >= 1]
    w_inner      = {u: np.sqrt(df_raw[u].mean()) for u in inner_nodes}
    totw_inner   = sum(w_inner.values())
    v_inner      = {u: w_inner[u] / totw_inner for u in inner_nodes}

    # Outer nodes = tous les autres
    outer_nodes  = [n for n in ELEMENTS if n not in inner_nodes]
    w_outer      = {u: np.sqrt(df_raw[u].mean()) for u in outer_nodes}
    totw_outer   = sum(w_outer.values())
    v_outer      = {u: w_outer[u] / totw_outer for u in outer_nodes}

    # ─── 5) Allocation angulaire ────────────────────────────────────────
    # part d'angle réservé à la somme des largeurs de disques
    reserved_inner = np.pi      # réservez π rad pour les inner
    reserved_outer = np.pi      # réservez π rad pour les outer
    # espacement “gap” uniforme sur chaque ellipse
    gap_inner  = (2*np.pi - reserved_inner) / len(inner_nodes)
    gap_outer  = (2*np.pi - reserved_outer) / len(outer_nodes)
    # angle alloué à chaque nœud
    D_inner = {u: v_inner[u] * reserved_inner for u in inner_nodes}
    D_outer = {u: v_outer[u] * reserved_outer for u in outer_nodes}

    # ─── 6) Placement des inner nodes ───────────────────────────────────
    # tri facultatif, ici par poids v_inner décroissant
    sorted_inner = sorted(inner_nodes, key=lambda u: v_inner[u], reverse=True)
    pos = {}
    θ = - reserved_inner / 2
    for u in sorted_inner:
        θ += D_inner[u] / 2
        pos[u] = np.array([CX + INNER_A * np.cos(θ),
                           CY + INNER_B * np.sin(θ)])
        θ += D_inner[u] / 2 + gap_inner

    # ─── 7) Placement des outer nodes ───────────────────────────────────
    sorted_outer = sorted(outer_nodes, key=lambda u: v_outer[u], reverse=True)
    θ = - reserved_outer / 2
    for u in sorted_outer:
        θ += D_outer[u] / 2
        pos[u] = np.array([CX + OUTER_A * np.cos(θ),
                           CY + OUTER_B * np.sin(θ)])
        θ += D_outer[u] / 2 + gap_outer

    # ─── 8) Tracé ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.margins(0.2)

    # séparer arêtes weak/strong
    weak_edges   = [(u,v) for u,v,d in G_plot.edges(data=True)
                    if SMALL_THRESHOLD <= d['weight'] < THRESHOLD]
    strong_edges = [(u,v) for u,v,d in G_plot.edges(data=True)
                    if d['weight'] >= THRESHOLD]

    # séparer nœuds low‐deg/high‐deg (contour)
    low_deg  = [n for n,d in strong_deg.items() if d < N_EDGE_AT]
    high_deg = [n for n,d in strong_deg.items() if d >= N_EDGE_AT]

    nx.draw_networkx_nodes(G_plot, pos, ax=ax,
                           nodelist=low_deg,
                           node_size=[areas[n] for n in low_deg],
                           node_color='skyblue', edgecolors='none', alpha=0.8)
    nx.draw_networkx_nodes(G_plot, pos, ax=ax,
                           nodelist=high_deg,
                           node_size=[areas[n] for n in high_deg],
                           node_color='skyblue',
                           edgecolors='navy', linewidths=1.0, alpha=0.8)

    if weak_edges:
        nx.draw_networkx_edges(G_plot, pos, ax=ax,
                               edgelist=weak_edges,
                               width=[G_plot[u][v]['weight']*EDGE_WIDTH_MULT
                                      for u,v in weak_edges],
                               edge_color='lightgray', alpha=0.6)
    if strong_edges:
        nx.draw_networkx_edges(G_plot, pos, ax=ax,
                               edgelist=strong_edges,
                               width=[G_plot[u][v]['weight']*EDGE_WIDTH_MULT
                                      for u,v in strong_edges],
                               edge_color='black', alpha=0.8)

    nx.draw_networkx_labels(G_plot, pos, ax=ax, font_size=9)
    ax.set_title("Correlation Graph Between Elements of HESA\nof the Pareto Front")
    ax.axis('off')

    # ─── A) Légende Corrélations (+ titre au-dessus) ───────────────────────
    border_handle      = Line2D([], [], marker='o', linestyle='None',
                                markerfacecolor='skyblue',
                                markeredgecolor='navy',
                                markeredgewidth=1.0,
                                markersize=10)
    weak_edge_handle   = Line2D([], [], color='lightgray',
                                linewidth=EDGE_WIDTH_MULT*SMALL_THRESHOLD,
                                alpha=0.6)
    strong_edge_handle = Line2D([], [], color='black',
                                linewidth=EDGE_WIDTH_MULT*THRESHOLD,
                                alpha=0.8)

    handles_corr = [
        border_handle,
        weak_edge_handle,
        strong_edge_handle
    ]
    labels_corr = [
        f"At least {N_EDGE_AT} |r|≥{THRESHOLD}",
        f"{SMALL_THRESHOLD} ≤ |r| < {THRESHOLD}",
        f"|r| ≥ {THRESHOLD}"
    ]

    leg_corr = ax.legend(
        handles_corr, labels_corr,
        title="Correlations & Mean element content (%):",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(handles_corr),
        frameon=False,
        handletextpad=0.8,
        columnspacing=1.5,
        borderaxespad=0.0
    )
    # titre un peu plus petit si besoin
    leg_corr.get_title().set_fontsize(10)
    # aligner les étiquettes à gauche sous chaque symbole
    for txt in leg_corr.get_texts():
        txt.set_ha('left')

    ax.add_artist(leg_corr)


    # ─── B) Légende Tailles (Mean element content) ────────────────────────
    ref_vals    = [1, 10]
    ref_areas   = [v * k for v in ref_vals]
    size_handles = [
        ax.scatter([], [], s=a, color='skyblue', alpha=0.8)
        for a in ref_areas
    ]
    size_labels  = [f"{v}%" for v in ref_vals]

    leg_sizes = ax.legend(
        size_handles, size_labels,
        loc='lower center',
        bbox_to_anchor=(0.303, -0.13),
        ncol=len(size_handles),
        frameon=False,
        handletextpad=0.8,
        columnspacing=5,
        borderaxespad=0.0
    )
    leg_sizes.get_title().set_fontsize(10)
    for txt in leg_sizes.get_texts():
        txt.set_ha('left')

    ax.add_artist(leg_sizes)

    plt.savefig(
        c_folder / "correlation_graph.png",
        dpi=900,
        bbox_inches='tight',
        pad_inches=0.15,
        bbox_extra_artists=(leg_corr, leg_sizes)
    )
    plt.show()


def main():
    # ─── A) Lecture & nettoyage
    df_raw = pd.read_csv(CSV_FILE, sep=';')[ELEMENTS].fillna(0.0)
    
    # ─── B) CLR transform
    #    on transforme df_raw.values en CLR, puis on remet en DataFrame
    clr_array = clr_transform(df_raw.values)
    df = pd.DataFrame(clr_array,
                      index=df_raw.index,
                      columns=df_raw.columns)
    
    # ─── C) Heatmap de corrélation sur données CLR
    corr = df.corr(method='pearson')
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(len(ELEMENTS)), ELEMENTS, rotation=90)
    plt.yticks(np.arange(len(ELEMENTS)), ELEMENTS)
    plt.title("Pearson Correlation Matrix (CLR-transformed)")
    plt.tight_layout()
    plt.show()

    # ─── D) Graphe de corrélation amélioré sur CLR
    plot_correlation_graph(corr, df_raw)  # si vous voulez garder les tailles basées sur df_raw
    #                                       ou passez df si vous préférez CLR pour tout

    # ─── E) Standardisation + PCA sur CLR
    X_std = StandardScaler().fit_transform(df)
    pca   = PCA(n_components=N_COMPONENTS)
    pca.fit(X_std)
    for i, (comp, vr) in enumerate(zip(pca.components_, pca.explained_variance_ratio_), start=1):
        pairs = sorted(zip(ELEMENTS, comp), key=lambda x: abs(x[1]), reverse=True)
        expr  = " ".join(f"{'+' if c>=0 else '-'}{abs(c):.3f}×{el}" for el, c in pairs)
        print(f"\nComponent {i}: variance explained = {vr:.4f}\n  {expr}")


if __name__ == "__main__":
    main()
