# pareto_clusters_ward.py
import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from numpy.linalg import norm

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# ---------------- I/O ----------------
IN_CSV  = PROJECT_FOLDER / "D-Beyond brute force" / "Front de Pareto complet.csv"
OUTDIR  = c_folder
os.makedirs(OUTDIR, exist_ok=True)

elements = ['Nb','Sc','Ta','Ti','Zr','Mo','Cr','Mn','Fe','Hf','Co','Si','Mg','V','W','Y','In','Ag','Ga','Cd','Sb','Te','Cu','Re','Ge','Hg','Ru','Al','As','Zn','Ni','Ir','Bi','Au','Pt','Pb','Rh','Pd','Sn']
properties = ['Melting_Temperature','LMP','Log_10(K_A)','Density','Bulk Modulus (GPa)','Shear Modulus (GPa)','GB ratio']

# ------------- Utils ---------------
def parse_composition(comp):
    d = {el:0.0 for el in elements}
    for el,val in re.findall(r'([A-Z][a-z]*)([0-9.]+)', str(comp)):
        d[el] = float(val)
    return d

def clr_transform(X):
    eps = 1e-12
    Xs  = X + eps
    gm  = np.exp(np.mean(np.log(Xs), axis=1, keepdims=True))
    return np.log(Xs/gm)

def wcss_from_labels(X, labels):
    # somme des carrés intra-clusters dans l'espace des features X (ici CLR)
    w = 0.0
    for lab in np.unique(labels):
        Xi = X[labels==lab]
        if Xi.shape[0] <= 1: 
            continue
        c  = Xi.mean(axis=0, keepdims=True)
        dif = Xi - c
        w += np.sum(dif*dif)
    return float(w)

# ------------- Load ---------------
df = pd.read_csv(IN_CSV, sep=';')
compo_df = df['Normalized Composition'].apply(parse_composition).apply(pd.Series)
props_df = df[properties].copy()

# ------------- CLR -----------------
Z_CLR = clr_transform(compo_df.values)   # (n, d)

# ------------- Ward linkage --------
# SciPy linkage avec 'ward' (distance euclidienne implicite)
Z = linkage(Z_CLR, method='ward')

# --- Dendrogramme propre et coloré par clusters Ward ---
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np

# Palette figée pour 1..4 comme ta PCA3D, puis cycle Matplotlib au-delà
palette_base = {1:"#1f77b4", 2:"#ff7f0e", 3:"#2ca02c", 4:"#d62728"}
mpl_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

def build_link_parents(Z):
    """Mappe l'index de nœud -> (enfants). Feuilles: 0..n-1 ; nœuds internes: n..2n-2."""
    n = Z.shape[0] + 1
    parents = {}
    for i, (a, b, *_rest) in enumerate(Z):
        node_id = n + i
        parents[node_id] = (int(a), int(b))
    return parents

def collect_leaves_iterative(node_id, parents, n_leaves):
    """Retourne l'ensemble des feuilles sous un nœud (version non récursive)."""
    stack = [node_id]
    leaves = set()
    while stack:
        nid = stack.pop()
        if nid < n_leaves:
            leaves.add(nid)
        else:
            a, b = parents[nid]
            stack.append(a)
            stack.append(b)
    return leaves

def make_link_color_func(Z, final_labels, cluster_colors):
    n = Z.shape[0] + 1
    parents = build_link_parents(Z)
    leaf_cache = {}

    def _link_color_func(node_id):
        if node_id < n:
            c = final_labels[node_id]
            return cluster_colors.get(c, "#666666")
        if node_id not in leaf_cache:
            leaf_cache[node_id] = collect_leaves_iterative(node_id, parents, n)
        leaves = list(leaf_cache[node_id])
        labs = {final_labels[i] for i in leaves}
        if len(labs) == 1:
            c = labs.pop()
            return cluster_colors.get(c, "#666666")
        return "#9aa0a6"

    return _link_color_func

def pretty_dendrogram(Z, labels_final, k_star, out_path=c_folder / "Dendrogram_pretty.png",
                      p_show=None):
    """
    Z: matrice de linkage SciPy
    labels_final: tableau long n d'étiquettes de clusters (1..k*)
    k_star: nombre de clusters retenu
    p_show: nombre de “feuilles contractées” à afficher (si None: 8*k_star, borné à [min(30,n)])
    """
    n = len(labels_final)
    # couleurs par cluster
    cluster_colors = {}
    for i, c in enumerate(sorted(np.unique(labels_final)), start=1):
        cluster_colors[c] = palette_base.get(c, mpl_cycle[(i-1) % max(1, len(mpl_cycle))])

    link_color_func = make_link_color_func(Z, np.asarray(labels_final, dtype=int), cluster_colors)

    # Déterminer un seuil de coupe correspondant à k*
    # (hauteur à laquelle fcluster(..., maxclust=k_star) couperait l'arbre)
    # On l'estime par la hauteur de la (k*-1)-ième fusion la plus haute.
    heights = Z[:, 2]
    # trier décroissant les sauts
    cut_h = np.partition(heights, - (k_star-1))[-(k_star-1)] if k_star > 1 else heights.max()

    # nombre de groupes à afficher en bas (lisibilité)
    if p_show is None:
        p_show = int(np.clip(8*k_star, 30, max(30, min(120, n))))

    plt.figure(figsize=(14, 4.8))
    dendrogram(
        Z,
        truncate_mode='lastp', p=p_show,  # compaction lisible
        color_threshold=cut_h + 1e-9,     # ne colore pas via seuil (on gère nous-mêmes)
        link_color_func=link_color_func,  # notre mapping couleurs
        no_labels=True,
        leaf_rotation=0,
        leaf_font_size=10,
        above_threshold_color="#9aa0a6",  # gris
        distance_sort='descending',
        count_sort=True,
        #show_contracted=True,             # triangles = effectifs approximatifs
    )

    # Ligne de coupe à k*
    plt.axhline(cut_h, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    plt.title(f"Dendrogram (Ward) — colored by final clusters (k={k_star})", fontsize=14, pad=10)
    plt.ylabel("Fusion distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"Saved prettified dendrogram to {out_path}")

# --- Appel (après avoir calculé 'labels' avec fcluster) ---
# pretty_dendrogram(Z, labels, k_star)  # décommente dans ton script

# ------------- Choix de k ----------
K_RANGE = range(2, 16)
records = []
for k in K_RANGE:
    labels = fcluster(Z, t=k, criterion='maxclust')
    wcss   = wcss_from_labels(Z_CLR, labels)
    # métriques additionnelles (silhouette difficile si k=1)
    sil, ch, db = np.nan, np.nan, np.nan
    try:
        sil = silhouette_score(Z_CLR, labels, metric='euclidean')
        ch  = calinski_harabasz_score(Z_CLR, labels)
        db  = davies_bouldin_score(Z_CLR, labels)
    except Exception:
        pass
    records.append({"k":k, "WCSS":wcss, "silhouette":sil, "calinski_harabasz":ch, "davies_bouldin":db})

met = pd.DataFrame(records)
met.to_csv(f"{OUTDIR}/Ward_metrics.csv", index=False, sep=';')

# Elbow sur WCSS
plt.figure(figsize=(5,4))
plt.plot(met["k"], met["WCSS"], 'o-')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Total within-cluster SSE (WCSS)")
plt.title("Elbow (Ward linkage)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/Elbow_WCSS.png", dpi=900)
plt.close()

# --- Choix de k : farthest-point-to-chord (robuste) + checks ---
x = met["k"].values.astype(float)
y = met["WCSS"].values.astype(float)

def farthest_point_elbow(x, y, k_min=3):
    # scale to [0,1] to make distances isotropic
    xs = (x - x.min()) / (x.max() - x.min())
    ys = (y - y.min()) / (y.max() - y.min())
    # chord endpoints
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    # distances point->line
    # area-based formula for distance to line segment (perp. to infinite line is enough here)
    denom = np.hypot(x2 - x1, y2 - y1)
    denom = 1.0 if denom == 0 else denom
    d = np.abs((y2 - y1)*xs - (x2 - x1)*ys + x2*y1 - y2*x1) / denom
    # forbid k < k_min when looking for elbow
    mask = x >= k_min
    idx = np.argmax(d * mask)  # zeros out k<k_min
    return int(x[idx]), d[idx]

k_fp, dmax = farthest_point_elbow(x, y, k_min=3)

def safe_argmax(metric_col):
    arr = np.asarray(metric_col, dtype=float)
    if np.all(~np.isfinite(arr)): return None
    return int(met.loc[np.nanargmax(arr), "k"])

k_sil = safe_argmax(met["silhouette"])
k_ch  = safe_argmax(met["calinski_harabasz"])

# agrégation prudente : priorité à la méthode géométrique robuste
cands_core = [k for k in [k_fp] if k is not None]
cands_info = [k for k in [k_sil, k_ch] if k is not None]

# règle simple : on retient k_fp, sauf si un indicateur (sil/CH) le contredit fortement
# (différence > 3 clusters et score du candidat alternatif est >95% de son max local autour de k_fp)
k_star = cands_core[0]
print(f"[elbow-chord] k_fp={k_fp} (dmax={dmax:.3g}), sil={k_sil}, CH={k_ch} -> k*={k_star}")

# ------------- Clustering final ----
labels = fcluster(Z, t=k_star, criterion='maxclust')
df["Cluster"] = labels
# --- Dendrogramme beau & coloré par clusters finaux ---
pretty_dendrogram(
    Z,
    labels,                # étiquettes finales 1..k*
    k_star,
    out_path=f"{OUTDIR}/Dendrogram_pretty.png"
)

# ------------- PCA pour VISU (optionnel) ----
# Visualisation seulement (ne change pas le clustering)
pca = PCA(n_components=3)
X3  = pca.fit_transform(Z_CLR)
plt.figure(figsize=(6,5))
from mpl_toolkits.mplot3d import Axes3D  # noqa
ax = plt.subplot(111, projection='3d')
for lab in np.unique(labels):
    idx = np.where(labels==lab)[0]
    ax.scatter(X3[idx,0], X3[idx,1], X3[idx,2], s=10, alpha=0.8, label=f"Cluster {lab}")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.set_title(f"Ward clusters (k={k_star}) — PCA 3D (visu)")
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/PCA3D_clusters.png", dpi=900)
plt.close()

# ------------- Caractérisation clusters ----
# mêmes sorties logiques que ton script "Properties by cluster"
cluster_stats = []
for lab in sorted(np.unique(labels)):
    sel = df["Cluster"]==lab
    prop_mean = props_df[sel].mean().add_suffix("_mean")
    prop_std  = props_df[sel].std(ddof=1).add_suffix("_std")
    comp_mean = compo_df[sel].mean().add_suffix("_mean")
    comp_std  = compo_df[sel].std(ddof=1).add_suffix("_std")
    out = pd.concat([pd.Series({"Cluster":lab}), comp_mean, comp_std, prop_mean, prop_std])
    cluster_stats.append(out)

out_df = pd.DataFrame(cluster_stats)
out_df.to_csv(f"{OUTDIR}/Cluster_characterization.csv", sep=';', index=False)

print(f"✔ Done. k*={k_star}. Files in {OUTDIR}/")
