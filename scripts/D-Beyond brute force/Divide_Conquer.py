#!/usr/bin/env python3
"""
ward_divide_and_conquer.py

Découpe un CSV massif en paquets, fait du Ward linkage+mé­dé­oïde sur chaque paquet,
concatène les représentants, et recommence jusqu'à ce qu'il ne reste plus qu'un lot
< CHUNK_SIZE. Enfin, une dernière passe Ward pour bien appliquer THRESHOLD globalement.
On arrête aussi si une itération intermédiaire n'apporte aucun progrès.
"""
from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# === À éditer à la main ===
INPUT_CSV   = PROJECT_FOLDER / 'C-HEA' / 'Alloys_list_Allfeats.csv'
OUTPUT_CSV  = c_folder / 'Divide_Conquer.csv'
THRESHOLD   = 0.3
CHUNK_SIZE  = 1000

# === Imports ===
import re
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances

# --- 1) Extraction de la liste de tous les éléments chimiques ---
all_elements = set()
for chunk in pd.read_csv(INPUT_CSV, sep=";", usecols=['Normalized Composition'], chunksize=CHUNK_SIZE):
    for formula in chunk['Normalized Composition']:
        for el, coeff in re.findall(r'([A-Z][a-z]*)([0-9]*\.?[0-9]+)', formula):
            all_elements.add(el)
all_elements = sorted(all_elements)

def parse_formula(formula: str) -> dict:
    """Parse 'Fe0.6Al0.1Au0.3' → {'Fe':0.6,'Al':0.1,'Au':0.3}."""
    return {el: float(coef) for el, coef in re.findall(r'([A-Z][a-z]*)([0-9]*\.?[0-9]+)', formula)}

def medoid_index(idxs: np.ndarray, X: np.ndarray) -> int:
    """Sur un sous-ensemble X[idxs], renvoie la position du médéoïde."""
    sub = X[idxs, :]
    D = pairwise_distances(sub, metric='euclidean')
    return idxs[np.argmin(D.sum(axis=1))]

def cluster_df(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Regroupe le DataFrame df par Ward+fcluster(distance=threshold)
    et renvoie les lignes médéoïdes de chaque cluster.
    """
    compo_dicts = df['Normalized Composition'].map(parse_formula)
    X = np.array([[d.get(el, 0.0) for el in all_elements] for d in compo_dicts])

    Z = linkage(X, method='ward')
    labels = fcluster(Z, t=threshold, criterion='distance')

    reps = []
    for cid in np.unique(labels):
        idxs = np.where(labels == cid)[0]
        midx = medoid_index(idxs, X)
        reps.append(df.iloc[midx])
    return pd.DataFrame(reps, columns=df.columns)

def main():
    # 2) Première passe : chunkize l'entrée
    reps_list = []
    for chunk in pd.read_csv(INPUT_CSV, sep=";", chunksize=CHUNK_SIZE):
        reps_list.append(cluster_df(chunk, THRESHOLD))
    df_reps = pd.concat(reps_list, ignore_index=True)

    # 3) Itérations intermédiaires avec détection de plateau
    prev_len = None
    iteration = 0
    while len(df_reps) > CHUNK_SIZE:
        iteration += 1
        current_len = len(df_reps)
        if prev_len is not None and current_len >= prev_len:
            print(f"🔶 Itération {iteration} n’a pas réduit la taille "
                  f"({current_len} ≥ {prev_len}), arrêt des passes intermédiaires.")
            break
        prev_len = current_len

        tmp = []
        for start in range(0, len(df_reps), CHUNK_SIZE):
            block = df_reps.iloc[start:start + CHUNK_SIZE]
            tmp.append(cluster_df(block, THRESHOLD))
        df_reps = pd.concat(tmp, ignore_index=True)
        print(f"→ Après itération {iteration}, {len(df_reps)} représentants")

    # 4) Dernière passe sur le petit lot (ou le lot figé) pour garantir le seuil global
    df_final = cluster_df(df_reps, THRESHOLD)

    # 5) Sauvegarde
    df_final.to_csv(OUTPUT_CSV, index=False, sep=";")

    # 6) Bilan
    print(f"→ État final : {len(df_final)} représentants écrits dans '{OUTPUT_CSV}'.")
    print(f"  (Seuil = {THRESHOLD}, chunk size = {CHUNK_SIZE})")

if __name__ == '__main__':
    main()
