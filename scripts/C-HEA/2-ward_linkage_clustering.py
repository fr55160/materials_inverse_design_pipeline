#!/usr/bin/env python3
"""
cluster_compositions_with_labels.py

Regroupe par Ward linkage des compositions normalisées (distance L2) et
ajoute, pour chaque matériau, un numéro de cluster dans une colonne
"Cluster" du CSV de sortie.

--- CONFIGURATION ---
Modifiez ces trois variables avant exécution :
  • INPUT_CSV  : chemin vers le CSV d'entrée (doit contenir la colonne
                 "Normalized Composition")
  • OUTPUT_CSV : chemin vers le CSV de sortie (sera créé ou écrasé)
  • THRESHOLD  : seuil de coupure pour fcluster (distance Euclidienne)
"""

# --- Imports ---
import re
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# --- Paramètres à modifier à la main ---
INPUT_CSV  = c_folder / 'Test_database.csv'
OUTPUT_CSV = c_folder / 'Test_database_clustered.csv'
THRESHOLD  = 12.0  # ← Seuil Euclidien pour fcluster

def medoid_index(cluster_idxs: np.ndarray, X: np.ndarray) -> int:
    """
    Renvoie l'indice (dans X) du point médéoïde du sous-ensemble cluster_idxs.
    """
    sub = X[cluster_idxs, :]
    D = pairwise_distances(sub, metric='euclidean')
    sums = D.sum(axis=1)
    return cluster_idxs[np.argmin(sums)]


def parse_formula(formula: str) -> dict:
    """
    Transforme une formule normalisée, ex. "Fe0.6Al0.1Au0.3",
    en dict {'Fe': 0.6, 'Al': 0.1, 'Au': 0.3}.
    """
    tokens = re.findall(r'([A-Z][a-z]*)([0-9]*\.?[0-9]+)', formula)
    return {elem: float(coeff) for elem, coeff in tokens}

def main():
    # --- Lecture du fichier ---
    df = pd.read_csv(INPUT_CSV, sep=";")
    if 'Normalized Composition' not in df.columns:
        raise KeyError("Colonne 'Normalized Composition' introuvable dans le CSV d'entrée.")
    
    # --- Parsing et construction de la matrice X ---
    compo_dicts = df['Normalized Composition'].map(parse_formula)
    all_elements = sorted({el for d in compo_dicts for el in d})
    X = np.vstack([[d.get(el, 0.0) for el in all_elements] for d in compo_dicts])

    # --- Clustering Ward (L2) + découpage par distance ---
    Z = linkage(X, method='ward')                          # Ward linkage
    labels = fcluster(Z, t=THRESHOLD, criterion='distance')  # Numéros de cluster

    # On reconstruit la matrice X comme avant :
    compo_dicts = df['Normalized Composition'].map(parse_formula)
    all_elements = sorted({el for d in compo_dicts for el in d})
    X = np.vstack([[d.get(el, 0.0) for el in all_elements] for d in compo_dicts])

    # Calcul des médéoïdes
    medoid_formula = {}
    for cid in np.unique(labels):
        idxs = np.where(labels == cid)[0]
        midx = medoid_index(idxs, X)
        # on stocke la formule normalisée du médéoïde
        # --- Reconstruction arrondie de la formule du médéoïde ---
        d = compo_dicts.iloc[midx]   # dict élément→coefficient
        formula = ''.join(
            f"{el}{format(d.get(el, 0.0), '.2f').rstrip('0').rstrip('.')}"
            for el in all_elements
            if d.get(el, 0.0) > 0.0
        )
        medoid_formula[cid] = formula 

    # --- Ajout de la colonne Cluster et écriture du CSV ---
    df['Cluster'] = [medoid_formula[c] for c in labels]
    df.to_csv(OUTPUT_CSV, index=False, sep=";")

    # --- Feedback utilisateur ---
    n_in  = len(df)
    n_cl  = len(np.unique(labels))
    print(
        f"→ {n_in} lignes lues, {n_cl} clusters formés (L2),\n"
        f"  tous les matériaux ont été étiquetés dans '{OUTPUT_CSV}'."
    )

if __name__ == '__main__':
    main()
