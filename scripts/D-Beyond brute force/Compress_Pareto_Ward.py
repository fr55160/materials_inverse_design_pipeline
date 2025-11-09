#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_compositions_numeric.py

Clusterise des compositions stœchiométriques représentées 
par des colonnes d'éléments (valeurs normalisées) et 
sélectionne, pour chaque cluster, la composition « médoïde ».
  
--- CONFIGURATION (à modifier avant exécution) ---
  • INPUT_CSV       : chemin vers le CSV d'entrée 
                      (exemple : r'./data/compositions.csv')
  • OUTPUT_CSV      : chemin vers le CSV de sortie 
                      (sera créé ou écrasé)
  • ELEMENT_COLUMNS : liste des noms de colonnes d'éléments
                      à prendre en compte pour la composition
  • THRESHOLD       : seuil de coupure pour fcluster 
                      (distance Euclidienne)
  • SEP             : séparateur de colonnes dans vos CSV
"""
from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# --- Paramètres à modifier à la main ---
INPUT_CSV = PROJECT_FOLDER / 'C-HEA' / 'Genuine_Pareto_Front.csv' 
OUTPUT_CSV = c_folder / 'Genuine_Pareto_Clustered.csv'
# Liste exhaustive des colonnes élémentaires dans votre CSV :
ELEMENT_COLUMNS = [
    'Nb','Sc','Ta','Ti','Zr','Mo','Cr','Mn','Fe','Hf','Co','Si','V','Mg','W','Y',
    'Ag','In','Ga','Cd','Sb','Te','Cu','Ge','Re','Hg','Ru','Al','As','Zn','Ni',
    'Ir','Pt','Au','Bi','Rh','Pd','Pb'
]
THRESHOLD = 0.5    # cut threshold to determine the clusters
SEP = ';'           # ajustez si nécessaire (',' ou '\t'...)

# --- Imports ---
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances

def medoid_index(cluster_idxs: np.ndarray, X: np.ndarray) -> int:
    """
    Renvoie l'indice (dans X) du point médéoïde du sous-ensemble cluster_idxs.
    """
    sub = X[cluster_idxs, :]
    D = pairwise_distances(sub, metric='euclidean')
    sums = D.sum(axis=1)
    # on retourne l'indice AVANT sous-échantillonnage
    return cluster_idxs[np.argmin(sums)]

def make_formula(row):
    return ''.join(f"{el}{row[el]:g}" for el in ELEMENT_COLUMNS if row[el] > 0)

#df_repr['Normalized Composition'] = df_repr.apply(make_formula, axis=1)

def main():
    # --- Lecture du fichier ---
    df = pd.read_csv(INPUT_CSV, sep=SEP)
    
    # --- Vérification des colonnes d'éléments ---
    missing = [el for el in ELEMENT_COLUMNS if el not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes dans le CSV d'entrée : {missing}")
    
    # --- Construction de la matrice X (n_rows × n_elements) ---
    # On remplace les NaN par 0.0 au cas où
    X = df[ELEMENT_COLUMNS].fillna(0.0).values
    
    # --- Clustering Ward + découpage par distance ---
    Z = linkage(X, method='ward')
    labels = fcluster(Z, t=THRESHOLD, criterion='distance')
    
    # --- Sélection des médéoïdes ---
    representatives = []
    for cid in np.unique(labels):
        idxs = np.where(labels == cid)[0]
        rep = medoid_index(idxs, X)
        representatives.append(rep)
    
    # --- Écriture du CSV de sortie ---
    df_repr = df.iloc[representatives].copy()
    df_repr['Normalized Composition'] = df_repr.apply(
        lambda row: ''.join(f"{el}{row[el]:g}" for el in ELEMENT_COLUMNS if row[el] > 0),
        axis=1
    )    
    df_repr.to_csv(OUTPUT_CSV, index=False, sep=SEP)
    
    # --- Feedback console ---
    n_in  = len(df)
    n_cl  = len(np.unique(labels))
    n_out = len(df_repr)
    print(f"→ {n_in} lignes lues, {n_cl} clusters formés, "
          f"{n_out} représentants écrits dans '{OUTPUT_CSV}'.")


if __name__ == '__main__':
    main()
