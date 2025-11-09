#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de sélection basé sur le front de Pareto à partir du fichier "HEA Database_Hephaistos.csv" :
1. Filtre les lignes dont les 10 colonnes d'estimateurs (g1 à g10) ont des valeurs > 0.3.
2. Extrait le front de Pareto sur ces colonnes.
3. Transforme la colonne "Normalized Composition" en colonnes séparées (une par élément) avec 
   la stœchiométrie arrondie à 2 décimales ; les éléments non présents apparaissent comme cellule vide.
4. Affiche et enregistre le résultat dans un CSV avec toutes les colonnes sélectionnées, en arrondissant :
   - les températures à la dizaine près,
   - le Creep (LMP) au dixième près,
   - les autres colonnes à 2 décimales.
   
Mis à jour pour conserver "Origin" et "Cluster"
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent
file_path = c_folder / "Test_database_clustered.csv"
output_file = c_folder / "Pareto_selection.csv"

# ---------------------------------------------------------------------------
# Fonction pour extraire la composition sous forme de dictionnaire {élément: quantité}
# ---------------------------------------------------------------------------
def parse_composition(comp_str):
    """
    Extrait les paires (élément, quantité) depuis comp_str.
    La quantité est convertie en float et arrondie à 2 décimales.
    Seuls les éléments avec une quantité strictement positive sont conservés.
    Retourne un dictionnaire {élément: quantité}.
    """
    matches = re.findall(r"([A-Z][a-z]*)([0-9]*\.?[0-9]+)", comp_str)
    comp_dict = {}
    for elem, num in matches:
        try:
            value = round(float(num), 2)
        except Exception:
            value = 0.0
        if value > 0:
            comp_dict[elem] = value
    return comp_dict

# ---------------------------------------------------------------------------
# Fonction pour extraire le front de Pareto (non dominé)
# ---------------------------------------------------------------------------
def pareto_front(df, cols):
    """
    Retourne le sous-ensemble de df correspondant au front de Pareto sur les colonnes cols.
    Une ligne est considérée comme dominée si une autre ligne a des valeurs supérieures ou égales sur toutes 
    les colonnes et strictement supérieure sur au moins une.
    """
    X = df[cols].values
    n_points = X.shape[0]
    is_dominated = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i != j and all(X[j] >= X[i]) and any(X[j] > X[i]):
                is_dominated[i] = True
                break
    return df[~is_dominated].copy()

# ---------------------------------------------------------------------------
# Chargement du fichier "HEA Database_Hephaistos.csv"
# ---------------------------------------------------------------------------
df = pd.read_csv(file_path, sep=";")

# Dictionnaire de renommage : clé = nom original, valeur = nouveau nom simple
rename_map = {
    "g1, fluage":         "g1",
    "g2, fusion":         "g2",
    "g3, densité":        "g3",
    "g4, phase BCC":      "g4",
    "g5, ductilité":      "g5",
    "g6, ductilité therm": "g6",
    "g7, formation":      "g7",
    "g8, Omega":          "g8",
    "g9, entropie":       "g9",
    "g10, HT oxydation":  "g10"
}

# Renommer uniquement si la colonne est présente
existing_renames = {old: new for old, new in rename_map.items() if old in df.columns}
if existing_renames:
    df = df.rename(columns=existing_renames)

# Liste des 10 colonnes d'estimateurs
g_columns = [
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "g6",
    "g7",
    "g8",
    "g9",
    "g10"
]


# Filtrer les lignes pour ne conserver que celles où chacune de ces 10 colonnes > 0.3
df_filtered = df[(df[g_columns] > 0.3).all(axis=1)].copy()

# Conserver le front de Pareto (non dominé) au regard des 10 colonnes
df_pareto = pareto_front(df_filtered, g_columns)

# ---------------------------------------------------------------------------
# Transformation de "Normalized Composition" en colonnes distinctes
# ---------------------------------------------------------------------------
# Appliquer parse_composition pour créer une nouvelle colonne "comp_dict"
df_pareto["comp_dict"] = df_pareto["Normalized Composition"].apply(parse_composition)

# Calculer la fréquence d'apparition de chaque élément dans la colonne "comp_dict"
element_freq = {}
for comp in df_pareto["comp_dict"]:
    for elem in comp.keys():
        element_freq[elem] = element_freq.get(elem, 0) + 1

# Ordonner les éléments par fréquence décroissante
sorted_elements = sorted(element_freq.keys(), key=lambda x: element_freq[x], reverse=True)

# Pour chaque élément, créer une colonne (laisser vide si non présent)
for elem in sorted_elements:
    df_pareto[elem] = df_pareto["comp_dict"].apply(lambda d: d.get(elem, np.nan))

# ---------------------------------------------------------------------------
# Préparation du tableau final
# ---------------------------------------------------------------------------
# Liste des colonnes quantitatives à afficher (à arrondir ensuite)
# On exclut "Decomposition Energy (eV/atom)" comme demandé.
other_cols = [
    "stoich_entropy", "delta", "avg_VEC",
    "Formation Energy (eV/atom)", "Bulk Modulus (GPa)",
    "Shear Modulus (GPa)", "Density",
    "Melting_Temperature", "LMP", "G/B ratio", "Omega", "Log_10(K_A)", "Origin", "Cluster"
]

# On crée le tableau final qui contient d'abord les colonnes de composition (une par élément), puis les autres colonnes
final_cols = sorted_elements + other_cols
df_final = df_pareto[final_cols].copy()

# Pour les colonnes de composition, remplacer NaN par une chaîne vide (cellule vide)
df_final[sorted_elements] = df_final[sorted_elements].fillna("")

# Arrondir les autres colonnes de manière appropriée :
# Pour "Melting Temperature (K)" arrondir à la dizaine près,
# Pour "Creep (LMP)" arrondir au dixième près,
# Pour les autres, arrondir à 2 chiffres après la virgule.
for col in other_cols:
    if col == "Melting Temperature (K)":
        df_final[col] = df_final[col].apply(lambda x: round(x/10)*10)
    elif col == "Creep (LMP)":
        df_final[col] = df_final[col].round(1)
    else:
        df_final[col] = df_final[col].round(2)

# ---------------------------------------------------------------------------
# Écriture du tableau final dans un fichier CSV
# ---------------------------------------------------------------------------
df_final.to_csv(output_file, sep=";", index=False)
print(f"\nFichier CSV '{output_file}' généré avec succès.")
