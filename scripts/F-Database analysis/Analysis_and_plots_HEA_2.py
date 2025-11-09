#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import warnings
import re

from plotting_utils2 import plot_ashby

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# ──────────────────────────────────────────────────────────────────────
# 0) Lecture des CSV (déjà enrichis en g₁…g₉, f₁…f₄)
# ──────────────────────────────────────────────────────────────────────
CSV_FILE     = PROJECT_FOLDER / "D-Beyond brute force" / "Genuine_Pareto_Clustered.csv"
LEGACY_CSV = PROJECT_FOLDER / "C-HEA" / "classic_superalloys_utf8.csv"
df_new     = pd.read_csv(CSV_FILE, sep=";")
df_classic = pd.read_csv(LEGACY_CSV, sep=";", encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────
# 0bis) Pour les “classiques”, expander Normalized Composition en colonnes
# ──────────────────────────────────────────────────────────────────────
def expand_normalized_composition(df, comp_col="Normalized Composition"):
    """
    Convertit une string style "Sc0.06Nb0.08Zr0.44…" en colonnes
      Sc  Nb   Zr   … 
      0.06 0.08 0.44 …
    """
    # 1) On extrait tous les couples (symbole, quantité)
    parsed = df[comp_col].fillna("").apply(
        lambda s: re.findall(r'([A-Z][a-z]?)(\d*\.?\d+)', s)
    )
    # 2) On transforme chaque liste de tuples en dict {El: float(qty)}
    comp_dicts = parsed.apply(lambda lst: {el: float(qty) for el, qty in lst})
    # 3) On passe en DataFrame, on remplace NaN par 0
    comp_df = pd.DataFrame(comp_dicts.tolist()).fillna(0.0)
    # 4) On concatène en colonnes
    return pd.concat([df.reset_index(drop=True), comp_df.reset_index(drop=True)], axis=1)

df_classic = expand_normalized_composition(df_classic)

# Renoms minimaux pour tracer
rename_map = {
    "Density"                      : "Density (g/cm³)",
    "Density (g/cm3)"              : "Density (g/cm³)",
    "Density (t/m3)"               : "Density (g/cm³)",
    "Melting Temperature (K)"      : "Melting Point (K)",
    "Formation Enthalpy (eV/atom)" : "Formation Energy (eV/atom)",
    "LMP"                 : "Creep (LMP)",
    "Melting_Temperature" : "Melting Point (K)",    
}
df_new.rename(columns=rename_map, inplace=True)
df_classic.rename(columns=rename_map, inplace=True)

# Flags
df_new["is_classic"]     = False
df_classic["is_classic"] = True

# Concaténation
df_all = pd.concat([df_new, df_classic], ignore_index=True)

# ──────────────────────────────────────────────────────────────────────
# 0ter) Calcul des colonnes Energy Footprint et Supply Risk
# ──────────────────────────────────────────────────────────────────────
# 1) Lecture du dictionnaire
risk_df = pd.read_csv(PROJECT_FOLDER / "G-risks" / "Risk&Extraction_Energy_Dictionary.csv", sep=";")

# 2) Création des mappings élément → valeurs
footprint_map = dict(zip(risk_df["Element"], 
                         risk_df["Energy footprint (MJ/mol)"]))
supply_map    = dict(zip(risk_df["Element"], 
                         risk_df["Supply risk"]))

# 3) Détection des colonnes stœchiométrie présentes
#    (tous les symboles d'éléments qui figurent dans le dictionnaire)
stoich_cols = [el for el in risk_df["Element"] if el in df_all.columns]

# 4) Calcul vectorisé
#    – Energy Footprint = ∑ (stoich_i * footprint_i)
df_all["Energy Footprint (MJ/mol)"] = (
    df_all[stoich_cols]
    .mul(pd.Series(footprint_map))
    .sum(axis=1)
)
#    – Supply Risk = ∑ supply_i pour chaque élément où stoich_i > 0
df_all["Supply Risk"] = (
    (df_all[stoich_cols] > 0)
    .mul(pd.Series(supply_map))
    .sum(axis=1)
)


# Calcul du module de Young si nécessaire
if "Young Modulus (GPa)" not in df_all.columns:
    df_all["Young Modulus (GPa)"] = (
         9 * df_all["Bulk Modulus (GPa)"] * df_all["Shear Modulus (GPa)"] /
         (3 * df_all["Bulk Modulus (GPa)"] + df_all["Shear Modulus (GPa)"] + 1e-12)
    )

# ──────────────────────────────────────────────────────────────────────
# 1) Préparation des masques et vecteurs de labels
# ──────────────────────────────────────────────────────────────────────
# On suppose que le CSV source contient déjà :
#   - une colonne "Cluster" (chaîne : formule médéoïde)
#   - une colonne "Origin" (texte brut : "Brute Force HEA", "Generative CVAE", "CVAE_hamming_augmented")

# on récupère les labels de cluster comme chaînes
cluster_labels = df_all["Cluster"].astype(str).to_numpy()
origin_labels  = df_all["Origin"].to_numpy(dtype=str)
classic_mask   = df_all["is_classic"].to_numpy()

# ──────────────────────────────────────────────────────────────────────
# 2) Tracés d’Ashby (4 combinaisons)
# ──────────────────────────────────────────────────────────────────────
plot_dir = c_folder / "Ashby plots"
os.makedirs(plot_dir, exist_ok=True)

# 4 cartes d’Ashby
plot_ashby(df_all,
           x_col="Density (g/cm³)",        y_col="Melting Point (K)",
           cluster_labels=cluster_labels, origin_labels=origin_labels,
           classic_mask=classic_mask,
           output_filename=os.path.join(plot_dir, "ashby_density_vs_melting2.png"))

plot_ashby(df_all,
           x_col="Shear Modulus (GPa)",    y_col="Bulk Modulus (GPa)",
           cluster_labels=cluster_labels, origin_labels=origin_labels,
           classic_mask=classic_mask,
           output_filename=os.path.join(plot_dir, "ashby_shear_vs_bulk2.png"))

plot_ashby(df_all,
           x_col="Young Modulus (GPa)",    y_col="Density (g/cm³)",
           cluster_labels=cluster_labels, origin_labels=origin_labels,
           classic_mask=classic_mask,
           output_filename=os.path.join(plot_dir, "ashby_young_vs_density2.png"))

plot_ashby(df_all,
           x_col="Creep (LMP)",            y_col="Formation Energy (eV/atom)",
           cluster_labels=cluster_labels, origin_labels=origin_labels,
           classic_mask=classic_mask,
           output_filename=os.path.join(plot_dir, "ashby_creep_vs_energy2.png"))

plot_ashby(df_all,
           x_col="Log_10(K_A)",            y_col="Melting Point (K)",
           cluster_labels=cluster_labels, origin_labels=origin_labels,
           classic_mask=classic_mask,
           output_filename=os.path.join(plot_dir, "ashby_HT_oxidation_vs_melting2.png"))

plot_ashby(df_all,
           x_col="Energy Footprint (MJ/mol)",
           y_col="Supply Risk",
           cluster_labels=cluster_labels,
           origin_labels=origin_labels,
           classic_mask=classic_mask,
           output_filename=os.path.join(plot_dir,
                                        "ashby_energy_footprint_vs_supply_risk.png"), log_x=True)

print("✅ Tracés réalisés avec couleurs par Cluster et formes par Origin.")
