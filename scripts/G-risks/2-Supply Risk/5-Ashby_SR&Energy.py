#!/usr/bin/env python
# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from plotting_utils2 import plot_ashby

# ────────────────────────────────────────────────
# 0) Définition des chemins
# ────────────────────────────────────────────────
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent

# Fichiers CSV à lire
CSV_FILE     = c_folder / "Genuine_Pareto_Clustered_withSR.csv"
LEGACY_CSV   = c_folder / "classic_superalloys_utf8_withSR.csv"
ENERGY_DICT  = PROJECT_FOLDER / "G-risks" / "Risk&Extraction_Energy_Dictionary.csv"

# ────────────────────────────────────────────────
# 1) Lecture des CSV
# ────────────────────────────────────────────────
df_new     = pd.read_csv(CSV_FILE, sep=";")
df_classic = pd.read_csv(LEGACY_CSV, sep=";", encoding="utf-8")

# Renommer SR → Supply Risk (si présent)
for df in (df_new, df_classic):
    if "SR" in df.columns:
        df.rename(columns={"SR": "Supply Risk"}, inplace=True)

# ────────────────────────────────────────────────
# 2) Lecture du dictionnaire d’énergie d’extraction
# ────────────────────────────────────────────────
try:
    df_energy = pd.read_csv(ENERGY_DICT, sep=";")
except FileNotFoundError:
    raise FileNotFoundError(f"Fichier énergie manquant : {ENERGY_DICT}")

if not {"Element", "Energy footprint (MJ/mol)"} <= set(df_energy.columns):
    raise ValueError("Le fichier Risk&Extraction_Energy_Dictionary.csv doit contenir "
                     "les colonnes 'Element' et 'Energy footprint (MJ/mol)'.")

energy_map = dict(zip(df_energy["Element"], df_energy["Energy footprint (MJ/mol)"]))

# ────────────────────────────────────────────────
# 3) Fonction pour extraire la composition
# ────────────────────────────────────────────────
def parse_composition(comp_str):
    """Retourne un dict {élément: fraction} à partir d'une string de composition."""
    if not isinstance(comp_str, str) or not comp_str:
        return {}
    pairs = re.findall(r"([A-Z][a-z]?)(\d*\.?\d+)", comp_str)
    return {el: float(qty) for el, qty in pairs}

# ────────────────────────────────────────────────
# 4) Calcul du Energy Footprint pondéré
# ────────────────────────────────────────────────
def compute_energy_footprint(df, comp_col="Normalized Composition"):
    energies = []
    for s in df[comp_col].fillna(""):
        comp = parse_composition(s)
        if not comp:
            energies.append(np.nan)
            continue
        e_sum, total = 0.0, 0.0
        for el, frac in comp.items():
            e_el = energy_map.get(el)
            if e_el is not None:
                e_sum += frac * e_el
                total += frac
        energies.append(e_sum / total if total > 0 else np.nan)
    df["Energy Footprint (MJ/mol)"] = energies

compute_energy_footprint(df_new)
compute_energy_footprint(df_classic)

# ────────────────────────────────────────────────
# 5) Harmonisation minimale pour le traçage
# ────────────────────────────────────────────────
rename_map = {
    "Density": "Density (g/cm³)",
    "Density (g/cm3)": "Density (g/cm³)",
    "Density (t/m3)": "Density (g/cm³)",
    "Melting Temperature (K)": "Melting Point (K)",
    "Formation Enthalpy (eV/atom)": "Formation Energy (eV/atom)",
    "LMP": "Creep (LMP)",
    "Melting_Temperature": "Melting Point (K)"
}
for df in (df_new, df_classic):
    df.rename(columns=rename_map, inplace=True)

# Indicateur “classique” ou “nouveau”
df_new["is_classic"] = False
df_classic["is_classic"] = True

# Fusion
df_all = pd.concat([df_new, df_classic], ignore_index=True)

# ────────────────────────────────────────────────
# 6) Préparation des labels et du tracé
# ────────────────────────────────────────────────
cluster_labels = df_all["Cluster"].astype(str).to_numpy()
origin_labels  = df_all["Origin"].astype(str).to_numpy()
classic_mask   = df_all["is_classic"].to_numpy()

plot_dir = c_folder / "Figures"
plot_dir.mkdir(exist_ok=True)

plot_ashby(
    df_all,
    x_col="Energy Footprint (MJ/mol)",
    y_col="Supply Risk",
    cluster_labels=cluster_labels,
    origin_labels=origin_labels,
    classic_mask=classic_mask,
    output_filename=plot_dir / "ashby_energy_footprint_vs_supply_risk.png",
    log_x=True
)

print("✅ Diagramme Ashby mis à jour avec 'Energy Footprint (MJ/mol)' et 'Supply Risk'.")
