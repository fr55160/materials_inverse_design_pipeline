#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flag_forbidden_and_energy.py  (bool + NaN si manque d'énergie)

- Lit un CSV (sep=';') avec la colonne 'Normalized Composition'.
- Ajoute :
    * 'Forbidden_elem' (bool) : True si ≥1 élément interdit est présent.
    * 'Energy_Footprint(MJ/mol)' : somme(x_i*EF_i) ou NaN si une EF manque.
    * 'Energy_Footprint_incRecy' : somme(x_i*EFrecy_i) ou NaN si une EF manque.
"""

import re
import pandas as pd
import numpy as np

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# ===================== Paramètres =====================
source_filename = "Genuine_Pareto_Clustered.csv"
INPUT_CSV  = PROJECT_FOLDER / "D-Beyond brute force" / source_filename
output_filename = (
    source_filename[:-4] + "_with_flags_energy.csv"
    if source_filename.lower().endswith(".csv")
    else source_filename + "_with_flags_energy.csv"
)
OUTPUT_CSV = c_folder / output_filename
SEP = ";"           # CSV en sep=';'

# ===================== Éléments interdits (dédupliqués) =====================
_forbidden_raw = """
Po, At, Rn, Fr, Ra, Lr, Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, Og,
Pm, Ac, Pa, Am, Bk, Cf, Es, Fm, Md, No, Np, Tc, Th, Pu, U, Pb, Cd, Hg, Tl, Au, Pt,
Ru, Rh, Pd, Os, Ir, Os, Ir, Rh, Sc, Ru, Re, Hf, Rb, Cs, Sc, Re, Te, Re, Ga, In, Cd,
Ru, Ir, Os, Ir, Rh, Ru, Ga, Hg, Y, Nb, W, Sc, Pt, Hg, Y, Bi, Nb, V, Ge, Co, Mg
"""
FORBIDDEN_SET = {s.strip() for s in _forbidden_raw.replace("\n", "").split(",") if s.strip()}

# ===================== Tables d'énergie (MJ/mol) =====================
ENERGY_FOOTPRINT = {
    "Os": 12298.3695, "Ir": 9149.672, "Rh": 134297.55, "Sc": 2465.8317732,
    "Ru": 2466.108, "Re": 1613.50965, "Hf": 197.23145, "Pt": 218494.08,
    "Pd": 19687.7, "Ga": 182.67426, "Te": 15.95, "In": 299.6802,
    "Ta": 779.8945, "Hg": 22.46608, "Au": 83612.3085405, "Y": 130.69182,
    "Bi": 30.302158, "Cd": 10.26874485, "Ag": 747.526626, "As": 0.37461,
    "Nb": 171.8761, "V": 189.2476725, "W": 106.25952, "Sb": 16.07232,
    "Ge": 71.97633, "Co": 124.9383734, "Mo": 14.6339, "Sn": 20.41812,
    "Mg": 9.3695775, "Zr": 146.868064, "Ti": 28.1218625, "Ni": 12.1201871,
    "Pb": 5.698, "Si": 3.63707225, "Zn": 2.968252, "Mn": 4.1368347885,
    "Cu": 6.2433945, "Cr": 25.1661124, "Al": 5.3828169507, "Fe": 1.1894985,
    "H": 0.0796385, "He": None, "Li": 4.07378, "Be": 57.5739,
    "B": 0.5518505, "C": 2.636195, "N": 0.0844803, "O": 0.2406495,
    "F": 3.74103, "Ne": None, "Na": 2.3162425, "P": 7.18504,
    "S": 0.0869097, "Cl": None, "Ar": None, "K": None,
    "Ca": 5.93184, "Se": 6.249684, "Br": None, "Kr": None,
    "Rb": 5.93184, "Sr": 5.93184, "Tc": None, "I": None,
    "Xe": None, "Cs": 5.93184, "Ba": 5.93184, "La": 37.6419,
    "Ce": None, "Pr": 90.38735, "Nd": 12.24979, "Pm": None,
    "Sm": 177.472, "Eu": 3146.4, "Gd": 208.4225, "Tb": 1460.291,
    "Dy": 229.125, "Ho": 319.0815, "Er": 494.3715, "Tm": 4551.855,
    "Yb": 971.395, "Lu": 7875, "Tl": 485.45, "Po": None,
    "At": None, "Rn": None, "Fr": None, "Ra": None,
    "Ac": None, "Th": 0.0, "Pa": None, "U": 308.21
}
ENERGY_FOOTPRINT_INC_RECY = {
    "Os": 523.1325, "Ir": 419.0396, "Rh": 2773.4245, "Sc": None,
    "Ru": 132.90705, "Re": 111.16737, "Hf": 22.31125, "Pt": 4682.016,
    "Pd": 650.7583, "Ga": 16.803243, "Te": 3.03688, "In": 27.5568,
    "Ta": 63.603925, "Hg": 4.392921, "Au": 2265.1155435, "Y": None,
    "Bi": 5.56932766, "Cd": 2.10770625, "Ag": 54.3655728, "As": None,
    "Nb": 17.18761, "V": 15.97016025, "W": 14.026992, "Sb": 3.025736,
    "Ge": None, "Co": 12.081304975, "Mo": 2.658092, "Sn": 3.608784,
    "Mg": 1.3635105, "Zr": 15.1885296, "Ti": 3.69772575, "Ni": 2.04839966,
    "Pb": 1.562288, "Si": None, "Zn": 0.722449, "Mn": 0.889996329,
    "Cu": 1.2613881, "Cr": 3.465540065, "Al": 0.89713615845, "Fe": 0.34679745,
    "H": None, "He": None, "Li": 0.535768, "Be": 4.27074,
    "B": None, "C": None, "N": None, "O": None,
    "F": None, "Ne": None, "Na": None, "P": None,
    "S": None, "Cl": None, "Ar": None, "K": None,
    "Ca": None, "Se": 1.330476, "Br": None, "Kr": None,
    "Rb": None, "Sr": None, "Tc": None, "I": None,
    "Xe": None, "Cs": None, "Ba": None, "La": None,
    "Ce": None, "Pr": None, "Nd": None, "Pm": None,
    "Sm": None, "Eu": None, "Gd": None, "Tb": None,
    "Dy": None, "Ho": None, "Er": None, "Tm": None,
    "Yb": None, "Lu": None, "Tl": 45.6834, "Po": None,
    "At": None, "Rn": None, "Fr": None, "Ra": None,
    "Ac": None, "Th": None, "Pa": None, "U": 35.224
}

# ===================== Parsing de la composition =====================
RE_TERM = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")

def parse_normalized_composition(s: str) -> dict:
    """Retourne {élément: fraction} à partir d'une chaîne 'Normalized Composition'."""
    if not isinstance(s, str):
        return {}
    s = s.strip()
    if not s:
        return {}
    comp = {}
    for sym, val in RE_TERM.findall(s):
        try:
            comp[sym] = float(val)
        except Exception:
            pass
    return comp

# ===================== Logiques demandées =====================
def has_forbidden_element(comp: dict) -> bool:
    return any(el in FORBIDDEN_SET for el in comp.keys())

def energy_sum_or_nan(comp: dict, table: dict) -> float:
    """
    Somme(x_i * EF_i) si TOUTES les EF requises sont connues (non-None) ;
    sinon retourne NaN.
    - 'Inconnue' = clé absente de la table OU valeur None.
    """
    total = 0.0
    for el, x in comp.items():
        key = el.strip()
        if key not in table or table[key] is None:
            return float("nan")
        total += x * float(table[key])
    return total

# ===================== Main =====================
def main():
    df = pd.read_csv(INPUT_CSV, sep=SEP, engine="python")
    if "Normalized Composition" not in df.columns:
        raise ValueError("Colonne 'Normalized Composition' introuvable dans le CSV d'entrée.")

    comps = df["Normalized Composition"].astype(str).tolist()

    forb = []
    ef_norecy = []
    ef_recy = []

    for s in comps:
        comp = parse_normalized_composition(s)
        forb.append(has_forbidden_element(comp))
        ef_norecy.append(energy_sum_or_nan(comp, ENERGY_FOOTPRINT))
        ef_recy.append(energy_sum_or_nan(comp, ENERGY_FOOTPRINT_INC_RECY))

    out = df.copy()
    out["Forbidden_elem"] = pd.Series(forb, dtype="boolean")   # booléen (Excel FR -> VRAI/FAUX)
    out["Energy_Footprint(MJ/mol)"] = ef_norecy                 # float ou NaN
    out["Energy_Footprint_incRecy"] = ef_recy                   # float ou NaN

    out_path = OUTPUT_CSV
    out.to_csv(out_path, sep=SEP, index=False)
    print(f"[OK] Écrit : {out_path}")

if __name__ == "__main__":
    main()
