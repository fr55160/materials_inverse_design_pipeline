#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_alloy_supply_risk.py  (édition: Normalized Composition + sep=';')

- Lit les fichiers CSV au séparateur ';'.
- Détecte les éléments présents pour chaque alliage à partir de la colonne
  'Normalized Composition' (ex: "Cr0.36Co0.29Mg0.17Cd0.04Ru0.14").
- Calcule SR_alloy selon (8-26) :
    SR_alloy = sum_i r_i  -  sum_{i<j} phi_bar(delta_ij) * min(r_i, r_j)
  où:
    r_i    = SR (ligne 'SR index' du fichier pre_p_star.csv)
    rho_ij = corrélation (matrice dans pre_p_star.csv)
    q_i    = phi_bar^{-1}(p_i), p_i = epsilon * r_i
    m_ij   = min(q_i, q_j), M_ij = max(q_i, q_j)
    delta  = (m_ij - M_ij * rho_ij) / sqrt(1 - rho_ij^2)

- Écrit un CSV de sortie avec une nouvelle colonne 'SR' (même sep=';').

Usage:
  python compute_alloy_supply_risk.py \
    --alloys from 'input_file' \
    --rho    "Sigma_E.csv"
    --sr     "SR_EU.csv"

Options:
  --epsilon  (défaut: 1e-3)
  --presence-threshold  (défaut: 0.0 ; seuil sur les fractions lues dans
                         Normalized Composition pour considérer qu'un élément
                         est présent)
  --inplace  (écrit dans le fichier d'entrée des alliages)
"""

from __future__ import annotations
import argparse
import math
import re
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent
input_file = PROJECT_FOLDER / "D-Beyond brute force" / "Genuine_Pareto_Clustered.csv"
output_file = c_folder / f"{input_file.stem}_withSR{input_file.suffix}"

# ===================== 0) Constantes =====================
EPSILON_DEFAULT: float = 1e-3
PRESENCE_THRESHOLD_DEFAULT: float = 0.0
RHO_EPS: float = 1e-12         # clip |rho| <= 1 - RHO_EPS pour stabilité
P_MIN: float = 1e-300          # bornes de sûreté pour p = epsilon * r
P_MAX: float = 1.0 - 1e-16

CSV_SEP = ";"                  # >>> Exigence: tous les CSV sont au sep=';'

# ===================== 1) Noyau gaussien =================
try:
    from scipy.stats import norm
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from math import erfc as _math_erfc
try:
    from numpy import special as _npsp
    _HAVE_NP_SPECIAL = hasattr(_npsp, "erfc")
except Exception:
    _HAVE_NP_SPECIAL = False

SQRT2 = math.sqrt(2.0)

def _erfc_any(x):
    """erfc compatible scalaires/ndarrays, sans np.erfc (fallback vectorisé)."""
    if isinstance(x, np.ndarray):
        if _HAVE_NP_SPECIAL:
            return _npsp.erfc(x)
        return np.vectorize(_math_erfc, otypes=[float])(x)
    return _math_erfc(float(x))

def phi_bar(x):
    """Survivance gaussienne standard: P[Z > x], Z ~ N(0,1)."""
    if isinstance(x, np.ndarray):
        return 0.5 * _erfc_any(x / SQRT2)
    return 0.5 * _erfc_any(x / SQRT2)

def phi_bar_inv(p: np.ndarray | float) -> np.ndarray | float:
    """Inverse de la survivance: phi_bar_inv(p). Utilise SciPy si dispo, sinon Acklam."""
    if _HAVE_SCIPY:
        return norm.isf(p)  # inverse survival function
    # ---- Fallback Acklam (inverse CDF) ----
    def _ppf(u: np.ndarray | float) -> np.ndarray | float:
        a = [ -3.969683028665376e+01,  2.209460984245205e+02,
              -2.759285104469687e+02,  1.383577518672690e+02,
              -3.066479806614716e+01,  2.506628277459239e+00 ]
        b = [ -5.447609879822406e+01,  1.615858368580409e+02,
              -1.556989798598866e+02,  6.680131188771972e+01,
              -1.328068155288572e+01 ]
        c = [ -7.784894002430293e-03, -3.223964580411365e-01,
              -2.400758277161838e+00, -2.549732539343734e+00,
               4.374664141464968e+00,  2.938163982698783e+00 ]
        d = [  7.784695709041462e-03,  3.224671290700398e-01,
               2.445134137142996e+00,  3.754408661907416e+00 ]
        plow, phigh = 0.02425, 1 - 0.02425

        if isinstance(u, np.ndarray):
            u = np.asarray(u, dtype=float)
            x = np.empty_like(u)

            mask_low = (u < plow)
            if np.any(mask_low):
                q = np.sqrt(-2*np.log(u[mask_low]))
                x[mask_low] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

            mask_mid = (u >= plow) & (u <= phigh)
            if np.any(mask_mid):
                q = u[mask_mid] - 0.5
                r = q*q
                x[mask_mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
                               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

            mask_high = (u > phigh)
            if np.any(mask_high):
                q = np.sqrt(-2*np.log(1.0 - u[mask_high]))
                x[mask_high] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
            return x

        u = float(u)
        if u < plow:
            q = math.sqrt(-2*math.log(u))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        if u <= phigh:
            q = u - 0.5
            r = q*q
            return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
        q = math.sqrt(-2*math.log(1.0 - u))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    if isinstance(p, np.ndarray):
        pp = np.clip(1.0 - p, 1e-300, 1.0 - 1e-16)
        return _ppf(pp)
    pp = min(max(1.0 - float(p), 1e-300), 1.0 - 1e-16)
    return _ppf(pp)

# ===================== 2) Lecture SR & rho (sep=';') =================
def read_rho_matrix(rho_csv: str) -> pd.DataFrame:
    """
    Lit le fichier Sigma_E.csv contenant la matrice de corrélation.
    - Première ligne : noms d'éléments.
    - Première colonne : noms d'éléments.
    - Format carré, sep=';'
    Retourne un DataFrame carré corr_df (index=colonnes=éléments).
    """
    df = pd.read_csv(rho_csv, sep=CSV_SEP, header=0, index_col=0)
    df = df.astype(float)

    # Harmonisation : ne garder que les intersections valides
    common = sorted(set(df.index).intersection(df.columns))
    rho = df.loc[common, common].copy()

    # Nettoyage
    np.fill_diagonal(rho.values, 1.0)
    rho.clip(lower=-1.0, upper=1.0, inplace=True)
    return rho

def read_sr_index(sr_csv: str) -> Dict[str, float]:
    """
    Lit le fichier SR_EU.csv :
    - 2 lignes : 'Element' et 'SR'
    - Colonnes = symboles chimiques
    Retourne un dict {élément: SR}.
    """
    df_sr = pd.read_csv(sr_csv, sep=CSV_SEP, header=None, index_col=0)

    if not {"Element", "SR"}.issubset(df_sr.index):
        raise ValueError("Le fichier SR_EU.csv doit contenir les lignes 'Element' et 'SR'.")

    elements = [str(e).strip() for e in df_sr.loc["Element"].tolist()]
    values = [float(v) for v in df_sr.loc["SR"].tolist()]

    return dict(zip(elements, values))


# ===================== 3) Lecture alliages (sep=';') =================
def read_alloys(alloys_csv: str) -> pd.DataFrame:
    """
    Lit le CSV des alliages (sep=';').
    Doit contenir une colonne 'Normalized Composition'.
    """
    df = pd.read_csv(alloys_csv, sep=CSV_SEP, engine="python")
    if "Normalized Composition" not in df.columns:
        raise ValueError("Colonne 'Normalized Composition' introuvable dans le fichier alliages.")
    return df

# ===================== 4) Parsing 'Normalized Composition' =================
# Exemple: "Nb0.03Ti0.05Cr0.23Mn0.04Fe0.03Co0.37Si0.07V0.17"
# Regex: symbole chimique ([A-Z][a-z]?) suivi d’un float
_RE_TERM = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")

def parse_normalized_composition(s: str) -> Dict[str, float]:
    """
    Parse une formule réduite: retourne {element: fraction} (floats).
    Ignore les espaces et lignes vides. Fractions non normalisées si l'entrée l'est déjà;
    on ne renormalise pas (seuil de présence utilisera ces valeurs).
    """
    if not isinstance(s, str):
        return {}
    s = s.strip()
    if not s:
        return {}
    comp: Dict[str, float] = {}
    for sym, val in _RE_TERM.findall(s):
        try:
            comp[sym] = float(val)
        except Exception:
            continue
    return comp

# ===================== 5) Calcul SR pour une liste d’éléments présents =========
def compute_SR_for_elements(
    elements_present: List[str],
    sr_map: Dict[str, float],
    rho_full: pd.DataFrame,
    epsilon: float
) -> float:
    """
    Calcule SR_alloy pour l’ensemble des éléments présents (aucun poids compositionnel).
    """
    if not elements_present:
        return 0.0

    # r_i: SR par élément (absents -> 0)
    r_vec = np.array([sr_map.get(e, 0.0) for e in elements_present], dtype=float)
    S1 = float(np.sum(r_vec))
    k = len(elements_present)
    if k < 2 or S1 == 0.0:
        return S1

    # rho_k: sous-matrice
    rho_k = np.zeros((k, k), dtype=float)
    for i, ei in enumerate(elements_present):
        for j, ej in enumerate(elements_present):
            try:
                rho_k[i, j] = float(rho_full.at[ei, ej])
            except Exception:
                rho_k[i, j] = 0.0
    np.clip(rho_k, -1.0 + RHO_EPS, 1.0 - RHO_EPS, out=rho_k)
    np.fill_diagonal(rho_k, 1.0 - RHO_EPS)

    # q_i = phi_bar_inv(p_i), p_i = epsilon * r_i
    p_vec = np.clip(epsilon * r_vec, P_MIN, P_MAX)
    q_vec = phi_bar_inv(p_vec)

    # m_ij & M_ij
    q_i = q_vec.reshape(-1, 1)
    q_j = q_vec.reshape(1, -1)
    m = np.minimum(q_i, q_j)
    M = np.maximum(q_i, q_j)

    denom = np.sqrt(1.0 - rho_k**2)
    delta = (m - M * rho_k) / denom
    phi_delta = phi_bar(delta)

    r_i = r_vec.reshape(-1, 1)
    r_j = r_vec.reshape(1, -1)
    rmin = np.minimum(r_i, r_j)

    upper = np.triu(np.ones_like(phi_delta, dtype=bool), k=1)
    S2 = float(np.sum(phi_delta[upper] * rmin[upper]))
    return S1 - S2

# ===================== 6) Pipeline principal ===================================
def compute_alloys_SR(
    alloys_csv: str,
    rho_csv: str,
    sr_csv: str,
    epsilon: float = EPSILON_DEFAULT,
    presence_threshold: float = PRESENCE_THRESHOLD_DEFAULT,
    inplace: bool = False
) -> str:
    """
    Lit SR & rho (sep=';'), lit alliages (sep=';'),
    parse 'Normalized Composition' pour chaque ligne, calcule SR, écrit 'SR'.
    """
    sr_map = read_sr_index(sr_csv)
    rho = read_rho_matrix(rho_csv)
    df_alloys = read_alloys(alloys_csv)

    SR_vals: List[float] = []
    missing_warned = set()

    for idx, s in enumerate(df_alloys["Normalized Composition"].astype(str).tolist()):
        comp = parse_normalized_composition(s)
        # Filtrer par présence (fraction > seuil)
        els_present = [e for e, frac in comp.items() if frac > presence_threshold]

        # Avertissement (une fois par élément) si e n'a pas de SR ni rho
        for e in els_present:
            if e not in sr_map and e not in rho.index and e not in missing_warned:
                warnings.warn(
                    f"Élément '{e}' présent mais sans SR/rho connu. Traité avec r=0, rho=0."
                )
                missing_warned.add(e)

        SR_vals.append(
            compute_SR_for_elements(els_present, sr_map, rho, epsilon)
        )

    out_df = df_alloys.copy()
    out_df["SR"] = SR_vals

    out_df.to_csv(output_file, sep=CSV_SEP, index=False)

    sr_series = pd.Series(SR_vals)
    print(f"[OK] Alliages traités : {len(SR_vals)}")
    print(f"     SR min/median/max : {sr_series.min():.6g} / {sr_series.median():.6g} / {sr_series.max():.6g}")
    if inplace:
        print(f"[OK] Écrasé en place : {output_file}")
    else:
        print(f"[OK] Écrit : {output_file}")

    return output_file

# ===================== 7) CLI ==================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calcule la colonne SR (supply risk) pour chaque alliage à partir de 'Normalized Composition'."
    )
    parser.add_argument("--alloys", type=str,
                        default=input_file,
                        help="CSV des alliages (doit contenir 'Normalized Composition'; sep=';').")
    parser.add_argument("--rho", type=str,
                        default=c_folder / "Sigma_E.csv",
                        help="Fichier CSV contenant la matrice de corrélation entre éléments (sep=';').")
    parser.add_argument("--sr", type=str,
                        default=c_folder / "SR_EU.csv",
                        help="Fichier CSV contenant les indices de Supply Risk en lignes ('Element' et 'SR').")
    parser.add_argument("--epsilon", type=float, default=EPSILON_DEFAULT,
                        help=f"Facteur p_i = epsilon * r_i (défaut {EPSILON_DEFAULT}).")
    parser.add_argument("--presence-threshold", type=float, default=PRESENCE_THRESHOLD_DEFAULT,
                        help=f"Seuil de présence sur les fractions de 'Normalized Composition' (défaut {PRESENCE_THRESHOLD_DEFAULT}).")
    parser.add_argument("--inplace", action="store_true",
                        help="Écrit directement dans le fichier d’entrée des alliages.")

    args = parser.parse_args()

    try:
        compute_alloys_SR(
            alloys_csv=args.alloys,
            rho_csv=args.rho,
            sr_csv=args.sr,
            epsilon=args.epsilon,
            presence_threshold=args.presence_threshold,
            inplace=args.inplace
        )
    except Exception as e:
        print(f"[ERREUR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
