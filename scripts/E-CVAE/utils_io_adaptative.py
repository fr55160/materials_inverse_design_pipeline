"""
utils_io_adaptative.py

Fonctions utilitaires pour le pipeline adaptative :
- parse_composition  : chaîne → vecteur de fractions
- clr_transformation : barycentre → log-ratio centré
- transform_g         : vecteur gᵢ → vecteur de condition Y
- vector_to_formula  : vecteur de fractions → formule chimique
- softmin            : soft-minimum pondéré (pour certaines modes)
"""

import re
import numpy as np
from config_adaptative import ELEMENTS, N_ELEMENTS, CLR_EPSILON

def parse_composition(comp_str: str) -> np.ndarray:
    vect = np.zeros(N_ELEMENTS, dtype=float)
    for elem, frac in re.findall(r'([A-Z][a-z]*)([0-9\.]+)', comp_str):
        if elem in ELEMENTS:
            vect[ELEMENTS.index(elem)] = float(frac)
    return vect

def clr_transformation(x_array: np.ndarray, epsilon: float = CLR_EPSILON) -> np.ndarray:
    x = x_array + epsilon
    sums = x.sum(axis=1, keepdims=True)
    log_x = np.log(x / sums)
    return log_x - log_x.mean(axis=1, keepdims=True)

def softmin(g_vec: np.ndarray, lambda_: float = 10.0) -> float:
    g = np.array(g_vec, dtype=float)
    w = np.exp(-lambda_ * g)
    return np.sum(g * w) / np.sum(w)

def transform_g(g_vec, mode: str = "classification") -> np.ndarray:
    g = np.array(g_vec, dtype=float)
    if mode == "g1":
        return g[:1]
    elif mode == "g1g2":
        return g[:2]
    elif mode == "S&P":
        return np.array([np.sum(g), np.log10(np.prod(g) + 1e-100), softmin(g)])
    elif mode == "identical":
        return g
    elif mode == "classification":
        lmin = 1 if np.min(g) >= 0.33 else 0
        lsum = 1 if np.sum(g) >= 5.0 else 0
        #llog = 1 if np.sum(np.log(g + 1e-8)) >= -9.0 else 0
        #return np.array([lmin, lsum, llog])
        return np.array([lmin, lsum])
    elif mode == "classification+":
        lmin = 1 if np.min(g) >= 0.4 else 0
        lsum = 1 if np.sum(g) >= 5.0 else 0
        llog = 1 if np.sum(np.log(g + 1e-8)) >= -10.0 else 0
        perf = 0 if ((g[0] < 0.5 or g[9] < 0.5) or (g[1] < 0.6 or g[2] < 0.6)) else 1
        return np.array([lmin, lsum, llog, perf])
    else:
        raise ValueError(f"Mode inconnu pour transform_g: {mode}")

def vector_to_formula(vect: np.ndarray, precision: int = 3) -> str:
    v = np.array(vect, dtype=float)
    if v.sum() > 0:
        v = v / v.sum()
    parts = []
    for i, frac in enumerate(v):
        if frac > 1e-8:
            parts.append(f"{ELEMENTS[i]}{round(frac, precision)}")
    return "".join(parts)
