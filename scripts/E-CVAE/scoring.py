"""
scoring.py – Calcul des scores g₁ à g₁₀ pour la qualité des matériaux

Ce module fournit une fonction `compute_scores(desc_dict, phys_dict)` qui prend en
entrée :
- un dictionnaire de descripteurs chimiques,
- un dictionnaire de propriétés physiques,

et renvoie un tableau numpy (shape: (10,)) contenant les scores `g₁ à g₁₀`,
chacun étant une valeur normalisée entre 0 et 1.

Ces scores sont conçus pour quantifier les qualités mécaniques, thermiques et chimiques
des matériaux, et sont utilisés comme **conditions de sortie (`Y`)** dans le CVAE.

Ce module est appelé dans `preprocess.py` après prédiction des propriétés physiques.
"""

import numpy as np

# Fonctions de filtrage
def GaussianWindow(xmin, xmax, xeval):
    xcenter = (xmin + xmax) / 2
    sigma = (xmax - xmin) / 2.355
    return np.exp(-((xeval - xcenter) ** 2) / (2 * sigma ** 2))

def SigmoidFilter(nook, ok, xeval):
    xcenter = (nook + ok) / 2
    return 1 / (1 + np.exp(-(xeval - xcenter) / (ok - nook)))

# Calcul des scores g1 à g10
def compute_scores(desc, phys):
    """
    Calcule les 10 estimateurs g₁ à g₁₀ à partir des descripteurs et propriétés.

    Entrées :
    - desc : dict contenant les descripteurs chimiques
    - phys : dict contenant les propriétés physiques prédites

    Sortie :
    - np.array shape (10,) : scores entre 0 et 1
    """
    g = []
    g.append(SigmoidFilter(20, 26, phys.get("Creep (LMP)", 0)))                            # g1
    g.append(SigmoidFilter(1200, 1800, phys.get("Melting Point (K)", 0)))                 # g2
    g.append(SigmoidFilter(-9, -6, -phys.get("Density (g/cm³)", 0)))                      # g3
    g.append(GaussianWindow(6.87, 8, desc.get("avg_VEC", 0)))                             # g4
    g.append(GaussianWindow(0.03, 0.066, desc.get("delta", 0)))                           # g5
    g.append(SigmoidFilter(-0.6, -0.5, -phys.get("G/B ratio", 0)))                        # g6
    g.append(GaussianWindow(-0.156, 0.052, phys.get("Formation Enthalpy (eV/atom)", 0)))  # g7
    g.append(SigmoidFilter(1.0, 1.2, phys.get("Omega", 0)))                               # g8
    g.append(GaussianWindow(1.31, 2.32, desc.get("stoich_entropy", 0)))                   # g9
    g.append(SigmoidFilter(-0.25, 0.25, -phys.get("log10(K_A)", 0)))                      # g10
    return np.array(g)
