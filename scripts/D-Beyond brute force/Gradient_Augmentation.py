# This script searches for optimal 5-element alloy compositions by combining elemental properties
# (melting point, radius, valence, etc.) into a multi-objective function. It uses finite-difference
# gradient ascent to maximise this objective over stoichiometric fractions, filters physically valid
# results, and saves the best candidate compositions to a CSV file.

import math
import itertools
import numpy as np
import pandas as pd
from mendeleev import element
import logging

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent
output_file = "gradient_augmentation_results.csv"

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
DEBUG = True   # Passez à False pour utiliser les 39 éléments complets

# En mode DEBUG, on restreint à 10 éléments pour tests rapides
if DEBUG:
    elements = ["Al", "Cu", "Au", "Ta", "Pd", "Zn", "Ni", "Mo", "Ir", "Ti"]
else:
    elements = [
        "Al", "Cu", "Au", "Ta", "Pd", "Zn", "Ni", "Mo", "Ir", "Ti",
        "V", "Y", "Zr", "W", "Co", "Pt", "Re", "Rh", "Ru", "Fe",
        "Si", "Ga", "Sb", "As", "Nb", "Te", "Sc", "Hf", "Ag", "Hg",
        "In", "Mn", "Cr", "Mg", "Bi", "Ge", "Sn", "Cd", "Pb"
    ]

MAX_STEPS = 50          # Nombre max. d'itérations de montée de gradient
STEP_NORM = 0.04        # Norme du vecteur de mise à jour à chaque pas
EPS_NUM = 1e-6          # Pas numérique pour gradient approché

# Logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# -------------------------------------------------------------------
# CHARGEMENT DES PROPRIÉTÉS ÉLÉMENTAIRES
# -------------------------------------------------------------------
def load_element_data(elem_list):
    data = {}
    for symb in elem_list:
        e = element(symb)
        # ––– Nombre d'électrons de valence robuste –––
        n_valence = None
        ec = getattr(e, "ec", None)
        if ec is not None and hasattr(ec, "get_valence"):
            try:
                val_cfg = ec.get_valence()
                if val_cfg is not None and hasattr(val_cfg, "conf") and val_cfg.conf:
                    n_valence = sum(val_cfg.conf.values())
                else:
                    n_valence = getattr(e, "nvalence", None)
            except Exception:
                n_valence = getattr(e, "nvalence", None)
        else:
            n_valence = getattr(e, "nvalence", None)

        # ––– Melting point –––
        mp = getattr(e, "melting_point", None)

        # ––– Rayon atomique –––
        ar = getattr(e, "atomic_radius", None)
        if ar is None:
            ar = getattr(e, "metallic_radius", None)
        if ar is None:
            ar = getattr(e, "atomic_radius_empirical", None)

        # ––– Masse atomique –––
        aw = getattr(e, "atomic_weight", None)

        data[symb] = {
            "n_valence":    n_valence,
            "melting_point": mp,
            "atomic_radius": ar,
            "atomic_weight": aw
        }
        logging.debug(f"{symb}: n_valence={n_valence}, mp={mp}, ar={ar}, aw={aw}")
        
    print("Toutes les données relatives aux éléments présents ont été collectées.")
    return data


# -------------------------------------------------------------------
# CALCUL DE L'OBJECTIF (f1…f5) ET DE LA SOMME DES LOGS
# -------------------------------------------------------------------
def compute_objective(x, props):
    """
    x : array shape (5,) des stœchiométries qui somment à 1
    props : liste de 5 dicts d'un même combo d'éléments
    renvoie (obj, f_vec) où obj = sum_i log(f_i), f_vec = [f1…f5]
    """

    # 1) reject non-physical compositions immediately
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        # on renvoie un objectif -inf et des f_vec à NaN
        return -np.inf, [np.nan]*5
    
    # extraire vecteurs de propriétés
    Tm   = np.array([p["melting_point"] for p in props], dtype=float)
    R    = np.array([p["atomic_radius"] for p in props], dtype=float)
    AW   = np.array([p["atomic_weight"] for p in props], dtype=float)
    NV   = np.array([p["n_valence"] for p in props], dtype=float)

    # --- f1 : moyenne pondérée des melting points
    f1 = float(np.dot(x, Tm))

    # --- f2 : entropie stœchiométrique
    f2 = -float(np.sum(x * np.log(x)))

    # --- f3 : inverse de la moyenne pondérée de (AW / R^3)
    A = AW / (R**3)
    avg_A = float(np.dot(x, A))
    f3 = 1.0 / avg_A

    # --- f4 : fonction sur le nombre moyen d'électrons de valence
    xv = float(np.dot(x, NV))
    u = max(0.0, xv - 8.0)
    v = max(0.0, 6.87 - xv)
    f4 = math.exp(-6.0 * (u + v))

    # --- f5 : radius mismatch
    avg_R = float(np.dot(x, R))
    var_R = float(np.dot(x, (R - avg_R)**2))
    std_R = math.sqrt(var_R)
    delta = std_R / avg_R if avg_R else np.nan
    u2 = max(0.0, delta - 0.066)
    v2 = max(0.0, 0.03 - delta)
    f5 = math.exp(-150.0 * (u2 + v2))

    f_vec = [f1, f2, f3, f4, f5]
    # si un fi est nan ou <= 0, l'objectif est invalide
    if any(not np.isfinite(fi) or fi <= 0 for fi in f_vec):
        return -np.inf, f_vec

    return sum(math.log(fi) for fi in f_vec), f_vec


# -------------------------------------------------------------------
# GRADIENT APPROCHÉ (différences finies)
# -------------------------------------------------------------------
def approximate_gradient(w, props):
    grad = np.zeros_like(w)
    # construire x_full initial
    for j in range(len(w)):
        w_plus  = w.copy(); w_minus = w.copy()
        w_plus[j]  += EPS_NUM
        w_minus[j] -= EPS_NUM

        # reconstituer x_full pour les 5 éléments
        x5p = 1.0 - w_plus.sum()
        x5m = 1.0 - w_minus.sum()
        x_p = np.concatenate([w_plus,  [x5p]])
        x_m = np.concatenate([w_minus, [x5m]])

        obj_p, _ = compute_objective(x_p, props)
        obj_m, _ = compute_objective(x_m, props)
        grad[j] = (obj_p - obj_m) / (2 * EPS_NUM)
    return grad


# -------------------------------------------------------------------
# BOUCLE SUR LARGE_SET ET MONTÉE DE GRADIENT
# -------------------------------------------------------------------
def search_alloys(element_data):
    eligible = []
    combos = list(itertools.combinations(element_data.keys(), 5))
    logging.info(f"Nombre de combinaisons à étudier : {len(combos)}")

    for combo in combos:
        props = [element_data[el] for el in combo]
        # vérifier que toutes les propriétés sont définies
        if any(None in p.values() for p in props):
            continue

        # stœchiométrie initiale (4 libres, 5ᵉ = 1−somme)
        w = np.array([0.2, 0.2, 0.2, 0.2], dtype=float)
        valid = True

        for step in range(MAX_STEPS):
            x_full = np.concatenate([w, [1.0 - w.sum()]])
            # arrêt si composition non physique
            if np.any(x_full <= 0):
                valid = False
                break

            # gradient approché
            grad = approximate_gradient(w, props)
            norm_g = np.linalg.norm(grad)
            if norm_g == 0:
                break

            # mise à jour le long de la direction du gradient, norme fixe
            direction = grad / norm_g
            w = w + direction * STEP_NORM

        # validation finale
        # composition complète
        x_final = np.concatenate([w, [1.0 - w.sum()]])

        # 1) on récupère f2 (entropie stœchiométrique) via compute_objective
        obj_final, f_vec_final = compute_objective(x_final, props)
        f2 = f_vec_final[1]

        # 2) on recalcule la moyenne d'électrons de valence xv
        NV = np.array([p["n_valence"] for p in props], dtype=float)
        xv = float(np.dot(x_final, NV))

        # 3) on recalcule le radius mismatch delta
        R = np.array([p["atomic_radius"] for p in props], dtype=float)
        avg_R   = float(np.dot(x_final, R))
        std_R   = float(np.sqrt(np.dot(x_final, (R - avg_R)**2)))
        delta   = std_R / avg_R if avg_R else np.nan

        # on impose les nouvelles bornes
        if f2    <= 1.2 \
        or xv    <  6.0 or xv   >  9.0 \
        or delta < 0.022 or delta>0.075:
            valid = False
        if valid and np.all(x_final > 0):
            # formater en chaîne « Fe0.100Cr0.100… »
            comp_str = "".join(f"{el}{x_final[i]:.3f}"
                                for i, el in enumerate(combo))
            # calculer la somme des log(f_i) pour la composition finale
            obj_final, _ = compute_objective(x_final, props)
            # stocker un tuple (composition, objective)
            eligible.append((comp_str, obj_final))
            print(f"Eligible : {comp_str}")

    return eligible


# -------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# -------------------------------------------------------------------
def main():
    data = load_element_data(elements)
    eligible_comps = search_alloys(data)

    # Sauvegarde en CSV
    df = pd.DataFrame(eligible_comps, columns=["composition", "objective"])
    df.to_csv(c_folder / output_file, index=False, sep=";")
    logging.info(f"{len(eligible_comps)} compositions éligibles enregistrées dans {output_file}")


if __name__ == "__main__":
    main()
