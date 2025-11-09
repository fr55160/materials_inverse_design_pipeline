"""
descriptors.py – Calcul des descripteurs et prédiction des propriétés physiques

Ce module gère :
- Le calcul des descripteurs chimiques à partir d’une composition textuelle,
  en utilisant la fonction `compute_descriptors` du module Descriptors_Hephaistos.py.
- La prédiction des propriétés physiques via des modèles de machine learning (.pkl)
  pré-entraînés, spécifiques à chaque grandeur.
- L'utilisation de listes de features sélectionnées (par modèle) pour construire
  les vecteurs d’entrée.

Ce module est utilisé dans `preprocess.py` pour enrichir les données d'entrée avec
les propriétés physiques (nécessaires au CVAE).
"""

import os
import pickle
import pandas as pd
import numpy as np
from Descriptors_Hephaistos import compute_descriptors
from config_adaptative import MODEL_DIR, FEATURE_SEL_DIR

# Mapping des modèles
MODEL_FILES = {
    "Formation Enthalpy (eV/atom)": "Hephaistos_Formation_Enthalpy.pkl",
    "Decomposition Energy Per Atom MP (eV/atom)": "Hephaistos_Decomposition_Energy.pkl",
    "Melting Point (K)": "Hephaistos_Melting_Point.pkl",
    "Shear Modulus (GPa)": "Hephaistos_Shear_Modulus.pkl",
    "Bulk Modulus (GPa)": "Hephaistos_Bulk_Modulus.pkl",
    "Density (g/cm³)": "Hephaistos_Density.pkl",
    "Creep (LMP)": "Hephaistos_Creep.pkl",
    "log10(K_A)": "Hephaistos_HT_Oxidation.pkl"
}

CASE_MAPPING = {
    "Formation Enthalpy (eV/atom)": "case_01_Formation_Energy",
    "Decomposition Energy Per Atom MP (eV/atom)": "case_02_Decomposition_Energy",
    "Melting Point (K)": "case_03_Melting_Point",
    "Shear Modulus (GPa)": "case_04_Shear_Modulus",
    "Bulk Modulus (GPa)": "case_05_Bulk_Modulus",
    "Density (g/cm³)": "case_06_Density",
    "Creep (LMP)": "case_07_Creep_(LMP)",
    "log10(K_A)": "case_08_HT_Oxidation"
}

EXTERNAL_DEFAULTS = {
    "LMP": {"Creep strength Stress (Mpa)": 100, "1-mr": 0},
    "log10(K_A)": {"inv_T": 1.0 / 1273.0}
}

# Chargement des modèles et des features
def load_models():
    models = {}
    for prop, file in MODEL_FILES.items():
        with open(os.path.join(MODEL_DIR, file), "rb") as f:
            loaded = pickle.load(f)
        model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded

        features_csv = os.path.join(FEATURE_SEL_DIR, CASE_MAPPING[prop], "final_selection.csv")
        feature_list = pd.read_csv(features_csv)["selected_features"].tolist()

        models[prop] = {"model": model, "features": feature_list}
    return models

# Calcul descripteurs (wrapper autour de compute_descriptors)
def get_descriptors(comp_str):
    try:
        desc = compute_descriptors(comp_str)
        return desc
    except Exception as e:
        print(f"[!] Erreur descripteurs pour '{comp_str}' : {e}")
        return {}

# Prédiction physique à partir des descripteurs
def predict_properties(desc_dict, models):
    results = {}
    for prop, entry in models.items():
        model = entry["model"]
        feats = getattr(model, "feature_names_in_", entry["features"])
        externals = EXTERNAL_DEFAULTS.get(prop, {})
        x_input = [desc_dict.get(f, externals.get(f, 0)) for f in feats]
        df_input = pd.DataFrame([x_input], columns=feats)
        try:
            results[prop] = model.predict(df_input)[0]
        except Exception as e:
            print(f"[!] Erreur prédiction '{prop}' : {e}")
            results[prop] = 0.0
    return results

from config_adaptative import ELEMENTS

def vector_to_formula(vec, threshold=1e-3):
    """
    Convertit un vecteur de fractions atomiques (longueur = N_ELEMENTS)
    en formule chimique normalisée.
    """
    vec = np.asarray(vec, dtype=np.float32)  # Conversion explicite en float
    parts = []
    for frac, elem in zip(vec, ELEMENTS):
        try:
            frac = float(frac)  # Sécurité supplémentaire
            if frac > threshold:
                parts.append(f"{elem}{frac:.3f}")
        except ValueError:
            continue  # En cas de valeur non convertible
    return "".join(parts)
