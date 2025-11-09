"""
François Rousseau, 20 mai 2025
Script complet pour la prédiction de grandeurs physiques et le calcul d'estimateurs pour des matériaux HEA. 
- Traitement par morceaux (chunks) pour gérer de grandes bases de données.
- Calcul parallèle des descripteurs.
- Prédiction des propriétés physiques via des modèles préalablement entraînés.
- Calcul d'estimateurs (valeurs entre 0 et 1) évaluant la qualité du matériau pour diverses contraintes.
- Fusion des résultats et export dans un nouveau fichier CSV.
"""
import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
from Descriptors_Hephaistos import compute_descriptors

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent
alloy_list_input = "Alloys_list.csv"
alloy_list_output = "Alloys_list_allfeats.csv"

BASE_PATH = PROJECT_FOLDER
MODEL_DIR = os.path.join(BASE_PATH, "B-supervised learning")
FEATURE_SEL_DIR = os.path.join(MODEL_DIR, "feature_selection_results")

# --------------------------
# Configuration
# --------------------------
N_WORKERS = cpu_count()                   # Tous les coeurs

# ---------------------------------
# Utilitaires de calcul
# ---------------------------------
def round_value(val):
    if isinstance(val, (float, int)):
        return float(format(val, ".3g"))
    return val

def safe_compute(formula, verbose=False):
    try:
        desc = compute_descriptors(formula)
        return {k: round_value(v) for k, v in desc.items()}
    except Exception as e:
        if verbose:
            print(f"Erreur avec {formula}: {e}")
        return None

def process_chunk(chunk):
    with Pool(N_WORKERS) as pool:
        results = pool.map(partial(safe_compute, verbose=True),
                           chunk["Normalized Composition"])
    return [r for r in results if r is not None]

def estimate_time(start, done, total):
    elapsed = time.time() - start
    prog = done / total if total else 0
    if prog > 0:
        rem = elapsed * (1 - prog) / prog
        return timedelta(seconds=int(rem))
    return timedelta(0)

# Estimateurs

def GaussianWindow(xmin, xmax, xeval):
    xcenter = (xmin + xmax) / 2
    sigma = (xmax - xmin) / 2.355
    return np.exp(-((xeval - xcenter) ** 2) / (2 * sigma ** 2))

def SigmoidFilter(nook, ok, xeval):
    xcenter = (nook + ok) / 2
    return 1 / (1 + np.exp(-(xeval - xcenter) / (ok - nook)))

# --------------------------
# Chargement des modèles et features
# --------------------------
model_files = {
    "Formation Energy (eV/atom)": "Hephaistos_Formation_Enthalpy.pkl",
    "Decomposition Energy Per Atom MP (eV/atom)": "Hephaistos_Decomposition_Energy.pkl",
    "Melting_Temperature": "Hephaistos_Melting_Point.pkl",
    "Shear Modulus (GPa)": "Hephaistos_Shear_Modulus.pkl",
    "Bulk Modulus (GPa)": "Hephaistos_Bulk_Modulus.pkl",
    "Density": "Hephaistos_Density.pkl",
    "LMP": "Hephaistos_Creep.pkl",
    "Log_10(K_A)": "Hephaistos_HT_Oxidation.pkl"
}

# Mapping vers dossiers de sélection de features
case_mapping = {
    "Formation Energy (eV/atom)": "case_01_Formation_Energy",
    "Decomposition Energy Per Atom MP (eV/atom)": "case_02_Decomposition_Energy",
    "Melting_Temperature": "case_03_Melting_Point",
    "Shear Modulus (GPa)": "case_04_Shear_Modulus",
    "Bulk Modulus (GPa)": "case_05_Bulk_Modulus",
    "Density": "case_06_Density",
    "LMP": "case_07_Creep_(LMP)",
    "Log_10(K_A)": "case_08_HT_Oxidation"
}

models = {}
# Valeurs fixes pour les prédictions
HT_TEMP = 1273.0  # 1000 °C en K
EXTERNAL_DEFAULTS = {
    "LMP": {
        "Creep strength Stress (Mpa)": 100, # valeur par défaut pour la contrainte
        "1-mr": 0 #tous les HEA sont 100% métalliques
    },
    "Log_10(K_A)": {
        "inv_T": 1.0 / HT_TEMP
    }
}
for prop, fname in model_files.items():
    # Chargement du modèle
    mdl_path = os.path.join(MODEL_DIR, fname)
    with open(mdl_path, "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, dict) and "model" in loaded:
        mdl = loaded["model"]
    else:
    # sinon on suppose que c'est déjà un XGBRegressor
        mdl = loaded
    #mdl.set_params(n_jobs=cpu_count())
    mdl.set_params(
        n_jobs=cpu_count(),
        tree_method="hist",
        device="cuda",
        predictor="gpu_predictor"  # optionnel, pour la prédiction GPU
    )
    # Chargement de la liste de features spécifiques
    feat_file = os.path.join(FEATURE_SEL_DIR,
                             case_mapping[prop],
                             "final_selection.csv")
    dff = pd.read_csv(feat_file, sep=",", engine='python')
    feature_list = dff["selected_features"].tolist()
    models[prop] = {"model": mdl, "features": feature_list}



# --------------------------
# Prédiction des propriétés
# --------------------------
def predict_properties(desc):
    preds = {}
    for prop, info in models.items():
        mdl = info["model"]
        # 1) Déterminer la liste EXACTE de features que le modèle attend
        if hasattr(mdl, "feature_names_in_"):
            feats = list(mdl.feature_names_in_)
        else:
            # fallback sur le CSV, mais en général XGBRegressor en possède toujours
            feats = info["features"]

        # 2) Récupérer les defaults externes pour ce modèle
        externals = EXTERNAL_DEFAULTS.get(prop, {})

        # 3) Construire la ligne de données dans l’ordre exact de `feats`
        values = [
            # si c’est une feature native de desc → desc[fname]
            # sinon si c’est une feature externe attendue → externals[fname]
            # sinon (au cas où) → 0
            desc.get(fname, externals.get(fname, 0))
            for fname in feats
        ]

        # 4) DataFrame nommée pour la prédiction
        df_in = pd.DataFrame([values], columns=feats)
        preds[prop] = mdl.predict(df_in)[0]
    return preds

# --------------------------
# Main
# --------------------------
def main():
    in_file = os.path.join(BASE_PATH, "C-HEA",alloy_list_input)
    out_file = os.path.join(BASE_PATH, "C-HEA",alloy_list_output)
    temp_file = os.path.join(BASE_PATH, "C-HEA","temp_enriched.csv")

    if not os.path.exists(in_file):
        print(f"Fichier CSV introuvable : {in_file}")
        return

    # Estimation dynamique du nombre de lignes et taille de chunk
    total_rows = sum(1 for _ in pd.read_csv(in_file, sep=";", engine='python'))
    global CHUNKSIZE
    CHUNKSIZE = max(20000, int(total_rows / N_WORKERS))

    # Préparation
    first = pd.read_csv(in_file, sep=";", nrows=1)
    if "Normalized Composition" not in first.columns:
        print('Colonne "Normalized Composition" manquante')
        return

    start = time.time()
    total_chunks = sum(1 for _ in pd.read_csv(in_file, sep=";", chunksize=CHUNKSIZE, engine='python'))
    element_counts = Counter()

    for i, chunk in enumerate(pd.read_csv(in_file, sep=";", engine='python', chunksize=CHUNKSIZE)):
        # Comptage éléments
        for comp in chunk["Normalized Composition"]:
            element_counts.update(set(re.findall(r'[A-Z][a-z]?', comp)))
        # Descripteurs
        desc_list = process_chunk(chunk)
        df_desc = pd.DataFrame(desc_list)
        if df_desc.empty:
            continue
        # Prédictions et estimateurs
        phys_list, est_list = [], []
        for desc in df_desc.to_dict('records'):
            phys = predict_properties(desc)
            # Ratios
            phys["G/B ratio"] = (phys.get("Shear Modulus (GPa)",0) /
                                  phys.get("Bulk Modulus (GPa)",1))
            phys["Omega"] = (phys.get("Melting_Temperature",0) * desc.get("stoich_entropy",0) * 8.62e-5 /
                              abs(phys.get("Formation Energy (eV/atom)",1)))
            phys_list.append(phys)
            # Estimateurs
            g1 = SigmoidFilter(20, 26, phys.get("LMP",0))
            g2 = SigmoidFilter(1200, 1800, phys.get("Melting_Temperature",0))
            g3 = SigmoidFilter(-9, -6, -phys.get("Density",0))
            g4 = GaussianWindow(6.87, 8, desc.get("avg_VEC",0))
            g5 = GaussianWindow(0.03, 0.066, desc.get("delta",0))
            g6 = SigmoidFilter(-0.6, -0.5, -phys.get("G/B ratio",0))
            g7 = GaussianWindow(-0.156, 0.052, phys.get("Formation Energy (eV/atom)",0))
            g8 = SigmoidFilter(1.0, 1.2, phys.get("Omega",0))
            g9 = GaussianWindow(1.31, 2.32, desc.get("stoich_entropy",0))
            # HT oxidation
            # A 1000°C, log k doit être dans [-14, -10] => K dans [-4.4, 0.4], plus négatif est mieux ; encadrement modulé avec échelle Barnett
            g10 = SigmoidFilter(-0.25, 0.25, -phys.get("Log_10(K_A)",0))
            est_list.append({
                "g1, fluage": g1,
                "g2, fusion": g2,
                "g3, densité": g3,
                "g4, phase BCC": g4,
                "g5, ductilité": g5,
                "g6, ductilité therm": g6,
                "g7, formation": g7,
                "g8, Omega": g8,
                "g9, entropie": g9,
                "g10, HT oxydation": g10
            })
        df_phys = pd.DataFrame(phys_list)
        df_est = pd.DataFrame(est_list)
        df_enr = pd.concat([chunk.reset_index(drop=True), df_desc, df_phys, df_est], axis=1)
        mode, hdr = ("w", True) if i == 0 else ("a", False)
        df_enr.to_csv(temp_file, sep=";", mode=mode, header=hdr, index=False)
        rem = estimate_time(start, (i+1)*CHUNKSIZE, total_chunks*CHUNKSIZE)
        usage = df_enr.memory_usage(deep=True).sum()/(1024**2)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Chunk {i+1}/{total_chunks} | "
              f"Lignes: {df_enr.shape[0]} | RAM: {usage:.1f}MB | Restant: {rem}")
    os.replace(temp_file, out_file)
    print(f"Terminé en {timedelta(seconds=int(time.time()-start))}")
    total = sum(element_counts.values())
    print("Fréquence éléments:")
    for el, cnt in element_counts.items():
        print(f"{el}: {cnt/total:.3f}")

if __name__ == "__main__":
    main()
