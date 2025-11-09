"""
preprocess.py – Pipeline de préparation des données pour le CVAE

Ce module orchestre la chaîne de traitement complète, depuis le fichier CSV brut
jusqu’aux matrices prêtes pour l’apprentissage :
- Chargement des compositions,
- Transformation CLR,
- Calcul descripteurs et propriétés physiques,
- Calcul des scores g₁ à g₁₀,
- Standardisation de la matrice X (composition + propriétés),
- Retour de X_scaled, Y, scaler.

Il utilise les modules :
- config.py        : paramètres globaux
- utils_io.py      : parsing + CLR
- descriptors.py   : descripteurs + prédictions
- scoring.py       : calcul des g_i
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from descriptors import load_models, get_descriptors, predict_properties
from scoring import compute_scores
from config_adaptative import INPUT_CSV, BASE_VAE_DIR, ELEMENTS
from utils_io_adaptative import parse_composition, clr_transformation

def load_and_preprocess(csv_file: str = None, max_rows: int = None):
    """
    Prétraitement adaptative pour le CVAE :

    1) Chargement du CSV (INPUT_CSV par défaut)
    2) Composition → CLR (N_ELEMENTS colonnes)
    3) Concaténation descripteurs atomiques + propriétés physiques
    4) Extraction Y_raw = scores g₁…g₁₀ bruts
    5) Construction de XY = [X_features | Y_raw]

    Retourne :
        XY     : np.ndarray, shape (n_samples, n_features + 10)
        Y_raw  : np.ndarray, shape (n_samples, 10)
        comps  : list[str], formules d’origine
    """
    if csv_file is None:
        csv_file = INPUT_CSV

    print(f"📥 Chargement du fichier : {csv_file}")
    df = pd.read_csv(csv_file, sep=";", engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]

    if max_rows is not None:
        df = df.sample(n=min(max_rows, len(df)), random_state=42).reset_index(drop=True)
        print(f"⚙️ Limitation à {len(df)} lignes (échantillon aléatoire).")

    # Renommage standardisé
    rename_map = {
        "Density": "Density",
        "LMP": "LMP",
        "Log_10(K_A)": "Log_10(K_A)",
        "Formation Energy (eV/atom)": "Formation Energy (eV/atom)",
        "Decomposition Energy Per Atom MP (eV/atom)": "Decomposition Energy Per Atom MP (eV/atom)",
        "Melting_Temperature": "Melting_Temperature",
        "g1, fluage": "g1", "g2, fusion": "g2", "g3, densité": "g3",
        "g4, phase BCC": "g4", "g5, ductilité": "g5", "g6, ductilité therm": "g6",
        "g7, formation": "g7", "g8, Omega": "g8", "g9, entropie": "g9", "g10, HT oxydation": "g10"
    }
    df.rename(columns=rename_map, inplace=True)


    # === Définition des groupes de colonnes ===

    clr_cols = [f"CLR_{el}" for el in ELEMENTS]

    desc_cols = [
        "DeltaH_mix", "stoich_entropy", "avg_radius", "std_radius", "max_r_ratio", "min_r_ratio", "delta",
        "avg_eneg", "std_eneg", "range_eneg", "avg_weight", "std_weight", "unique_elements",
        "avg_Z", "std_Z", "avg_VEC", "std_VEC", "avg_d", "std_d", "frac_d",
        "avg_s", "avg_p", "avg_f", "std_s", "std_p", "std_f",
        "avg_d_shell_n", "std_d_shell_n", "avg_group", "std_group", "avg_period", "std_period",
        "avg_mendeleev_no", "std_mendeleev_no", "avg_en_allen", "std_en_allen",
        "avg_IE1", "std_IE1", "avg_EA", "std_EA",
        "avg_melting_point", "std_melting_point", "d_virt"
    ]

    phys_cols = [
        "Formation Energy (eV/atom)", "Decomposition Energy Per Atom MP (eV/atom)", "Melting_Temperature",
        "Shear Modulus (GPa)", "Bulk Modulus (GPa)", "Density", "LMP",
        "Log_10(K_A)", "G/B ratio", "Omega"
    ]

    gi_cols = [f"g{i+1}" for i in range(10)]

    # Vérifie la présence des colonnes attendues
    required_cols = ["Normalized Composition"] + desc_cols + phys_cols + gi_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Colonnes manquantes dans le CSV : {missing}")
        raise ValueError("Le fichier CSV ne contient pas toutes les colonnes nécessaires.")

    # 1) Composition → CLR
    clr_cols = [f"CLR_{el}" for el in ELEMENTS]
    print("🔄 Transformation CLR des compositions...")
    comp_raw = np.vstack(df["Normalized Composition"].apply(parse_composition).values)
    comp_clr = clr_transformation(comp_raw)
    clr_df   = pd.DataFrame(comp_clr, columns=clr_cols)

    # 2) Concaténation X_features
    # liste desc_cols, phys_cols, gi_cols = [f"g{i+1}" for i in range(10)]
    X_features = pd.concat([clr_df.reset_index(drop=True),
                             df[desc_cols + phys_cols].reset_index(drop=True)],
                            axis=1).values

    # 3) Extraction Y_raw
    Y_raw  = df[gi_cols].values
    comps  = df["Normalized Composition"].tolist()

    # 4) Construction XY
    XY = np.hstack([X_features, Y_raw])

    print("✅ Données prêtes pour le CVAE :")
    print(f"   ➤ XY shape = {XY.shape}  (features + g₁…g₁₀)")
    print(f"   ➤ Y_raw shape = {Y_raw.shape}  (g₁…g₁₀)")
    
    # 5) Sauvegarde colonnes X si besoin
    os.makedirs(os.path.join(BASE_VAE_DIR, "scalers"), exist_ok=True)
    with open(os.path.join(BASE_VAE_DIR, "scalers", "X_columns.txt"), "w") as f:
        f.write("\n".join(clr_cols + desc_cols + phys_cols + gi_cols))

    return XY, Y_raw, comps



def recalculate_all_properties(comp_list):
    """
    Recalcule les descripteurs, propriétés physiques et g-scores
    à partir d'une liste de compositions normalisées.
    
    Retourne :
    - df : DataFrame avec colonnes [composition, descripteurs, propriétés, g_scores]
    """
    models = load_models()
    records = []

    for comp in comp_list:
        try:
            descriptors = get_descriptors(comp)
            properties = predict_properties(descriptors, models)

            # Vérification des valeurs numériques
            required_keys = ["Shear Modulus (GPa)", "Bulk Modulus (GPa)", "Formation Enthalpy (eV/atom)"]
            for key in required_keys:
                val = properties.get(key)
                if not np.issubdtype(type(val), np.number) or not np.isfinite(val):
                    print(f"❌ Propriété invalide: {key} = {val} pour {comp}")
                    raise ValueError("Propriété manquante")

            bulk = properties.get("Bulk Modulus (GPa)", 0)
            shear = properties.get("Shear Modulus (GPa)", 0)
            enthalpy = properties.get("Formation Enthalpy (eV/atom)", 0)
            if abs(bulk) < 1e-12 or abs(enthalpy) < 1e-12:
                raise ZeroDivisionError("Dénominateur nul détecté.")

            # Calculs secondaires
            properties["G/B ratio"] = shear / bulk
            properties["Omega"] = (
                properties.get("Melting Point (K)", 0) *
                descriptors.get("stoich_entropy", 0) *
                8.62e-5 / abs(enthalpy)
            )

            g_scores = compute_scores(descriptors, properties)

            records.append({
                "Normalized Composition": comp,
                **descriptors,
                **properties,
                **{f"g{i+1}": g for i, g in enumerate(g_scores)}
            })

        except Exception as e:
            print(f"⚠️ Ignoré : {comp} | Raison : {e}")
            continue

    return pd.DataFrame(records)

def rebuild_X_from_df(df, base_dir=BASE_VAE_DIR):
    """
    Reconstitue la matrice X (CLR + descripteurs + propriétés physiques)
    à partir d'un DataFrame contenant toutes les colonnes nécessaires.
    
    Arguments :
        df : pd.DataFrame – peut contenir un sous-ensemble des colonnes attendues.
        base_dir : str – chemin vers le dossier contenant 'X_columns.txt'
    
    Retourne :
        np.ndarray – matrice X_recalc (shape = [n_samples, n_features])
    """
    path = os.path.join(base_dir, "scalers", "X_columns.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier X_columns.txt introuvable à {path}")
    
    with open(path, "r") as f:
        X_columns = [line.strip() for line in f.readlines()]

    # Ajoute les colonnes manquantes avec des zéros
    for col in X_columns:
        if col not in df.columns:
            df[col] = 0.0  # Valeur par défaut

    # Réordonne selon l'ordre d'origine
    df_ordered = df[X_columns]

    return df_ordered.values


