# François Rousseau, may 11th, 2025

"""
Hybrid hyperparameter tuning script combining:
 1) HalvingGridSearchCV for coarse search
 2) OptunaSearchCV for fine tuning (using new FloatDistribution/IntUniformDistribution)
Features:
 - GPU/CPU auto-detection
 - Detailed progress prints with timestamps and ETA
 - Conditional model saving based on R² improvement (with logging)
 - High‑dpi (900) plots saved to per‑study directories
 - SHAP bar plots showing signed mean SHAP values, sorted ascending
"""

import os
import sys
import pickle
import time
import datetime
import logging
import psutil

import numpy as np
import pandas as pd
import math
#import cupy as cp

from multiprocessing import cpu_count
from GPUtil import showUtilization

from xgboost import XGBRegressor

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    HalvingGridSearchCV, train_test_split, KFold, GroupKFold, cross_val_score
)
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import parallel_backend

import optuna
from optuna_integration import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution

# Analyse Graphique
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import shap

from common_filters import filter_for_case
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# Disable Optuna default log handler to avoid duplicate logs
optuna.logging.disable_default_handler()

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(c_folder / "hybrid_tuning.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

BASE_PATH = PROJECT_FOLDER

def gpu_available():
    return False

# ========================================
# Parallélisme : on découple XGB / joblib
# ========================================
gpu_ok = gpu_available()
#  • XGB ne spinne qu’un seul thread à l’intérieur
xgb_threads = 1
#  • joblib utilisera tous les cœurs dispo pour la recherche
search_jobs = 1 if gpu_ok else cpu_count()  # Réduire le parallélisme CPU si GPU actif

print(f"[CONFIG] Base path détecté : {BASE_PATH}")
logger.info(f"Configuration GPU activée : {gpu_ok}")

# Adapt here the source filenames if needed
source_names = {}
source_names[1] = "Learning_Database_Hephaistos_outliers_score.csv"
source_names[2] = "MP_Materials_MeltingPoint_Hephaistos_outliers_score.csv"
source_names[3] = "alloy_creep_Hephaistos_outliers_score.csv"
source_names[4] = "HT_Oxidation_Hephaistos_outliers_score.csv"
source_names[4] = "test_database_2_outliers_score.csv" # only for testing purpose, remove later

# ========================================
# Liste des cas d’étude
# ========================================

STUDY_CASES = [
    { 'case': 1, 'name': 'Formation Energy', 'target': 'Formation Energy Per Atom (eV/atom)',
      'extra_features': [], 'filter': {'column': 'Material ID', 'method': 'startswith', 'arg': 'mp-'},
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[1],
      'outlier_col': 'Outlier_Score_stc1', 'max_outlier': 2 },
    { 'case': 2, 'name': 'Decomposition Energy', 'target': 'Decomposition Energy Per Atom MP (eV/atom)',
      'extra_features': [], 'filter': None,
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[1],
      'outlier_col': 'Outlier_Score_stc2', 'max_outlier': 2 },
    { 'case': 3, 'name': 'Melting Point', 'target': 'Melting_Temperature',
      'extra_features': [], 'filter': None,
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[2],
      'outlier_col': 'Outlier_Score_stc3', 'max_outlier': 2 },
    { 'case': 4, 'name': 'Shear Modulus', 'target': 'Shear Modulus VRH (GPa)',
      'extra_features': [], 'filter': {'column': 'Material ID', 'method': 'startswith', 'arg': 'mp-'},
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[1],
      'outlier_col': 'Outlier_Score_stc4', 'max_outlier': 0 },
    { 'case': 5, 'name': 'Bulk Modulus', 'target': 'Bulk Modulus VRH (GPa)',
      'extra_features': [], 'filter': {'column': 'Material ID', 'method': 'startswith', 'arg': 'mp-'},
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[1],
      'outlier_col': 'Outlier_Score_stc5', 'max_outlier': 1 },
    { 'case': 6, 'name': 'Density', 'target': 'Density',
      'extra_features': [], 'filter': None,
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[1],
      'outlier_col': 'Outlier_Score_stc6', 'max_outlier': 4 },
    { 'case': 7, 'name': 'Creep (LMP)', 'target': 'LMP',
      'extra_features': ['Creep strength Stress (Mpa)', '1-mr'], 'filter': None,
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[3],
      'outlier_col': 'Outlier_Score_stc7', 'max_outlier': 0 },
    { 'case': 8, 'name': 'HT Oxidation', 'target': 'Log_10(K_A)',
      'extra_features': ['inv_T'], 'filter': None,
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[4],
      'outlier_col': 'Outlier_Score_stc8', 'max_outlier': 0 }
]


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Data file and dtypes
DATA_FILE = os.path.join(BASE_PATH, "Database", "Learning_Database_Hephaistos_outliers_score.csv")
DTYPES = {
        "stoich_entropy": np.float32, "avg_radius": np.float32, "std_radius": np.float32, "max_r_ratio": np.float32, "min_r_ratio": np.float32, "delta": np.float32,
        "avg_eneg": np.float32, "std_eneg": np.float32, "range_eneg": np.float32, "avg_weight": np.float32, "std_weight": np.float32, 
        "avg_Z": np.float32, "std_Z": np.float32, "avg_VEC": np.float32, "std_VEC": np.float32, "avg_d": np.float32, "std_d": np.float32, "frac_d": np.float32,
        "avg_s": np.float32, "avg_p": np.float32, "avg_f": np.float32, "std_s": np.float32, "std_p": np.float32, "std_f": np.float32, "avg_d_shell_n": np.float32,
        "std_d_shell_n": np.float32, "avg_group": np.float32, "std_group": np.float32, "avg_period": np.float32, "std_period": np.float32, "avg_mendeleev_no": np.float32,
        "std_mendeleev_no": np.float32, "avg_en_allen": np.float32, "std_en_allen": np.float32, "avg_IE1": np.float32, "std_IE1": np.float32, "avg_EA": np.float32,
        "std_EA": np.float32, "avg_melting_point": np.float32, "std_melting_point": np.float32, "DeltaH_mix": np.float32, "d_virt": np.float32, "Material ID": "category"
}

FEATURES = [
        "stoich_entropy", "avg_radius", "std_radius", 'max_r_ratio', 'min_r_ratio', "delta", "avg_eneg", "std_eneg", "range_eneg", "avg_weight", "std_weight",
        "unique_elements", "avg_Z", "std_Z", "avg_VEC", "std_VEC", "avg_d", "std_d", "frac_d", "avg_s", "avg_p", "avg_f", "std_s", "std_p", "std_f", "avg_d_shell_n",
        "std_d_shell_n", "avg_group", "std_group", "avg_period", "std_period", "avg_mendeleev_no", "std_mendeleev_no", "avg_en_allen", "std_en_allen",
        "avg_IE1", "std_IE1", "avg_EA", "std_EA", "avg_melting_point", "std_melting_point", "DeltaH_mix", "d_virt"
]
MATERIAL_ID = "Material ID"

R2_TRACKER = PROJECT_FOLDER / "B-supervised learning" / "model_performance_history.csv"
if not os.path.exists(R2_TRACKER):
    pd.DataFrame(columns=["Study_case","Date","R2_test","Best"]).to_csv(R2_TRACKER,index=False)

# Load full dataset
logger.info(f"{now_str()} - Loading data from {DATA_FILE}")
df_full = pd.read_csv(DATA_FILE, sep=";", engine="python", dtype=DTYPES)
if "Density (t/m³)" in df_full.columns:
    df_full.rename(columns={"Density (t/m³)": "Density"}, inplace=True)
logger.info(f"{now_str()} - Data loaded: {df_full.shape[0]} rows, {df_full.shape[1]} cols")

total_cases = 8# REMPLACER 8
case_times = []

cases_to_run = [8]
total_cases = len(cases_to_run)
for study_case in cases_to_run:
#for study_case in range(1, total_cases+1):
    case_start = time.time()
    logger.info(f"{now_str()} - Starting Study_case {study_case}/{total_cases}")
    print(f"[{now_str()}] === Study_case {study_case}/{total_cases} start ===")

    # --- 1) Charger les données brutes depuis cfg['data_file'] ---
    cfg = STUDY_CASES[study_case-1].copy()
    # on récupère le chemin du CSV dédié à ce study_case
    data_path = cfg['data_file']
    raw = pd.read_csv(data_path, sep=";", engine="python", dtype=DTYPES)
    print(f"[{now_str()}] Loaded raw data for case {study_case} from {data_path}: "
          f"{raw.shape[0]} rows, {raw.shape[1]} cols")

    # --- 2) Préparer cfg et appliquer le filtrage centralisé ---
    #   On ajoute les extra_features à la liste de base
    cfg['features'] = FEATURES + cfg.get('extra_features', [])
    #   filter_for_case gère :
    #     • suppression des NaN sur la target
    #     • filtrage des outliers via cfg['outlier_col'] et cfg['max_outlier']
    #     • éventuels filtres sur Material ID
    df = filter_for_case(raw, cfg).reset_index(drop=True)
    print(f"[{now_str()}] After filter_for_case: {df.shape[0]} rows, {df.shape[1]} cols")

    # --- 3) Ajustements spécifiques à chaque cas ---
    # Par exemple, pour les cas 1,4,5 on ne conserve que les mp-*
    if study_case in [1, 4, 5]:
        df = df.query("`Material ID`.str.startswith('mp-')")
        print(f"[{now_str()}] After mp-* filter: {df.shape[0]} rows")

    # --- 4) Charger la liste de features et GRID_PARAMS spécifiques ---
    base_feat_dir = os.path.join(BASE_PATH, "B-supervised learning", "feature_selection_results")

    # 2) Configure target, features, grid
    if study_case == 1:
        TARGET, MODEL_NAME = "Formation Energy Per Atom (eV/atom)", "Hephaistos_Formation_Enthalpy.pkl"
        df = df.query("`Material ID`.str.startswith('mp-')")
        GRID_PARAMS = {"n_estimators":[100,300,600,800,1000,1300],"max_depth":[4,6,8,10,12],"subsample":[9.0/23,0.7,1.0],"gamma":[0,1,5],
    "min_child_weight": [1, 3, 5]}
        # 2.a) Load final selected features instead of stability run 
        feat_file = os.path.join(
            base_feat_dir,
            f"case_01_Formation_Energy", "final_selection.csv"
        )
        df_feats     = pd.read_csv(feat_file, sep=",", engine="python")
        feature_list = df_feats["selected_features"].tolist()       
    elif study_case == 2:
        TARGET, MODEL_NAME = "Decomposition Energy Per Atom MP (eV/atom)", "Hephaistos_Decomposition_Energy.pkl"
        GRID_PARAMS = {"n_estimators":[500,1000,1500],"max_depth":[4,7,10],"subsample":[16.0/27,1.0],"gamma":[0,1,2],
    "min_child_weight": [1, 3, 5]}
        # load case_02
        feat_file = os.path.join(base_feat_dir, "case_02_Decomposition_Energy", "final_selection.csv")
        df_feats     = pd.read_csv(feat_file, sep=",", engine="python")
        feature_list = df_feats["selected_features"].tolist()         
    elif study_case == 3:
        TARGET, MODEL_NAME = "Melting_Temperature", "Hephaistos_Melting_Point.pkl"
        GRID_PARAMS = {"n_estimators":[200,400,600,800],"max_depth":[3,5,7],"subsample":[7.0/23,0.7,1.0],"gamma":[0,1,5],
    "min_child_weight": [1, 3, 5]}
        feat_file = os.path.join(base_feat_dir, "case_03_Melting_Point", "final_selection.csv")
        df_feats     = pd.read_csv(feat_file, sep=",", engine="python")
        feature_list = df_feats["selected_features"].tolist()        
    elif study_case == 4:
        TARGET, MODEL_NAME = "Shear Modulus VRH (GPa)", "Hephaistos_Shear_Modulus.pkl"
        df = df.query("`Material ID`.str.startswith('mp-')")
        GRID_PARAMS = {"n_estimators":[50,150,300,500,800,1200],"max_depth":[2,4,6,8],"subsample":[8.0/20,0.7,1.0],"gamma":[0,1,5],
    "min_child_weight": [1, 3, 5]}
        feat_file = os.path.join(base_feat_dir, "case_04_Shear_Modulus", "final_selection.csv")
        df_feats     = pd.read_csv(feat_file, sep=",", engine="python")
        feature_list = df_feats["selected_features"].tolist()      
    elif study_case == 5:
        TARGET, MODEL_NAME = "Bulk Modulus VRH (GPa)", "Hephaistos_Bulk_Modulus.pkl"
        df = df.query("`Material ID`.str.startswith('mp-')")
        GRID_PARAMS = {"n_estimators":[200,350,500,800,1200],"max_depth":[3,5,7,9],"subsample":[12.0/30,0.7,1.0],"gamma":[0,1,5],
    "min_child_weight": [1, 3, 5]}
        feat_file = os.path.join(base_feat_dir, "case_05_Bulk_Modulus", "final_selection.csv")
        df_feats     = pd.read_csv(feat_file, sep=",", engine="python")
        feature_list = df_feats["selected_features"].tolist()       
    elif study_case == 6:
        TARGET, MODEL_NAME = "Density", "Hephaistos_Density.pkl"
        GRID_PARAMS = {"n_estimators":[100,400,800],"max_depth":[3,6,9],"subsample":[10.0/30,0.6,1.0],"gamma":[0,1,5],
    "min_child_weight": [1, 3, 5]}
        feat_file = os.path.join(base_feat_dir, "case_06_Density", "final_selection.csv")
        df_feats     = pd.read_csv(feat_file, sep=",", engine="python")
        feature_list = df_feats["selected_features"].tolist()        
    elif study_case == 7:
        # 7) Creep (LMP)
        TARGET     = "LMP"
        MODEL_NAME = "Hephaistos_Creep.pkl"
        feat_file = os.path.join(base_feat_dir, "case_07_Creep_(LMP)", "final_selection.csv")

        # 7.a) Read the dedicated creep CSV
        csv_creep = (
            os.path.join(BASE_PATH, "A-Database", source_names[3])
        )
        df = pd.read_csv(csv_creep, sep=";", encoding="latin1")

        # 7.b) Define features: base + two creep-specific
        FEATURES_CREEP = FEATURES + [
            "Creep strength Stress (Mpa)",
            "1-mr"
        ]

        # 7.c) Coerce to numeric and drop missing
        for c in FEATURES_CREEP + [TARGET]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=FEATURES_CREEP + [TARGET]).reset_index(drop=True)

        # 7.d) Outlier filtering
        #    Column “Outlier_Score_stc7” must already have been merged
        df = df[df["Outlier_Score_stc7"] <= 0].reset_index(drop=True)

        # 7.e) Load final stable features from your feature_selection_results
        feat_path = os.path.join(
            os.path.join(BASE_PATH, "B-supervised learning", "feature_selection_results",
            "case_07_Creep_(LMP)",
            "final_selection.csv")
        )
        df_feats     = pd.read_csv(feat_file, sep=",", engine="python")
        feature_list = df_feats["selected_features"].tolist() 

        # 7.f) Grid parameters for coarse tuning
        GRID_PARAMS = {
            "n_estimators":    [50, 100, 200,400,600,800,1000],
            "max_depth":       [3, 5, 7,9],
            "subsample":       [4.0/11, 0.7,1.0],
            "gamma":           [0, 1, 5],
            "min_child_weight":[1, 3, 5]
        }

    elif study_case == 8:
        # —————————————————————————————————————————
        # 8) High-Temperature Oxidation
        TARGET     = "Log_10(K_A)"
        MODEL_NAME = "Hephaistos_HT_Oxidation.pkl"
        MODEL_NAME = "Hephaistos_test_on_HT_Oxidation.pkl"
        feat_file = os.path.join(base_feat_dir, "case_08_HT_Oxidation", "final_selection.csv")

        # 8.a) Read dedicated HT-Ox CSV
        csv_ox = (
            os.path.join(BASE_PATH, "A-Database", source_names[4])
        )
        df = pd.read_csv(csv_ox, sep=";", engine="python")

        # 8.b) Create the inverse-temperature feature
        df["inv_T"] = 1.0 / df["Temperature (K)"]

        # 8.c) Define the full feature list
        FEATURES_HT = FEATURES + ["inv_T"]

        # 8.d) Coerce to numeric & drop missing
        for c in FEATURES_HT + [TARGET]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=FEATURES_HT + [TARGET]).reset_index(drop=True)

        # 8.e) Outlier filtering (stc8 must exist)
        df = df[df["Outlier_Score_stc8"] <= 0].reset_index(drop=True)

        # 8.f) Load final stable features
        feat_path = os.path.join(
            os.path.join(BASE_PATH, "B-supervised learning", "feature_selection_results",
            "case_08_HT_Oxidation",
            "final_selection.csv")
        )
        df_feats     = pd.read_csv(feat_file, sep=",", engine="python")
        feature_list = df_feats["selected_features"].tolist() 

        # 8.g) Grid params
        GRID_PARAMS = {
            "n_estimators":    [50,100,200,400,800, 1300],
            "max_depth":       [3, 5, 7, 9],
            "subsample":       [14.0/28, 0.75, 1.0],
            "gamma":           [0, 1, 5],
            "min_child_weight":[1, 3, 5]
        }

    else:
        raise ValueError(f"Unsupported study_case: {study_case}")
    # Dossier plots spécifique à ce modèle
    #plots_dir = MODEL_NAME.replace(".pkl", "_plots")
    plots_dir = os.path.join(BASE_PATH, "B-supervised learning", MODEL_NAME.replace(".pkl", "_plots"))
    os.makedirs(plots_dir, exist_ok=True)

    # 5) Train/test split
    X = df[feature_list].copy()
    y = df[TARGET].copy()
    if study_case in [1,4,5]:
        X['Material ID'] = df['Material ID']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5) XGB base args : mono-thread pour XGB, on évite l’over-subscription
    base_args = {"objective":"reg:squarederror", "n_jobs": xgb_threads}
    if gpu_ok:
        # on passe au tree_method hist + device cuda (v2.0+ de XGBoost)
        base_args.update({
             "tree_method": "gpu_hist",
             "predictor":   "gpu_predictor",
             "device":      "cuda:0",
             "sampling_method":  "gradient_based",
             #"verbosity":   1
        })        
        print(f"[{now_str()}] GPU detected, using CUDA XGB")
    else:
        base_args.update({
            "tree_method": "hist",
            "predictor":   "cpu_predictor",
            "n_jobs":      cpu_count()
            #"verbosity":   1
            })
        ans = input("No GPU available. Continue using CPU? (y/n) ").strip().lower()
        if ans not in ("y","yes"):
            print("Exiting by user request.")
            sys.exit(1)
        print(f"[{now_str()}] Using CPU XGB with {cpu_count()} threads")

    # 6) CV setup
    cv_coarse = KFold(n_splits=10, shuffle=True, random_state=42) 
    cv_fine   = GroupKFold(n_splits=7) if study_case in [1,4,5] else KFold(n_splits=7, shuffle=True, random_state=42) #REMPLACER n_splits=5

    # --- On prépare X sans la colonne catégorie pour XGBoost ---
    X_train_feat = X_train.drop(columns=[MATERIAL_ID], errors='ignore')
    use_cpu_for_coarse = True
    if gpu_ok and not use_cpu_for_coarse:
        # Conversion en cupy array avec méta-données DMatrix
        import cupy as cp
        Xg = cp.asarray(X_train_feat.values, dtype=cp.float32)
        yg = cp.asarray(y_train.values,       dtype=cp.float32)
    else:
        Xg = X_train_feat.values.astype(np.float32)
        yg = y_train.values.astype(np.float32)
    # groups reste dispo pour OptunaSearchCV
    groups_train = X_train[MATERIAL_ID] if study_case in [1,4,5] else None

    # 7) Coarse tuning
    score_coarse = "neg_root_mean_squared_error" if study_case==7 else "r2"
    print(f"[{now_str()}] Starting coarse tuning (scoring={score_coarse})...")
    cv_coarse = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)  

    # Coarse tuning sur CPU (plus rapide que le GPU pour HalvingGridSearchCV ?)===
    # Choix des arguments XGB en fonction du device et du mode
    if use_cpu_for_coarse:
        args_coarse = {
            "tree_method": "hist",
            "predictor":   "cpu_predictor",
            "n_jobs":      search_jobs,
            #"verbosity":   1
        }
        backend, nbb = "loky", search_jobs
    else:
        args_coarse = {
            "tree_method": "gpu_hist",
            "predictor":   "gpu_predictor",
            "device":      "cuda:0",
            #"verbosity":   1
        }
        backend, nbb = "threading", 1

    # On reconstruit l’estimateur coarse avec args_coarse
    hf = HalvingGridSearchCV(
        XGBRegressor(random_state=42, **args_coarse),
        param_grid=GRID_PARAMS,
        cv=cv_coarse,
        factor=1.5,
        scoring=score_coarse,
        n_jobs=-1,#search_jobs,
        error_score="raise",
        return_train_score=True,
    )

    if (gpu_ok and not use_cpu_for_coarse):
        # GPU: on force la conversion .get() pour obtenir un np.ndarray
        with parallel_backend("threading"):
            from GPUtil import showUtilization
            hf.fit(
                Xg.get(),          # cupy → numpy
                yg.get()
            )
            showUtilization(all=True)
    else:
        # CPU: on utilise loky et numpy classique
        with parallel_backend("loky", n_jobs=cpu_count()):
            hf.fit(
                Xg,                # déjà numpy
                yg
            )

    # Lancement
    with parallel_backend(backend, n_jobs=nbb):
        hf.fit(
            Xg.get() if (gpu_ok and not use_cpu_for_coarse) else Xg,
            yg.get() if (gpu_ok and not use_cpu_for_coarse) else yg
        )
        print(f"[Coarse tuning] done on {'CPU' if (use_cpu_for_coarse or not gpu_ok) else 'GPU'}")
    if (use_cpu_for_coarse or not gpu_ok):
        # Monitoring CPU
        load = psutil.cpu_percent(percpu=True)
        print(f"[CPU USAGE] Per core usage: {load}")

# ────────────────────────────────────────────────────────────────────────────────
#  → GESTION DU FICHIER Best_Coarse_Tuning.csv
# ────────────────────────────────────────────────────────────────────────────────

    import json
    coarse_file = os.path.join(BASE_PATH, "B-supervised learning", "Best_Coarse_Tuning.csv")

    # 1) Lire ou créer le DataFrame
    if os.path.exists(coarse_file):
        df_coarse = pd.read_csv(coarse_file, index_col="Study_case")
    else:
        df_coarse = pd.DataFrame(
            columns=["Study_case", "R2", "params"]
        ).set_index("Study_case")

    # 2) Récupérer le R² courant
    current_r2 = hf.best_score_
    best_coarse = hf.best_params_

    # 3) Comparaison et mise à jour
    if study_case in df_coarse.index:
        stored_r2 = df_coarse.at[study_case, "R2"]
        if current_r2 > stored_r2:
            # meilleure performance → mettre à jour le CSV
            df_coarse.at[study_case, "R2"] = current_r2
            df_coarse.at[study_case, "params"] = json.dumps(best_coarse)
            print(f"📈 New best R² for case {study_case}: {current_r2:.4f} (was {stored_r2:.4f})")
        else:
            # pas mieux → réutiliser les anciens paramètres
            stored_params = json.loads(df_coarse.at[study_case, "params"])
            best_coarse = stored_params
            print(f"📉 R² {current_r2:.4f} not better than stored {stored_r2:.4f}, reusing old params")
    else:
        # première exécution de ce case → on crée la ligne
        df_coarse.loc[study_case] = {
            "R2": current_r2,
            "params": json.dumps(best_coarse)
        }
        print(f"🆕 First coarse record for case {study_case}: R²={current_r2:.4f}")

    # 4) Sauvegarde du CSV
    df_coarse.to_csv(coarse_file)
    print(f"[{now_str()}] Coarse done, best: {best_coarse}")



    # 8) Fine tuning
    if gpu_ok:
        xgb_threads = 1  # on repasse en mono-thread pour GPU
        search_jobs=1
    print(f"[{now_str()}] Starting fine tuning (OptunaSearchCV, n_trials=200)...")
    
    # Conversion des données pour GPU
    if gpu_ok:
        import cupy as cp
        X_opt = cp.asarray(X_train_feat.values.astype(np.float32))
        y_opt = cp.asarray(y_train.to_numpy().astype(np.float32))
        free, total = cp.cuda.Device().mem_info
        used = total - free
        print(f"[GPU MEMORY] Alloué : {used/1e9:.2f} GB / {total/1e9:.2f} GB")
    else:
        X_opt = X_train_feat.values.astype(np.float32)
        y_opt = y_train.to_numpy().astype(np.float32)

    # Paramètres
    optuna_params = {
        "n_estimators":    IntDistribution(
            max(50, math.floor(best_coarse["n_estimators"]/1.2)),  # Plage élargie pour GPU
            math.ceil(best_coarse["n_estimators"]*1.5),
            step=50  # Pas plus grand pour profiter du parallélisme GPU
        ),
        "max_depth": IntDistribution(
            max(3, best_coarse["max_depth"]-1),  # Les GPUs aiment les arbres plus profonds
            best_coarse["max_depth"]+2
        ),
        "learning_rate": FloatDistribution(0.005, 0.1, log=True),  # Plage élargie
        "subsample": FloatDistribution(best_coarse["subsample"] * 0.75, min(best_coarse["subsample"] * 1.33, 1.0)),  # Éviter les sous-échantillons trop petits
        "colsample_bytree": FloatDistribution(0.6, 1.0),
        "gamma": FloatDistribution(best_coarse["gamma"]/3+1e-10, max(0.1,best_coarse["gamma"]*3), log=True),  
        "min_child_weight": IntDistribution(
            low  = max(1, math.floor(best_coarse["min_child_weight"] - 1)),
            high = math.ceil (best_coarse["min_child_weight"] * 1.5),
            step = 1
        ),
    }

    # Configuration GPU-spécifique
    def debug_callback(study, trial):
        print(f"[Optuna trial {trial.number}] params={trial.params}")
        print("  intermediate values:", trial.intermediate_values)

    opt = OptunaSearchCV(
        XGBRegressor(
            random_state=42,
            **base_args
        ),
        param_distributions=optuna_params,
        cv=cv_fine,
        n_trials=300,       # REMPLACER
        timeout=7200,       # REMPLACER
        scoring="r2",
        n_jobs=1 if gpu_ok else search_jobs,
        random_state=42,
        error_score=float("-inf"),
        return_train_score=False,
        callbacks=[debug_callback],
        verbose=1
    )

    # Configuration monitoring adaptatif
    if gpu_ok:
        # Callback spécifique GPU
        def hardware_monitor(study, trial):
            import cupy as cp
            mem = cp.cuda.Device().mem_info
            used_gb = (mem[1] - mem[0]) / 1e9
            print(f"[GPU] Trial {trial.number} | Mem: {used_gb:.2f} GB | Utilisation: {cp.cuda.runtime.deviceGetUtilizationRates(0)[0]}%")
            
        backend = "threading"
        X_data = X_opt
        y_data = y_opt
        callbacks = [hardware_monitor]
    else:
        # Callback spécifique CPU
        def hardware_monitor(study, trial):
            import psutil
            cpu = psutil.cpu_percent(percpu=True)
            mem = psutil.virtual_memory()
            print(f"[CPU] Trial {trial.number} | Charge: {np.mean(cpu):.1f}% | Mémoire: {mem.used/1e9:.1f} GB")
            
        backend = "loky"
        X_data = X_train_feat.values.astype(np.float32)
        y_data = y_train.to_numpy().astype(np.float32)
        callbacks = [hardware_monitor]
    fit_kwargs = {'groups': groups_train} if study_case in [1,4,5] else {}
 


    print("=== DEBUG CV SPLITS ===")
    for fold, (train_idx, test_idx) in enumerate(cv_fine.split(X_data, y_data, groups_train if 'groups_train' in locals() else None)):
        y_tr, y_te = y_data[train_idx], y_data[test_idx]
        print(f"Fold {fold}: train y min/max = ({y_tr.min():.3f},{y_tr.max():.3f}), "
            f"test y min/max = ({y_te.min():.3f},{y_te.max():.3f}), "
            f"len train={len(y_tr)}, len test={len(y_te)}")
    print("========================")




    # Exécution adaptative


    print("=== DEBUG OPTUNA INPUT ===")
    print(f"X_data shape: {X_data.shape}, dtype: {X_data.dtype}")
    print(f"y_data shape: {y_data.shape}, dtype: {y_data.dtype}")
    print(f"NaNs in X_data: {np.isnan(X_data).sum()}  |  NaNs in y_data: {np.isnan(y_data).sum()}")
    print("==========================")




    with parallel_backend(backend, n_jobs=search_jobs if not gpu_ok else 1):
        print(f"[OPTUNA] Démarrage optimisation ({'GPU' if gpu_ok else f'CPU {search_jobs} coeurs'})")
        if gpu_ok:
            # transformez en numpy
            X_input = X_data.get()  
            y_input = y_data.get()
            showUtilization(all=True)
        else:
            X_input = X_data
            y_input = y_data
        opt.fit(
            X_input,
            y_input,
            **fit_kwargs
        )

    best_params = opt.best_params_
    print(f"[{now_str()}] Study case completed, best params: {best_params}")

    # 9) Train final avec optimisations GPU
    final_model_args = base_args.copy()
    #final_model_args.update(gpu_args)
    final_model_args.update(best_params)
    
    model = XGBRegressor(
        random_state=42,
        **final_model_args
    )
    
    # Conversion finale des données si GPU
    if gpu_ok:
        X_tr = cp.asarray(
            X_train.drop(columns=[MATERIAL_ID], errors='ignore') 
            if study_case in [1,4,5] 
            else X_train_feat.values
        )
        X_te = cp.asarray(X_test.drop(columns=[MATERIAL_ID], errors='ignore')) if study_case in [1,4,5] else X_test.values
    else:
        X_tr = X_train.drop(columns=[MATERIAL_ID], errors='ignore') if study_case in [1,4,5] else X_train
        X_te = X_test.drop(columns=[MATERIAL_ID], errors='ignore') if study_case in [1,4,5] else X_test

    print(f"[TRAINING] Starting final training on {'GPU' if gpu_ok else 'CPU'}...")
    start_train = time.time()
    model.fit(X_tr, y_train)
    print(f"[TRAINING] Completed in {(time.time()-start_train)/60:.2f} minutes")

    # 10) Metrics
    y_train_pred = model.predict(X_tr)
    y_test_pred  = model.predict(X_te)

    import sklearn.metrics as skm

    # Calcul du MSE
    mse_train = skm.mean_squared_error(y_train, y_train_pred)
    mse_test  = skm.mean_squared_error(y_test,  y_test_pred)

    # Calcul du RMSE
    rmse_train = np.sqrt(mse_train)
    rmse_test  = np.sqrt(mse_test)

    # Conversion CPU pour les métriques si nécessaire
    if gpu_ok:
        y_train_pred = cp.asnumpy(y_train_pred)
        y_test_pred  = cp.asnumpy(y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test,  y_test_pred)
    print(f"[{now_str()}] Train R²={r2_train:.4f}, Test R²={r2_test:.4f}")
    if gpu_ok:
        import cupy as cp
        print(f"[GPU MEMORY] After training: {cp.cuda.Device().mem_info[0]/1e9:.2f} GB free")

    # 11) Conditional save + plots + logging
    history = pd.read_csv(R2_TRACKER)
    series = history.loc[history.Study_case == study_case, "R2_test"]
    raw_best = series.max()
    prev_best = -np.inf if np.isnan(raw_best) else raw_best

    delta = r2_test - prev_best
    print(f"[{now_str()}] Previous best R²={prev_best:.4f}, Delta ={delta:+.4f}")

    save_model = (r2_test > prev_best)

    if save_model:
        # 11a) dump model
        model_path = os.path.join(BASE_PATH, "B-supervised learning", MODEL_NAME)
        #with open(MODEL_NAME, "wb") as f:
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "best_params": best_params,
                "feature_columns": feature_list,
                "target_column": TARGET,
                "performance": {
                    "R2_train":   r2_train,
                    "R2_test":    r2_test,
                    "RMSE_train": rmse_train,
                    "RMSE_test":  rmse_test
                }
            }, f)
        logger.info(f"Study_case {study_case} - New best model saved: R2_test={r2_test:.4f} (Delta {delta:+.4f})")
        print(f"✅ Model saved to {MODEL_NAME} (improved by Delta R²={delta:+.4f})")

        # ——————————————————————————————————
        # 11.a) Parity plot
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_test_pred, s=15, alpha=0.6, label="Predicted")
        mn, mx = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
        plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="1:1 line")
        plt.xlabel(f"True {TARGET}")
        plt.ylabel(f"Predicted {TARGET}")
        plt.title(f"Parity Plot – {TARGET} - R²_test={r2_test:.4f}")
        plt.legend()
        plt.tight_layout()
        safe_target = TARGET.replace("/", "_")
        plt.savefig(os.path.join(plots_dir, f"Hephaistos - parity – {safe_target}.png"), dpi=900)
        plt.close()

        # ——————————————————————————————————
        # 11.b) Residuals histogram
        resid = y_test - y_test_pred
        plt.figure(figsize=(7,4))
        plt.hist(resid, bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(resid.mean(), color="red", linestyle="--", label=f"Mean={resid.mean():.3f}")
        plt.xlabel(f"Residuals of {TARGET} (true – pred)")
        plt.ylabel("Count")
        plt.title(f"Residuals Histogram – {TARGET}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Hephaistos - residuals_hist – {safe_target}.png"), dpi=900)
        plt.close()

        # ——————————————————————————————————
        # 11.c) Feature importances
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        plt.figure(figsize=(8,6))
        plt.barh([feature_list[i] for i in idx], imp[idx])
        plt.xlabel("Importance")
        plt.title(f"Feature Importances – {TARGET}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Hephaistos - feature_importances – {safe_target}.png"), dpi=900)
        plt.close()

        # ——————————————————————————————————
        # 11.d) Partial Dependence (Top 3 features)
        #top3 = [feature_list[i] for i in idx[:3]]
        #disp = PartialDependenceDisplay.from_estimator(
        #    model, X_tr, features=top3_idx, kind="average",
        #    grid_resolution=20, n_jobs=1  # mono-thread pour PDP
        #)
        # --- Conversion CuPy → NumPy DataFrame pour PDP ---
        X_tr_np = cp.asnumpy(X_tr) if gpu_ok else X_tr
        df_tr   = pd.DataFrame(X_tr_np, columns=feature_list)
        top3 = [feature_list[i] for i in idx[:3]]
        top3_idx = [feature_list.index(f) for f in top3]
        disp = PartialDependenceDisplay.from_estimator(
            model, df_tr, features=top3_idx, kind="average",
            grid_resolution=20, n_jobs=1
        )
        plt.suptitle(f"Partial Dependence – {TARGET} (Top 3)", y=1.02)
        # Les axes sont déjà étiquetés par sklearn, on renforce juste le titre
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Hephaistos - partial_dependence – {safe_target}.png"), dpi=900)
        plt.close()

        # ——————————————————————————————————
        # 11.e) SHAP summary (Beeswarm)
        if gpu_ok:
            X_tr_cpu = cp.asnumpy(X_tr)
            df_tr = pd.DataFrame(X_tr_cpu, columns=feature_list, index=y_train.index)
            #sample = X_tr_cpu[:1000] if X_tr_cpu.shape[0] > 1000 else X_tr_cpu
        else:
            #sample = X_tr.sample(1000, random_state=42) if len(X_tr) > 1000 else X_tr
            df_tr = X_train_feat.copy()
        sample = df_tr.sample(n=min(1000, len(df_tr)), random_state=42)
        expl   = shap.TreeExplainer(model)
        #sample_np = np.asarray(sample)
        #sv     = expl(sample_np)
        #shap.summary_plot(sv.values, sample, show=False)
        sv     = expl.shap_values(sample)
        shap.summary_plot(sv, sample, show=False)
        plt.title(f"SHAP Analysis: How Features Impact {TARGET} Predictions")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Hephaistos - shap_beeswarm – {safe_target}.png"), dpi=900)
        plt.close()

        # ——————————————————————————————————
        # 11.f) SHAP summary bar (signed mean)
        #shap.summary_plot(sv.values, sample, plot_type="bar", show=False)
        shap.summary_plot(sv, sample, plot_type="bar", show=False)
        plt.xlabel("Mean SHAP value (signed)")
        plt.title(f"SHAP Summary (Signed Mean) – {TARGET}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Hephaistos - shap_bar_signed – {safe_target}.png"), dpi=900)
        plt.close()

        # ——————————————————————————————————
        # 11.g) Residuals vs features heatmap
        X_te_cpu = cp.asnumpy(X_te) if gpu_ok else X_te
        df_r = pd.DataFrame(X_te_cpu, columns=feature_list, index=y_test.index)
        df_r["resid"] = resid
        plt.figure(figsize=(10,8))
        sns.heatmap(df_r.corr(), center=0, cmap="vlag", linewidths=0.5)
        plt.title(f"Residuals Correlation – {TARGET}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Hephaistos - residuals_corr_heatmap – {safe_target}.png"), dpi=900)
        plt.close()

    else:
        print("❌ Model not saved (no improvement)")

    # 11.h) Feature selection stability
    logger.info(f"Features selected for training: {feature_list}")


    # 12) Update history
    new_row = pd.DataFrame({
        "Study_case":[study_case],
        "Date":[now_str()],
        "R2_test":[r2_test],
        "Best":[save_model]
    })
    history.loc[len(history)] = [study_case, now_str(), r2_test, save_model]
    history.to_csv(R2_TRACKER, index=False)
    history = pd.read_csv(R2_TRACKER, index_col=None)

    # 13) Timing & ETA
    case_elapsed = time.time() - case_start
    case_times.append(case_elapsed)
    eta = (total_cases - study_case) * np.mean(case_times)
    print(f"[{now_str()}] Completed in {case_elapsed/60:.2f} min. ETA {eta/60:.2f} min\n")

    if gpu_ok:
        cp.get_default_memory_pool().free_all_blocks()
