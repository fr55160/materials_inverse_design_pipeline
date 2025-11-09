#La démarche commence par la transformation des données de composition chimique, qui se présentent initialement sous la forme de chaînes de caractères décrivant la
#fraction molaire de chaque élément (par exemple : “Al0.2Fe0.2Cr0.6”). Un script de parsing repère alors, dans chaque chaîne, les paires (élément ; fraction) et reconstruit
#un tableau numérique où chaque colonne correspond à un élément donné et chaque ligne à un alliage précis. Dans ce tableau, toute absence d’un élément dans la composition se
#traduit par un 0. Les alliages deviennent donc des vecteurs de fractions atomiques, leur somme étant égale à 1.

#Cette contrainte “la somme de toutes les fractions vaut 1” peut poser problème aux méthodes statistiques classiques. Pour y remédier, on applique une transformation log-ratio
#centrée, abrégée en CLR. Concrètement, on ajoute un très léger pseudo-count à chaque fraction pour éviter toute valeur nulle, puis on divise chaque ligne par sa somme afin de
#s’assurer que l’alliage reste bien dans un cadre compositionnel. Ensuite, on prend le logarithme de chaque fraction, puis on retire la moyenne de ces logarithmes pour recentrer
#la ligne. L’objectif est de se retrouver avec des variables libres, sans la contrainte initiale de somme à 1, et d’améliorer ainsi la pertinence des analyses qui suivent.

#Une fois les variables transformées, on souhaite relier ces données de composition, devenues des variables CLR, aux différents scores de performance notés g₁ à g₉. Chacun de ces
#scores peut être considéré comme une variable cible indépendante, reflétant une caractéristique particulière du matériau (par exemple, sa résistance au fluage ou sa densité).

#Pour expliquer ou prédire un score donné, on utilise un algorithme de Random Forest, c’est-à-dire un ensemble d’arbres de décision construits de manière aléatoire pour modéliser
#la relation entre les variables explicatives CLR et le score. Afin d’être sûr de choisir les paramètres les plus adaptés (nombre d’arbres, profondeur maximale, etc.), on fait appel
#à une validation croisée. Cette technique consiste à diviser l’ensemble des données en plusieurs sous-ensembles, à entraîner le modèle sur la plupart d’entre eux et à mesurer sa
#performance sur le sous-ensemble restant. Les performances sur chaque repli sont ensuite moyennées, ce qui permet de sélectionner les paramètres du Random Forest offrant la
#meilleure robustesse.

#Le modèle étant entraîné, on peut alors l’interpréter pour comprendre le rôle de chacun des éléments chimiques dans les performances. D’abord, la Random Forest fournit une mesure
#d’“importance” de chaque variable, qui renseigne sur l’impact global de la présence (ou de la proportion) de tel ou tel élément, après la transformation CLR. Mais cette importance
#interne peut parfois manquer de nuance. C’est pourquoi on calcule également l’importance par permutation : on évalue la dégradation de la performance prédictive du modèle lorsqu’on
#réarrange aléatoirement les valeurs d’une variable pour briser son lien avec la cible. Plus la performance chute, plus cette variable était déterminante. On ajoute enfin une étape
#d’interprétation via la méthode SHAP (SHapley Additive exPlanations). SHAP découle de la théorie des jeux et fournit, pour chaque alliage et chaque variable, une valeur décrivant
#dans quelle mesure cette variable contribue positivement ou négativement à la prédiction. C’est un outil précieux pour révéler la complexité des interactions dans un modèle non linéaire.

#Toutes ces informations sont consignées et sauvegardées par le script. Les performances du modèle (par exemple, le score R² et l’erreur quadratique moyenne) sont journalisées
#et complétées par des graphiques : on trace des diagrammes à barres montrant l’importance des variables, on affiche les résultats de permutation importance, et on génère des
#summary plots de SHAP. On met également en mémoire, par la sérialisation (pickle), la version finale de chaque Random Forest entraînée, permettant ainsi de reproduire ou d’approfondir
#l’analyse plus tard. L’ensemble de la procédure vise donc à transformer soigneusement les données de composition, à entraîner un modèle prédictif robuste et à fournir une
#interprétation détaillée des rôles joués par les différents éléments chimiques, le tout dans un cadre formel respectant les bonnes pratiques de validation croisée et de
#reproductibilité."

#Logging
#    Le script crée un fichier de logs (hea_analysis.log) où toutes les informations sont écrites.
#    Les messages sont également affichés à l’écran (via StreamHandler).
#Lecture des données
#    Charge le CSV avec séparateur ";".
#    Sélectionne les 20 éléments focalisés.
#    (Si le CSV ne contient pas directement ces colonnes, il faut les construire en parsant “Normalized Composition”.)
#Transformation CLR
#    La fonction clr_transformation applique un pseudo-count epsilon=1e-7, renormalise pour que chaque ligne somme à 1, puis effectue la transformation CLR.
#    Le résultat est rangé dans df_clr.
#Boucle sur les 9 scores
#    Pour chaque score score_name, on récupère y.
#    On fait un GridSearchCV (10-fold) sur un RandomForestRegressor, en cherchant parmi 2 valeurs pour n_estimators et 3 pour max_depth.
#    Le meilleur modèle est sélectionné et on calcule son R² sur l’ensemble d’apprentissage.
#Bar plot des importances directes
#    On utilise la propriété feature_importances_ du Random Forest.
#    On enregistre le graphique feature_importance_{score_name}.png et on logue également le “Top 5” des features.
#Permutation Importance
#    On utilise permutation_importance de scikit-learn, scoring = “r2”.
#    On fait un bar plot, sauvegardé dans un fichier perm_importance_{score_name}.png.
#SHAP
#    On construit un TreeExplainer pour le best_rf.
#    Pour des raisons de performance, on échantillonne 2 000 lignes (paramètre shap_sample_size) si le dataset est trop grand.
#    On produit un summary plot au format bar, puis un summary plot standard.
#    Les graphiques sont enregistrés au format PNG.
#Sauvegarde du modèle
#    On enregistre le meilleur Random Forest au format pickle, sous un nom distinct pour chaque score.


import os
import sys
import re
import time
import pickle
import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd

# GPU/CPU monitoring
import subprocess
import psutil
import pynvml as nvml

# ==============================
# GPU / CPU import sécurisée (CuPy → fallback NumPy)
# ==============================
try:
    import cupy as cp
    try:
        n_gpu = cp.cuda.runtime.getDeviceCount()
        print(f"[INFO] CuPy detected, using GPU ({n_gpu} device(s)).")
    except Exception as e:
        print(f"[WARN] CuPy import ok but GPU unavailable ({e}), switching to CPU (NumPy).")
        import numpy as cp
except Exception as e:
    import numpy as cp
    print(f"[WARN] CuPy unavailable ({e}), fallback to NumPy (CPU mode).")

# XGBoost (GPU compatible)
import xgboost as xgb
# from cuml.metrics import r2_score as r2_cuml  # optional

# Optuna for hyperparameter search
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution
from sklearn.model_selection import train_test_split

# SHAP
import shap

# Suppress Optuna and other warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

# Graphiques
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

@contextmanager
def nvml_context():
    """Gestion sécurisée des ressources NVML"""
    try:
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        yield handle
    except nvml.NVMLError as e:
        logging.error(f"Erreur NVML: {str(e)}")
        yield None
    finally:
        try:
            nvml.nvmlShutdown()
        except:
            pass

# Utilisation :
with nvml_context() as gpu_handle:
    if gpu_handle:
        util = nvml.nvmlDeviceGetUtilizationRates(gpu_handle)


# ==============================
# Logging configuration
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(c_folder / "SHAP Analysis" / "hea_shap_analysis_v2.log", mode='w'),
        logging.StreamHandler()
    ]
)

# ==============================
# 1. Mode DEBUG vs NORMAL
# ==============================
DEBUG = False

if DEBUG:
    SAMPLE_SIZE = 1000
    SHAP_SAMPLE = 1000
    N_TRIALS = 4
    N_EST_HIGH = 600
    R2_THRESHOLD = 0.15
    MAX_NUM_BOOST  = 300
    XGB_VERBOSITY = 2
else:
    SAMPLE_SIZE = None
    SHAP_SAMPLE = 15700
    N_TRIALS = 100
    N_EST_HIGH = 1500
    R2_THRESHOLD = 0.85
    MAX_NUM_BOOST  = 1500
    XGB_VERBOSITY = 0

optuna_param_distributions = {
    "n_estimators": IntDistribution(low=300, high=N_EST_HIGH, step=300),
    "max_depth":    IntDistribution(low=4,   high=12,   step=4)
}

ARGS_GPU = {
    "tree_method": "gpu_hist",
    "predictor":   "gpu_predictor",
    "device":      "cuda:0",
    "n_jobs":      1
}

XGB_THREADS = 1
SEARCH_JOBS = 1


def r2_gpu(y_true_gpu, y_pred_gpu):
    """Compute R² (works on GPU or CPU seamlessly)."""
    ss_res = cp.sum((y_true_gpu - y_pred_gpu) ** 2)
    y_mean = cp.mean(y_true_gpu)
    ss_tot = cp.sum((y_true_gpu - y_mean) ** 2)
    return 1 - ss_res / ss_tot


@contextmanager
def gpu_memory_context():
    """Context manager for safe GPU cleanup."""
    try:
        yield
    finally:
        if hasattr(cp, "get_default_memory_pool"):
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.Stream.null.synchronize()
                logging.debug("Nettoyage GPU effectué")
            except Exception:
                pass


# ==============================
# 2. Data loading and CLR parsing
# ==============================
start_time = time.time()
logging.info("=== Début du script de SHAP analysis v2 ===")

try:
    nvml.nvmlInit()
    gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
    logging.info("NVML initialisé avec succès")
except Exception as e:
    logging.warning(f"NVML indisponible ({e})")
    gpu_handle = None

data_file = c_folder / "Hephaistos_Compressed_LearningDB.csv"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Fichier non trouvé: {data_file}")

logging.info(f"Chargement des données depuis {data_file}...")
df = pd.read_csv(data_file, sep=';')
TOTAL_ROWS = df.shape[0]
logging.info(f"Données: {TOTAL_ROWS} matériaux, {df.shape[1]} colonnes")

if DEBUG:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    logging.info(f"Mode DEBUG: subset de {SAMPLE_SIZE} matériaux utilisé")
else:
    logging.info("Mode NORMAL: tous les matériaux seront analysés")

# Parsing compositions
def parse_composition(comp_str):
    pattern = r"([A-Z][a-z]?)(\d*\.\d+|\d+)"
    return {elem: float(val) for elem, val in re.findall(pattern, comp_str)}

comps = df["Normalized Composition"].apply(parse_composition)
elements = sorted({e for comp in comps for e in comp})
df_comp = pd.DataFrame([{e: comp.get(e, 0) for e in elements} for comp in comps],
                       index=df.index)

def clr_transform(X, eps=1e-7):
    Xc = X + eps
    row_sums = Xc.sum(axis=1, keepdims=True)
    Xn = Xc / row_sums
    logX = np.log(Xn)
    return logX - logX.mean(axis=1, keepdims=True)

X_clr = clr_transform(df_comp.values)
df_clr = pd.DataFrame(X_clr, columns=elements, index=df.index)

# ==============================
# 3. Scores et train/test
# ==============================
score_columns = [
    "Melting_Temperature",
    "Density",
    "LMP",
    "Log_10(K_A)",
    "Bulk Modulus (GPa)",
]

# 1) Split train/test
X = df_clr
Y = df[score_columns]
X_train, X_test, y_train_all, y_test_all = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
logging.info(f"Split train/test: {X_train.shape[0]}/{X_test.shape[0]} lignes")

out_dir = c_folder / "SHAP analysis"
os.makedirs(out_dir, exist_ok=True)


def xgb_objective(trial, dtrain, MAX_NUM_BOOST, base_params):
    """Objective function pour Optuna, paramétrable via arguments"""
    
    # 1.1 Hyperparamètres à optimiser
    params = base_params.copy()
    params.update({
        "max_depth": trial.suggest_int("max_depth", 4, 12, step=4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.2),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.2),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)
    })
    
    # 1.2 Paramètre de boosting
    num_boost = trial.suggest_int("num_boost_round", 300, MAX_NUM_BOOST, step=300)
    
    # 1.3 Validation croisée
    cvres = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost,
        nfold=5,
        metrics="rmse",
        early_stopping_rounds=10,
        seed=42,
        as_pandas=True
    )
    
    # 1.4 Meilleure itération (early stopping)
    best_iteration = cvres["test-rmse-mean"].idxmin() + 1
    trial.set_user_attr("best_iteration", best_iteration)
    
    return float(cvres["test-rmse-mean"].iloc[-1])


# Boucle sur chaque score g1…g10
for i, score_name in enumerate(score_columns, start=1):
    with gpu_memory_context():  # <-- Encapsule TOUT le traitement du score
        start = time.time()
        logging.info(f"--- [{i}/{len(score_columns)}] Analyse du score: {score_name} ---")

        # Recréation des arrays GPU à chaque itération
        X_train_gpu = cp.asarray(X_train.values)
        X_test_gpu  = cp.asarray(X_test.values)

        # a) Labels GPU
        y_train = y_train_all[score_name].values
        y_test  = y_test_all[score_name].values
        y_train_gpu = cp.asarray(y_train)
        y_test_gpu  = cp.asarray(y_test)

        # b) Créer les DMatrix GPU EN SPÉCIFIANT feature_names
        feature_names = X_train.columns.tolist()
        dtrain = xgb.DMatrix(
            X_train_gpu,
            label=y_train_gpu,
            feature_names=feature_names
        )
        dtest  = xgb.DMatrix(
            X_test_gpu,
            label=y_test_gpu,
            feature_names=feature_names
        )

        # c) Paramètres de base pour GPU
        base_params = {
            "tree_method":   "gpu_hist",
            "predictor":     "gpu_predictor",
            "objective":     "reg:squarederror",
            "verbosity":     XGB_VERBOSITY,
            "enable_categorical": True,
            "sampling_method": "gradient_based",
            **ARGS_GPU       
        }

        # e) Lancer l'optimisation  
        study = optuna.create_study(direction="minimize")
        try:
            # Appel avec lambda pour passer les arguments contextuels
            study.optimize(
                lambda trial: xgb_objective(
                    trial, 
                    dtrain=dtrain, 
                    MAX_NUM_BOOST=MAX_NUM_BOOST, 
                    base_params=base_params
                ), 
                n_trials=N_TRIALS
            )
        except optuna.exceptions.TrialPruned:
            logging.error("Optimisation interrompue pour score %s", score_name)
            continue

        best_params    = study.best_trial.params
        best_iteration = study.best_trial.user_attrs["best_iteration"]
        logging.info(f"Best params: {best_params}, best_iteration={best_iteration}")

        # f) Entraînement final sur GPU
        final_params = {**base_params, **best_params}
        bst = xgb.train(
            final_params,
            dtrain,
            num_boost_round=best_iteration
        )
        # --- Contrôle rapide ---
        logging.debug("Noms de features dans le Booster : %s", bst.feature_names)        

        # g) Prédictions
        y_pred_train = bst.predict(dtrain)       # renvoie numpy
        y_pred_test  = bst.predict(dtest)

        # Libération mémoire DMatrix IMMÉDIATEMENT après utilisation
        del dtrain, dtest

        # h) Rapatriement minimal pour métriques GPU
        y_pred_train_gpu = cp.asarray(y_pred_train)
        y_pred_test_gpu  = cp.asarray(y_pred_test)

        # i) Calcul du R2 sur GPU via fonction crée plus haut
        r2_tr = float(r2_gpu(y_train_gpu, y_pred_train_gpu))
        r2_te = float(r2_gpu(y_test_gpu,  y_pred_test_gpu))
        logging.info(f"R2 train: {r2_tr:.3f}, test: {r2_te:.3f}")


        # j) Contrôle de performance
        if r2_te < R2_THRESHOLD:
            logging.warning(f"R2 ({r2_te:.3f}) < threshold, rechargement du modèle existant")
            safe_name  = score_name.replace(",", "").replace(" ", "_")
            model_path = os.path.join(out_dir, f"best_xgb_{safe_name}.pkl")
            if os.path.exists(model_path):
                bst_old = xgb.Booster()
                bst_old.load_model(model_path)
                bst = bst_old
                logging.info(f"Rechargé le modèle existant depuis {model_path}")
            else:
                logging.error(f"Fichier introuvable : {model_path}, fin de la boucle")
                continue  # Passer au score suivant

    # === Sortie du with : tout ce qui suit est hors contexte GPU ===

    #model_sklearn = xgb.XGBRegressor(**final_params)
    #model_sklearn._Booster = bst
    #model_sklearn.n_features_in_   = X_train.shape[1]
    #model_sklearn.feature_names_in_ = X_train.columns.tolist()

    # Création du wrapper avec les noms de colonnes
    model_sklearn = xgb.XGBRegressor(**final_params)
    # Fit factice sur une seule ligne pour que l’estimator se « fitted »
    model_sklearn.fit(
        X_train.iloc[:1],  # une seule ligne
        y_train[:1]        # un seul label
    )
    # Injecter le Booster complet entraîné
    model_sklearn._Booster = bst

    # --- 1) Scatter Observed vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, s=1, alpha=0.6)
    mn, mx = y_test.min(), y_test.max()
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
    plt.xlabel(f"Observed {score_name}")
    plt.ylabel(f"Predicted {score_name}")
    plt.title(f"Observed vs Predicted – {score_name} | R²={r2_te:.3f}")
    path = os.path.join(out_dir, f"obs_vs_pred_{score_name.replace(',', '').replace(' ', '_')}.png")
    plt.tight_layout(); plt.savefig(path, dpi=900); plt.close()
    logging.info(f"Saved Obs vs Pred: {path}")

    # --- 2) Histogramme des résidus
    residuals = y_test - y_pred_test
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, density=True, alpha=0.7)
    plt.xlabel("Residuals (Observed vs Predicted)")
    plt.ylabel("Density")
    plt.title(f"Residuals Dist – {score_name}")
    path = os.path.join(out_dir, f"residuals_{score_name.replace(',', '').replace(' ', '_')}.png")
    plt.tight_layout(); plt.savefig(path, dpi=900); plt.close()
    logging.info(f"Saved residuals dist: {path}")

    # --- 3.1) Feature importances XGBoost (weight) via Booster.get_score()
    imp_dict = bst.get_score(importance_type="weight")
    names    = X_train.columns.tolist()
    importances = np.array([imp_dict.get(f"f{i}", 0.0) for i in range(len(names))])
    idxs = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,6))
    plt.bar(range(len(idxs)), importances[idxs])
    plt.xticks(range(len(idxs)), [names[i] for i in idxs], rotation=90)
    plt.ylabel("Importance (weight)")
    plt.title(f"Feature Importance – {score_name}")
    path = os.path.join(out_dir, f"feat_imp_{score_name.replace(',', '').replace(' ', '_')}.png")
    plt.tight_layout(); plt.savefig(path, dpi=900); plt.close()
    logging.info(f"Saved feature importances: {path}")

    # --- 3.1.1) Partial Dependence pour les 3 top features ---   
    # Construire un XGBRegressor scikit-learn réellement fitted ---
    model_sklearn = xgb.XGBRegressor(**final_params)
    # Fit factice sur un seul point pour initialiser
    # n_features_in_ et feature_names_in_
    model_sklearn.fit(
        X_train.iloc[:1],  # une seule ligne
        y_train[:1]        # un seul label
    )
    # Injecter le Booster complet entraîné
    model_sklearn._Booster = bst


    top3 = [names[i] for i in idxs[:3]]
    PartialDependenceDisplay.from_estimator(
        model_sklearn,
        X_train,
        features=top3,
        kind="average",
        grid_resolution=50,
        n_jobs=-1
    )
    plt.suptitle(f"Partial Dependence – {score_name}", y=0.98)
    pdp_path = os.path.join(out_dir, f"pdp_{score_name.replace(',', '').replace(' ', '_')}.png")
    plt.tight_layout(rect=[0,0,1,0.90]); plt.savefig(pdp_path, dpi=900); plt.close()
    logging.info(f"Saved PDP: {pdp_path}")

    # --- 3.2) Permutation Importance (ΔR²) ---
    perm = permutation_importance(
        model_sklearn,
        X_test.values, y_test,
        scoring="r2",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )
    means      = perm.importances_mean
    idxs_perm  = np.argsort(means)[::-1]
    plt.figure(figsize=(8,6))
    plt.bar(range(len(means)), means[idxs_perm])
    plt.xticks(range(len(means)), [names[i] for i in idxs_perm], rotation=90)
    plt.ylabel("Mean ΔR²")
    plt.title(f"Permutation Importance – {score_name}")
    path = os.path.join(out_dir, f"perm_imp_{score_name.replace(',', '').replace(' ', '_')}.png")
    plt.tight_layout(); plt.savefig(path, dpi=900); plt.close()
    logging.info(f"Saved permutation importance: {path}")

    # --- 3.3) Snapshot GPU/CPU
    with nvml_context() as gpu_handle:
        gpu_util = "N/A"
        if gpu_handle:
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                temp = nvml.nvmlDeviceGetTemperature(gpu_handle, nvml.NVML_TEMPERATURE_GPU)
                gpu_util = f"{util.gpu}% (Temp: {temp}°C)"
            except nvml.NVMLError:
                pass
    cpu_util = psutil.cpu_percent(interval=1)
    elapsed = time.time() - start
    logging.info(
        f"Step done in {elapsed:.1f}s | GPU util: {gpu_util.strip()}% | CPU util: {cpu_util}%"
    )

    # --- 3.4) SHAP explain using Booster
    logging.info("Calcul des valeurs SHAP...")
    if X_train.shape[0] > SHAP_SAMPLE:
        idx = np.random.choice(X_train.shape[0], SHAP_SAMPLE, replace=False)
        X_shap = X_train.iloc[idx]
    else:
        X_shap = X_train.copy()

    explainer = shap.TreeExplainer(bst)
    shap_vals = explainer(X_shap, check_additivity=False).values

    # 3.4.1 Mean Absolute SHAP
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    feat_names = X_shap.columns.tolist()
    order = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(8,6))
    plt.barh([feat_names[i] for i in order], mean_abs[order], alpha=0.8)
    plt.xlabel("mean(|SHAP value|)")
    plt.title(f"SHAP Mean Abs – {score_name}")
    barp = os.path.join(out_dir, f"abs_shap_bar_{score_name.replace(',', '').replace(' ', '_')}.png")
    plt.tight_layout(); plt.savefig(barp, dpi=900); plt.close()
    logging.info(f"Saved SHAP Mean Abs: {barp}")

    # 3.4.2 Summary Plot
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_vals, X_shap, show=False)
    ax = plt.gca()  # récupère l'axe créé par summary_plot

    # réduire la taille des points à e.g. 2 (valeur en points²)
    for coll in ax.collections:
        coll.set_sizes([1])
    plt.title(f"SHAP Summary – {score_name}")
    sump = os.path.join(out_dir, f"shap_summary_{score_name.replace(',', '').replace(' ', '_')}.png")
    plt.tight_layout(); plt.savefig(sump, dpi=900); plt.close()
    logging.info(f"Saved SHAP Summary: {sump}")

    # 3.4.3 Force Plot for best alloy
    # Trouver l'index du matériau avec la meilleure prédiction
    #idx_best = np.argmax(y_pred_test_np)
    best_pos = np.argmax(y_pred_test)
    idx_best = y_test_all.index[best_pos]
    #idx_best = y_test.iloc[np.argmax(y_pred_test)].name  
    sample = X_test.loc[[idx_best]]
    sv_sample = explainer(sample, check_additivity=False).values
    idx_top   = np.argsort(np.abs(sv_sample).flatten())[::-1][:10]
    sample_top = sample.iloc[:, idx_top].round(3)
    sv_top     = sv_sample[:, idx_top]

    fig = shap.force_plot(
        explainer.expected_value, sv_top, sample_top,
        matplotlib=True, show=False
    )
    for ax in fig.axes:
        ax.set_xlabel("")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    fig.suptitle(
        f"SHAP Force Plot – Alloy {idx_best} – {score_name} (top 10)",
        fontsize=12
    )
    fp = os.path.join(out_dir, f"force_plot_{idx_best}_{score_name.replace(',', '').replace(' ', '_')}.png")
    fig.savefig(fp, bbox_inches='tight', dpi=900)
    plt.close(fig)
    logging.info(f"Saved SHAP Force Plot: {fp}")

    # --- 3.5) Sauvegarde du Booster
    model_file = os.path.join(out_dir, f"best_xgb_{score_name.replace(',', '').replace(' ', '_')}.pkl")
    bst.save_model(model_file)
    #with open(model_file, "wb") as mf:
    #    pickle.dump(bst, mf)
    logging.info(f"Saved model: {model_file}")

    # Suppression EXPLICITE des variables GPU (optionnel mais recommandé)
    del X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu

    #break #je veux juste tester sur le premier score pour l'instant


logging.info("=== Fin du script ===")
if gpu_handle is not None:
    try:
        nvml.nvmlShutdown()
    except nvml.NVMLError as e:
        logging.error(f"Erreur arrêt NVML: {str(e)}")
print(f"Total runtime: {time.time() - start_time:.1f} seconds")
