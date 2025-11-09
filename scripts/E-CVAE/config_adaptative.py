"""
config_adaptative.py

Configuration générale pour le pipeline CVAE « adaptative » :
- Réglage des chemins et du mode (DEBUG vs NORMAL)
- Seuils pour reconstruction (RECON_THRESHOLD), KL (KL_THRESHOLD), entropie (ENTROPY_THRESHOLD)
- Hyperparamètres adaptatifs : β (beta_var), γ (gamma_var) et α (alpha_vars)
- Paramètres Active Learning :
    NB_POINTS       : nombre de candidats générés par epoch
    M_UNCERTAINTY   : nombre de décodages répétés par candidat pour estimer l’incertitude
    TOP_FRACTION    : fraction (0–1) des candidats les plus incertains retenus
- Condition cible pour la génération (TARGET_MODE, test_target)
- Fichiers de sortie (CSV)
"""

import os
import tensorflow as tf
import numpy as np

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR      = PROJECT_FOLDER / "C-HEA"
BASE_VAE_DIR  = c_folder
INPUT_CSV = os.path.join(c_folder, "HEA_database_Hephaistos_augmented.csv")
MODEL_DIR = PROJECT_FOLDER / "B-supervised learning"
FEATURE_SEL_DIR = os.path.join(MODEL_DIR, "feature_selection_results")

# ── Mode & Training ───────────────────────────────────────────────────────────
DEBUG         = True
BATCH_SIZE    = 64
EPOCHS_DEBUG  = 200
EPOCHS_NORMAL = 150

# ── Composition & CLR ─────────────────────────────────────────────────────────
ELEMENTS       = [
    'Al','Cu','Au','Ta','Pd','Zn','Ni','Mo','Ir','Ti','V','Y','Zr','W','Co',
    'Pt','Re','Rh','Ru','Fe','Si','Ga','Sb','As','Nb','Te','Sc','Hf','Ag','Hg',
    'In','Mn','Cr','Mg','Bi','Ge','Sn','Cd','Pb'
]
N_ELEMENTS     = len(ELEMENTS)
CLR_EPSILON    = 1e-7
MIN_FRAC       = 0.02

# ── Seuils activation récompense / génération ─────────────────────────────────
ENTROPY_STO_MIN = 1.2
ENTROPY_STO_MAX = 2.2
RECON_THRESHOLD   = 0.9
KL_THRESHOLD      = 0.5 # KL max pour déclencher la génération de nouveaux candidats
ENTROPY_THRESHOLD = 0.1
alpha_trigger     = 0.055 # déclenchement de la génération d'éléments
min_g = 0.3 # seuil minimum pour les g_i d'un matériau éligible

# ── Hyperparamètres adaptatifs β & γ & α ───────────────────────────────────────
# β adaptatif
BETA_MIN   = 0.02
BETA_MAX   = 0.8
KL_MIN     = 0.095
KL_MAX     = KL_THRESHOLD
BETA       = 0.2 # initialisation de beta au début de la boucle, habituellement 0.1
beta_var   = tf.Variable(initial_value=BETA,    trainable=False, dtype=tf.float32)

# γ adaptatif
GAMMA      = 0.1 # initialisation de gamma au début de la boucle
gamma_var  = tf.Variable(initial_value=GAMMA,   trainable=False, dtype=tf.float32)
gamma_max  = tf.constant(1.0, dtype=tf.float32)

# α adaptatif (vecteur de 10 pondérations gᵢ)
ALPHA = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float) # initialisation des alpha_i au début de la boucle
NORM_ALPHA = np.sum(ALPHA)
if NORM_ALPHA > 0:
    ALPHA = 0.055 * ALPHA / NORM_ALPHA
else:
    raise ValueError("La norme de ALPHA est nulle, impossible de normaliser.")
alpha_vars = tf.Variable(initial_value=ALPHA.tolist(),
                         trainable=False, dtype=tf.float32)
# g1 : fluage
# g2 : melting point
# g3 : densité
# g4 : phase BCC
# g5 : ductilité
# g6 : ductilité
# g7 : formation intermétalliques
# g8 : Omega
# g9 : entropie
# g10  :oxydation HT

# ── Active Learning ───────────────────────────────────────────────────────────
COARSE_LATENTS  = [16]
NB_POINTS       = 200
M_UNCERTAINTY   = 1 # nombre de décodages répétés pour chaque candidat pour estimer l'incertitude
TOP_FRACTION    = 1.0 # fraction des candidats les plus "incertains" (éloignés du centre de la gaussienne) retenus

# ── Condition cible pour génération ──────────────────────────────────────────
TARGET_MODE  = "classification"
test_target  = [0.9] * 10

# ── Fichiers de sortie ────────────────────────────────────────────────────────
GRID_RESULTS_CSV   = os.path.join(BASE_VAE_DIR, "grid_adaptative_scores.csv")
POINTS_RESULTS_CSV = os.path.join(BASE_VAE_DIR, "generated_adaptative_candidates.csv")
# ── Chemin du CSV brut ─────────────────────────────────────────────────────────
