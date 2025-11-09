"""
Simplex Steepest Ascent Optimization

Recherche du meilleur alliage par montée de gradient projeté sur la simplexe
(ensemble des compositions stœchiométriques normalisées).

Auteur : François Rousseau — juillet 2025
"""

import logging, warnings, os, re, pickle
from networkx import optimize_graph_edit_distance
import numpy as np
import pandas as pd
from Descriptors_Hephaistos import compute_descriptors
from xgboost import XGBRegressor

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent
# The starting point here is the Pareto-front, to be optimized further:
pareto_csv = PROJECT_FOLDER / "C-HEA" / "Pareto_selection_test.csv"
output_file = c_folder / "finite_difference_optimization.csv"

# ─── 0) Silence XGBoost & Matminer & Mendeleev warnings ─────────────────────
logging.getLogger('xgboost').setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=".*please export the model by calling `Booster.save_model`.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays on .fillna.*",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message="Sn has multiple allotropes.*",
    category=UserWarning
)

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ELEMENTS = [
    'Al','Cu','Au','Ta','Pd','Zn','Ni','Mo','Ir','Ti','V','Y','Zr','W','Co',
    'Pt','Re','Rh','Ru','Fe','Si','Ga','Sb','As','Nb','Te','Sc','Hf','Ag',
    'Hg','In','Mn','Cr','Mg','Bi','Ge','Sn','Cd','Pb'
]

max_elem            = 6        # nombre max d'éléments dans l’alliage
eps                 = 1e-2     # epsilon pour le gradient
step                = 0.04     # pas **initial** = 5% de la norme
gradient_threshold  = 1e-3     # critère de convergence sur ||grad||
MAX_ITERS           = 500      # itérations max
VALUE_THRESHOLD     = 1e-6     # gain F minimal pour accepter un pas
beta = 10.0   # paramètre de lissage : plus grand → plus proche du min exact

Ponderation         = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
start_alloy         = "Fe0.5Co0.5"

# mapping propriété → dossier de selection de features
case_mapping = {
    "Formation Energy (eV/atom)": "case_01_Formation_Energy",
    "Melting_Temperature"        : "case_03_Melting_Point",
    "Shear Modulus (GPa)"        : "case_04_Shear_Modulus",
    "Bulk Modulus (GPa)"         : "case_05_Bulk_Modulus",
    "Density"                    : "case_06_Density",
    "LMP"                        : "case_07_Creep_(LMP)",
    "Log_10(K_A)"                : "case_08_HT_Oxidation"
}

EXTERNAL_DEFAULTS = {
    "LMP":         {"Creep strength Stress (Mpa)": 100, "1-mr": 0},
    "Log_10(K_A)": {"inv_T": 1.0/1273.0}
}

# chemins
BASE_PATH       = r"C:\Users\François\Documents\Hephaistos"
MODEL_DIR       = os.path.join(BASE_PATH, "Apprentissage supervisé")
FEATURE_SEL_DIR = os.path.join(MODEL_DIR, "feature_selection_results")

model_files = {
    "Formation Energy (eV/atom)": "Hephaistos_Formation_Enthalpy.pkl",
    "Melting_Temperature":        "Hephaistos_Melting_Point.pkl",
    "Shear Modulus (GPa)":        "Hephaistos_Shear_Modulus.pkl",
    "Bulk Modulus (GPa)":         "Hephaistos_Bulk_Modulus.pkl",
    "Density":                    "Hephaistos_Density.pkl",
    "LMP":                        "Hephaistos_Creep.pkl",
    "Log_10(K_A)":                "Hephaistos_HT_Oxidation.pkl"
}

# ─────────────────────────────────────────────────────────────────────────────
# 2) CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
models = {}
for prop, fname in model_files.items():
    # 2.1) charger le XGB
    with open(os.path.join(MODEL_DIR, fname), "rb") as f:
        loaded = pickle.load(f)
        mdl    = loaded.get("model", loaded)
    # 2.2) récupérer la liste des features
    feat_csv = os.path.join(FEATURE_SEL_DIR, case_mapping[prop], "final_selection.csv")
    feat_df  = pd.read_csv(feat_csv, sep=",", engine="python")
    features = feat_df["selected_features"].tolist()
    # 2.3) stocker
    models[prop] = {
        "model"    : mdl,
        "features" : features,
        "externals": EXTERNAL_DEFAULTS.get(prop, {})
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3) FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def GaussianWindow(xmin, xmax, xeval):
    x0    = (xmin + xmax) / 2
    sigma = (xmax - xmin) / 2.355
    return np.exp(-((xeval - x0)**2)/(2*sigma**2))

def SigmoidFilter(nook, ok, xeval):
    x0 = (nook + ok)/2
    return 1/(1 + np.exp(-(xeval - x0)/(ok - nook)))

def parse_formula(formula: str) -> dict[str,float]:
    """Lit 'Fe0.2Al0.2…' → {'Fe':0.2,'Al':0.2,…}"""
    out = {}
    for el, qty in re.findall(r'([A-Z][a-z]*)([0-9.]+)', formula):
        out[el] = float(qty)
    return out

def _formula_for_calc(alloy: dict[str,float]) -> str:
    """Formule haute précision (8 décimales) pour compute_descriptors."""
    return "".join(f"{el}{v:.8f}" for el,v in alloy.items() if v>0.0)

def alloy_to_display(alloy: dict[str,float]) -> str:
    """Formate pour l’affichage (2 décimales, supprime <0.005)."""
    parts = []
    for el,v in alloy.items():
        if v<5e-3: continue
        parts.append(f"{el}{v:.2f}")
    return "".join(parts)

def calculate_scores(alloy: dict[str,float]) -> np.ndarray:
    # 1) descriptors
    desc = compute_descriptors(_formula_for_calc(alloy))
    # 2) prédictions physiques
    phys = {}
    for prop, info in models.items():
        row = [ desc.get(f, info["externals"].get(f,0.0))
                for f in info["features"] ]
        df_in = pd.DataFrame([row], columns=info["features"])
        phys[prop] = info["model"].predict(df_in)[0]
    # 3) ratio G/B & Omega
    phys["G/B ratio"] = phys["Shear Modulus (GPa)"]/phys["Bulk Modulus (GPa)"]
    phys["Omega"    ] = phys["Melting_Temperature"] * desc["stoich_entropy"] \
                       * 8.62e-5 / abs(phys["Formation Energy (eV/atom)"])
    # 4) g_i
    g = np.array([
        SigmoidFilter(20,   26,    phys["LMP"]),
        SigmoidFilter(1200, 1800,  phys["Melting_Temperature"]),
        SigmoidFilter(-9,   -6,   -phys["Density"]),
        GaussianWindow(6.87, 8,    desc["avg_VEC"]),
        GaussianWindow(0.03, 0.066, desc["delta"]),
        SigmoidFilter(-0.6, -0.5, -phys["G/B ratio"]),
        GaussianWindow(-0.156,0.052, phys["Formation Energy (eV/atom)"]),
        SigmoidFilter(1.0,  1.2,   phys["Omega"]),
        GaussianWindow(1.31, 2.32,  desc["stoich_entropy"]),
        SigmoidFilter(-0.25,0.25, -phys["Log_10(K_A)"])
    ], dtype=float)
    return np.clip(g, 1e-8, 1.0)

def F(alloy: dict[str,float]) -> float:
    """Fonction objectif = Σ wᵢ log gᵢ."""
    g = calculate_scores(alloy)
    #sm = - (1.0 / beta) * np.log( np.sum( Ponderation * np.exp(-beta * g) ) )
    #return float(sm)
    return float(np.dot(Ponderation, np.log(g)))

def grad_F(alloy: dict[str,float], elems: list[str]) -> dict[str,float]:
    """
    Gradient approché sur la simplexe :
      x_eps = (1-ε)x + ε eᵢ
    """
    base = F(alloy)
    out  = {}
    for el in elems:
        # 1) construire x_eps
        x_eps = {k:(1.0-eps)*v for k,v in alloy.items()}
        x_eps[el] += eps
        # 2) diff finie
        out[el] = (F(x_eps) - base)/eps
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 4) BOUCLE PRINCIPALE D’OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
def optimize_alloy(start: str):
    alloy = parse_formula(start)
    g0    = calculate_scores(alloy)
    print(f"[init] Alloy = {alloy_to_display(alloy)}   g = {np.round(g0,2)}")

    for it in range(1, MAX_ITERS+1):
        elems    = list(alloy.keys())
        grad_raw = grad_F(alloy, elems)

        # — a) ajout d’un élément si < max_elem —
        if len(elems) < max_elem:
            base = F(alloy)
            cand = {}
            for el in ELEMENTS:
                if el in alloy: continue
                # perturbation uniquement sur el
                x_eps = {k:(1.0-eps)*v for k,v in alloy.items()}
                x_eps[el] = eps
                cand[el] = (F(x_eps) - base)/eps
            best = max(cand, key=cand.get)
            if cand[best] > 0:
                alloy[best] = 0.0
                grad_raw[best] = cand[best]
                elems.append(best)

        # — b) projection sur la simplexe & test convergence —
        mean_g   = np.mean(list(grad_raw.values()))
        grad_proj = {e:g-mean_g for e,g in grad_raw.items()}
        norm_grad = np.linalg.norm(list(grad_proj.values()))
        print(f"[{it:03d}] ||grad|| = {norm_grad:.3e}")
        if norm_grad < gradient_threshold:
            print(f"✅ Convergence atteinte en {it} itérations (||grad||={norm_grad:.3e}).")
            break

        # — c) back-tracking le long de grad_proj unitaire —
        direction     = {e:grad_proj[e]/norm_grad for e in elems}
        step_adaptive = step
        F_old         = F(alloy)
        while step_adaptive > VALUE_THRESHOLD:
            candidate = {
                e: alloy[e] + step_adaptive*direction[e]
                for e in elems
            }
            # clip négatifs & écrêtage très petites fractions → 0
            candidate = {e:max(v,0.0) for e,v in candidate.items()}
            candidate = {e:(0.0 if v<1e-4 else v) for e,v in candidate.items()}
            # renorm
            s = sum(candidate.values())
            candidate = {e:v/s for e,v in candidate.items()}

            F_new = F(candidate)
            print(f"    essai step={step_adaptive:.3e}  F_old={F_old:.5f}  F_new={F_new:.5f}")
            if F_new > F_old + VALUE_THRESHOLD:
                alloy = candidate
                break
            step_adaptive *= 0.5
        else:
            print("🔴 Aucun progrès même en back-tracking → arrêt.")
            break

        # — d) affichage du nouvel alliage & de ses g_i —
        gi = calculate_scores(alloy)
        print(f"[{it:03d}] Alloy = {alloy_to_display(alloy)}   g = {np.round(gi,2)}")

    # ─ e) export final
    pd.DataFrame([{"Normalized Composition": alloy_to_display(alloy)}]) \
      .to_csv(output_file,
              mode="a",
              header=not os.path.exists(output_file),
              index=False)

def main():
    # 2) Lecture du fichier
    df = pd.read_csv(pareto_csv, sep=";", engine="python")  # adapte le sep si besoin

    # 2bis) Vérification / création de la colonne "Normalized Composition"
    if "Normalized Composition" not in df.columns:
        print("⚠️  Colonne 'Normalized Composition' absente, création automatique...")

        element_cols = [col for col in df.columns if col.strip() in ELEMENTS]
        print(f"Colonnes élémentaires détectées : {element_cols}")

        def build_normalized_formula(row):
            parts = []
            for el in element_cols:
                val = row[el]
                if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                    parts.append(f"{el}{val:.3f}")
            return "".join(parts)

        df["Normalized Composition"] = df.apply(build_normalized_formula, axis=1)
        tmp_path = c_folder / "Pareto_with_Normalized_Composition.csv"
        df.to_csv(tmp_path, sep=";", index=False)
        print(f"✅ Colonne 'Normalized Composition' ajoutée et sauvegardée dans : {tmp_path}")

    # 3) Boucle sur chaque composition non-NA
    for comp in df["Normalized Composition"].dropna().unique():
        print("\n" + "="*60)
        print(f"Optimisation à partir de : {comp}")
        optimize_alloy(comp)

    print("\n✔️ Toutes les compositions du Pareto front ont été traitées.")


if __name__ == "__main__":
    main()
