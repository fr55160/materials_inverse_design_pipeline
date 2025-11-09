import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import re

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# === CONFIGURATION ===
DEBUG               = False
N_SIMULATIONS       = 10000
RESULTS_DIR         = c_folder / "PDP"
DB_PATH             = PROJECT_FOLDER / "D-Beyond brute force" / "Divide_Conquer.csv"
OUT_CSV             = RESULTS_DIR / "random_composition_analysis.csv"

# seuil pour filtrer les éléments dans la DB
THRESHOLD_FILTER_DB = 0.2
MIN_COUNT_DB        = 100

# fraction fixe qu’on impose en simulation
FIXED_SIM_FRAC      = 0.3

SCORE_MAP = {
    "P1": "Density",
    "P2": "LMP",
    "P3": "Log_10(K_A)",
    "P4": "Bulk Modulus (GPa)",
    "P5": "Melting_Temperature"
}

# === Logging ===
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

def parse_composition(comp_str: str) -> dict[str,float]:
    pattern = r"([A-Z][a-z]?)(\d*\.?\d+)"
    return {el: float(val) for el,val in re.findall(pattern, comp_str)}

# === Charger un modèle XGB ===
def load_booster(score_name: str) -> xgb.Booster:
    path = c_folder / "SHAP Analysis" / f"best_xgb_{score_name.replace(' ', '_')}.pkl"
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster

# === Génération aléatoire de compositions ===
def simulate_random_compositions(
    elements:     list[str],
    n_sim:        int,
    min_el:       int = 5,
    max_el:       int = 7,
    fixed_el:     str|None = None,
    fixed_frac:   float|None = None
) -> pd.DataFrame:
    comps = []
    for _ in range(n_sim):
        k = np.random.randint(min_el, max_el+1)
        if fixed_el:
            others = [e for e in elements if e != fixed_el]
            chosen = [fixed_el] + list(np.random.choice(others, k-1, replace=False))
        else:
            chosen = list(np.random.choice(elements, k, replace=False))

        if fixed_el and fixed_frac is not None:
            alpha = np.ones(len(chosen)-1)
            sub   = np.random.dirichlet(alpha) * (1-fixed_frac)
            fracs = np.insert(sub, 0, fixed_frac)
        else:
            alpha = np.ones(len(chosen))
            fracs = np.random.dirichlet(alpha)

        comp = dict.fromkeys(elements, 0.0)
        for el, f in zip(chosen, fracs):
            comp[el] = f
        comps.append(comp)

    return pd.DataFrame(comps, columns=elements)

# === Statistiques à extraire ===
def compute_stats(preds: np.ndarray, global_mean: float) -> dict:
    return {
        "Mean":          preds.mean(),
        "Std":           preds.std(ddof=0),
        "Median":        np.median(preds),
        "Prop_Positive": float((preds > global_mean).mean()),
        "Delta":         preds.mean() - global_mean
    }

# === Main ===
def main():
    logging.info("▶️ Début analyse par simulations aléatoires")

    # 0) lire la DB **une fois** pour tout
    df_db     = pd.read_csv(DB_PATH, sep=";")
    comps_db  = df_db["Normalized Composition"].apply(parse_composition)

    all_results = []

    for key, score_name in SCORE_MAP.items():
        logging.info(f"— Traitement: {score_name}")
        booster    = load_booster(score_name)
        full_feats = booster.feature_names

        # 1) construire df_comp_db
        df_comp_db = pd.DataFrame(
            [{el: comp.get(el,0.0) for el in full_feats} for comp in comps_db],
            index=df_db.index
        )
        # 2) filtrage
        counts      = (df_comp_db >= THRESHOLD_FILTER_DB).sum(axis=0)
        valid_feats = [el for el,c in counts.items() if c >= MIN_COUNT_DB]
        if not valid_feats:
            logging.warning("Aucun élément passe le filtre pour %s, skip", score_name)
            continue
        logging.info("Éléments retenus: %s", valid_feats)

        # 3) simulation globale **sur full_feats**
        df_global = simulate_random_compositions(full_feats, N_SIMULATIONS)
        dmat_g    = xgb.DMatrix(df_global)
        gmean     = booster.predict(dmat_g).mean()
        logging.info("  Moyenne globale = %.4f", gmean)

        # 4) simulations fixes, toujours **full_feats**, mais on ne rapporte
        #    les stats que pour valid_feats
        for el in valid_feats:
            df_fix = simulate_random_compositions(
                full_feats,
                N_SIMULATIONS,
                fixed_el=el,
                fixed_frac=FIXED_SIM_FRAC
            )
            preds_f = booster.predict(xgb.DMatrix(df_fix))
            stats   = compute_stats(preds_f, gmean)
            all_results.append([
                el, score_name,
                stats["Mean"], stats["Std"], stats["Median"],
                stats["Prop_Positive"], stats["Delta"]
            ])


    # 3) assemblage et export
    df_res = pd.DataFrame(
        all_results,
        columns=["Element","Score","Mean","Std","Median","Prop_Positive","Delta"]
    )
    df_res.to_csv(OUT_CSV, sep=";", index=False)
    logging.info(f"✅ Terminé — résultats dans {OUT_CSV}")

if __name__ == "__main__":
    main()
