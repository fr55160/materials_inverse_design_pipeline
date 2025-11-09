# This script performs  robust feature selection process for multiple study cases using hierarchical clustering, bootstrap stability selection, and elbow-point detection.
"""
feature_selection_pipeline_elbow.py

Robust descriptor selection for multiple study cases using:
  1) Redundancy reduction via hierarchical clustering on descriptor correlations
  2) Stability selection across B bootstraps for four ranking methods
  3) Automated cutoff by elbow‐point detection on each method and on aggregated stability

Outputs per study case:
  - Correlation heatmap and dendrogram
  - CSV of reduced descriptor clusters
  - CSV of mean scores, binary selections, and stability S
  - Barplots of mean scores with selected features highlighted
  - RFECV CV‐score curve (last bootstrap)
  - Stability curve with elbow cutoff
  - Final list of selected features
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
from common_filters import filter_for_case
from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# ========================================
# Configuration générale
# ========================================
FEATURES = [
    "stoich_entropy", "avg_radius", "std_radius", "max_r_ratio", "min_r_ratio",
    "delta", "avg_eneg", "std_eneg", "range_eneg", "avg_weight", "std_weight",
    "unique_elements", "avg_Z", "std_Z", "avg_VEC", "std_VEC", "avg_d", "std_d",
    "frac_d", "avg_s", "avg_p", "avg_f", "std_s", "std_p", "std_f",
    "avg_d_shell_n", "std_d_shell_n", "avg_group", "std_group", "avg_period",
    "std_period", "avg_mendeleev_no", "std_mendeleev_no", "avg_en_allen",
    "std_en_allen", "avg_IE1", "std_IE1", "avg_EA", "std_EA",
    "avg_melting_point", "std_melting_point", "DeltaH_mix", "d_virt"
]
B = 50              # Number of bootstrap iterations
SAMPLE_FRAC = 0.8   # Fraction of data per bootstrap sample

# Adapt here the source filenames if needed
source_names = {}
source_names[1] = "Learning_Database_Hephaistos_outliers_score.csv"
source_names[2] = "MP_Materials_MeltingPoint_Hephaistos_outliers_score.csv"
source_names[3] = "alloy_creep_Hephaistos_outliers_score.csv"
source_names[4] = "HT_Oxidation_Hephaistos_outliers_score.csv"
source_names[4] = "test_database_1_outliers_score.csv" # only for testing purpose, remove later

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
      'outlier_col': 'Outlier_Score_stc4', 'max_outlier': 2 },
    { 'case': 5, 'name': 'Bulk Modulus', 'target': 'Bulk Modulus VRH (GPa)',
      'extra_features': [], 'filter': {'column': 'Material ID', 'method': 'startswith', 'arg': 'mp-'},
      'data_file': PROJECT_FOLDER / "A-Database" / source_names[1],
      'outlier_col': 'Outlier_Score_stc5', 'max_outlier': 2 },
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

# ========================================
# Utility: elbow detection
# ========================================
def detect_elbow(y):
    """
    Return index of maximum distance from the line between endpoints of y.
    """
    n = len(y)
    x1, y1 = 0, y[0]
    x2, y2 = n - 1, y[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    distances = [abs((y2 - y1)*i - (x2 - x1)*yi + x2*y1 - y2*x1)/denom
                 for i, yi in enumerate(y)]
    return int(np.argmax(distances))

# ========================================
# Core pipeline per study case
# ========================================
def run_feature_selection(case_cfg):
    out_dir = c_folder / "feature_selection_results" / f"case_{case_cfg['case']:02d}_{case_cfg['name'].replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load & pre‐filter **uniquement** via common_filters
    import pandas as pd
    df0 = pd.read_csv(case_cfg['data_file'], sep=';', decimal='.', low_memory=False)
    # on assemble la config complète de filtrage
    full_cfg = {
        'target'         : case_cfg['target'],
        'filter'         : case_cfg['filter'],           # ex. startswith 'mp-'
        'extra_features' : case_cfg['extra_features'],   # ex. ['inv_T']
        'outlier_col'    : case_cfg['outlier_col'],
        'max_outlier'    : case_cfg['max_outlier'],
        # on passe la liste **complète** des features (de base + extra)
        'features'       : FEATURES + case_cfg['extra_features']
    }
    from common_filters import filter_for_case
    # ce df a déjà :
    #   - target non‐NaN
    #   - Energy Above Hull ≤ 0
    #   - startswith/mp‐-, inv_T, outlier ≤ max_outlier
    df = filter_for_case(df0, full_cfg).reset_index(drop=True)

    # et on conserve maintenant la vraie liste des colonnes à traiter
    features = FEATURES + case_cfg['extra_features']

    # 2) Correlation matrix & clustering
    corr = df[features].corr().abs()
    plt.figure(figsize=(12,10))
    plt.imshow(corr, vmin=0, vmax=1)
    plt.colorbar(label='|r|')
    plt.xticks(range(len(features)), features, rotation=90)
    plt.yticks(range(len(features)), features)
    plt.title('Abs. Corr. Matrix'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'corr_heatmap.png'), dpi=900); plt.close()

    dist = 1 - corr
    link = linkage(squareform(dist), method='average')
    plt.figure(figsize=(10,6))
    dendrogram(link, labels=features, leaf_rotation=90)
    plt.title('Hierarchical Clustering'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'dendrogram.png'), dpi=900); plt.close()

    clusters = fcluster(link, t=0.05, criterion='distance')
    rep = {}
    reduced = []
    for feat, cl in zip(features, clusters):
        if cl not in rep:
            rep[cl] = feat
            reduced.append(feat)
    pd.DataFrame({'cluster':clusters,'feature':features}).to_csv(os.path.join(out_dir,'feature_clusters.csv'),index=False)

    # 3) Bootstrap for mean scores
    n = len(reduced)
    mean_spearman = np.zeros(n)
    mean_mi       = np.zeros(n)
    mean_xgb      = np.zeros(n)
    mean_rfe_rank = np.zeros(n)
    last_rfecv_scores = None

    for b in range(B):
        samp = df.sample(frac=SAMPLE_FRAC, replace=True, random_state=1000+b)
        Xb = samp[reduced]
        yb = samp[case_cfg['target']]

        # --- Copie des données sur GPU ---
        Xb_gpu = cp.asarray(Xb.values)
        yb_gpu = cp.asarray(yb.values)

        # Spearman
        rho = Xb.apply(lambda col: abs(spearmanr(col, yb)[0]))
        mean_spearman += rho.values
        # MI
        mi = mutual_info_regression(Xb, yb, random_state=1000+b)
        mean_mi += mi
        # XGBoost (CPU mode to ensure compatibility with RTX 5080)
        model = XGBRegressor(
            tree_method='hist',           # fast CPU method
            predictor='cpu_predictor',    # force CPU
            random_state=1000+b,
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=4
        )
        model.fit(Xb, yb)  # CPU arrays
        mean_xgb += model.feature_importances_

        # RFECV (CPU only)
        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=KFold(5, shuffle=True, random_state=1000+b),
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        rfecv.fit(Xb, yb)
        mean_rfe_rank += rfecv.ranking_
        last_rfecv_scores = rfecv.cv_results_['mean_test_score']

    # Normalize means
    mean_spearman /= B; mean_mi /= B; mean_xgb /= B; mean_rfe_rank /= B

    # 4) Per-method elbow cutoffs
    # Spearman descending
    idx_s = np.argsort(mean_spearman)[::-1]
    s_sorted = mean_spearman[idx_s]
    elbow_s = detect_elbow(s_sorted)
    sel_s = set(idx_s[:elbow_s+1])
    # MI descending
    idx_m = np.argsort(mean_mi)[::-1]
    m_sorted = mean_mi[idx_m]
    elbow_m = detect_elbow(m_sorted)
    sel_m = set(idx_m[:elbow_m+1])
    # XGB descending
    idx_x = np.argsort(mean_xgb)[::-1]
    x_sorted = mean_xgb[idx_x]
    elbow_x = detect_elbow(x_sorted)
    sel_x = set(idx_x[:elbow_x+1])
    # RFECV: invert rank so lower->higher
    r_scores = mean_rfe_rank.max() - mean_rfe_rank
    idx_r = np.argsort(r_scores)[::-1]
    r_sorted = r_scores[idx_r]
    elbow_r = detect_elbow(r_sorted)
    sel_r = set(idx_r[:elbow_r+1])

    # 5) Aggregated stability
    I = np.zeros((n,4), dtype=int)
    for j in sel_s: I[j,0]=1
    for j in sel_m: I[j,1]=1
    for j in sel_x: I[j,2]=1
    for j in sel_r: I[j,3]=1
    S = I.mean(axis=1)

    # 6) Export data
    df_out = pd.DataFrame({
        'feature': reduced,
        'mean_spearman': mean_spearman,
        'mean_mi':       mean_mi,
        'mean_xgb':      mean_xgb,
        'mean_rfe_rank': mean_rfe_rank,
        'sel_spearman':  [int(i in sel_s) for i in range(n)],
        'sel_mi':        [int(i in sel_m) for i in range(n)],
        'sel_xgb':       [int(i in sel_x) for i in range(n)],
        'sel_rfe':       [int(i in sel_r) for i in range(n)],
        'stability_S':   S
    })
    df_out.to_csv(os.path.join(out_dir,'selection_data.csv'), index=False)

    # 7) Plot barplots per method
    methods = [
        ('Spearman', mean_spearman, sel_s),
        ('Mutual Information', mean_mi, sel_m),
        ('XGBoost', mean_xgb, sel_x),
        ('RFECV', r_scores, sel_r)
    ]
    for name, scores, sel_set in methods:
        idx = np.argsort(scores)[::-1]
        feats = [reduced[i] for i in idx]
        vals  = scores[idx]
        cols = ['tab:red' if i in sel_set else 'grey' for i in idx]
        plt.figure(figsize=(10,5))
        plt.bar(feats, vals, color=cols)
        plt.xticks(rotation=90); plt.title(f'{name} mean scores'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f'bar_{name}.png'), dpi=900); plt.close()

    # 8) Plot RFECV CV‐score curve (last bootstrap)
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(1,len(last_rfecv_scores)+1), last_rfecv_scores)
    plt.xlabel('Num features'); plt.ylabel('neg MSE CV score')
    plt.title('RFECV curve'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'rfecv_curve.png'), dpi=900); plt.close()

    # 9) Plot stability curve with elbow
    order   = np.argsort(S)[::-1]
    S_sorted = S[order]
    elbow    = detect_elbow(S_sorted)
    S_elbow  = S_sorted[elbow]

    # on colorie toutes les barres dont la valeur >= S_elbow
    colors = ['tab:red' if s_val >= S_elbow else 'grey' for s_val in S_sorted]

    plt.figure(figsize=(10,5))
    plt.bar(range(len(S_sorted)), S_sorted, color=colors)
    # Étiquettes : nom des features dans l'ordre décroissant de S
    plt.xticks(
        range(len(S_sorted)),
        [reduced[i] for i in order],
        rotation=90
    )
    # Ligne pour situer le coude
    plt.axvline(elbow, linestyle='--', color='black')
    plt.xlabel('Feature (ranked by S)'); plt.ylabel('Stability S')
    plt.title('Stability Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'stability_curve.png'), dpi=900)
    plt.close()

    # 10) Final selected features up to elbow
    #   include every feature whose S >= S_elbow
    final_feats = [reduced[i] for i, s_val in enumerate(S) if s_val >= S_elbow]
    pd.Series(final_feats, name='selected_features').to_csv(os.path.join(out_dir,'final_selection.csv'), index=False)

    return final_feats

# ========================================
# Execute for all study cases
# ========================================
if __name__ == '__main__':
    #for cfg in STUDY_CASES: # uncomment to run all cases
    for cfg in [STUDY_CASES[7]]: # only for testing purpose, remove later
        print(f"Processing case {cfg['case']}: {cfg['name']}")
        sel = run_feature_selection(cfg)
        print(f"Selected {len(sel)} features: {sel}\n")
