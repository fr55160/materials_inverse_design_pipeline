# François Rousseau, may 11th, 2025
# Adapt the values of variables target_source (lines 30 to 35) to your environment!
"""
batch_outlier_scoring_merged.py

For each of 8 study cases, compute an outlier score (0–4) using four methods:
  1) Local Outlier Factor
  2) IsolationForest
  3) kNN residuals
  4) HuberRegressor residuals

This script reads each base CSV only once, adds one column per study case:
  Outlier_Score_stc<case>
and writes back a single enriched CSV per original file with suffix _outliers_score.
Also prints a summary table of counts of score values 0–4 per study case.
"""

import os
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from common_filters import filter_for_case
from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent
target1_source = "Learning_Database_Hephaistos.csv"
target2_source = "MP_Materials_MeltingPoint_Hephaistos.csv"
target3_source = "alloy_creep_Hephaistos.csv"
target4_source = "HT_Oxidation_Hephaistos.csv"
# For test purposes only, all target_source4 is set to a small test file test_database_1.csv; other target sources do not exist yet and the program will fail if used as is.
target4_source = "test_database_1.csv"

# ----------------------------------------
# Function to compute combined outlier score
# ----------------------------------------
def clean_dataset_count(X: pd.DataFrame, y: pd.Series,
                        lof_frac=0.05, iso_frac=0.05,
                        knn_k=10, huber_thresh=2.5,
                        huber_max_iter=1000) -> np.ndarray:
    """
    Returns an integer array of length len(X) with values in {0,1,2,3,4},
    the count of methods that flag each sample as outlier.
    Methods:
      - Local Outlier Factor
      - Isolation Forest
      - kNN residual threshold
      - HuberRegressor residual threshold
    """
    # Convert to numpy and standardize
    X_np = X.values.astype(float)
    y_np = y.values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    # 1) Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=lof_frac)
    lof_labels = lof.fit_predict(X_scaled)
    out_lof = (lof_labels < 0).astype(int)

    # 2) Isolation Forest
    iso = IsolationForest(contamination=iso_frac, random_state=0).fit(X_scaled)
    out_iso = (iso.predict(X_scaled) < 0).astype(int)

    # 3) kNN residuals
    knn = NearestNeighbors(n_neighbors=knn_k).fit(X_scaled)
    _, idx = knn.kneighbors(X_scaled)
    y_knn = np.mean(y_np[idx], axis=1)
    resid_knn = np.abs(y_np - y_knn)
    thresh_knn = resid_knn.mean() + 2 * resid_knn.std()
    out_knn = (resid_knn > thresh_knn).astype(int)

    # 4) Huber residuals
    hub = HuberRegressor(max_iter=huber_max_iter).fit(X_scaled, y_np)
    resid_hub = np.abs(y_np - hub.predict(X_scaled))
    resid_hub /= np.std(y_np - hub.predict(X_scaled))
    out_hub = (resid_hub > huber_thresh).astype(int)

    # Sum flags to get a 0–4 score per sample
    return out_lof + out_iso + out_knn + out_hub

# ----------------------------------------
# Main processing
# ----------------------------------------
def main():
    # Common descriptor list
    FEATURES = [
        "stoich_entropy","avg_radius","std_radius","max_r_ratio","min_r_ratio","delta",
        "avg_eneg","std_eneg","range_eneg","avg_weight","std_weight","unique_elements",
        "avg_Z","std_Z","avg_VEC","std_VEC","avg_d","std_d","frac_d","avg_s","avg_p",
        "avg_f","std_s","std_p","std_f","avg_d_shell_n","std_d_shell_n","avg_group",
        "std_group","avg_period","std_period","avg_mendeleev_no","std_mendeleev_no",
        "avg_en_allen","std_en_allen","avg_IE1","std_IE1","avg_EA","std_EA",
        "avg_melting_point","std_melting_point","DeltaH_mix","d_virt"
    ]

    # Define the 8 study cases
    studies = [
        { 'case':1, 'target':"Formation Energy Per Atom (eV/atom)",
           'features':FEATURES,
           'pre_query': lambda df: df[df["Material ID"].str.startswith('mp-')],
           'data_file': c_folder / target1_source},
        { 'case':2, 'target':"Decomposition Energy Per Atom MP (eV/atom)",
           'features':FEATURES,
           'pre_query': None,
           'data_file': c_folder / target1_source},
        { 'case':3, 'target':"Melting_Temperature",
           'features':FEATURES,
           'pre_query': None,
           'data_file': c_folder / target2_source},
        { 'case':4, 'target':"Shear Modulus VRH (GPa)",
           'features':FEATURES,
           'pre_query': lambda df: df[df["Material ID"].str.startswith('mp-')],
           'data_file': c_folder / target1_source},
        { 'case':5, 'target':"Bulk Modulus VRH (GPa)",
           'features':FEATURES,
           'pre_query': lambda df: df[df["Material ID"].str.startswith('mp-')],
           'data_file': c_folder / target1_source},
        { 'case':6, 'target':"Density",
           'features':FEATURES,
           'pre_query': None,
           'data_file': c_folder / target1_source},
        { 'case':7, 'target':"LMP",
           'features':FEATURES + ["Creep strength Stress (Mpa)", "1-mr"],
           'pre_query': None,
           'data_file': c_folder / target3_source},
        { 'case':8, 'target':"Log_10(K_A)",
           'features':FEATURES + ['inv_T'],
           'pre_query': None,
           'data_file': c_folder / target4_source}
    ]

    # 1) Read each distinct input file only once
    master = {}
    for s in studies:
        path = s['data_file']
        if path not in master:
            # Read CSV with specified separators and NaN markers
            master[path] = pd.read_csv(path, sep=';', decimal='.', na_values=["#NOMBRE!"])

    # 2) Prepare summary container
    summary = []  # to collect counts per case

    # 3) Loop over study cases
    for s in studies:
        mdf = master[s['data_file']]
        # 3.1) filtrage identique à feature_selection & learning
        cfg = {
            'target':      s['target'],
            'filter':      None,
            'pre_query':   s.get('pre_query'),
            'extra_features': ['inv_T'] if s['case']==8 else [],
            'outlier_col': None,
            'max_outlier': 0,
            'features':    s['features']
        }
        df = filter_for_case(mdf.copy(), cfg)

        # 3.2) on score les lignes valides
        scores = clean_dataset_count(
            df[s['features']],
            df[s['target']]
        )
        col_name = f"Outlier_Score_stc{s['case']}"
        if col_name not in mdf.columns:
            mdf[col_name] = np.nan
        mdf.loc[df.index, col_name] = scores

        # résumé avec np.bincount
        counts = np.bincount(scores, minlength=5)
        summary.append([
            s['case'],
            int(counts[0]),
            int(counts[1]),
            int(counts[2]),
            int(counts[3]),
            int(counts[4]),
        ])
        

    # 4) Print summary table
    summary_df = pd.DataFrame(
        summary,
        columns=["Study_case","Count_0","Count_1","Count_2","Count_3","Count_4"]
    )
    print("\nOutlier scores distribution by Study_case:\n")
    print(summary_df.to_string(index=False))

    # 5) Write back each enriched DataFrame exactly once
    for path, mdf in master.items():
        # — remplace NaN → 0 et cast int
        for col in mdf.columns:
            if col.startswith("Outlier_Score_stc"):
                mdf[col] = mdf[col].fillna(0).astype(int)

        base, ext = os.path.splitext(path)
        out_file = f"{base}_outliers_score{ext}"
        mdf.to_csv(out_file, sep=';', decimal='.', index=False)

if __name__ == '__main__':
    main()
