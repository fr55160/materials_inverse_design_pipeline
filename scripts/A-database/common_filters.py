# common_filters.py

import pandas as pd
import numpy as np

def filter_for_case(df: pd.DataFrame, case_cfg: dict) -> pd.DataFrame:
    """
    Applique aux données de base tous les filtres requis avant :
      1) Target non‐NaN
      2) Energy Above Hull <= 0 ou NaN
      3) Filtre spécifique (dict 'filter') ou fonction 'pre_query'
      4) Ajout colonne inv_T pour le case 8
      5) Outlier score <= max_outlier (si la colonne existe)
      6) Suppression des NaN sur features+target

    Conserve les indices d’origine de df.
    """
    # 1) Target non‐NaN
    df = df[df[case_cfg['target']].notna()]

    # 2) Energy Above Hull <= 0 ou NaN
    if 'Energy Above Hull (eV)' in df.columns:
        hull = df['Energy Above Hull (eV)']
        df = df[(hull <= 0) | (hull.isna())]

    # 3) Filtre spécifique :
    #   - si case_cfg['filter'] est un dict, on utilise getattr sur str
    #   - si case_cfg['pre_query'] est une fonction, on l'applique
    flt_dict = case_cfg.get('filter')
    pre_q     = case_cfg.get('pre_query')

    if isinstance(flt_dict, dict):
        col    = flt_dict['column']
        method = flt_dict['method']
        arg    = flt_dict['arg']
        series = df[col].astype(str).str
        mask   = getattr(series, method)(arg)
        df     = df[mask]

    elif callable(pre_q):
        # on suppose que pre_query(df) renvoie un df filtré
        df = pre_q(df)

    # 4) Ajout de inv_T pour le case 8
    if 'inv_T' in case_cfg.get('extra_features', []):
        df = df.copy()  # éviter SettingWithCopyWarning
        df['inv_T'] = 1.0 / df['Temperature (K)']

    # 5) Outlier score <= max_outlier si la colonne existe
    out_col = case_cfg.get('outlier_col')
    maxo    = case_cfg.get('max_outlier', np.inf)
    if out_col and out_col in df.columns:
        df = df[df[out_col] <= maxo]

    # 6) Suppression des NaN sur features + extras + target
    feats     = case_cfg.get('features', []) + case_cfg.get('extra_features', [])
    all_cols  = feats + [case_cfg['target']]
    exist_cols = [c for c in all_cols if c in df.columns]
    # convertir chaque colonne individuellement
    for col in exist_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # puis supprimer les lignes où l'une de ces colonnes est NaN
    df = df.dropna(subset=exist_cols)

    return df
