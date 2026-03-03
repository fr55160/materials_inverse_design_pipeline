#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd


SEP = ";"  # tes CSV sont en ';'
PRICES_FILE = "prices_production.csv"
COMP_FILE = "M_E_over_HE.csv"
OUT_XLSX = "M_elast_build.xlsx"
OUT_CSV = "M_elast.csv"


def parse_num_or_nan(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() in {"nd", "n.d.", "na", "n/a", "nan"}:
        return np.nan

    s = s.replace("\xa0", " ").replace(",", ".")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9eE\.\-\+]", "", s)

    try:
        return float(s)
    except Exception:
        return np.nan


def clean_element_name(name: str) -> str:
    return re.sub(r"\*", "", str(name)).strip()


def load_prices_production(path: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    df = pd.read_csv(path, sep=SEP, engine="python")
    df = df.set_index(df.columns[0])

    prod_row = None
    price_row = None
    for idx in df.index:
        key = str(idx).strip().lower()
        if key.startswith("annual"):
            prod_row = idx
        if key.startswith("price"):
            price_row = idx

    if prod_row is None or price_row is None:
        raise ValueError("Rows 'Annual Production' and/or 'Price USD per ton' not found in prices_production.csv")

    production = df.loc[prod_row].apply(parse_num_or_nan)
    price = df.loc[price_row].apply(parse_num_or_nan)

    production.index = production.index.map(clean_element_name)
    price.index = price.index.map(clean_element_name)

    production = production.groupby(level=0).first()
    price = price.groupby(level=0).first()

    return production, price, df


def load_companionality(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, sep=SEP, engine="python", index_col=0)
    s.index = s.index.map(clean_element_name)
    s.columns = s.columns.map(clean_element_name)

    if s.index.duplicated().any():
        s = s.groupby(level=0).sum()
    if pd.Index(s.columns).duplicated().any():
        s = s.groupby(axis=1, level=0).sum()

    return s.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def build(production: pd.Series, price: pd.Series, s_host_to_element: pd.DataFrame):
    elements_all = sorted(set(s_host_to_element.index) | set(s_host_to_element.columns) |
                          set(production.index) | set(price.index))

    s_sq = s_host_to_element.reindex(index=elements_all, columns=elements_all, fill_value=0.0)

    H_i = production.reindex(elements_all).astype(float)  # host production [t/year]
    Q_j = production.reindex(elements_all).astype(float)  # element production [t/year]
    P_j = price.reindex(elements_all).astype(float)       # price [USD/t]

    # y_ij = s_ij * Q_j / H_i
    y_ij = s_sq.multiply(Q_j, axis=1).divide(H_i, axis=0)

    # y_ij * P_j
    yP_ij = y_ij.multiply(P_j, axis=1)

    # rho_ij = (yP_ij) / sum_j(yP_ij)
    denom = yP_ij.sum(axis=1).replace({0.0: np.nan})
    rho_ij = yP_ij.div(denom, axis=0)

    # M_elast = s_ij * rho_ij
    # --- Step 4 (propre) : M_elast_ij = s_ij * rho_ij
    # Cas particulier : si la ligne i est vide (sum(s_i·)=0), on force M_i· = 0
    row_has_any_share = (s_sq.sum(axis=1) > 0)

    M_sq = pd.DataFrame(0.0, index=s_sq.index, columns=s_sq.columns)

    # On ne calcule le produit que pour les lignes "non vides"
    idx = row_has_any_share[row_has_any_share].index
    M_sq.loc[idx, :] = s_sq.loc[idx, :].to_numpy() * rho_ij.loc[idx, :].to_numpy()

    # Filter: keep only elements with known price and positive production (avoid nd and 0/0)
    valid = (P_j.notna()) & (Q_j.notna()) & (Q_j > 0)
    keep = [e for e in elements_all if bool(valid.loc[e])]

    M = M_sq.reindex(index=keep, columns=keep)

    return {
        "s_ij": s_sq,
        "H_i_t_per_year": H_i.to_frame("Annual production [t/year]"),
        "Q_j_t_per_year": Q_j.to_frame("Annual production [t/year]"),
        "P_j_usd_per_t": P_j.to_frame("Price [USD/t]"),
        "y_ij_tj_per_ti": y_ij,
        "yP_ij_usd_per_ti": yP_ij,
        "rho_ij": rho_ij,
        "M_elast_sq": M_sq,
        "M_elast_final": M,
    }


def ndify(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).where(lambda x: ~x.isna(), other="nd")


def main():
    here = Path(__file__).resolve().parent
    prices_path = here / PRICES_FILE
    comp_path = here / COMP_FILE

    if not prices_path.exists():
        raise FileNotFoundError(f"Missing {prices_path}")
    if not comp_path.exists():
        raise FileNotFoundError(f"Missing {comp_path}")

    production, price, _ = load_prices_production(prices_path)
    s = load_companionality(comp_path)

    out = build(production, price, s)

    # Excel with intermediate sheets
    with pd.ExcelWriter(here / OUT_XLSX, engine="openpyxl") as writer:
        for name, df in out.items():
            ndify(df).to_excel(writer, sheet_name=name[:31])

    # Final CSV (square)
    out["M_elast_final"].to_csv(here / OUT_CSV, sep=SEP, float_format="%.10g")

    print(f"OK: wrote {OUT_XLSX} and {OUT_CSV} in {here}")


if __name__ == "__main__":
    main()
