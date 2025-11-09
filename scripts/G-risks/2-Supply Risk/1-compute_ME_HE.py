#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build M_(E/HE) from a companionality matrix C (column-stochastic), following the
absorbing Markov chain with virtual absorbing states for partial primary shares.

Input CSV format (semicolon-separated):
- First row: header with element names; cell (0,0) may be empty or a label.
- First column: element names (same order as header).
- Entry (i,j) = fraction of X_j sourced from X_i.
"""

import sys
import numpy as np
import pandas as pd

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent
raw_companionality = c_folder / "companionality_matrix.csv"
global_companionality = c_folder / "M_E_over_HE.csv"

def load_companionality_csv(path: str, sep=";"):
    df = pd.read_csv(path, sep=sep, header=0, index_col=0)
    # Remplacer les cases vides ou NaN par 0
    df = df.fillna(0.0)
    # Assurer le typage en float
    C = df.values.astype(float)
    elems = list(df.columns)
    return elems, C

def reorder_by_diagonal(elems, C):
    diag = np.diag(C)
    order = np.argsort(diag, kind="stable")
    C_ord = C[np.ix_(order, order)]
    elems_ord = [elems[k] for k in order]
    return elems_ord, C_ord, order

def build_Tstar_D_S(elems_ord, C_ord, tol=1e-12):
    """
    Given column-stochastic C (reordered), build:
      - T*: transient->transient with diagonal removed
      - D: rows = virtual absorbing states for transients with p_j>0, cols = transients
      - S: rows = original absorbing elements (C_jj ~ 1), cols = transients
    """
    n = C_ord.shape[0]
    p = np.diag(C_ord)  # primary shares

    # Classify states
    is_abs_original = np.isclose(p, 1.0, atol=tol)
    is_transient    = ~is_abs_original

    idx_trans = np.where(is_transient)[0].tolist()
    idx_abs   = np.where(is_abs_original)[0].tolist()

    u = len(idx_trans)
    v = len(idx_abs)

    # Submatrix on transient rows/cols
    T = C_ord[np.ix_(idx_trans, idx_trans)].copy()  # includes the diagonal p_j for transients
    # T* = T with diagonal zeroed (remove immediate primary absorption from transient columns)
    T_star = T.copy()
    np.fill_diagonal(T_star, 0.0)

    # D: one virtual absorbing row per transient with p_j>0
    trans_p = p[idx_trans]
    has_virtual = trans_p > tol
    idx_trans_with_virtual = np.where(has_virtual)[0].tolist()
    u_star = len(idx_trans_with_virtual)

    D = np.zeros((u_star, u), dtype=float)
    for row_pos, col_pos in enumerate(idx_trans_with_virtual):
        D[row_pos, col_pos] = trans_p[col_pos]  # place the primary share on its own column

    # S: original absorbing rows × transient cols, entries from C for those absorbing rows
    S = np.zeros((v, u), dtype=float)
    if v > 0:
        S = C_ord[np.ix_(idx_abs, idx_trans)].copy()

    # Names for hosts (rows of [D; S]): virtual absorbers first, then original absorbing elements
    host_names_virtual = [f"{elems_ord[idx_trans[j]]}*" for j in idx_trans_with_virtual]
    host_names_abs     = [elems_ord[i] for i in idx_abs]
    host_names = host_names_virtual + host_names_abs

    return {
        "idx_trans": idx_trans,
        "idx_abs": idx_abs,
        "u": u, "v": v, "u_star": u_star,
        "T_star": T_star,
        "D": D,
        "S": S,
        "host_names": host_names,
        "host_names_virtual": host_names_virtual,
        "host_names_abs": host_names_abs,
    }

def compute_block_product(D, S, T_star):
    u = T_star.shape[0]
    Iu = np.eye(u)
    # Fundamental matrix on transients with diagonal removed
    N = np.linalg.inv(Iu - T_star)
    # Stack (D; S) and right-multiply
    stacked = np.vstack([D, S]) if D.size or S.size else np.zeros((0,u))
    prod = stacked @ N if u > 0 else stacked
    return prod, N

def assemble_M_E_over_HE(elems_ord, info, prod):
    """
    Assemble M_(E/HE) of size (u_star + v) × n.
    Columns are the reordered elements (elems_ord).
    Rows are host events: virtual (for p_j>0 among transients) then original absorbing elements.
    For transient columns j: column = prod[:, position_in_transients].
    For original absorbing columns j_abs: column has 1 on the corresponding original-absorbing row, 0 elsewhere.
    """
    n = len(elems_ord)
    u = info["u"]
    v = info["v"]
    u_star = info["u_star"]
    idx_trans = info["idx_trans"]
    idx_abs = info["idx_abs"]
    host_names = info["host_names"]

    # Map transient index (in global space) -> position in transient subspace [0..u-1]
    trans_pos = {idx_trans[k]: k for k in range(u)}
    # Map original absorbing global index -> row in the lower block [u_star .. u_star+v-1]
    abs_row_pos = {idx_abs[r]: u_star + r for r in range(v)}

    M = np.zeros((u_star + v, n), dtype=float)

    for j_global in range(n):
        if j_global in trans_pos:
            jt = trans_pos[j_global]
            if prod.size:  # prod has shape (u_star+v, u)
                M[:, j_global] = prod[:, jt]
        else:
            # original absorbing element: put a 1 on its absorbing row
            row = abs_row_pos[j_global]
            M[row, j_global] = 1.0

    return M, host_names

def pretty_print_matrix(name, mat, row_names=None, col_names=None, floatfmt="{:8.4f}"):
    print(f"\n=== {name} ===")
    if col_names is not None:
        print(" " * 15 + " ".join([f"{c:>10s}" for c in col_names]))
    for i in range(mat.shape[0]):
        row_label = (row_names[i] if row_names is not None else f"r{i}")
        vals = " ".join(floatfmt.format(x) for x in mat[i, :])
        print(f"{row_label:>12s} : {vals}")

def main(path_csv=raw_companionality, sep=";", out_csv=global_companionality, tol=1e-12):
    # 1) Load
    print (f"Loading companionality matrix C from: {path_csv}")
    elems, C = load_companionality_csv(path_csv, sep=sep)

    # 2) Reorder by non-decreasing diagonal
    elems_ord, C_ord, order = reorder_by_diagonal(elems, C)

    print("list_elem (reordered by non-decreasing diagonal C_jj):")
    print(elems_ord)
    pretty_print_matrix("Reordered Companionality Matrix C (row i, col j = fraction of X_j from X_i)",
                        C_ord, row_names=elems_ord, col_names=elems_ord)

    # 3) Build T*, D, S
    info = build_Tstar_D_S(elems_ord, C_ord, tol=tol)
    T_star, D, S = info["T_star"], info["D"], info["S"]

    # Display T*, D, S
    trans_names = [elems_ord[i] for i in info["idx_trans"]]
    abs_names   = [elems_ord[i] for i in info["idx_abs"]]
    pretty_print_matrix("T* (transient→transient, diagonal removed)", T_star,
                        row_names=trans_names, col_names=trans_names)
    if D.size:
        pretty_print_matrix("D (virtual absorbers for partial primaries → columns are transients)",
                            D, row_names=info["host_names_virtual"], col_names=trans_names)
    else:
        print("\n=== D is empty (no transient state has p_j>0) ===")
    if S.size:
        pretty_print_matrix("S (original absorbing elements → columns are transients)",
                            S, row_names=abs_names, col_names=trans_names)
    else:
        print("\n=== S is empty (no original absorbing element) ===")

    # 4) Compute (D;S)·(I_u - T*)^{-1}
    prod, N = compute_block_product(D, S, T_star)
    if prod.size:
        pretty_print_matrix("(D; S) · (I_u - T*)^{-1}",
                            prod, row_names=(info["host_names_virtual"] + abs_names), col_names=trans_names)
    else:
        print("\n=== (D;S)·(I_u - T*)^{-1} is empty (no transients) ===")

    # 5) Assemble M_(E/HE) and save
    M, host_names = assemble_M_E_over_HE(elems_ord, info, prod)

    # Save to CSV with row/col names
    df_M = pd.DataFrame(M, index=host_names, columns=elems_ord)
    df_M.to_csv(out_csv, sep=";", encoding="utf-8")

    print(f"\nSaved M_(E/HE) to: {out_csv}")
    pretty_print_matrix("M_(E/HE) (rows: host events [virtual*, then original]; cols: elements)",
                        M, row_names=host_names, col_names=elems_ord)

if __name__ == "__main__":
    # Usage: python script.py [companionality_matrix.csv] [;] [M_E_over_HE.csv]
    # All args optional.
    argv = sys.argv
    path_csv=raw_companionality
    sep=";"
    out_csv=global_companionality
    main(path_csv, sep, out_csv)
