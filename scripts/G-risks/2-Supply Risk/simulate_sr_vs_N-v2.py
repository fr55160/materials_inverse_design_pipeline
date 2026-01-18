#!/usr/bin/env python3
"""
simulate_sr_vs_N.py
===================

Alloy supply risk vs number of elements (N=1..10), with a hybrid sampling design:

- If C(M, N) <= K: exhaustive enumeration of all alloys of size N (noise-free).
- Else: nested / common-random-numbers design ("trajectories"):
        sample K trajectories of length N_MAX (10 distinct elements each),
        and for each N take the prefix set of each trajectory.

This reduces sampling noise for percentile curves while keeping computation feasible.

Additionally, the plot includes a horizontal reference line equal to the maximum
elemental SR among the element universe (read from SR_EU.csv), to compare alloy SR
against the "worst single element" SR.

Inputs (same folder):
- alloy_supply_risk.py  -> provides compute_alloy_supply_risk(formula: str) -> float
- SR_EU.csv             (sep=';')  -> element SR values
- Sigma_E.csv           (sep=';')  -> used internally by alloy_supply_risk.py

Outputs (created in ./sr_vs_n_outputs_hybrid/):
- L_N01_SR.csv ... L_N10_SR.csv   (sep=';') : alloy_formula ; SR_alloy
- summary_sr_vs_n.csv             (sep=';') : N ; mode ; n_samples ; mean ; p05 ; p95
- sr_vs_n.png                     : plot

Reviewer notes:
- Equiatomic alloys within each N: each element is written with coefficient 0.1.
  Upstream code normalizes the composition, so only the set of elements matters.
"""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from alloy_supply_risk import compute_alloy_supply_risk


# -----------------------------
# User configuration
# -----------------------------

ELEMENTS: List[str] = [
    "Sc","Ga","Zr","Ru","Rh","Cd","In","Te","Hf","Re","Ir","Pd","As","Bi","Co","Sb","V",
    "Ag","Ge","Mo","Y","Ta","Pt","Au","Zn","Pb","Cu","W","Mn","Sn","Cr","Ni","Nb","Mg",
    "Al","Si","Ti","Fe"
]

N_MIN = 1
N_MAX = 10

# Chosen K
K = 8436

# Stoichiometry fixed (equiatomic after normalization upstream)
FIXED_STOICH = 0.1

# Random seed for reproducibility (only used when C(M,N) > K)
RNG_SEED = 12345

# File containing element SR (used to compute max elemental SR for horizontal line)
SR_CSV_PATH = Path("SR_EU.csv")

# Output folder
OUTPUT_DIR = Path("sr_vs_n_outputs_hybrid")


# -----------------------------
# Helpers
# -----------------------------

def alloy_formula_from_elements(elts: Sequence[str], fixed_stoich: float = FIXED_STOICH) -> str:
    """Build a canonical alloy formula string, e.g., 'Cr0.1Fe0.1Ni0.1'."""
    elts_sorted = sorted(elts)
    return "".join([f"{e}{fixed_stoich:g}" for e in elts_sorted])

from dataclasses import asdict, is_dataclass

def extract_sr_scalar(res) -> float:
    """
    Extract the scalar SR value from the result returned by compute_alloy_supply_risk().

    Supported cases:
    - float / int
    - dataclass (e.g., AlloySRResult) with at least one numeric field
    - object exposing a dict-like interface (__dict__)
    - object exposing common attribute names (sr_alloy, sr, value, etc.)

    If multiple numeric fields exist, we prioritize likely SR names.
    """
    # Already a scalar
    if isinstance(res, (int, float, np.floating)):
        return float(res)

    # Try common attribute names first
    for attr in ("sr_alloy", "SR_alloy", "SR", "sr", "value"):
        if hasattr(res, attr):
            v = getattr(res, attr)
            if isinstance(v, (int, float, np.floating)):
                return float(v)

    # Dataclass case: inspect fields
    if is_dataclass(res):
        d = asdict(res)
        # prioritize likely keys
        for key in ("sr_alloy", "SR_alloy", "SR", "sr", "value"):
            if key in d and isinstance(d[key], (int, float)):
                return float(d[key])
        # otherwise: take the first numeric field
        for _, v in d.items():
            if isinstance(v, (int, float)):
                return float(v)

    # Generic object: inspect __dict__
    if hasattr(res, "__dict__"):
        d = res.__dict__
        for key in ("sr_alloy", "SR_alloy", "SR", "sr", "value"):
            if key in d and isinstance(d[key], (int, float, np.floating)):
                return float(d[key])
        for v in d.values():
            if isinstance(v, (int, float, np.floating)):
                return float(v)

    raise TypeError(
        f"Could not extract a numeric SR scalar from object of type {type(res)}. "
        f"Available attributes: {dir(res)}"
    )


def sample_unique_trajectories(
    elements: Sequence[str],
    k: int,
    n_max: int,
    rng: np.random.Generator
) -> List[Tuple[str, ...]]:
    """
    Sample k trajectories (ordered tuples of length n_max), all distinct elements in a trajectory.
    Enforce uniqueness of the underlying n_max-element SET (order ignored) to avoid duplicates at N=n_max.
    """
    if n_max > len(elements):
        raise ValueError(f"n_max={n_max} cannot exceed available elements={len(elements)}.")

    seen_sets = set()
    trajectories: List[Tuple[str, ...]] = []

    max_attempts = k * 2000  # safety
    attempts = 0

    while len(trajectories) < k and attempts < max_attempts:
        attempts += 1
        traj = tuple(rng.choice(elements, size=n_max, replace=False).tolist())
        key = frozenset(traj)
        if key in seen_sets:
            continue
        seen_sets.add(key)
        trajectories.append(traj)

    if len(trajectories) < k:
        raise RuntimeError(
            f"Only sampled {len(trajectories)} unique trajectories out of requested {k}. "
            "Reduce K or increase the element universe."
        )

    return trajectories


def read_max_element_sr(sr_csv_path: Path, allowed_elements: Sequence[str]) -> float:
    """
    Read SR_EU.csv (sep=';') and return the maximum SR among 'allowed_elements'.

    Robust to two common layouts:
    1) Wide format: header row = element symbols, next row = SR values.
    2) Long format: two columns like "Element;SR" (or similar), one row per element.

    If extra columns exist, this tries to pick a numeric SR in each row.
    """
    allowed = set(allowed_elements)
    values: List[float] = []

    with sr_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if len(rows) == 0:
        raise ValueError(f"{sr_csv_path} is empty or unreadable.")

    # Try wide format: header row elements, second row numeric values
    if len(rows) >= 2:
        header = [h.strip() for h in rows[0]]
        second = [c.strip() for c in rows[1]]
        # Check if many of second row cells are floats
        floatable = 0
        for c in second:
            try:
                float(c.replace(",", "."))
                floatable += 1
            except Exception:
                pass

        if floatable >= max(3, len(second) // 2):
            # Treat as wide
            for elem, val in zip(header, second):
                elem = elem.strip()
                if elem in allowed:
                    try:
                        v = float(val.replace(",", "."))
                        values.append(v)
                    except Exception:
                        continue
            if values:
                return float(max(values))

    # Else: treat as long / row-wise
    # Heuristic: first column may be element symbol; find any numeric in the row as SR
    for row in rows[1:] if len(rows) > 1 else rows:
        cells = [c.strip() for c in row]
        if not cells:
            continue

        # Find element symbol in row
        elem = None
        for c in cells:
            if c in allowed:
                elem = c
                break
        if elem is None:
            continue

        # Take the first numeric cell (excluding the element symbol itself)
        for c in cells:
            if c == elem:
                continue
            try:
                v = float(c.replace(",", "."))
                values.append(v)
                break
            except Exception:
                continue

    if not values:
        raise ValueError(
            f"Could not extract SR values for the provided element set from {sr_csv_path}. "
            "Please check the CSV structure."
        )

    return float(max(values))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    # Pre-sample trajectories once (used only when C(M,N) > K)
    # Common random numbers across N for the sampled regime.
    trajectories = sample_unique_trajectories(ELEMENTS, K, N_MAX, rng)

    # Cache SR computations to avoid recomputation (prefixes/combinations can repeat).
    sr_cache: Dict[str, float] = {}

    # Max elemental SR horizontal reference
    max_elem_sr = read_max_element_sr(SR_CSV_PATH, ELEMENTS)

    Ns: List[int] = []
    means: List[float] = []
    p05s: List[float] = []
    p95s: List[float] = []
    sample_sizes: List[int] = []
    modes: List[str] = []

    M = len(ELEMENTS)

    for N in range(N_MIN, N_MAX + 1):
        total_combinations = math.comb(M, N)

        formulas_N: List[str] = []
        sr_values_N: List[float] = []

        if total_combinations <= K:
            mode = "exhaustive"
            iterator = itertools.combinations(ELEMENTS, N)
            for combo in iterator:
                formula = alloy_formula_from_elements(combo)
                if formula in sr_cache:
                    sr_val = sr_cache[formula]
                else:
                    res = compute_alloy_supply_risk(formula)
                    sr_val = extract_sr_scalar(res)
                    sr_cache[formula] = sr_val
                formulas_N.append(formula)
                sr_values_N.append(sr_val)
        else:
            mode = "nested"
            for traj in trajectories:
                prefix = traj[:N]
                formula = alloy_formula_from_elements(prefix)
                if formula in sr_cache:
                    sr_val = sr_cache[formula]
                else:
                    res = compute_alloy_supply_risk(formula)
                    sr_val = extract_sr_scalar(res)
                    sr_cache[formula] = sr_val
                formulas_N.append(formula)
                sr_values_N.append(sr_val)

        sr_arr = np.asarray(sr_values_N, dtype=float)

        # Save per-N data
        out_csv = OUTPUT_DIR / f"L_N{N:02d}_SR.csv"
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("alloy_formula;SR_alloy\n")
            for form, val in zip(formulas_N, sr_values_N):
                f.write(f"{form};{val:.12g}\n")

        # Summary stats
        Ns.append(N)
        modes.append(mode)
        sample_sizes.append(int(sr_arr.size))
        means.append(float(np.mean(sr_arr)))
        p05s.append(float(np.quantile(sr_arr, 0.05)))
        p95s.append(float(np.quantile(sr_arr, 0.95)))
        #CHOIX A FAIRE !!
        #p05s.append(float(np.min(sr_arr)))   # now "lower envelope"
        #p95s.append(float(np.max(sr_arr)))   # now "upper envelope"

        print(
            f"N={N:2d} | mode={mode:10s} | C(M,N)={total_combinations:8d} | "
            f"n_samples={sr_arr.size:5d} | mean={means[-1]:.4g} | "
            f"p05={p05s[-1]:.4g} | p95={p95s[-1]:.4g}"
        )

    # Save summary CSV
    summary_csv = OUTPUT_DIR / "summary_sr_vs_n.csv"
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("N;mode;n_samples;mean;p05;p95\n")
        for N, mode, ns, m, q05, q95 in zip(Ns, modes, sample_sizes, means, p05s, p95s):
            f.write(f"{N};{mode};{ns};{m:.12g};{q05:.12g};{q95:.12g}\n")

    # -----------------------------
    # Plot (add origin (0,0))
    # -----------------------------
    Ns_plot   = [0] + Ns
    mean_plot = [0.0] + means
    p05_plot  = [0.0] + p05s
    p95_plot  = [0.0] + p95s

    plt.figure(figsize=(12, 6))
    plt.plot(Ns_plot, mean_plot, marker="o", linewidth=2.5, label="Mean SR_alloy")
    plt.plot(Ns_plot, p05_plot, linestyle="--", color="green", linewidth=2.0, label="5th percentile")
    plt.plot(Ns_plot, p95_plot, linestyle="--", color="red",  linewidth=2.0, label="95th percentile")

    # Horizontal reference: worst single-element SR
    plt.axhline(
        y=max_elem_sr,
        linestyle="-.",
        linewidth=2.0,
        color="black",
        label=f"Max elemental SR (over {M} elements)"
    )

    plt.title("Alloy Supply Risk vs number of elements")
    plt.xlabel("Number of elements in the alloy (N)")
    plt.ylabel("Alloy Supply Risk indicator (SR_alloy)")

    # Force axes to start at 0
    ax = plt.gca()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    #plt.grid(True, which="both", linestyle=":", linewidth=1.0)
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_png = OUTPUT_DIR / "sr_vs_n-2.png"
    plt.savefig(out_png, dpi=900)
    plt.close()

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
    print(f"- Summary CSV: {summary_csv}")
    print(f"- Plot PNG:    {out_png}")
    print(f"- Max elemental SR used for reference line: {max_elem_sr:.6g}")


if __name__ == "__main__":
    main()
