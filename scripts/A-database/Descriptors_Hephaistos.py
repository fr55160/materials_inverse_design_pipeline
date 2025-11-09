"""Compute composition-based descriptors for metallic alloys (Hephaistos class).

This module enriches the original Oracle descriptor set by using the `mendeleev` library
for all elemental data (no hard‑coded dicts) and adds new composition features:
    • avg/std subshell valence counts (s, p, d, f)
    • frac_d, avg/std d_shell_n
    • avg/std group, period, mendeleev_no
    • avg/std Allen electronegativity, IE1, electron affinity
    • avg/std total valence electrons (nvalence)
    • avg/std melting point

Public API
----------
parse_formula(formula: str) -> Dict[str, float]
compute_descriptors(formula: str) -> Dict[str, float]

Author: François Rousseau – 2025-04-20
"""
from __future__ import annotations
import re, math
from typing import Dict, List
import numpy as np
from mendeleev import element
import pandas as pd  
from pymatgen.core import Composition
from matminer.featurizers.composition.alloy import Miedema

# regex for formula parsing
def parse_formula(formula: str) -> Dict[str, float]:
    comp: Dict[str, float] = {}
    for el, num in re.findall(r"([A-Z][a-z]*)([0-9]*\.?[0-9]*)", formula):
        comp[el] = comp.get(el, 0.0) + (float(num) if num else 1.0)
    return comp

# cache elemental data
element_cache: Dict[str, Dict] = {}

def _elem_data(sym: str) -> Dict:
    if sym in element_cache:
        return element_cache[sym]
    e = element(sym)

    # —–– 1) Ionisation énergie (inchangé)
    ie1 = None
    # mendeleev stocke les IE dans un dict {ordre: valeur en eV} :contentReference[oaicite:4]{index=4}
    if hasattr(e, "ionenergies") and isinstance(e.ionenergies, dict):
        ie1 = e.ionenergies.get(1)

    # —–– 2) Nombre total d’électrons de valence (robuste) —––
    n_valence = None
    ec = getattr(e, "ec", None)
    if ec is not None and hasattr(ec, "get_valence"):
        try:
            val_cfg = ec.get_valence()
            # si get_valence() renvoie un objet EC avec un dict .conf
            if val_cfg is not None and hasattr(val_cfg, "conf") and val_cfg.conf:
                n_valence = sum(val_cfg.conf.values())
            else:
                # fallback sur la propriété calculée
                n_valence = getattr(e, "nvalence", None)
        except Exception:
            # en cas d’erreur interne à mendeleev, on retombe sur e.nvalence
            n_valence = getattr(e, "nvalence", None)
    else:
        # si pas d’EC, on utilise directement l’attribut
        n_valence = getattr(e, "nvalence", None)

    # —–– 3) Répartition par sous‑couche via ec.conf :contentReference[oaicite:4]{index=4}
    # ec.conf est OrderedDict [((n,'s'), occ), …]
    ec = getattr(e, "ec", None)
    n_s = n_p = n_d = n_f = None
    if ec is not None and hasattr(ec, "conf"):
        subs = {"s": 0, "p": 0, "d": 0, "f": 0}
        for (_n, sub), occ in ec.conf.items():
            if sub in subs:
                subs[sub] += occ
        n_s, n_p, n_d, n_f = subs["s"], subs["p"], subs["d"], subs["f"]

    # —–– 4) Autres attributs classiques
    ar = getattr(e, "atomic_radius", None) or getattr(e, "metallic_radius", None) or getattr(e, "atomic_radius_empirical", None)
    data = {
        "atomic_weight": getattr(e, "atomic_weight", None),
        "atomic_radius": ar,
        "en_Pauling": getattr(e, "en_pauling", None),
        "en_Allen":   getattr(e, "en_allen",   None),
        "IE1": ie1,
        "EA": getattr(e, "electron_affinity", None),
        "group": getattr(e, "group_id", None),
        "period": getattr(e, "period", None),
        "mendeleev_no": getattr(e, "mendeleev_number", None),
        "n_s": n_s, "n_p": n_p, "n_d": n_d, "n_f": n_f,
        "n_valence": n_valence,
        "d_shell_n": (e.period - 1) if getattr(e, "block", None) == 'd' else None,
        "melting_point": getattr(e, "melting_point", None),
        "atomic_number": getattr(e, "atomic_number", None),
    }
    element_cache[sym] = data
    return data


# main descriptor function
def compute_descriptors(formula: str) -> Dict[str, float]:
    comp = parse_formula(formula)
    total = sum(comp.values())
    frac = {el: v/total for el, v in comp.items()}

    # initialize sums and variances
    keys = ["atomic_radius","atomic_weight","en_Pauling","en_Allen","IE1","EA",
            "group","period","mendeleev_no","n_s","n_p","n_d","n_f","n_valence",
            "d_shell_n","melting_point", "atomic_number"]
    avg = {k:0.0 for k in keys}
    var = {k:0.0 for k in keys}

    # collect weighted means
    for el, x in frac.items():
        d = _elem_data(el)
        for k in keys:
            val = d.get(k)
            if val is None or callable(val):
                continue
            avg[k] += x * val
    # collect variances
    for el, x in frac.items():
        d = _elem_data(el)
        for k in keys:
            val = d.get(k)
            if val is None or callable(val):
                continue
            var[k] += x * (val - avg[k])**2
    std = {k: math.sqrt(var[k]) for k in keys}

    # stoichiometric entropy
    stoich_entropy = -sum(x * math.log(x) for x in frac.values() if x>0)

    # radius mismatch
    delta = std["atomic_radius"]/avg["atomic_radius"] if avg["atomic_radius"] else np.nan

    # d_virt descriptor
    d_virt = (1e7 / 6.02) * avg["atomic_weight"] / (avg["atomic_radius"] ** 3) if avg["atomic_radius"] else None

    # unconditional ranges for radius
    radii = [d["atomic_radius"] for d in (_elem_data(el) for el in comp) if isinstance(d.get("atomic_radius"), (int,float))]
    if radii:
        max_radius = max(radii)
        min_radius = min(radii)
        avg_r = avg["atomic_radius"]
        max_r_ratio = max_radius/avg_r if avg_r else None
        min_r_ratio = min_radius/avg_r if avg_r else None
    else:
        max_r_ratio = min_r_ratio = None

    # frac_d
    frac_d = None
    if isinstance(avg["n_valence"], (int, float)) and avg["n_valence"] > 0:
        frac_d = avg["n_d"] / avg["n_valence"]

    # electronegativity range
    eneg_vals: List[float] = []
    for el in comp:
        val = _elem_data(el).get("en_Pauling")
        if isinstance(val, (int, float)):
            eneg_vals.append(val)
    range_eneg = max(eneg_vals) - min(eneg_vals) if eneg_vals else np.nan

    # Miedema mixing enthalpy
    try:
        comp_obj=Composition(formula)
        mix=Miedema(impute_nan=True).featurize(comp_obj)
        DeltaHmix=sum(mix[:3])
    except:
        DeltaHmix=np.nan    

    desc = {
        "stoich_entropy": stoich_entropy,
        "avg_radius": avg["atomic_radius"], "std_radius": std["atomic_radius"],
        'max_r_ratio': max_r_ratio, 'min_r_ratio': min_r_ratio,
        "delta": delta,
        "avg_eneg": avg["en_Pauling"], "std_eneg": std["en_Pauling"], 'range_eneg': range_eneg,
        "avg_weight": avg["atomic_weight"], "std_weight": std["atomic_weight"],
        "unique_elements": len(comp),
        "avg_Z": avg["atomic_number"], "std_Z": std["atomic_number"],
        # VEC
        "avg_VEC": avg["n_valence"], "std_VEC": std["n_valence"],
        "avg_d": avg["n_d"], "std_d": std["n_d"],
        "frac_d": frac_d,
        # subshell avg/std
        **{f"avg_{s}": avg[cpy] for s,cpy in [("s","n_s"),("p","n_p"),("f","n_f")]},
        **{f"std_{s}": std[cpy] for s,cpy in [("s","n_s"),("p","n_p"),("f","n_f")]},
        # d_shell
        "avg_d_shell_n": avg["d_shell_n"], "std_d_shell_n": std["d_shell_n"],
        # group/period
        "avg_group": avg["group"], "std_group": std["group"],
        "avg_period": avg["period"], "std_period": std["period"],
        # Pettifor
        "avg_mendeleev_no": avg["mendeleev_no"], "std_mendeleev_no": std["mendeleev_no"],
        # Allen, IE, EA
        "avg_en_allen": avg["en_Allen"], "std_en_allen": std["en_Allen"],
        "avg_IE1": avg["IE1"], "std_IE1": std["IE1"],
        "avg_EA": avg["EA"], "std_EA": std["EA"],
        # melting point
        "avg_melting_point": avg["melting_point"], "std_melting_point": std["melting_point"],
        # mixing descriptor
        "DeltaH_mix": DeltaHmix,
        "d_virt": d_virt
    }
    return desc
