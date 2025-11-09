import pandas as pd
import numpy as np
import random
from collections import defaultdict
import re
import sys
import os

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent
INPUT_PATH = c_folder / "Pareto_with_Normalized_Composition.csv"
OUTPUT_PATH = c_folder / "pareto_plus_noise.csv"

# Crée du bruit pour démultiplier des échantillons faibles

# === Paramètres globaux ===
# Number of new alloys to generate versus the number of added new elements
# Here: 50 with only noise on stoechiometries (0 new elements), 50 with 1 new element, 25 with 2 new elements
AUGMENT_BY_ELEMENTS = {
    0: 50,
    1: 50,
    2: 25,
}

# Noise levels for existing stoechiometries (percentage of variation)
NOISE_ALL = 0.05
# Noise level for newly added elements (fixed fraction)
NOISE_NEW = 0.03

# Accès aux éléments depuis le config du projet
ELEMENTS = [
    'Al','Cu','Au','Ta','Pd','Zn','Ni','Mo','Ir','Ti','V','Y','Zr','W','Co',
    'Pt','Re','Rh','Ru','Fe','Si','Ga','Sb','As','Nb','Te','Sc','Hf','Ag','Hg',
    'In','Mn','Cr','Mg','Bi','Ge','Sn','Cd','Pb'
]

N_ELEMENTS = len(ELEMENTS)


def parse_formula(formula: str) -> dict:
    pattern = r"([A-Z][a-z]*)([0-9.]+)"
    matches = re.findall(pattern, formula)
    comp = defaultdict(float)
    for elem, frac in matches:
        comp[elem] += float(frac)
    total = sum(comp.values())
    if total > 0:
        for elem in comp:
            comp[elem] /= total
    return dict(comp)


def vector_to_formula(vec, threshold=1e-4):
    formula = ""
    for el, frac in zip(ELEMENTS, vec):
        if frac >= threshold:
            formula += f"{el}{round(frac, 6)}"
    return formula


def load_compositions(path):
    df = pd.read_csv(path, sep=";")
    return df["Normalized Composition"].tolist()


def add_elements_and_perturb(base_vec, n_new_elems: int, perturbation=NOISE_NEW):
    vec = base_vec.copy()
    available = [i for i in range(N_ELEMENTS) if vec[i] == 0.0]
    if len(available) < n_new_elems:
        return None  # Pas assez d’éléments disponibles à ajouter

    added_indices = random.sample(available, n_new_elems)
    for i in added_indices:
        vec[i] = perturbation

    # Diminuer les dominants pour compenser
    scale_down = 1.0 - n_new_elems * perturbation
    for i in range(N_ELEMENTS):
        if i not in added_indices and vec[i] > 0:
            vec[i] *= scale_down

    vec = np.clip(vec, 0, 1)
    vec /= vec.sum()

    return vec


def apply_relative_noise(vec, noise_level=0.1):
    noise = np.random.uniform(1 - noise_level, 1 + noise_level, size=vec.shape)
    noisy_vec = vec * noise
    noisy_vec /= noisy_vec.sum()
    return noisy_vec


def enrich_pareto(input_csv, output_csv):
    base_formulas = load_compositions(input_csv)
    enriched = []

    for formula in base_formulas:
        parsed = parse_formula(formula)
        base_vec = np.zeros(N_ELEMENTS)
        for i, el in enumerate(ELEMENTS):
            base_vec[i] = parsed.get(el, 0.0)

        for n_new_elems, n_variants in AUGMENT_BY_ELEMENTS.items():
            for _ in range(n_variants):
                if n_new_elems == 0:
                    vec = base_vec.copy()
                else:
                    vec = add_elements_and_perturb(base_vec, n_new_elems)
                    if vec is None:
                        continue

                vec = apply_relative_noise(vec, NOISE_ALL)
                enriched.append(vector_to_formula(vec))

    df = pd.DataFrame({"Normalized Composition": enriched})
    df.to_csv(output_csv, index=False)
    print(f"✅ Fichier enrichi sauvegardé : {output_csv}")
    print(f"🧪 {len(base_formulas)} originaux → {len(enriched)} enrichis")


# === Script exécutable directement
if __name__ == "__main__":
    enrich_pareto(INPUT_PATH, OUTPUT_PATH)
