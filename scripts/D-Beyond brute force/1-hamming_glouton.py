#!/usr/bin/env python3

import itertools
import csv
import random

from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

output_file = c_folder / "Hamming_filled_space.csv"

# List of the potential chemical elements
elements = [
    "Al", "Cu", "Au", "Ta", "Pd", "Zn", "Ni", "Mo", "Ir", "Ti", 
    "V", "Y", "Zr", "W", "Co", "Pt", "Re", "Rh", "Ru", "Fe", 
    "Si", "Ga", "Sb", "As", "Nb", "Te", "Sc", "Hf", "Ag", "Hg", 
    "In", "Mn", "Cr", "Mg", "Bi", "Ge", "Sn", "Cd", "Pb"
]

# Parameters to adjust
n = len(elements)      # total number of available elements: 39
w = 5                  # number of elements in each alloy
d_min = 6             # minimum Hamming distance between two alloys

# Seuil d'intersection maximal :
# d = 2*(w - t) >= d_min  =>  t <= w - d_min/2
intersection_max = w - d_min // 2

# Génération de toutes les combinaisons d'alliages de poids w
combinaisons = list(itertools.combinations(elements, w))
# On peut mélanger pour éviter un biais séquentiel
grandom_seed = 0
random.seed()
random.shuffle(combinaisons)

# Algorithme glouton de sélection
selection = []
for c in combinaisons:
    # Vérifier la condition de distance minimale avec chaque alliage déjà sélectionné
    if all(len(set(c) & set(sel)) <= intersection_max for sel in selection):
        selection.append(c)

# Écriture du résultat dans un fichier CSV
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f, delimiter=';')
    for alloy in selection:
        writer.writerow(alloy)

# Affichage du nombre d'alliages générés
print(f"Nombre d'alliages générés : %d\n", len(selection))
