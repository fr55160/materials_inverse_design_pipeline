# Adapt with the proper name of 'source' and 'outfile' variables!

"""
generate_descriptors.py

Lit le fichier HT_oxidation.csv (séparateur ';', décimal ','),
applique les compute_descriptors du module Descriptors_Hephaistos
sur la colonne 'Normalized Composition', et écrit en sortie
outfile (séparateur ',', décimal '.').
"""

import pandas as pd
from Descriptors_Hephaistos import compute_descriptors
from pathlib import Path

c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

def main():
    print("Project folder:", PROJECT_FOLDER)
    source = 'test_database_0.csv'
    outfile = 'test_database_1.csv'
    # 1) Lecture du fichier d'entrée
    df = pd.read_csv(c_folder / source, sep=';', decimal=',', encoding='latin-1', engine='python')
    
    # 2) Calcul des descripteurs pour chaque formule normalisée
    #    compute_descriptors attend un string comme "Cr0.259Al0.004..."
    desc_series = df['Normalized Composition'].apply(compute_descriptors)
    
    # 3) Conversion en DataFrame et concaténation
    desc_df = pd.DataFrame(desc_series.tolist())
    df_out = pd.concat([df, desc_df], axis=1)
    
    # 4) Sauvegarde du résultat
    df_out.to_csv(c_folder / outfile, index=False, sep=";")
    print(f"Fichier généré : {outfile}")

if __name__ == '__main__':
    main()

