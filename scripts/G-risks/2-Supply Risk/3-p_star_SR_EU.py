# p_star_builder.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent

# -----------------------
# 1) LIRE LES FICHIERS D'ENTRÉE
# -----------------------
# Fichier 1 : matrice de corrélation (Sigma_E.csv)
IN_CSV_CORR = c_folder / "Sigma_E.csv"   # sep=';'
df_raw = pd.read_csv(IN_CSV_CORR, sep=";", header=0, index_col=0)

# Fichier 2 : supply risk (SR_EU.csv)
# Format : deux lignes ("Element" et "SR")
IN_CSV_SR = c_folder / "SR_EU.csv"
df_sr = pd.read_csv(IN_CSV_SR, sep=";", header=None, index_col=0)

# Vérification minimale
if not {"Element", "SR"}.issubset(df_sr.index):
    raise ValueError("Le fichier SR_EU.csv doit contenir deux lignes intitulées 'Element' et 'SR'.")

# -----------------------
# 1 bis) PRÉPARATION DES DONNÉES
# -----------------------
# Éléments présents (colonnes) dans Sigma_E
elements_in_file = [c.strip() for c in df_raw.columns.tolist()]
row_labels = [r.strip() for r in df_raw.index.tolist()]

# Matrice de corrélation (nettoyée)
corr_df = df_raw.astype(float)

# Extraction des noms d'éléments et des valeurs SR
elements_from_sr = [str(e).strip() for e in df_sr.loc["Element"].tolist()]
values_sr = [float(v) for v in df_sr.loc["SR"].tolist()]

# Création de la série SR
sr_series = pd.Series(values_sr, index=elements_from_sr, dtype=float)

# Harmoniser l'ordre/alignement : ne garder que les éléments communs
common = [e for e in elements_in_file if e in corr_df.index and e in corr_df.columns and e in sr_series.index]

corr_df = corr_df.loc[common, common]
sr_series = sr_series.loc[common]

# Messages de contrôle
print(f"[OK] Matrice de corrélation : {corr_df.shape}, Série SR : {sr_series.shape}")
print(f"[OK] Éléments communs ({len(common)}): {', '.join(common)}")



# -----------------------
# 2) CALCULER theta et kappa
# -----------------------
# Clamp des corrélations très légèrement hors bornes par arrondis
eps_clip = 1e-12
C_raw = corr_df.to_numpy(dtype=float)
C = np.clip(C_raw, -1.0 + eps_clip, 1.0 - eps_clip)

# theta_ij = arccos(c_ij) en radians
theta = np.arccos(C)

# kappa_ij = ln(max(SR_i, SR_j)/min(SR_i, SR_j))
SR = sr_series.to_numpy(dtype=float)
if np.any(SR <= 0):
    raise ValueError("Tous les SR doivent être strictement positifs.")
SR_i = SR.reshape(-1, 1)
SR_j = SR.reshape(1, -1)
kappa = np.log(np.maximum(SR_i, SR_j) / np.minimum(SR_i, SR_j))

# -----------------------
# 3) CALCULER p_star
#    p_star_ij = exp(-kappa_ij / tan(theta_ij)^2)
# -----------------------
# Gestion num. : tan(0) = 0 -> division par zéro ; on mettra la diagonale à 1 ensuite.
tan_theta = np.tan(theta)
den = tan_theta ** 2

# Protéger contre den ~ 0 par sécurité (hors diagonale cela ne devrait pas arriver si |c_ij| < 1)
den = np.where(den <= 0, np.finfo(float).tiny, den)

p_star = np.exp(-kappa / den)

# --- FORCER p_star_ij = 0 quand c_ij = 1 (à tolérance près) ---
tol = 1e-12  # tolérance numérique pour "1"
mask_c_eq_1 = (C_raw >= 1.0 - tol)
p_star[mask_c_eq_1] = 0.0

p_star_df = pd.DataFrame(p_star, index=common, columns=common)

# -----------------------
# 4) RÉORDONNER ET ENREGISTRER p_star.csv
# -----------------------
target_order_str = """Mn	Pd	Cr	Rh	Pt	Ru	Ir	Ta	Nb	Sn	Sc	Co	Ni	Sm	V	La	Ce	Fe	Au	Ti	Zr	Hf	Ge	As	Ag	In	Cd	Zn	Te	Cu	Re	Mo	Y	Pb	Bi	Sb	Ga	Al	Si	Hg	W	Mg"""
target_order = [e.strip() for e in target_order_str.split()]

# Ne conserver que les éléments présents + respecter l'ordre cible
order_present = [e for e in target_order if e in p_star_df.index]
p_star_ord = p_star_df.loc[order_present, order_present]

# Sauvegarde CSV (sep=';')
OUT_CSV = c_folder / "p_star.csv"
p_star_ord.to_csv(OUT_CSV, sep=';', float_format="%.10g")

# -----------------------
# 5) HEATMAP LOG-SCALE ET ENREGISTREMENT 900 dpi
# -----------------------
# Norme logarithmique : p_star ∈ (0, 1]; on fixe vmin au min non-nul observé
vals = p_star_ord.to_numpy()
vals_pos = vals[vals > 0]
# --- forcer l'échelle de couleur ---
VMIN, VMAX = 1e-6, 1.0  # échelle log de 1e-10 à 1e0

vals = p_star_ord.to_numpy(dtype=float)
# En log-scale, les zéros sont interdits : on clippe très légèrement au-dessus de 0
vals = np.clip(vals, VMIN*1e-6, VMAX)

plt.figure(figsize=(10, 8))
im = plt.imshow(vals, norm=LogNorm(vmin=VMIN, vmax=VMAX))
plt.xticks(ticks=np.arange(len(order_present)), labels=order_present, rotation=90)
plt.yticks(ticks=np.arange(len(order_present)), labels=order_present)
plt.title("Matrix $p^*_{i,j}$ (log-scale)")
cbar = plt.colorbar(im)
cbar.set_label("$p^*_{i,j}$ (log scale)")

plt.tight_layout()
plt.savefig(c_folder / "Figures" / "p_star_heatmap.png", dpi=900, bbox_inches="tight")
plt.close()

print(f"✅ Terminé. Fichiers écrits : {OUT_CSV} et p_star_heatmap.png")
