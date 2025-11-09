import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import textwrap

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent

# --- 1) Lecture et pivot des données ---

# Chargement du CSV source
df_path = c_folder / "PDP" / "random_composition_analysis.csv"
df = pd.read_csv(df_path, sep=";", decimal=".")

# Vérification et création des colonnes manquantes
# ------------------------------------------------

# 1️⃣ General Mean = Mean - Delta
if "General Mean" not in df.columns:
    if {"Mean", "Delta"} <= set(df.columns):
        df["General Mean"] = df["Mean"] - df["Delta"]
    else:
        raise ValueError("Impossible de créer 'General Mean' : colonnes 'Mean' et 'Delta' manquantes.")

# 2️⃣ Définir le signe de contribution selon le type de Score
if "Relative Delta" not in df.columns:
    # Par défaut, signe positif
    df["sign_contrib"] = 1
    df.loc[df["Score"].isin(["Density", "Log_10(K_A)"]), "sign_contrib"] = -1

    if {"Delta", "General Mean"} <= set(df.columns):
        df["Relative Delta"] = df["sign_contrib"] * df["Delta"] / df["General Mean"]
    else:
        raise ValueError("Impossible de créer 'Relative Delta' : colonnes 'Delta' ou 'General Mean' manquantes.")

# 3️⃣ Relative std = Std / General Mean
if "Relative std" not in df.columns:
    if {"Std", "General Mean"} <= set(df.columns):
        df["Relative std"] = df["Std"] / df["General Mean"]
    else:
        raise ValueError("Impossible de créer 'Relative std' : colonnes 'Std' ou 'General Mean' manquantes.")

# Nettoyage : on peut supprimer la colonne temporaire sign_contrib si elle n’existait pas
if "sign_contrib" in df.columns:
    df.drop(columns=["sign_contrib"], inplace=True, errors="ignore")

# Sauvegarde du fichier mis à jour
df.to_csv(df_path, sep=";", decimal=".", index=False)
print(f"✅ Fichier mis à jour et sauvegardé avec les colonnes nécessaires : {df_path}")

# Pivot pour reconstituer les tableaux (n_elem × n_score)
mean_df = df.pivot(index="Element", columns="Score", values="Relative Delta")
std_df  = df.pivot(index="Element", columns="Score", values="Relative std")
prop_df = df.pivot(index="Element", columns="Score", values="Prop_Positive")

elements = mean_df.index.tolist()
scores   = mean_df.columns.tolist()

# --- 2) Tri des éléments par numéro atomique ---
atomic_numbers = {
    "Ag":47, "Al":13, "As":33, "Au":79, "Bi":83, "Cd":48,
    "Co":27, "Cr":24, "Cu":29, "Fe":26, "Ga":31, "Ge":32,
    "Hf":72, "Hg":80, "In":49, "Ir":77, "Mg":12, "Mn":25,
    "Mo":42, "Nb":41, "Ni":28, "Pb":82, "Pd":46, "Pt":78,
    "Re":75, "Rh":45, "Ru":44, "Sb":51, "Sc":21, "Si":14,
    "Sn":50, "Ta":73, "Te":52, "Ti":22, "V":23, "W":74,
    "Y":39,  # si Y présent
    "Zn":30, "Zr":40
}

# ne garder que les atomes effectivement dans le csv
elements = [el for el in elements if el in atomic_numbers]
elements_sorted = sorted(elements, key=lambda e: atomic_numbers[e])

mean_df = mean_df.loc[elements_sorted]
std_df  = std_df.loc[elements_sorted]
prop_df = prop_df.loc[elements_sorted]

elements = elements_sorted
n_elems, n_scores = mean_df.shape

# --- 3) Extraction des matrices numpy ---
c = mean_df.values    # Δ_rel
s = std_df.values     # σ_rel

# --- 4) Masques de signe certain / ambigu ---
mask_pos = (c - s) > 0                 # vert
mask_neg = (c + s) < 0                 # rouge
mask_unc = ~mask_pos & ~mask_neg       # jaune

# --- 5) Intensités [0..1] pour chaque zone ---
I_g = np.zeros_like(c)   # pour le vert
I_r = np.zeros_like(c)   # pour le rouge
I_y = np.zeros_like(c)   # pour le jaune

for j in range(n_scores):
    # 5.1 vert ∝ c
    idx = mask_pos[:, j] & ~mask_unc[:, j]
    if idx.any():
        M = c[idx, j].max()
        if M>0:
            I_g[idx, j] = c[idx, j] / M

    # 5.2 rouge ∝ |c|
    idx = mask_neg[:, j] & ~mask_unc[:, j]
    if idx.any():
        M = np.abs(c[idx, j]).max()
        if M>0:
            I_r[idx, j] = np.abs(c[idx, j]) / M

    # 5.3 jaune ∝ min(|c−s|,|c+s|)
    idx = mask_unc[:, j]
    if idx.any():
        ab = np.minimum(np.abs(c[:,j] - s[:,j]),
                        np.abs(c[:,j] + s[:,j]))
        M  = ab[idx].max()
        if M>0:
            I_y[idx, j] = ab[idx] / M

# --- 6) Définition des couleurs de base ---
GREEN  = np.array([0x2E, 0x8B, 0x57]) / 255  # sequoia-like
YELLOW = np.array([0xFF, 0xFF, 0x33]) / 255  # lemon
RED    = np.array([0xC4, 0x1E, 0x3A]) / 255  # écarlate

# --- 7) Construction de l’image RGB (blanc par défaut) ---
img = np.ones((n_elems, n_scores, 3))

# overlay vert
idx = mask_pos & ~mask_unc
v   = I_g[idx]
img[idx] = img[idx] * (1-v)[:,None] + GREEN  * v[:,None]

# overlay rouge
idx = mask_neg & ~mask_unc
v   = I_r[idx]
img[idx] = img[idx] * (1-v)[:,None] + RED    * v[:,None]

# overlay ambigu (jaune mélangé au vert/rouge selon signe)
idx = mask_unc
vy  = I_y[idx]
sign_col = np.zeros((vy.shape[0],3))
cvals    = c[idx]
sign_col[cvals>0] = GREEN
sign_col[cvals<0] = RED
img[idx] = sign_col * (1-vy)[:,None] + YELLOW * vy[:,None]

# ─── Renommage spécifique des abréviations ────────────────────────────────
rename_map = {
    "LMP":                 "Creep Resistance (LMP)",
    "Log_10(K_A)":         "HT Oxidation Resistance",
    "Melting_Temperature": "Melting Point"
}
# on remplace dans la liste scores
scores = [ rename_map.get(s, s) for s in scores ]

# --- 8) Affichage A4 portrait ---
fig, ax = plt.subplots(figsize=(8.27, 11.69))
ax.imshow(img, aspect="auto", origin="lower")

# abscisses = scores physiques
xticks = np.arange(n_scores)
wrapped = [ "\n".join(textwrap.wrap(s, width=12)) for s in scores ]
ax.set_xticks(xticks)
ax.set_xticklabels(wrapped, fontsize=9, fontname="Arial", rotation=0)
ax.set_xlim(-0.5, n_scores-0.5)

# ordonnées = éléments
yticks = np.arange(n_elems)
ax.set_yticks(yticks)
ax.set_yticklabels(elements, fontsize=9, fontname="Arial")
ax.set_ylim(-0.5, n_elems-0.5)

ax.set_xlabel("Physical Property",        fontname="Arial", fontsize=10)
ax.set_ylabel("Element",                  fontname="Arial", fontsize=10)
ax.set_title(
    "Relative Δmean and σ for Physical Properties",
    fontname="Arial", fontsize=12, pad=15
)

# légende
legend_handles = [
    Patch(facecolor=GREEN,  edgecolor="black",
          label=r'Positive ($\Delta_{\mathrm{rel}}-\sigma_{\mathrm{rel}}>0$)'),
    Patch(facecolor=RED,    edgecolor="black",
          label=r'Negative ($\Delta_{\mathrm{rel}}+\sigma_{\mathrm{rel}}<0$)'),
    Patch(facecolor=YELLOW, edgecolor="black",
          label=r'Ambiguous ($\Delta_{\mathrm{rel}}-\sigma_{\mathrm{rel}}<0<\Delta_{\mathrm{rel}}+\sigma_{\mathrm{rel}}$)'),
    Patch(facecolor="white",edgecolor="black",
          label="Neutral / no significant effect")
]
leg = ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.21),
    ncol=1,
    frameon=False
)
for txt in leg.get_texts():
    txt.set_fontname("Arial")
    txt.set_fontsize(9)

plt.tight_layout()
fig.subplots_adjust(bottom=0.22, left=0.18, right=0.95, top=0.92)

# --- 9) Sauvegarde haute déf. ---
plt.savefig(c_folder / "PDP" / "heatmap_PDP_relative.png", dpi=900, bbox_inches="tight")
plt.show()

# --- Générer une seconde heatmap pour Prop_Positive ---
fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))

# j'utilise prop_df.values qui a la même shape (n_elems × n_scores)
# on choisit par exemple une palette continue du blanc au bleu
im = ax2.imshow(
    prop_df.values,
    aspect='auto',
    origin='lower',
    cmap='hot',
    vmin=0.0,
    vmax=1.0
)

# mêmes ticks que pour la première heatmap
ax2.set_xticks(np.arange(n_scores))
ax2.set_xticklabels(
    [ "\n".join(textwrap.wrap(s, width=12)) for s in scores ],
    fontsize=9, fontname="Arial", rotation=0
)
ax2.set_xlim(-0.5, n_scores-0.5)

ax2.set_yticks(np.arange(n_elems))
ax2.set_yticklabels(elements, fontsize=9, fontname="Arial")
ax2.set_ylim(-0.5, n_elems-0.5)

ax2.set_xlabel("Physical Property", fontname="Arial", fontsize=10)
ax2.set_ylabel("Element",          fontname="Arial", fontsize=10)
ax2.set_title(
    "Proportion of Positive Contributions",
    fontname="Arial", fontsize=12, pad=15
)

# ajout d'une barre de couleur
cbar = fig2.colorbar(
    im,
    ax=ax2,
    orientation='horizontal',
    fraction=0.046,
    pad=0.04
)
cbar.set_label("Positive Proportion", fontname="Arial", fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
fig2.subplots_adjust(bottom=0.15, left=0.18, right=0.95, top=0.92)

# sauvegarde haute résolution
plt.savefig(c_folder / "PDP" / "heatmap_Prop_Positive.png", dpi=900, bbox_inches="tight")
plt.show()
