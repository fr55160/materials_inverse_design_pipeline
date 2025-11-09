# evaluate_adaptative.py

"""
evaluate_adaptative.py

Évaluation « adaptative » du CVAE sur plusieurs dimensions latentes :
- Nettoyage des anciens artefacts
- Boucle : entraînement (train_adaptative), suivi epoch-wise
- Adaptation de α_vars et γ à chaque epoch
- Condition Active Learning (recon, KL, entropie, α trigger)
- Génération de NB_POINTS candidats, estimation incertitude (M_UNCERTAINTY décodages)
- Sélection des TOP_FRACTION candidats les plus incertains
- Filtrage (entropie + gᵢ ≥ min_g), stratégie corrective si aucun candidat valide
- Enregistrement des métriques et des compositions valides dans deux CSV
"""

import os, sys, warnings, shutil, csv
import numpy as np, joblib, tensorflow as tf
from sklearn.ensemble import IsolationForest

# pour N_ELEMENTS
from config_adaptative import (
    BASE_VAE_DIR, COARSE_LATENTS, NB_POINTS, M_UNCERTAINTY, TOP_FRACTION,
    RECON_THRESHOLD, KL_THRESHOLD, ENTROPY_THRESHOLD, alpha_trigger,
    beta_var, gamma_var, gamma_max, MIN_FRAC, test_target, TARGET_MODE,
    N_ELEMENTS, alpha_vars, min_g, BETA, GAMMA, ALPHA, BETA_MAX, GRID_RESULTS_CSV, POINTS_RESULTS_CSV
)

# vos modules
from preprocess_adaptative import load_and_preprocess, recalculate_all_properties, rebuild_X_from_df
from model_cvae_adaptative import build_cvae, get_decoder
from train_adaptative import train_cvae_adaptative
from utils_io_adaptative import vector_to_formula, parse_composition, transform_g

# ── 1) Nettoyage ──────────────────────────────────────────────────────────────
def safe_clean():
    print("🧹 Nettoyage adaptative : suppression des anciens artefacts…")
    for d in ["models","scalers","plots_training"]:
        p = os.path.join(BASE_VAE_DIR, d)
        if os.path.exists(p): shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)
    for f in [GRID_RESULTS_CSV,POINTS_RESULTS_CSV]:
        #fp = os.path.join(BASE_VAE_DIR, f)
        if os.path.exists(f): os.remove(f)
    print("    ✓ Fichiers et dossiers recréés.")
safe_clean()

# ── 3) Chargement des données ─────────────────────────────────────────────────
print("📥 Chargement et prétraitement…")
X_raw, Y_raw, _ = load_and_preprocess(max_rows=None)
# masque True pour les lignes où *toutes* les features de X_raw et de Y_raw sont finies
mask_finite = np.all(np.isfinite(X_raw), axis=1) & np.all(np.isfinite(Y_raw), axis=1)

n_total = X_raw.shape[0]
n_good  = mask_finite.sum()
n_bad   = n_total - n_good

if n_bad > 0:
    print(f"⚠️  Suppression de {n_bad} échantillons non-finis "
          f"({n_total} → {n_good})")
    X_raw = X_raw[mask_finite]
    Y_raw = Y_raw[mask_finite]
else:
    print("✅  Aucune ligne NaN/Inf détectée dans X_raw et Y_raw.")

# Filtrage par Isolation Forest pour X (pour éliminer les valeurs aberrantes)
clf = IsolationForest(contamination=0.001, random_state=42)
inlier = clf.fit_predict(X_raw) == 1
print(f"→ IsoForest : on retire {(~inlier).sum()} lignes sur {len(inlier)}")
X_raw, Y_raw = X_raw[inlier], Y_raw[inlier]

print(f"    X_raw.shape = {X_raw.shape}, Y_raw.mode = '{TARGET_MODE}', Y_dim = {transform_g(Y_raw[0], TARGET_MODE).shape[0]}")

# Affichage des paramètres Active Learning
print(f"▶️  ACTIVE LEARNING PARAMS : NB_POINTS={NB_POINTS}, "
      f"M_UNCERTAINTY={M_UNCERTAINTY}, TOP_FRACTION={TOP_FRACTION:.2f}")

# ── 2) Initialisation des CSV ─────────────────────────────────────────────────
res_csv = GRID_RESULTS_CSV
pts_csv = POINTS_RESULTS_CSV

# header pour les métriques
with open(res_csv, 'w', newline='') as f:
    csv.writer(f).writerow([
        "latent_dim","epoch",
        "recon_loss","kl_loss","entropy_loss",
        *[f"alpha_{i+1}" for i in range(10)],
        "n_valid","accuracy","diversity"
    ])

# header complet pour les candidats
n_features = X_raw.shape[1]  # 102
with open(pts_csv, 'w', newline='') as f:
    csv.writer(f).writerow([
        "latent_dim","epoch",
        "beta","gamma","alpha_norm",
        "composition",
        *[f"g{i+1}" for i in range(10)],
        *[f"x_feat_{i}" for i in range(n_features)]
    ])

# ── 4) Générateur d’incertitude ────────────────────────────────────────────────
def generate_active_candidates(model, test_target, n_to_generate):
    """
    → génère n_to_generate candidats
    → M_UNCERTAINTY décodages par candidat pour estimer l’incertitude
    → sélection des TOP_FRACTION les plus incertains
    Retourne (indices, formules, scores_g, entropies_moyennes)
    """
    decoder    = get_decoder(model, cond_dim=model.input[1].shape[-1])
    latent_dim = int(model.get_layer("z").output_shape[-1])
    z_samples  = np.random.normal(size=(n_to_generate, latent_dim))

    # condition Y
    y_cond  = transform_g(test_target, mode=TARGET_MODE)
    scaler_Y= joblib.load(os.path.join(BASE_VAE_DIR, "scalers", "scaler_Y.pkl"))
    y_scaled= scaler_Y.transform([y_cond])
    y_repeat= np.repeat(y_scaled, n_to_generate, axis=0)

    # allocation
    all_stoich    = np.zeros((n_to_generate, M_UNCERTAINTY, N_ELEMENTS))
    all_entropies = np.zeros((n_to_generate, M_UNCERTAINTY))

    scaler_X= joblib.load(os.path.join(BASE_VAE_DIR, "scalers", "scaler_X_std.pkl"))
    mean_std= scaler_X.mean_[:N_ELEMENTS]
    scale_std= scaler_X.scale_[:N_ELEMENTS]

    for m in range(M_UNCERTAINTY):
        x_decoded = decoder.predict([z_samples, y_repeat], batch_size=256)
        clr = x_decoded[:, :N_ELEMENTS]
        clr_u = clr * scale_std + mean_std
        expv = np.exp(np.clip(clr_u, -10, 10))
        sto  = expv / (expv.sum(axis=1, keepdims=True) + 1e-8)
        all_stoich[:, m, :]    = sto
        ent_m = -np.sum(sto * np.log(sto + 1e-8), axis=1)
        all_entropies[:, m]    = ent_m

    uncert = all_stoich.std(axis=1).mean(axis=1)
    k      = max(1, int(TOP_FRACTION * n_to_generate))
    sel_idx= np.argsort(uncert)[-k:]

    sto_sel = all_stoich[sel_idx, 0, :]
    forms   = [vector_to_formula(v) for v in sto_sel]
    df_props= recalculate_all_properties(forms)
    g_scores = df_props[[f"g{i+1}" for i in range(10)]].to_numpy()
    ent_sel  = all_entropies[sel_idx].mean(axis=1)

    print(f"    → Générés {n_to_generate} candidats, sélection {k} incertains.")
    # on renvoie aussi df_props pour pouvoir reconstruire X complet
    return sel_idx, forms, g_scores, ent_sel, df_props

# ── 5) Boucle principale ───────────────────────────────────────────────────────
for ld in COARSE_LATENTS:
    print(f"\n🔬 Évaluation adaptative pour latent_dim = {ld}")
    beta_var.assign(BETA)
    gamma_var.assign(GAMMA)
    alpha_vars.assign(ALPHA)

    # 5.1) entraînement
    model, history = train_cvae_adaptative(X_raw, Y_raw, latent_dim=ld, nb_points=NB_POINTS, min_frac=MIN_FRAC)
    print(f"    ✓ Modèle latent_dim={ld} entraîné, {len(history)} epochs enregistrées.")

    for rec in history:
        ep, r, k, e = rec["epoch"], rec["recon_loss"], rec["kl_loss"], rec["entropy_loss"]

        # condition active learning
        cond_AL = (k < KL_THRESHOLD) and (r < RECON_THRESHOLD) and (e < ENTROPY_THRESHOLD) and np.any(alpha_vars.numpy()>alpha_trigger)
        if cond_AL:
            print(f"    🎯 Génération de {NB_POINTS} candidats bruts…")
            sel, forms, gs, ents, df_props = generate_active_candidates(model, test_target, NB_POINTS)

            # 2️⃣ Filtrage entropie + comptage
            mask_ent = ents >= ENTROPY_THRESHOLD
            n_ent    = mask_ent.sum()
            print(f"    🔍 Filtre entropie (seuil={ENTROPY_THRESHOLD:.2f}) : {n_ent}/{len(sel)}")

            # 3️⃣ Filtrage gᵢ > min_g + comptage
            mask_g = (gs >= min_g).all(axis=1)
            n_g    = mask_g.sum()
            print(f"    🔍 Filtre gᵢ (min_g={min_g:.2f}) : {n_g}/{n_ent}")

            # Combinaison finale
            valid    = mask_ent & mask_g
            n_valid  = valid.sum()

            if n_valid == 0:
                # aucun candidat valide → on ne fait rien, tout est déjà ajusté dans train_adaptative
                n_valid = 0
            else:
                # 4️⃣ Stratégie de poursuite
                print(f"    ✅ Epoch {ep} : {n_valid}/{len(sel)} validés.")

                # ─── Métadonnées ───────────────────────────
                beta_val   = float(beta_var.numpy())
                gamma_val  = float(gamma_var.numpy())
                alpha_norm = float(tf.norm(alpha_vars).numpy())

                # ─── Extrait les lignes valides du DataFrame ─
                valid_idx = np.where(valid)[0]
                df_valid  = df_props.iloc[valid_idx]

                # ─── Reconstruit la matrice X (102 features) ─
                X_valid   = rebuild_X_from_df(df_valid)  # shape = (n_valid, 102)

                # ─── Écriture détaillée des candidats ──────
                with open(pts_csv, 'a', newline='') as f:
                    w = csv.writer(f)
                    for i, idx in enumerate(valid_idx):
                        comp = forms[idx]
                        gvec = gs[idx].tolist()
                        xvec = X_valid[i].tolist()
                        w.writerow([
                            ld,          # latent_dim
                            ep,          # epoch
                            beta_val,    # β
                            gamma_val,   # γ
                            alpha_norm,  # ||α||
                            comp,        # formule
                            *gvec,       # g1…g10
                            *xvec        # 102 features
                        ])
        else:
            # Pas d’AL cette époque : on force n_valid=0
            n_valid = 0

        # ── Calcul des métriques & écriture dans grid_adaptative_scores.csv ────────
        if n_valid > 0:
            acc = np.mean(np.linalg.norm(gs[valid] - test_target, axis=1))
            sto_v = np.vstack([parse_composition(f) for f in np.array(forms)[valid]])
            div   = np.std(sto_v, axis=0).mean() * n_valid / NB_POINTS
        else:
            acc, div = np.nan, 0.0

        row = [ld, ep, r, k, e, *alpha_vars.numpy().tolist(), n_valid, acc, div]
        with open(res_csv, 'a', newline='') as f:
            csv.writer(f).writerow(row)

print("✅ Évaluation adaptative terminée.")
