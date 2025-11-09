"""
train_adaptative.py

Entraînement du CVAE « adaptative » avec :
- Warm-up et adaptation multiplicative de β (KL), γ (entropie) et α_vars (récompense gᵢ)
- Logging epoch-wise des pertes [recon, KL, entropie]
- Sauvegarde des scalers et du modèle final
"""

import os, math, joblib, numpy as np, tensorflow as tf
import csv
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from config_adaptative import (
    BASE_VAE_DIR, BATCH_SIZE, EPOCHS_DEBUG, EPOCHS_NORMAL, DEBUG,
    KL_MIN, KL_MAX, BETA_MIN, BETA_MAX, beta_var,
    ENTROPY_THRESHOLD, gamma_var, gamma_max, ENTROPY_STO_MIN,
    RECON_THRESHOLD, alpha_vars, TARGET_MODE, ENTROPY_STO_MAX,
    NB_POINTS, M_UNCERTAINTY, TOP_FRACTION,
    KL_THRESHOLD, alpha_trigger, min_g, test_target, N_ELEMENTS,
    MIN_FRAC as min_frac
)
from model_cvae_adaptative import build_cvae, CVAELossLayer, get_decoder
from utils_io_adaptative import transform_g, vector_to_formula, parse_composition
from preprocess_adaptative import recalculate_all_properties

def train_cvae_adaptative(X_raw, Y_raw, latent_dim, nb_points, min_frac):
    # scalers X/Y
    scaler_dir = os.path.join(BASE_VAE_DIR, "scalers")
    os.makedirs(scaler_dir, exist_ok=True)

    scaler_X = StandardScaler().fit(X_raw)
    X = scaler_X.transform(X_raw)
    joblib.dump(scaler_X, os.path.join(scaler_dir, "scaler_X_std.pkl"))

    Y_t = np.vstack([transform_g(y, mode=TARGET_MODE) for y in Y_raw])
    scaler_Y = StandardScaler().fit(Y_t)
    Y = scaler_Y.transform(Y_t)
    joblib.dump(scaler_Y, os.path.join(scaler_dir, "scaler_Y.pkl"))

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    cvae, loss_name = build_cvae(
        input_dim=X.shape[1], cond_dim=Y.shape[1],
        latent_dim=latent_dim, beta=beta_var, gamma=gamma_var
    )
    loss_layer = cvae.get_layer(loss_name)

    history = []
    class HistoCB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            r, k, e, *_ = loss_layer.get_losses()
            history.append({"epoch": epoch, "recon_loss": float(r), "kl_loss": float(k), "entropy_loss": float(e)})

    class BetaCB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            # récupère recon, kl, entropie, reward, total
            recon, kl, entropy, reward, total = loss_layer.get_losses()
            old = float(beta_var.numpy())

            if kl > KL_MAX:
                new = min(old * 1.1, BETA_MAX)
                beta_var.assign(new)
                print(f"   🔄 KL={kl:.4f} > {KL_MAX:.3f} → β ↑ {old:.4f}→{new:.4f}")
            elif kl < KL_MIN:
                new = max(old * 0.95, BETA_MIN)
                beta_var.assign(new)
                print(f"   🔄 KL={kl:.4f} < {KL_MIN:.3f} → β ↓ {old:.4f}→{new:.4f}")

    class GammaCB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            # récupère recon, kl, entropie, reward, total
            recon, kl, entropy, reward, total = loss_layer.get_losses()
            old = float(gamma_var.numpy())

            if entropy > ENTROPY_THRESHOLD/2:
                new = min(old * 1.03, gamma_max.numpy())
                gamma_var.assign(new)
                print(f"   🔄 Entropy={entropy:.4f} > {ENTROPY_THRESHOLD/2:.2f} → γ ↑ {old:.4f}→{new:.4f}")
            # (sinon, on peut laisser γ stable ou ajouter un else pour décroître)

    class AlphaCB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            # récupère recon et kl de la couche de loss
            recon, kl, *_ = loss_layer.get_losses()
            # mise à jour ×1.03 uniquement si les deux seuils sont passés
            if (kl < KL_THRESHOLD) and (recon < RECON_THRESHOLD):
                old = alpha_vars.numpy().sum()
                new_vec = tf.minimum(alpha_vars * 1.03, 3.0)
                alpha_vars.assign(new_vec)
                new = alpha_vars.numpy().sum()
                print(f"   🔄 |α| = {old:.5f} → {new:.5f}")

    class EpochLossLogger(Callback):
        def __init__(self,
                     loss_layer_name,
                     beta_var,
                     gamma_var,
                     latent_dim,
                     recon_th,    
                     kl_th):      
            super().__init__()
            self.loss_layer_name = loss_layer_name
            self.beta_var   = beta_var
            self.gamma_var  = gamma_var
            self.latent_dim = latent_dim
            self.recon_th        = recon_th   
            self.kl_th           = kl_th
            self.reward_active = False

        def on_epoch_begin(self, epoch, logs=None):
            # remet les métriques à zéro au début de l’époque
            layer = self.model.get_layer(self.loss_layer_name)
            layer.reset_losses()

        def on_epoch_end(self, epoch, logs=None):
            layer = self.model.get_layer(self.loss_layer_name)
            # récupère toutes les pertes : recon, kl, entropy, reward, total
            recon, kl, entropy, reward_loss, total_loss = layer.get_losses()

            # 1) si on repasse sous les deux seuils, on active la reward pour toujours
            if (kl < self.kl_th) and (recon < self.recon_th):
                self.reward_active = True
                        
            # calcule la somme des alphaᵢ
            sum_alpha = float(alpha_vars.numpy().sum())

            # affiche uniquement |α|, reward et total
            print(f"[Epoch {epoch+1}] |α| = {sum_alpha:.4f}   "
                  f"reward_loss = {reward_loss:.4f}   total_loss = {total_loss:.4f}")

            # ——————— Détail de la décomposition ———————
            beta_val  = float(self.beta_var.numpy())
            gamma_val = float(self.gamma_var.numpy())
            # on décide comment afficher reward_loss
            if self.reward_active:
                # la reward est gelée (active ou non à cette epoch), on l'affiche toujours
                reward_part = f"{reward_loss:.4f} (reward_loss {'active' if (kl<self.kl_th and recon<self.recon_th) else 'frozen'})"
            else:
                # jamais activée, on continue de masquer
                reward_part = "0.0000 (reward_loss deactivated)"

            print(
                f"    breakdown: total_loss = {total_loss:.4f}  =  "
                f"{recon:.4f} (recon) + "
                f"{beta_val:.4f} (beta) * {kl:.4f} (kl_loss) + "
                f"{gamma_val:.4f} (gamma) * {entropy:.4f} (entropy_loss) + "
                f"{reward_part}"
            )

    class ActiveLearningCallback(Callback):
        def __init__(self,
                     base_dir,
                     test_target,
                     nb_points,
                     m_uncertainty,
                     top_fraction,
                     recon_th,
                     kl_th,
                     ent_th,
                     ent_sto_min,            
                     ent_sto_max,   
                     alpha_trigger,
                     min_frac,
                     min_g,
                     loss_name):
            super().__init__()
            self.base_dir       = base_dir
            self.test_target    = test_target
            self.nb_points      = nb_points
            self.m_uncertainty  = m_uncertainty
            self.top_fraction   = top_fraction
            self.recon_th       = recon_th
            self.kl_th          = kl_th
            self.ent_th         = ent_th
            self.ent_sto_min  = ent_sto_min    
            self.ent_sto_max  = ent_sto_max    
            self.alpha_trigger  = alpha_trigger
            self.min_frac          = min_frac
            self.min_g          = min_g

            # définit où écrire les candidats validés
            self.pts_csv   = os.path.join(self.base_dir, "generated_adaptative_candidates.csv")
            self.loss_name = loss_name
            if not os.path.exists(self.pts_csv):
                with open(self.pts_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "latent_dim","epoch","beta","gamma","alpha_norm","composition",
                        *[f"g{i+1}" for i in range(10)]
                    ])

        def on_epoch_end(self, epoch, logs=None):
            # récupère les losses du epoch
            layer = self.model.get_layer(self.loss_name)  
            # ou stockez le nom du loss layer dans self.loss_name
            recon, kl, ent, _, _ = layer.get_losses()

            # récupère α_vars et test_target
            alpha_vals = alpha_vars.numpy()
            sum_alpha = alpha_vals.sum()
            cond_AL = (
                (kl    < self.kl_th)
                and (recon < self.recon_th)
                and (ent   < self.ent_th)
                and (sum_alpha > self.alpha_trigger)
            )

            if not cond_AL:
                return  # on ne génère pas

            print(f"    🎯 [AL] Epoch {epoch+1}: génération de {self.nb_points} candidats…")

            # --- Génération & incertitude ---
            decoder = get_decoder(self.model,
                                  cond_dim=self.model.input[1].shape[-1])
            latent_dim = int(decoder.input[0].shape[-1])
            z_samples  = np.random.normal(size=(self.nb_points, latent_dim))

            # condition scale+repeat
            y_cond   = transform_g(self.test_target, mode=TARGET_MODE)
            scaler_Y = joblib.load(os.path.join(self.base_dir, "scalers", "scaler_Y.pkl"))
            y_scaled = scaler_Y.transform([y_cond])
            y_repeat = np.repeat(y_scaled, self.nb_points, axis=0)

            # échantillonne M fois pour l’incertitude
            sto_all = np.zeros((self.nb_points, self.m_uncertainty, N_ELEMENTS))
            ent_all = np.zeros((self.nb_points, self.m_uncertainty))
            scaler_X = joblib.load(os.path.join(self.base_dir, "scalers","scaler_X_std.pkl"))
            mean_std = scaler_X.mean_[:N_ELEMENTS]
            scale_std= scaler_X.scale_[:N_ELEMENTS]

            for m in range(self.m_uncertainty):
                x_dec = decoder.predict([z_samples, y_repeat], batch_size=256)
                clr   = x_dec[:, :N_ELEMENTS]
                clr_u = clr * scale_std + mean_std
                expv  = np.exp(np.clip(clr_u, -10, 10))
                sto   = expv / (expv.sum(axis=1, keepdims=True)+1e-8)
                sto_all[:, m, :] = sto
                ent_all[:, m]    = -np.sum(sto*np.log(sto+1e-8), axis=1)

            uncert = sto_all.std(axis=1).mean(axis=1)
            k      = max(1, int(self.top_fraction * self.nb_points))
            sel_idx= np.argsort(uncert)[-k:]

            # 1) on supprime les éléments trace < min_frac et on renormalise
            sto_sel    = sto_all[sel_idx, 0, :]            # shape = (k, N_ELEMENTS)
            mask_big   = sto_sel >= self.min_frac          # True si composant >= min_frac
            sto_thresh = sto_sel * mask_big                # passe à zéro les traces
            sums       = sto_thresh.sum(axis=1, keepdims=True) + 1e-8
            sto_norm   = sto_thresh / sums                 # renorm en 1

            k = len(sel_idx)
            print(f"    → {k}/{self.nb_points} candidats incertains retenus")

            # 2) calcul de l'entropie stœchiométrique sur sto_norm (renormalisé)
            #    et filtrage dans [ENTROPY_STO_MIN, ENTROPY_STO_MAX]
            ent_sto = -np.sum(sto_norm * np.log(sto_norm + 1e-8), axis=1)
            mask_ent = (ent_sto >= ENTROPY_STO_MIN) & (ent_sto <= ENTROPY_STO_MAX)
            n_ent    = int(mask_ent.sum())
            print(
                f"    🔍 Filtre entropie stœchio "
                f"([{ENTROPY_STO_MIN:.1f},{ENTROPY_STO_MAX:.1f}]) : {n_ent}/{k}"
            )

            # 3) on ne calcule les gᵢ QUE pour ceux qui ont passé l’entropie
            sto_ent   = sto_norm[mask_ent]                 # (n_ent, N_ELEMENTS)
            forms_ent = [vector_to_formula(v) for v in sto_ent]

            # recalcul des propriétés + g-scores
            df_props  = recalculate_all_properties(forms_ent)
            # 0.a) Cas où aucun matériau n'a pu être featuré 
            if df_props.empty:
                print(f"⚠️ [AL] Aucune propriété calculable (df_props vide) → β×1.1, |α|=0.05, γ×1.0")
                # 1) |α|=0.05
                old_alpha = float(tf.reduce_sum(alpha_vars).numpy())
                #alpha_vars.assign(alpha_vars * 0.9)
                alpha_vars.assign(
                    alpha_vars * (0.05 / tf.reduce_sum(alpha_vars))
                )
                new_alpha = float(tf.reduce_sum(alpha_vars).numpy())
                print(f"   🔄 |α| = {old_alpha:.4f} → {new_alpha:.4f}")
                # 2) augmente γ de 0 %
                old_gamma = float(gamma_var.numpy())
                new_gamma = min(old_gamma * 1.0, gamma_max.numpy())
                gamma_var.assign(new_gamma)
                print(f"   🔄 γ = {old_gamma:.4f} → {new_gamma:.4f}")
                # 3) augmente β de 20 %
                old_beta = float(beta_var.numpy())
                new_beta = min(old_beta * 1.10, BETA_MAX)
                beta_var.assign(new_beta)
                print(f"   🔄 β = {old_beta:.4f} → {new_beta:.4f}")
                return

            # on extrait dynamiquement les colonnes g1…g10
            g_cols = [c for c in df_props.columns
                      if c.lower().startswith('g') and c[1:].isdigit()]

            # 1.b) Si on n’en a pas 10, on traite aussi comme « aucun valide »
            if len(g_cols) != 10:
                print(f"    ⚠️ [AL] Colonnes g manquantes → stratégie corrective")
                # idem corrective…
                old_alpha = float(tf.reduce_sum(alpha_vars).numpy())
                alpha_vars.assign(alpha_vars * 0.9)
                new_alpha = float(tf.reduce_sum(alpha_vars).numpy())
                old_gamma = float(gamma_var.numpy())
                gamma_var.assign(min(old_gamma * 1.10, gamma_max.numpy()))
                old_beta = float(beta_var.numpy())
                beta_var.assign(min(old_beta * 1.10, BETA_MAX))
                return

            # sinon, on récupère les g-scores
            gs = df_props[g_cols].to_numpy()

            # 4) filtrage gᵢ
            mask_g = (gs >= self.min_g).all(axis=1)
            n_g    = int(mask_g.sum())
            print(f"    🔍 Filtre gᵢ (≥{self.min_g:.2f}) : {n_g}/{n_ent}")

            # positions relatives dans forms_ent/gs
            valid_positions = np.where(mask_g)[0]
            n_valid = len(valid_positions)

            if n_valid == 0:
                print(f"    🚨 Epoch {epoch+1} : aucun candidat valide → β←β×lambda_KL, α←|α|=0.05, γ←γ×1.0")
                # — α ← α×1.0
                old_alpha = float(tf.reduce_sum(alpha_vars).numpy())
                #alpha_vars.assign(alpha_vars * 1.0)
                alpha_vars.assign(
                    alpha_vars * (0.05 / tf.reduce_sum(alpha_vars))
                )
                new_alpha = float(tf.reduce_sum(alpha_vars).numpy())
                print(f"   🔄 |α| = {old_alpha:.4f} → {new_alpha:.4f}")
                # — γ ← γ×1.0
                old_gamma = float(gamma_var.numpy())
                new_gamma = min(old_gamma * 1.0, gamma_max.numpy())
                gamma_var.assign(new_gamma)
                print(f"   🔄 γ = {old_gamma:.4f} → {new_gamma:.4f}")
                # — adaptation de β en fonction de la perte KL
                old_beta = float(beta_var.numpy())
                if kl > 0.2:
                    # KL trop élevée → on augmente β pour contraindre davantage
                    new_beta = min(old_beta * 1.1, BETA_MAX)
                    beta_var.assign(new_beta)
                    print(f"   🔄 KL={kl:.4f} > 0.20 → β ↑ {old_beta:.4f}→{new_beta:.4f}")
                elif kl < 0.10:
                    # KL trop faible → on baisse β pour laisser plus de créativité
                    new_beta = max(old_beta * 0.9, BETA_MIN)
                    beta_var.assign(new_beta)
                    print(f"   🔄 KL={kl:.4f} < 0.10 → β ↓ {old_beta:.4f}→{new_beta:.4f}")
                else:
                    # KL dans la bande de tolérance : on ne touche pas à β
                    print(f"   🔄 KL={kl:.4f} ∈ [0.10,0.20] → β stable ({old_beta:.4f})")
            else:
                print(f"    ✅ [AL] {n_valid}/{k} candidats validés → écriture CSV")
                with open(self.pts_csv, 'a', newline='') as f:
                    w = csv.writer(f)
                    b      = float(beta_var.numpy())
                    g_val  = float(gamma_var.numpy())
                    a_norm = float(tf.norm(alpha_vars).numpy())
                    for rel_pos in valid_positions:
                        w.writerow([
                            latent_dim,
                            epoch+1,
                            b,
                            g_val,
                            a_norm,
                            forms_ent[rel_pos],      # formule renormalisée
                            *gs[rel_pos].tolist()    # g-scores correspondants
                        ])



    epochs = EPOCHS_DEBUG if DEBUG else EPOCHS_NORMAL

    print(f"▶️  Lancement entraînement adaptative (latent_dim={latent_dim})")
    print(f"    β init = {beta_var.numpy():.4f}, γ init = {gamma_var.numpy():.4f}")
    print(f"    α init =", [f"{v:.4f}" for v in alpha_vars.numpy()])
    print(f"    X_train.shape = {X_train.shape}, Y_train.shape = {Y_train.shape}")
    print(f"    Époques = {epochs}, batch_size = {BATCH_SIZE}")

    al_cb = ActiveLearningCallback(
        base_dir      = BASE_VAE_DIR,
        test_target   = test_target,
        nb_points     = NB_POINTS,
        m_uncertainty = M_UNCERTAINTY,
        top_fraction  = TOP_FRACTION,
        recon_th      = RECON_THRESHOLD,
        kl_th         = KL_THRESHOLD,
        ent_th        = ENTROPY_THRESHOLD,
        ent_sto_min   = ENTROPY_STO_MIN,
        ent_sto_max   = ENTROPY_STO_MAX,
        alpha_trigger = alpha_trigger,
        min_g         = min_g,
        min_frac      = min_frac,
        loss_name     = loss_name
    )

    cvae.fit(
        [X_train, Y_train], X_train,
        validation_data=([X_val, Y_val], X_val),
        epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=[
            EpochLossLogger(
                loss_name,
                beta_var,
                gamma_var,
                latent_dim,
                RECON_THRESHOLD,  
                KL_THRESHOLD
            ),
            AlphaCB(),           
            HistoCB(),
            BetaCB(),
            GammaCB(),
            al_cb                       # ActiveLearningCallback
        ],
        verbose=0
    )

    model_dir = os.path.join(BASE_VAE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"cvae_adaptative_latent{latent_dim}.keras")
    cvae.save(model_path)

    return cvae, history
