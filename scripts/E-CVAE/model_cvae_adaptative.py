"""
model_cvae_adaptative.py

Définition du CVAE conditionnel avec couche de perte personnalisée :
- reconstruction MSE + β·KL + γ·entropie
- récompense linéaire adaptative –αᵢ·ĝᵢ dès que recon < RECON_THRESHOLD ET KL < KL_THRESHOLD
- Permet l’Active Learning sur les scores gᵢ
"""

import os
import joblib
import numpy as np
import tensorflow as tf
from uuid import uuid4
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.layers import Layer, Input
from keras.saving import register_keras_serializable

from config_adaptative import (
    ELEMENTS, N_ELEMENTS, BASE_VAE_DIR, TARGET_MODE,
    ALPHA, alpha_vars,
    RECON_THRESHOLD, KL_THRESHOLD
)

@register_keras_serializable()
class CVAELossLayer(Layer):
    """
    Couche de perte personnalisée pour CVAE adaptatif :
    - recon + β KL + γ entropie
    - récompense linéaire sur gᵢ (avec alpha_vars) quand recon et KL sont satisfaisants
    """
    def __init__(
        self,
        beta=1.0,
        gamma=0.0,
        clr_mean=None,
        clr_scale=None,
        y_scaler_mean=None,
        y_scaler_scale=None,
        g_mean=None, g_scale=None,
        alpha_vars=None,
        alpha_baseline=None,
        recon_threshold=0.9,
        kl_threshold=0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        # poids adaptatifs
        self.beta = beta
        self.gamma = gamma
        self.clr_mean = clr_mean
        self.clr_scale = clr_scale
        self.y_scaler_mean = y_scaler_mean
        self.y_scaler_scale = y_scaler_scale
        self.g_mean       = g_mean      
        self.g_scale      = g_scale
        self.alpha_vars = alpha_vars                  # tf.Variable vectoriel
        self.alpha_baseline = tf.constant(alpha_baseline, dtype=tf.float32)
        # seuils pour activation de la récompense
        self.recon_th = tf.constant(recon_threshold, dtype=tf.float32)
        self.kl_th = tf.constant(kl_threshold, dtype=tf.float32)
        # métriques
        self.recon_tracker   = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_tracker      = tf.keras.metrics.Mean(name="kl_loss")
        self.entropy_tracker = tf.keras.metrics.Mean(name="entropy_loss")
        self.reward_tracker  = tf.keras.metrics.Mean(name="reward_loss")
        self.total_tracker   = tf.keras.metrics.Mean(name="total_loss")

    def call(self, inputs):
        # inputs : [x_input, x_decoded, y_input, y_decoded, mu, log_sigma]
        x_input, x_out, y_input, y_out, mu, log_sigma = inputs

        # === 1) Reconstruction MSE sur X ===
        recon_loss = tf.reduce_mean(tf.square(x_input - x_out))

        # === 2) Divergence KL ===
        log_sigma_clipped = tf.clip_by_value(log_sigma, -5.0, 5.0)
        mu_clipped        = tf.clip_by_value(mu,        -5.0, 5.0)
        sigma_clipped     = tf.clip_by_value(tf.exp(log_sigma_clipped), 1e-6, 1e+1)
        kl_per_sample = -0.5 * tf.reduce_sum(
            1.0
            + log_sigma_clipped
            - tf.square(mu_clipped)
            - sigma_clipped,
            axis=1
        )
        kl_loss = tf.reduce_mean(kl_per_sample)

        # === 3) Entropie de Shannon sur la partie compositionnelle ===
        clr_part = x_out[:, :N_ELEMENTS]
        clr_unscaled = clr_part * self.clr_scale + self.clr_mean
        exp_vals = tf.exp(tf.clip_by_value(clr_unscaled, -10.0, 10.0))
        stoich = exp_vals / (tf.reduce_sum(exp_vals, axis=1, keepdims=True) + 1e-8)
        stoich_clipped = tf.clip_by_value(stoich, 1e-4, 1.0)
        ent_per_sample = -tf.reduce_sum(stoich_clipped * tf.math.log(stoich_clipped), axis=1)
        entropy_loss = tf.reduce_mean(1.0 - tf.exp(- tf.square(ent_per_sample - 1.75) / 0.2))

        # === 4) Récompense linéaire adaptative sur gᵢ bruts ===
        # on récupère les dernières 10 dimensions de x_out (gᵢ reconstruits, scaled)
        g_rec_scaled = x_out[:, -10:]  
        # on inverse le scaling de X pour obtenir g_pred réels
        g_pred = g_rec_scaled * self.g_scale + self.g_mean

        # condition d’activation
        cond_reward = tf.logical_and(recon_loss < self.recon_th,
                                     kl_loss    < self.kl_th)

        # perte négative = – mean_i Σ (αᵢ × g_predᵢ)
        reward = tf.cond(
            cond_reward,
            lambda: -tf.reduce_mean(tf.reduce_sum(self.alpha_vars * g_pred, axis=1)),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

        # === 5) Perte totale ===
        total_loss = recon_loss + self.beta * kl_loss + self.gamma * entropy_loss + reward
        self.add_loss(total_loss)

        # === 6) Mise à jour des métriques ===
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_loss)
        self.entropy_tracker.update_state(entropy_loss)
        self.reward_tracker.update_state(reward)
        self.total_tracker.update_state(total_loss)

        return x_out  # on ne sort que la partie X reconstruite

    def get_losses(self):
        """
        Retourne un tuple de numpy floats :
          (recon_loss, kl_loss, entropy_loss, reward_loss, total_loss)
        """
        return (
            self.recon_tracker.result().numpy(),
            self.kl_tracker.result().numpy(),
            self.entropy_tracker.result().numpy(),
            self.reward_tracker.result().numpy(),
            self.total_tracker.result().numpy()
        )

    def reset_losses(self):
        """
        Reset les compteurs pour les 5 métriques.
        """
        self.recon_tracker.reset_state()
        self.kl_tracker.reset_state()
        self.entropy_tracker.reset_state()
        self.reward_tracker.reset_state()
        self.total_tracker.reset_state()

    def get_config(self):
        cfg = super().get_config()
        # Sérialisation minimale
        cfg.update({
            "beta": float(self.beta.numpy()) if isinstance(self.beta, tf.Variable) else float(self.beta),
            "gamma": float(self.gamma.numpy()) if isinstance(self.gamma, tf.Variable) else float(self.gamma),
            "clr_mean":    self.clr_mean.numpy().tolist(),
            "clr_scale":   self.clr_scale.numpy().tolist(),
            "g_mean": self.g_mean.numpy().tolist(),
            "g_scale": self.g_scale.numpy().tolist(),
            "y_scaler_mean":  self.y_scaler_mean.numpy().tolist(),
            "y_scaler_scale": self.y_scaler_scale.numpy().tolist(),
            "alpha_baseline": self.alpha_baseline.numpy().tolist(),
            "recon_threshold": float(self.recon_th.numpy()),
            "kl_threshold":    float(self.kl_th.numpy())
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        # Reconstruction du layer depuis config
        beta    = tf.Variable(config.pop("beta"), trainable=False, dtype=tf.float32)
        gamma   = tf.Variable(config.pop("gamma"), trainable=False, dtype=tf.float32)
        clr_mean  = tf.constant(config.pop("clr_mean"), dtype=tf.float32)
        clr_scale = tf.constant(config.pop("clr_scale"), dtype=tf.float32)
        g_mean = tf.constant(config.pop("g_mean"), dtype=tf.float32)
        g_scale= tf.constant(config.pop("g_scale"),dtype=tf.float32)
        y_mean    = tf.constant(config.pop("y_scaler_mean"), dtype=tf.float32)
        y_scale   = tf.constant(config.pop("y_scaler_scale"), dtype=tf.float32)
        alpha_baseline = config.pop("alpha_baseline")
        recon_th = config.pop("recon_threshold")
        kl_th    = config.pop("kl_threshold")
        return cls(
            beta=beta,
            gamma=gamma,
            clr_mean=clr_mean,
            clr_scale=clr_scale,
            g_mean=g_mean,
            g_scale=g_scale,
            y_scaler_mean=y_mean,
            y_scaler_scale=y_scale,
            alpha_vars=alpha_vars,
            alpha_baseline=alpha_baseline,
            recon_threshold=recon_th,
            kl_threshold=kl_th,
            **config
        )

@register_keras_serializable()
def sample_z(args):
    """Reparameterization trick: z = μ + σ * ε"""
    mu, log_sigma = args
    epsilon = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(log_sigma / 2) * epsilon

def build_cvae(input_dim, cond_dim, latent_dim, beta=1.0, gamma=0.0):
    """
    Construit le CVAE avec deux têtes (X et gᵢ) et la couche de perte adaptative.
    Retourne (model, loss_layer_name).
    """
    # ── Entrées ──────────────────────────────────────────────────────────────────
    x_input = Input(shape=(input_dim,), name="x_input")
    y_input = Input(shape=(cond_dim,),  name="y_input")

    # ── Encodeur ─────────────────────────────────────────────────────────────────
    enc = layers.Concatenate()([x_input, y_input])
    h = layers.Dense(64, activation="relu")(enc)
    h = layers.Dense(32, activation="relu")(h)
    mu = layers.Dense(latent_dim, name="mu")(h)
    log_sigma = layers.Dense(latent_dim, name="log_sigma")(h)
    z = layers.Lambda(sample_z, name="z")([mu, log_sigma])

    # ── Décodeur partagé ─────────────────────────────────────────────────────────
    dec_in = layers.Concatenate()([z, y_input])
    d = layers.Dense(32, activation="relu", name="decoder_dense_1")(dec_in)
    d = layers.BatchNormalization()(d)
    d = layers.Dense(64, activation="relu", name="decoder_dense_2")(d)
    d = layers.BatchNormalization()(d)

    # ── Têtes de sortie ──────────────────────────────────────────────────────────
    x_decoded = layers.Dense(input_dim, activation="linear", name="x_decoded")(d)
    y_decoded = layers.Dense(cond_dim,  activation="linear", name="y_decoded")(d)

    # ── 1) Chargement du StandardScaler (X) ──────────────────────────────
    std_path     = os.path.join(BASE_VAE_DIR, "scalers", "scaler_X_std.pkl")
    scaler_X_std = joblib.load(std_path)
    if not hasattr(scaler_X_std, "mean_"):
        raise ValueError("❌ scaler_X_std.pkl n’est pas un StandardScaler")

    # vecteurs complets pour toute la X reconstruite (features + g₁…g₁₀)
    x_scaler_mean  = tf.constant(scaler_X_std.mean_,  dtype=tf.float32)
    x_scaler_scale = tf.constant(scaler_X_std.scale_, dtype=tf.float32)

    # uniquement la partie CLR pour la loss
    clr_mean  = x_scaler_mean[:N_ELEMENTS]
    clr_scale = x_scaler_scale[:N_ELEMENTS]

    # partie gᵢ (dernières 10 colonnes de X) pour la reward
    gi_mean  = x_scaler_mean[-10:]
    gi_scale = x_scaler_scale[-10:]

    y_scaler_path = os.path.join(BASE_VAE_DIR, "scalers", "scaler_Y.pkl")
    y_scaler = joblib.load(y_scaler_path)
    y_mean  = tf.constant(y_scaler.mean_,  dtype=tf.float32)
    y_scale = tf.constant(y_scaler.scale_, dtype=tf.float32)

    # ── Couche de perte adaptative ───────────────────────────────────────────────
    unique_id = str(uuid4())[:8]
    layer_name = f"cvae_loss_adaptive_{unique_id}"
    x_out = CVAELossLayer(
        beta=beta,
        gamma=gamma,
        clr_mean=clr_mean,
        clr_scale=clr_scale,
        g_mean=gi_mean,
        g_scale=gi_scale,
        y_scaler_mean=y_mean,
        y_scaler_scale=y_scale,
        alpha_vars=alpha_vars,
        alpha_baseline=ALPHA,
        recon_threshold=RECON_THRESHOLD,
        kl_threshold=KL_THRESHOLD,
        name=layer_name
    )([x_input, x_decoded, y_input, y_decoded, mu, log_sigma])

    # ── Compilation du modèle ────────────────────────────────────────────────────
    cvae = Model(inputs=[x_input, y_input], outputs=x_out, name="CVAE_adaptive")
    cvae.compile(optimizer="adam")
    return cvae, layer_name

def get_decoder(cvae, cond_dim):
    """
    Extrait la partie décodeur (composition seulement) pour la génération
    active de nouvelles stoich.
    """
    # Récupère le tenseur z (Lambda layer) et sa dimension
    z_tensor = cvae.get_layer("z").output
    z_dim    = int(z_tensor.shape[-1])

    # entrées du décodeur
    z_input = Input(shape=(z_dim,),    name="z_input")
    y_input = Input(shape=(cond_dim,), name="y_input_decoder")

    # concat & réutilisation des couches du décodeur
    d = layers.Concatenate()([z_input, y_input])
    d = cvae.get_layer("decoder_dense_1")(d)
    d = cvae.get_layer("decoder_dense_2")(d)
    x_decoded = cvae.get_layer("x_decoded")(d)

    return Model([z_input, y_input], x_decoded, name="Decoder_adaptive")
