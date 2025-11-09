# phibar_delta_p_theta_guides_2panels_logx_labels_ARIAL.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from pathlib import Path
c_folder = Path(__file__).resolve().parent
PROJECT_FOLDER = c_folder.parent.parent

# ----------- global style -----------
mpl.rcParams.update({
    # Arial pour tout le texte hors maths
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans"],
    # Maths en Computer Modern (classique)
    "mathtext.fontset": "cm",
})

# ----------- options figure -----------
COL       = False         # True: 1x2 ; False: 2x1
LOGX      = True          # p en échelle log
CLIP_NEG  = True          # clamp >=0 sous les racines (évite trous/bruit)
TAU_GUIDES = [-np.log(10.0), np.log(10.0)]  # lignes τ = constantes (ROUGE dans ce script)

# ----------- outils numériques -----------
def _make_erfc():
    if hasattr(np, "erfc"): return np.erfc
    else:
        from math import erfc as _erfc_scalar
        return np.vectorize(_erfc_scalar)
_erfc = _make_erfc()

def Phi_bar(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * _erfc(x / np.sqrt(2.0))

def p_star(theta, kappa):
    t = np.tan(theta)
    return np.exp(-kappa / (t * t))

# δ via τ = log(p/p*) — forme rationalisée (stable)
def delta_from_p_theta_kappa(p, theta, kappa, clip_neg=CLIP_NEG):
    rho = np.cos(theta)
    s   = np.sin(theta)
    s2  = np.maximum(1.0 - rho*rho, 1e-15)     # = sin^2(theta)
    Lk  = kappa / s2

    pst = np.maximum(p_star(theta, kappa), 1e-300)
    tau = np.log(np.maximum(p, 1e-300)) - np.log(pst)

    A1 = rho*rho * Lk - tau
    A2 = Lk - tau
    if clip_neg:
        A1 = np.maximum(A1, 0.0)
        A2 = np.maximum(A2, 0.0)
    else:
        valid = (A1 >= 0.0) & (A2 >= 0.0)
        A1 = np.where(valid, A1, np.nan)
        A2 = np.where(valid, A2, np.nan)

    denom = np.sqrt(A1) + rho*np.sqrt(A2)
    denom = np.maximum(denom, 1e-300)

    # δ = -sqrt(2) * τ * sinθ / (sqrt(A1) + ρ sqrt(A2))
    return -np.sqrt(2.0) * tau * s / denom

# ----------- placement intelligent des étiquettes -----------
def _to_metric_coords(x, y, logx=True):
    """Espace métrique pour les distances: (log10 p, theta) si logx, sinon (p, theta)."""
    X = np.log10(x) if logx else x
    return X, y

def _min_dist_to_polylines(px, py, other_segs, logx=True):
    if not other_segs:
        return np.inf
    PX, PY = _to_metric_coords(px, py, logx)
    dmin = np.inf
    for seg in other_segs:
        x, y = seg[:, 0], seg[:, 1]
        X, Y = _to_metric_coords(x, y, logx)
        d = np.hypot(X - PX, Y - PY)
        m = np.nanmin(d)
        if m < dmin:
            dmin = m
    return dmin

def smart_label_placement(ax, cs, levels, p_min, p_max, theta_min, theta_max,
                          logx=True, n_per_seg=12,
                          min_sep=0.08,  # distance min entre étiquettes (dans l'espace métrique)
                          inner_margin_x=0.02, inner_margin_y=0.02,
                          fontsize=9, color="k"):
    levels_arr = np.asarray(cs.levels)
    all_segs_by_level = []
    for i, L in enumerate(levels_arr):
        seg_lists = cs.allsegs[i]
        segs = [seg for seg in seg_lists if seg is not None and len(seg) >= 2]
        all_segs_by_level.append((L, segs))

    placed_metric_pts = []

    for L in levels:
        idx = np.where(np.isclose(levels_arr, L))[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        segs_L = all_segs_by_level[idx][1]
        if not segs_L:
            continue

        other_segs = []
        for j, (Lj, segs_j) in enumerate(all_segs_by_level):
            if j == idx: 
                continue
            other_segs.extend(segs_j)

        # Génère des candidats
        candidates = []
        for seg in segs_L:
            xy = seg
            n = len(xy)
            k = max(2, n_per_seg)
            idxs = np.linspace(0, n - 1, k).astype(int)
            for ii in idxs:
                px, py = xy[ii, 0], xy[ii, 1]
                # évite les bords
                if not (p_min*(1+1e-12) < px < p_max*(1-1e-12)): continue
                if not (theta_min + inner_margin_y < py < theta_max - inner_margin_y): continue
                if logx:
                    X = np.log10(px); Xmin, Xmax = np.log10(p_min), np.log10(p_max)
                    if not (Xmin + inner_margin_x < X < Xmax - inner_margin_x): continue
                else:
                    if not (p_min + inner_margin_x < px < p_max - inner_margin_x): continue

                score = _min_dist_to_polylines(px, py, other_segs, logx=logx)
                candidates.append((score, px, py))

        if not candidates:
            continue

        # trie par score décroissant (plus isolé d'abord)
        candidates.sort(key=lambda t: (-t[0], t[2]))

        # choisit un candidat qui ne chevauche pas les labels déjà posés
        chosen = None
        for score, px, py in candidates:
            Xp, Yp = _to_metric_coords(px, py, logx)
            ok = True
            for (Xq, Yq) in placed_metric_pts:
                if np.hypot(Xp - Xq, Yp - Yq) < min_sep:
                    ok = False; break
            if ok:
                chosen = (px, py)
                placed_metric_pts.append((Xp, Yp))
                break
        if chosen is None:
            # si tous chevauchent, on prend tout de même le meilleur
            _, px, py = candidates[0]
            chosen = (px, py)
            placed_metric_pts.append(_to_metric_coords(px, py, logx))

        # place le label sans rectangle, avec halo blanc
        txt = ax.text(chosen[0], chosen[1], f"{L:.2f}",
                      fontsize=fontsize, color=color, ha="center", va="center")
        txt.set_path_effects([pe.Stroke(linewidth=0.5, foreground="white"), pe.Normal()])

# ----------- figure -----------
if __name__ == "__main__":
    # Domaine p
    if LOGX:
        p_min, p_max = 1e-5, 1.0
        N_P = 1200
        p = np.logspace(np.log10(p_min), np.log10(p_max), N_P)
    else:
        p_min, p_max = 1e-6, 0.999
        N_P = 900
        p = np.linspace(p_min, p_max, N_P)

    # Domaine theta
    EPS_TH = 3e-2
    theta_min, theta_max = EPS_TH, np.pi/2 - EPS_TH
    N_TH = 600
    theta = np.linspace(theta_min, theta_max, N_TH)

    P, THETA = np.meshgrid(p, theta)

    kappas = [0.1, 1.0]

    levels_fill  = np.linspace(0.0, 1.0, 41)
    levels_lines = [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90]

    nrows = 1 if COL else 2
    ncols = 2 if COL else 1
    figsize = (12, 4.8) if COL else (7.2, 9.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.12, w_pad=0.06, hspace=0.08, wspace=0.06)
    axes = np.atleast_1d(axes).ravel()

    last_cf = None
    for ax, kappa in zip(axes, kappas):
        Z = Phi_bar(delta_from_p_theta_kappa(P, THETA, kappa))

        # Heatmap + contours
        cf = ax.contourf(P, THETA, Z, levels=levels_fill, cmap="viridis", antialiased=True)
        cs = ax.contour(P, THETA, Z, levels=levels_lines, colors="k",
                        linewidths=0.9, corner_mask=True)

        # Labels intelligents (un par niveau, sans recouvrement)
        smart_label_placement(
            ax, cs, levels_lines, p_min, p_max, theta_min, theta_max,
            logx=LOGX, n_per_seg=12, min_sep=0.08,
            inner_margin_x=0.02, inner_margin_y=0.02, fontsize=10, color="k"
        )

        # τ = 0 (ORANGE)
        th = np.linspace(theta_min, theta_max, 1200)
        p0 = np.clip(p_star(th, kappa), p_min, p_max)
        ax.plot(p0, th, color="orange", ls="--", lw=1.6)

        # τ = ±a (ROUGE)
        for tau_c in TAU_GUIDES:
            pg = p_star(th, kappa) * np.exp(tau_c)
            pg = np.where((pg >= p_min) & (pg <= p_max), pg, np.nan)
            ax.plot(pg, th, color="red", ls="--", lw=1.4)

        if LOGX:
            ax.set_xscale("log")
            ax.set_xlim(p_min, p_max)

        ax.set_title(rf"$\kappa={kappa:g}$")
        last_cf = cf

    # Axes + colorbar
    axes[-1].set_xlabel(r"$p$")
    if not COL:
        axes[1].set_xlabel(r"$p$")
    axes[0].set_ylabel(r"$\theta$ (rad)")
    axes[1].set_ylabel(r"$\theta$ (rad)")
    if COL:
        axes[1].set_ylabel(r"$\theta$ (rad)")

    cbar = fig.colorbar(last_cf, ax=axes.tolist(), shrink=0.9, pad=0.02)
    cbar.set_label(r"$\overline{\Phi}(\delta)$")

    # Légende
    tau_abs = abs(TAU_GUIDES[0])
    orange_label = r"$\tau=0:\ p=p^\ast(\theta)$"
    red_label    = r"$\tau=\pm {0:.1f}:\ p=10^{{\pm 1}}\,p^\ast(\theta)$".format(tau_abs)
    fig.legend(
        handles=[
            Line2D([0], [0], color="orange", ls="--", lw=1.6, label=orange_label),
            Line2D([0], [0], color="red",    ls="--", lw=1.4, label=red_label),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05)
    )

    fig.suptitle(r"Correction Factor $\overline{\Phi}(\delta)$ versus $p$, $\kappa$ and $\theta$",
                 y=1.04)

    plt.savefig(c_folder / "Phibar_delta_p_theta_guides_2panels_logx_labels_ARIAL.png", dpi=900,
                bbox_inches="tight", pad_inches=0.3)
    plt.show()
