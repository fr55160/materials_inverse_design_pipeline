import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Données ===
x = np.linspace(-4, 4, 2000)
y = norm.pdf(x)  # N(0,1)

# Seuillage (exemple) : Q_ij < 0 => Phi(Q_ij) < 1/2
Q_ij = +1.618  # exemple cohérent avec Phi(Q_ij) < 1/2
R = norm.cdf(Q_ij)

# === Figure ===
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label='Probability density')

# Surfaces (on remplit jusqu’à y=0)
ax.fill_between(x, y, 0, where=(x <= Q_ij), color='lightblue', alpha=0.5,
                label='No risk realization')
ax.fill_between(x, y, 0, where=(x >= Q_ij), color='red', alpha=0.5,
                label=fr'Risk realization: $z>Q_{{ij}}$ '
                      fr'($R_i(X_j)=\bar\Phi(Q_{{ij}})$')

# Marqueur vertical et étiquette sur l’axe x
ax.axvline(Q_ij, color='red', linestyle='--')
ax.annotate(r'$Q_{ij}$', xy=(Q_ij, 0), xycoords='data',
            xytext=(0, -22), textcoords='offset points',
            ha='center', va='top',
            arrowprops=dict(arrowstyle='-|>', lw=0.8, shrinkA=0, shrinkB=0))

# Axes passant par l’origine
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Habillage
ax.set_xlim(-3, 3)
ax.yaxis.set_ticks([])     # on cache les graduations y
ax.set_xlabel('z', fontsize=12)
ax.set_title('Standard Normal Distribution', fontsize=14)

# Légende en dessous
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=2, frameon=False)
plt.subplots_adjust(bottom=0.25)

# Sauvegarde 900 dpi
fig.savefig('normal_gaussian_distribution_900dpi.png', dpi=900, bbox_inches='tight')

plt.show()
