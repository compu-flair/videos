import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- Dark Quantum Aesthetic ---
plt.style.use('dark_background')

# --- Static energy values ---
E = np.linspace(0, 1, 100)

# --- Figure setup ---
fig = plt.figure(figsize=(15, 13))
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)
ax.set_facecolor("black")

# --- Initial beta range and mesh ---
beta = np.linspace(0.1, 10, 100)
E_grid, beta_grid = np.meshgrid(E, beta)

# --- Compute Boltzmann probabilities ---
def compute_P(beta_values):
    Z = np.sum(np.exp(-beta_values[:, None] * E), axis=1)
    P = np.exp(-beta_grid * E_grid) / Z[:, None]
    return P

P = compute_P(beta)

# --- Surface plot (initial frame) ---
surf = ax.plot_surface(
    E_grid, beta_grid, P,
    cmap='plasma',
    linewidth=0,
    antialiased=False,
    alpha=0.9,
    rstride=1,
    cstride=1,
    shade=False,
    zorder=1
)

ax.set_xlabel(r'Energy $E$', color='white')
ax.set_ylabel(r'Inverse Temperature $\beta$', color='white')
ax.set_zlabel(r'Probability $P(E)$', color='white')
# ax.set_title(r'$P(E_n) = \frac{e^{-\beta E_n}}{Z}$', fontsize=14, color='white', pad=15)
ax.set_zscale('log')
ax.view_init(elev=25, azim=125)

# --- Colorbar ---
# cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
# cbar.set_label(r'$P(E)$', color='white')
# cbar.ax.yaxis.set_tick_params(color='white')
# plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# --- Animation update function ---
def update(frame):
    # Create gentle oscillation in β range (simulate temperature breathing)
    beta_dynamic = np.linspace(0.1 + 0.05*np.sin(frame/10), 10 + 0.2*np.sin(frame/15), 100)
    E_grid, beta_grid = np.meshgrid(E, beta_dynamic)
    Z = np.sum(np.exp(-beta_dynamic[:, None] * E), axis=1)
    P_new = np.exp(-beta_grid * E_grid) / Z[:, None]

    # Clear and redraw surface
    ax.clear()
    ax.set_facecolor("black")
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.set_xlabel(r'Energy $E$', color='white')
    ax.set_ylabel(r'Inverse Temperature $\beta$', color='white')
    ax.set_zlabel(r'Probability $P(E)$', color='white')
    ax.set_zscale('log')
    # ax.set_title(r'$P(E_n) = \frac{e^{-\beta E_n}}{Z}$', fontsize=14, color='white', pad=15)

    # Dynamic rotation (slow cinematic orbit)
    ax.view_init(elev=25 + 2*np.sin(frame/30), azim=125 + frame/3)

    surf = ax.plot_surface(
        E_grid, beta_grid, P_new,
        cmap='plasma',
        linewidth=0,
        antialiased=False,
        alpha=0.9,
        rstride=1,
        cstride=1,
        shade=False,
        zorder=1
    )
    return surf,

# --- Create animation ---
anim = FuncAnimation(fig, update, frames=1200, interval=50, blit=False)

# --- Save or display ---
# Uncomment below to save as video (requires ffmpeg)
anim.save("boltzmann_distribution_animation.mp4", writer='ffmpeg', dpi=200, fps=30)

plt.show()
