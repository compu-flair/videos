# Third attempt with very lightweight rendering (wireframes) and fewer frames for a fast GIF export

import os
import numpy as np
import matplotlib as mpl
# Use a non-interactive backend to prevent GUI blocking when running headless
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def Y_s(theta, phi): return np.ones_like(theta)
def Y_p(kind, theta, phi):
    if kind == "pz": return np.cos(theta)
    if kind == "px": return np.sin(theta) * np.cos(phi)
    if kind == "py": return np.sin(theta) * np.sin(phi)
def Y_d(kind, theta, phi):
    if kind == "dz2": return 0.5*(3*np.cos(theta)**2 - 1.0)
    if kind == "dx2y2": return (np.sin(theta)**2) * np.cos(2*phi)
    if kind == "dxy": return (np.sin(theta)**2) * np.sin(2*phi)
def Y_f(kind, theta, phi):
    if kind == "f_x(x2-3y2)": return (np.sin(theta)**3) * np.cos(3*phi)
    if kind == "f_z(5z2-3r2)": return np.cos(theta)*(5*np.cos(theta)**2 - 3.0)

def surface_from_Y(Yval, r_base=0.6, r_scale=0.8):
    Y_abs = np.abs(Yval)
    # Use np.ptp for NumPy >= 2.0 compatibility
    Y_norm = (Y_abs - Y_abs.min()) / (np.ptp(Y_abs) + 1e-12)
    return r_base + r_scale * Y_norm

def spherical_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Extra-light mesh
N_theta, N_phi = 32, 48
theta = np.linspace(0, np.pi, N_theta)
phi = np.linspace(0, 2*np.pi, N_phi)
TH, PH = np.meshgrid(theta, phi, indexing="ij")

orbitals = {
    "s": Y_s(TH, PH),
    "p (pz)": Y_p("pz", TH, PH),
    "d (x²−y²)": Y_d("dx2y2", TH, PH),
    "f (x(x²−3y²))": Y_f("f_x(x2-3y2)", TH, PH),
}

surfaces = {}
for name, Y in orbitals.items():
    r = surface_from_Y(Y)
    X, Yc, Z = spherical_to_cart(r, TH, PH)
    surfaces[name] = (X, Yc, Z)

# Static preview using wireframes to keep it lightweight
# Apply black background and warm colors
fig = plt.figure(figsize=(7, 7))
fig.patch.set_facecolor("black")
axs = []
warm_colors = [
    "#ff6b6b",  # warm red
    "#f59e0b",  # amber
    "#f97316",  # orange
    "#fde047",  # warm yellow
]
for i, key in enumerate(surfaces.keys(), 1):
    ax = fig.add_subplot(2, 2, i, projection="3d")
    ax.set_facecolor("black")
    X, Yc, Z = surfaces[key]
    color = warm_colors[(i-1) % len(warm_colors)]
    ax.plot_wireframe(X, Yc, Z, rstride=2, cstride=2, linewidth=0.6, color=color, alpha=0.95)
    # Remove all texts
    # ax.set_title(key, pad=6)  # removed per request
    ax.set_box_aspect((1,1,1))
    ax.set_axis_off()
    ax.view_init(elev=25, azim=35)
    axs.append(ax)
plt.tight_layout()
# plt.show()  # Disabled for headless/non-interactive runs

# Per-orbital animations (split into 4 separate files)

def _slugify(name: str) -> str:
    # simple slug: lowercase, replace spaces with underscores, keep alnum and underscore only
    import re
    s = name.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

DURATION_SEC = 15
FPS = 24
frames = DURATION_SEC * FPS
interval = int(1000 / FPS)  # ms per frame
base_dir = os.path.dirname(__file__)

for i, (key, (X, Yc, Z)) in enumerate(surfaces.items(), 1):
    color = warm_colors[(i-1) % len(warm_colors)]
    fig_o = plt.figure(figsize=(6, 6))
    fig_o.patch.set_facecolor("black")
    ax_o = fig_o.add_subplot(1, 1, 1, projection="3d")
    ax_o.set_facecolor("black")
    ax_o.plot_wireframe(X, Yc, Z, rstride=2, cstride=2, linewidth=0.7, color=color, alpha=0.95)
    ax_o.set_box_aspect((1,1,1))
    ax_o.set_axis_off()
    ax_o.view_init(elev=25, azim=35)

    def animate_single(frame):
        # One full rotation over the whole animation duration
        az = 35 + (360.0 * frame / max(frames - 1, 1))
        ax_o.view_init(elev=25, azim=az)
        return []

    ani_single = animation.FuncAnimation(fig_o, animate_single, frames=frames, interval=interval, blit=False)

    safe = _slugify(key)
    # Save GIF (15s at FPS)
    gif_path = os.path.join(base_dir, f"{safe}_orbitals_rotation.gif")
    try:
        ani_single.save(
            gif_path,
            writer="pillow",
            fps=FPS,
            dpi=72,
            savefig_kwargs={"facecolor": "black", "edgecolor": "none"},
        )
        print(f"Saved animation to {gif_path}")
    except Exception as e:
        print("Could not save GIF for", key, ":", e)

    # Save MP4 using ffmpeg if available (15s at FPS)
    mp4_path = os.path.join(base_dir, f"{safe}_orbitals_rotation.mp4")
    try:
        ani_single.save(
            mp4_path,
            writer="ffmpeg",
            fps=FPS,
            dpi=300,
            savefig_kwargs={"facecolor": "black", "edgecolor": "none"},
        )
        print(f"Saved animation to {mp4_path}")
    except Exception as e:
        print("Could not save MP4 for", key, ":", e, "\nIf missing, install ffmpeg in base: conda install -n base -c conda-forge ffmpeg")

# plt.show()  # Disabled for headless/non-interactive runs
