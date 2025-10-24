"""
Train a simple softmax (multinomial logistic regression) classifier on a
synthetic random 2D dataset with three classes labeled as -1, 0, 1.

No external ML libraries are required; this uses NumPy only.

Usage:
	Set the configuration constants below, then run:
		python temp.py

Config constants:
	N_PER_CLASS, EPOCHS, LR, REG, SEED, STD, PLOT
"""

from __future__ import annotations
import sys
from dataclasses import dataclass

import numpy as np
import os
from pathlib import Path


# ----------------------------
# Configuration (edit here)
# ----------------------------

N_PER_CLASS = 200   # Number of samples per class
EPOCHS = 500        # Training epochs
LR = 0.1            # Learning rate
REG = 1e-3          # L2 regularization strength
SEED = 42           # Random seed
STD = 0.8           # Std-dev for Gaussian clusters
PLOT = False        # Set True to visualize decision regions (requires matplotlib)
# Export controls
EXPORT = True       # Set True to export dataset and learned weights
EXPORT_DIR = "exports"  # Relative to this file's directory

# 3D plot controls for w^T x visualization
PLOT3D = True
CLASS_FOR_3D = 1           # class label to visualize (e.g., -1, 0, or 1)
USE_BIAS_FOR_3D = False    # if True, plot z = w^T x + b; else z = w^T x
SAVE_3D = True             # save figure to PNG under EXPORT_DIR
SHOW_3D = True             # show interactive window (may block)
ANIMATE_3D = True          # animate with dynamic camera movements
ANIMATION_FRAMES = 1020    # number of frames (about 34 seconds at 30fps)
ANIMATION_FPS = 30         # frames per second
SAVE_ANIMATION = True      # save animation as MP4/GIF

# Partition function Z(x) = sum_n exp(w_n^T x + b_n) plot controls
PLOT_Z2D = True            # 2D heatmap of Z (or log Z)
PLOT_Z3D = False           # 3D surface plot of Z (or log Z)
Z_AS_LOG = True            # if True, plot log Z(x) for stability
SAVE_Z = True              # save Z plot(s) under EXPORT_DIR
SHOW_Z = True              # show interactive window (may block)

# Probability P(n|x) plots where P(n|x) = exp(w_n^T x (+ b_n?)) / Z(x)
# By default, numerator uses NO bias to match the user's formula, and denominator Z uses bias (as above)
PLOT_P2D = True            # heatmaps of class probabilities
P_CLASSES_TO_PLOT = [-1, 0, 1]  # which class labels to render
PROB_NUM_USE_BIAS = False  # numerator uses b_n if True; else only w_n^T x
PROB_DEN_USE_BIAS = True   # denominator Z uses b_n if True; else only w_k^T x
SAVE_P = True              # save probability heatmaps
SHOW_P = True              # show probability heatmaps


# ----------------------------
# Data generation
# ----------------------------


def make_data(n_per_class: int = 200,
			  centers: list[tuple[float, float]] | None = None,
			  std: float = 0.8,
			  seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
	"""Generate a 2D dataset with three Gaussian clusters labeled -1, 0, 1.

	Returns:
		X: (N, 2) float32 features
		y: (N,) int labels in {-1, 0, 1}
	"""
	if centers is None:
		centers = [(-2.0, -2.0), (0.0, 2.0), (2.5, -0.5)]

	rng = np.random.default_rng(seed)

	class_labels = np.array([-1, 0, 1], dtype=int)
	xs = []
	ys = []
	for label, (cx, cy) in zip(class_labels, centers):
		x = rng.normal(loc=cx, scale=std, size=(n_per_class, 1))
		y = rng.normal(loc=cy, scale=std, size=(n_per_class, 1))
		xy = np.concatenate([x, y], axis=1)
		xs.append(xy)
		ys.append(np.full((n_per_class,), label, dtype=int))

	X = np.concatenate(xs, axis=0).astype(np.float32)
	y = np.concatenate(ys, axis=0)

	# Shuffle dataset
	perm = rng.permutation(X.shape[0])
	X = X[perm]
	y = y[perm]
	return X, y


# ----------------------------
# Softmax Regression Model
# ----------------------------


@dataclass
class SoftmaxRegressor:
	input_dim: int
	num_classes: int
	lr: float = 0.1
	reg: float = 1e-3
	seed: int = 42

	def __post_init__(self):
		rng = np.random.default_rng(self.seed)
		# Heuristic small random init
		self.W = rng.normal(0.0, 0.01, size=(self.input_dim, self.num_classes)).astype(np.float32)
		self.b = np.zeros((self.num_classes,), dtype=np.float32)

	@staticmethod
	def _softmax(logits: np.ndarray) -> np.ndarray:
		# logits: (N, K)
		z = logits - logits.max(axis=1, keepdims=True)  # stability
		e = np.exp(z)
		return e / (e.sum(axis=1, keepdims=True) + 1e-12)

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		logits = X @ self.W + self.b  # (N, K)
		return self._softmax(logits)

	def predict(self, X: np.ndarray) -> np.ndarray:
		probs = self.predict_proba(X)
		return probs.argmax(axis=1)

	def loss_and_grads(self, X: np.ndarray, y_onehot: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
		N = X.shape[0]
		logits = X @ self.W + self.b
		probs = self._softmax(logits)
		# Cross-entropy loss
		ce = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
		# L2 regularization (on W only)
		reg_loss = 0.5 * self.reg * np.sum(self.W * self.W)
		loss = ce + reg_loss

		# Gradients
		dZ = (probs - y_onehot) / N  # (N, K)
		dW = X.T @ dZ + self.reg * self.W  # (D, K)
		db = dZ.sum(axis=0)  # (K,)
		return loss, dW, db

	def fit(self, X: np.ndarray, y_idx: np.ndarray, epochs: int = 500, verbose: bool = True) -> list[float]:
		# y_idx: integer class indices in [0, K)
		K = self.num_classes
		y_onehot = np.eye(K, dtype=np.float32)[y_idx]
		history = []
		for ep in range(1, epochs + 1):
			loss, dW, db = self.loss_and_grads(X, y_onehot)
			# SGD update
			self.W -= self.lr * dW
			self.b -= self.lr * db
			history.append(loss)
			if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
				print(f"Epoch {ep:4d}/{epochs}  loss={loss:.4f}")
		return history


# ----------------------------
# Utilities
# ----------------------------


def map_labels_to_indices(y: np.ndarray, ordered_labels: np.ndarray) -> np.ndarray:
	"""Map labels in ordered_labels to 0..K-1 indices.
	Example: ordered_labels=[-1,0,1] maps -1->0, 0->1, 1->2
	"""
	mapping = {lab: i for i, lab in enumerate(ordered_labels)}
	return np.array([mapping[int(v)] for v in y], dtype=int)


def maybe_plot(X: np.ndarray, y: np.ndarray, model: SoftmaxRegressor, label_values: np.ndarray) -> None:
	try:
		import matplotlib.pyplot as plt
	except Exception as e:  # pragma: no cover
		print("matplotlib not available, skipping plot.")
		return

	# Scatter plot of points
	colors = { -1: "tab:red", 0: "tab:blue", 1: "tab:green" }
	for lab in label_values:
		mask = (y == lab)
		plt.scatter(X[mask, 0], X[mask, 1], s=12, alpha=0.7, label=f"class {lab}", color=colors.get(int(lab), "gray"))

	# Decision regions
	x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
	y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
	grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
	idx_pred = model.predict(grid)
	zz = label_values[idx_pred].reshape(xx.shape)

	from matplotlib.colors import ListedColormap
	cmap = ListedColormap(["#ffcccc", "#cce0ff", "#ccffcc"])  # light tints for -1,0,1
	# Map labels -1,0,1 to 0,1,2 for contourf
	zz_idx = map_labels_to_indices(zz.ravel(), label_values).reshape(zz.shape)
	plt.contourf(xx, yy, zz_idx, alpha=0.2, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap)

	# plt.legend()
	plt.title("Softmax Classifier Decision Regions")
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.tight_layout()
	plt.show()


def export_dataset_and_weights(
	X: np.ndarray,
	y: np.ndarray,
	model: SoftmaxRegressor,
	label_values: np.ndarray,
	out_dir: str | os.PathLike = EXPORT_DIR,
) -> dict:
	"""Export dataset and learned weights in convenient formats.

	Creates files under out_dir relative to this script:
	- dataset.csv   with columns: x1,x2,y
	- weights.csv   with columns: class_label,w1,w2,b
	- data_weights.npz containing arrays: X, y, W, b, label_values

	Returns a dict of created file paths.
	"""
	script_dir = Path(__file__).resolve().parent
	out_path = script_dir / str(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	# Save dataset
	dataset_csv = out_path / "dataset.csv"
	ds = np.column_stack([X.astype(np.float32), y.astype(int)])
	headers_ds = "x1,x2,y"
	np.savetxt(dataset_csv, ds, delimiter=",", header=headers_ds, comments="", fmt=["%.6f", "%.6f", "%d"])

	# Save weights per class in the same order as label_values
	# model.W shape: (D, K), model.b shape: (K,)
	weights_rows = []
	for class_index, class_label in enumerate(label_values):
		w = model.W[:, class_index]
		b = model.b[class_index]
		weights_rows.append([int(class_label), float(w[0]), float(w[1]), float(b)])
	weights_csv = out_path / "weights.csv"
	np.savetxt(
		weights_csv,
		weights_rows,
		delimiter=",",
		header="class_label,w1,w2,b",
		comments="",
		fmt=["%d", "%.9f", "%.9f", "%.9f"],
	)

	# Also save a NumPy bundle for direct loading
	npz_path = out_path / "data_weights.npz"
	with open(npz_path, "wb") as f:
		np.savez(
			f,
			X=X.astype(np.float32),
			y=y.astype(int),
			W=model.W.astype(np.float32),
			b=model.b.astype(np.float32),
			label_values=label_values.astype(int),
		)

	print("\nExported files:")
	print(f"- {dataset_csv}")
	print(f"- {weights_csv}")
	print(f"- {npz_path}")

	return {
		"dataset_csv": str(dataset_csv),
		"weights_csv": str(weights_csv),
		"npz": str(npz_path),
	}


## main execution (uses config constants above)

# 1) Data
X, y = make_data(n_per_class=N_PER_CLASS, std=STD, seed=SEED)
label_values = np.array([-1, 0, 1], dtype=int)
y_idx = map_labels_to_indices(y, label_values)

# 2) Model
model = SoftmaxRegressor(input_dim=X.shape[1], num_classes=3, lr=LR, reg=REG, seed=SEED)

# 3) Train
print("Training softmax classifier...")
_ = model.fit(X, y_idx, epochs=EPOCHS, verbose=True)

# 4) Evaluate
idx_pred = model.predict(X)
y_pred = label_values[idx_pred]
acc = float((y_pred == y).mean())
print(f"\nTraining accuracy: {acc*100:.2f}%")

# Show a few predictions
print("\nSample predictions (first 10):")
for i in range(min(10, X.shape[0])):
	print(f"x={X[i].tolist()}  y_true={int(y[i])}  y_pred={int(y_pred[i])}")

# Optional plot
if PLOT:
	maybe_plot(X, y, model, label_values)


# 5) Export for downstream use (w_n^T x)
if EXPORT:
	export_dataset_and_weights(X, y, model, label_values, out_dir=EXPORT_DIR)


# 6) Load exported files and perform dot products w_n^T x (and scores with bias)
def _load_exports_and_demo_dot_products():
	"""Load weights.csv and dataset.csv then compute w_n^T x for all classes.

	Also computes scores = w_n^T x + b_n and verifies predictions match training.
	"""
	script_dir = Path(__file__).resolve().parent
	weights_csv = script_dir / EXPORT_DIR / "weights.csv"
	dataset_csv = script_dir / EXPORT_DIR / "dataset.csv"

	if not weights_csv.exists() or not dataset_csv.exists():
		print("\nExport files not found; skipping dot product demo.")
		return

	# Load weights
	weights = np.genfromtxt(weights_csv, delimiter=",", names=True)
	# Ensure 1D array even when a single row is present
	if weights.shape == ():
		weights = np.array([weights], dtype=weights.dtype)

	class_labels = weights["class_label"].astype(int)
	# Build W matrix (D x K) in the order rows appear in weights.csv
	W = np.vstack([
		[float(weights["w1"][i]), float(weights["w2"][i])] for i in range(len(weights))
	]).T
	b = np.array([float(weights["b"][i]) for i in range(len(weights))], dtype=float)

	# Load dataset
	ds = np.genfromtxt(dataset_csv, delimiter=",", names=True)
	X_loaded = np.vstack([ds["x1"], ds["x2"]]).T.astype(np.float32)
	y_loaded = ds["y"].astype(int)

	# Compute dot products (no bias) and scores (with bias)
	dots = X_loaded @ W                 # shape (N, K)
	scores = dots + b                   # broadcast b over rows

	print("\nDemo: w_n^T x and (w_n^T x + b_n) for first 5 samples:")
	for i in range(min(5, X_loaded.shape[0])):
		xi = X_loaded[i]
		di = dots[i]
		si = scores[i]
		parts = [f"n={int(class_labels[j])}: w^T x={di[j]:.4f}, score={si[j]:.4f}" for j in range(len(class_labels))]
		print(f"x={xi.tolist()} -> " + ", ".join(parts))

	# Verify predictions using loaded files
	pred_idx = scores.argmax(axis=1)
	y_pred_loaded = class_labels[pred_idx]
	acc_loaded = float((y_pred_loaded == y_loaded).mean())
	print(f"\nVerification using loaded files: accuracy={acc_loaded*100:.2f}%")


_load_exports_and_demo_dot_products()


def _plot_3d_from_exports(
	class_label: int = CLASS_FOR_3D,
	use_bias: bool = USE_BIAS_FOR_3D,
	save_png: bool = SAVE_3D,
	show: bool = SHOW_3D,
):
	"""Make a 3D plot of x, w, and their dot product z = w^T x (+ b optional).

	- Loads exports/weights.csv and exports/dataset.csv
	- Plots a plane z = w1*x1 + w2*x2 (+ b)
	- Plots dataset points at height z_i = w^T x_i (+ b)
	- Draws the weight vector w on the base plane (z=0)
	- Draws a few vertical segments to illustrate z for sample points
	"""
	try:
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D
	except Exception as e:
		print("matplotlib not available, skipping 3D plot.")
		return

	script_dir = Path(__file__).resolve().parent
	weights_csv = script_dir / EXPORT_DIR / "weights.csv"
	dataset_csv = script_dir / EXPORT_DIR / "dataset.csv"
	if not weights_csv.exists() or not dataset_csv.exists():
		print("Exports not found; run training/export first.")
		return

	weights = np.genfromtxt(weights_csv, delimiter=",", names=True)
	if weights.shape == ():
		weights = np.array([weights], dtype=weights.dtype)
	class_labels = weights["class_label"].astype(int)

	# Pick class index
	if class_label in class_labels:
		k = int(np.where(class_labels == class_label)[0][0])
	else:
		print(f"Requested class {class_label} not found; defaulting to {int(class_labels[0])}.")
		k = 0
		class_label = int(class_labels[0])

	w = np.array([float(weights["w1"][k]), float(weights["w2"][k])], dtype=np.float64)
	b = float(weights["b"][k])

	ds = np.genfromtxt(dataset_csv, delimiter=",", names=True)
	X_loaded = np.vstack([ds["x1"], ds["x2"]]).T.astype(np.float64)
	y_loaded = ds["y"].astype(int)

	# Compute dot products (suppress any spurious warnings)
	with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
		z_vals = X_loaded @ w + (b if use_bias else 0.0)

	# Grid for plane
	x_min, x_max = float(X_loaded[:, 0].min()), float(X_loaded[:, 0].max())
	y_min, y_max = float(X_loaded[:, 1].min()), float(X_loaded[:, 1].max())
	pad_x = 0.5 * (x_max - x_min)
	pad_y = 0.5 * (y_max - y_min)
	x_lin = np.linspace(x_min - pad_x, x_max + pad_x, 60)
	y_lin = np.linspace(y_min - pad_y, y_max + pad_y, 60)
	XX, YY = np.meshgrid(x_lin, y_lin)
	ZZ = w[0] * XX + w[1] * YY + (b if use_bias else 0.0)

	# --- Dark Quantum Aesthetic ---
	plt.style.use('dark_background')
	
	# Figure
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection='3d')
	
	# --- Remove panes & grids for clean dark look ---
	ax.grid(False)
	ax.xaxis.pane.set_visible(False)
	ax.yaxis.pane.set_visible(False)
	ax.zaxis.pane.set_visible(False)
	ax.set_facecolor("black")
	
	ax.set_xlabel('$x_1$', color='white')
	ax.set_ylabel('$x_2$', color='white')
	ax.set_zlabel(r'$z = w^T x$' + (' + b' if use_bias else ''), color='white')
	# ax.set_title(f"3D: class {class_label} — w=({w[0]:.3f},{w[1]:.3f})", color='white', fontsize=14, pad=15)

	# Surface plane
	ax.plot_surface(XX, YY, ZZ, cmap='magma', alpha=0.25, linewidth=0, antialiased=False)

	# Scatter points colored by label
	colors = { -1: 'tab:red', 0: 'tab:blue', 1: 'tab:green' }
	for lab in np.unique(y_loaded):
		mask = (y_loaded == lab)
		ax.scatter(X_loaded[mask, 0], X_loaded[mask, 1], z_vals[mask], s=8, color=colors.get(int(lab), 'gray'), alpha=0.8, label=f"y={int(lab)}")
	# ax.legend(loc='upper left')
	
	# --- Axis limits & viewing angle ---
	ax.view_init(elev=25, azim=125)

	# Save/show
	if save_png and not ANIMATE_3D:
		out_dir = script_dir / EXPORT_DIR
		out_dir.mkdir(parents=True, exist_ok=True)
		fname = out_dir / f"dotproduct_3d_class_{class_label}{'_bias' if use_bias else ''}.png"
		plt.tight_layout()
		plt.savefig(fname, dpi=1000, bbox_inches='tight', facecolor='black')
		print(f"Saved 3D plot to: {fname}")

	if show and not ANIMATE_3D:
		plt.show()
	elif not ANIMATE_3D:
		plt.close(fig)
	
	return fig, ax, X_loaded, y_loaded, z_vals, XX, YY, ZZ


def _animate_3d_rotation():
	"""Animate the 3D plot with dynamic camera movements: rotation, zoom, top view, etc."""
	try:
		import matplotlib.pyplot as plt
		from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
	except Exception as e:
		print(f"Animation libraries not available: {e}")
		return
	
	print("\nGenerating 3D animation with dynamic camera movements...")
	
	# Get the figure and data from _plot_3d_from_exports
	result = _plot_3d_from_exports(
		class_label=CLASS_FOR_3D,
		use_bias=USE_BIAS_FOR_3D,
		save_png=False,
		show=False,
	)
	
	if result is None:
		print("Could not generate 3D plot for animation.")
		return
	
	fig, ax, X_loaded, y_loaded, z_vals, XX, YY, ZZ = result
	
	# Store initial limits for zoom
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	zlim = ax.get_zlim()
	
	# Define keyframes: (frame_num, elevation, azimuth, zoom_factor, description)
	total_frames = ANIMATION_FRAMES
	keyframes = [
		# Start: standard view, full rotation
		(0, 25, 0, 1.0, "Standard view - rotation start"),
		(90, 25, 90, 1.0, "Quarter turn"),
		(180, 25, 180, 1.0, "Half turn"),
		(270, 25, 270, 1.0, "Three-quarter turn"),
		
		# Top view sequence
		(360, 90, 0, 1.0, "Top view - bird's eye"),
		(450, 90, 180, 1.0, "Top view - rotated"),
		
		# Side views
		(540, 0, 0, 1.0, "Front view"),
		(600, 0, 90, 1.0, "Side view"),
		
		# Zoom in sequence
		(660, 25, 45, 0.5, "Zoom in close"),
		(720, 25, 135, 0.5, "Zoom in - different angle"),
		
		# Low angle view
		(780, 10, 225, 1.0, "Low angle view"),
		(840, 10, 315, 1.0, "Low angle - opposite side"),
		
		# High angle view with zoom out
		(900, 60, 45, 1.5, "High angle - zoomed out"),
		(960, 60, 225, 1.5, "High angle - opposite"),
		
		# Return to start
		(1020, 25, 360, 1.0, "Return to start"),
	]
	
	# Interpolate between keyframes
	def interpolate_params(frame):
		# Find surrounding keyframes
		for i in range(len(keyframes) - 1):
			f1, e1, a1, z1, _ = keyframes[i]
			f2, e2, a2, z2, _ = keyframes[i + 1]
			
			if f1 <= frame <= f2:
				# Linear interpolation
				t = (frame - f1) / (f2 - f1) if f2 != f1 else 0
				elev = e1 + t * (e2 - e1)
				azim = a1 + t * (a2 - a1)
				zoom = z1 + t * (z2 - z1)
				return elev, azim, zoom
		
		# Default to last keyframe
		return keyframes[-1][1], keyframes[-1][2], keyframes[-1][3]
	
	# Animation function with dynamic camera
	def update(frame):
		elev, azim, zoom = interpolate_params(frame)
		ax.view_init(elev=elev, azim=azim)
		
		# Apply zoom by adjusting axis limits
		x_center = (xlim[0] + xlim[1]) / 2
		y_center = (ylim[0] + ylim[1]) / 2
		z_center = (zlim[0] + zlim[1]) / 2
		
		x_range = (xlim[1] - xlim[0]) / 2 * zoom
		y_range = (ylim[1] - ylim[0]) / 2 * zoom
		z_range = (zlim[1] - zlim[0]) / 2 * zoom
		
		ax.set_xlim(x_center - x_range, x_center + x_range)
		ax.set_ylim(y_center - y_range, y_center + y_range)
		ax.set_zlim(z_center - z_range, z_center + z_range)
		
		return fig,
	
	# Use total frames from last keyframe
	max_frame = keyframes[-1][0]
	frames_to_use = min(max_frame, total_frames)
	
	# Create animation
	anim = FuncAnimation(
		fig, 
		update, 
		frames=np.linspace(0, frames_to_use, frames_to_use),
		interval=1000/ANIMATION_FPS,
		blit=False
	)
	
	print(f"Animation sequence: {len(keyframes)} keyframes, {frames_to_use} total frames")
	
	# Save animation
	if SAVE_ANIMATION:
		script_dir = Path(__file__).resolve().parent
		out_dir = script_dir / EXPORT_DIR
		out_dir.mkdir(parents=True, exist_ok=True)
		
		# Try to save as MP4 first, fallback to GIF
		try:
			fname = out_dir / f"dotproduct_3d_class_{CLASS_FOR_3D}_rotation.mp4"
			writer = FFMpegWriter(fps=ANIMATION_FPS, bitrate=1800)
			anim.save(fname, writer=writer, dpi=150)
			print(f"Saved animation to: {fname}")
		except Exception as e:
			print(f"Could not save as MP4 ({e}), trying GIF...")
			try:
				fname = out_dir / f"dotproduct_3d_class_{CLASS_FOR_3D}_rotation.gif"
				writer = PillowWriter(fps=ANIMATION_FPS)
				anim.save(fname, writer=writer, dpi=100)
				print(f"Saved animation to: {fname}")
			except Exception as e2:
				print(f"Could not save animation: {e2}")
	
	if SHOW_3D:
		plt.show()
	else:
		plt.close(fig)


# if PLOT3D:
# 	if ANIMATE_3D:
# 		_animate_3d_rotation()
# 	else:
# 		_plot_3d_from_exports(
# 			class_label=CLASS_FOR_3D,
# 			use_bias=USE_BIAS_FOR_3D,
# 			save_png=SAVE_3D,
# 			show=SHOW_3D,
# 		)


def _compute_grid_and_scores_from_exports():
	"""Utility: load dataset/weights, build a grid, and compute class scores.

	Returns a dict with:
	  - XX, YY: meshgrid arrays (shape [Ny, Nx])
	  - scores: array of shape (Ny, Nx, K) with scores s_k(x) = w_k^T x + b_k
	  - X_loaded, y_loaded: dataset arrays
	  - class_labels: labels in the same order as scores' last dimension
	  - extent: (x_min, x_max, y_min, y_max) for imshow
	"""
	script_dir = Path(__file__).resolve().parent
	weights_csv = script_dir / EXPORT_DIR / "weights.csv"
	dataset_csv = script_dir / EXPORT_DIR / "dataset.csv"
	if not weights_csv.exists() or not dataset_csv.exists():
		print("Exports not found; run training/export first.")
		return None

	weights = np.genfromtxt(weights_csv, delimiter=",", names=True)
	if weights.shape == ():
		weights = np.array([weights], dtype=weights.dtype)
	class_labels = weights["class_label"].astype(int)
	# Build W (D x K) and b (K,) in the order they appear in weights.csv
	W = np.vstack([[float(weights["w1"][i]), float(weights["w2"][i])] for i in range(len(weights))]).T
	b = np.array([float(weights["b"][i]) for i in range(len(weights))], dtype=float)

	ds = np.genfromtxt(dataset_csv, delimiter=",", names=True)
	X_loaded = np.vstack([ds["x1"], ds["x2"]]).T.astype(np.float64)
	y_loaded = ds["y"].astype(int)

	# Grid over data extent with padding
	x_min, x_max = float(X_loaded[:, 0].min()), float(X_loaded[:, 0].max())
	y_min, y_max = float(X_loaded[:, 1].min()), float(X_loaded[:, 1].max())
	pad_x = 0.5 * (x_max - x_min + 1e-9)
	pad_y = 0.5 * (y_max - y_min + 1e-9)
	x_lin = np.linspace(x_min - pad_x, x_max + pad_x, 200)
	y_lin = np.linspace(y_min - pad_y, y_max + pad_y, 200)
	XX, YY = np.meshgrid(x_lin, y_lin)
	grid = np.c_[XX.ravel(), YY.ravel()].astype(np.float64)  # (Ny*Nx, 2)

	# Scores for all classes: S = XW + b
	# With bias
	S = grid @ W + b  # (Ny*Nx, K)
	scores = S.reshape(XX.shape + (W.shape[1],))
	# Without bias
	S_nb = grid @ W  # (Ny*Nx, K)
	scores_nb = S_nb.reshape(XX.shape + (W.shape[1],))

	extent = (x_lin.min(), x_lin.max(), y_lin.min(), y_lin.max())
	return {
		"XX": XX,
		"YY": YY,
		"scores": scores,
		"scores_no_bias": scores_nb,
		"X_loaded": X_loaded,
		"y_loaded": y_loaded,
		"class_labels": class_labels,
		"W": W,
		"b": b,
		"grid": grid,
		"extent": extent,
	}


def _plot_Z_heatmap_from_exports(as_log: bool = Z_AS_LOG, save_png: bool = SAVE_Z, show: bool = SHOW_Z):
	"""2D heatmap of Z(x) = sum_n exp(w_n^T x + b_n).

	If as_log is True, plot log Z(x) for numerical stability.
	"""
	try:
		import matplotlib.pyplot as plt
	except Exception:
		print("matplotlib not available, skipping Z heatmap.")
		return

	bundle = _compute_grid_and_scores_from_exports()
	if bundle is None:
		return

	XX, YY = bundle["XX"], bundle["YY"]
	scores = bundle["scores"]  # (Ny, Nx, K)
	X_loaded, y_loaded = bundle["X_loaded"], bundle["y_loaded"]
	extent = bundle["extent"]

	# Compute Z via log-sum-exp for stability
	s_max = np.max(scores, axis=-1, keepdims=True)
	with np.errstate(over='ignore', under='ignore', invalid='ignore'):
		sum_exp = np.sum(np.exp(scores - s_max), axis=-1)  # (Ny, Nx)
		logZ = (s_max[..., 0] + np.log(sum_exp + 1e-12))   # (Ny, Nx)
	Z = np.exp(logZ)

	# Dark background styling
	plt.style.use('dark_background')
	fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
	data_to_plot = logZ if as_log else Z
	hm = ax.imshow(
		data_to_plot,
		origin='lower',
		extent=extent,
		aspect='auto',
		cmap='magma',
	)
	cbar = plt.colorbar(hm, ax=ax)
	cbar.set_label('log Z(x)' if as_log else 'Z(x)', color='white')

	# Overlay dataset points
	colors = {-1: 'tab:red', 0: 'tab:blue', 1: 'tab:green'}
	for lab in np.unique(y_loaded):
		mask = (y_loaded == lab)
		ax.scatter(X_loaded[mask, 0], X_loaded[mask, 1], s=8, c=colors.get(int(lab), 'gray'), alpha=0.7, label=f"y={int(lab)}")
	# ax.legend(loc='upper left')
	ax.set_xlabel('x1', color='white')
	ax.set_ylabel('x2', color='white')
	ax.set_title('Heatmap of ' + (r'$\log Z(x)$' if as_log else r'$Z(x)$'), color='white')
	ax.set_facecolor('black')
	plt.tight_layout()

	if save_png:
		out_dir = Path(__file__).resolve().parent / EXPORT_DIR
		out_dir.mkdir(parents=True, exist_ok=True)
		fname = out_dir / f"Z_heatmap{'_log' if as_log else ''}.png"
		plt.savefig(fname, dpi=1000, facecolor='black', bbox_inches='tight')
		print(f"Saved Z heatmap to: {fname}")

	if show:
		plt.show()
	else:
		plt.close(fig)


def _plot_Z_surface_from_exports(as_log: bool = Z_AS_LOG, save_png: bool = SAVE_Z, show: bool = SHOW_Z):
	"""3D surface plot of Z(x) = sum_n exp(w_n^T x + b_n) (or log Z)."""
	try:
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
	except Exception:
		print("matplotlib not available, skipping Z 3D surface.")
		return

	bundle = _compute_grid_and_scores_from_exports()
	if bundle is None:
		return

	XX, YY = bundle["XX"], bundle["YY"]
	scores = bundle["scores"]  # (Ny, Nx, K)

	# Compute Z via log-sum-exp
	s_max = np.max(scores, axis=-1, keepdims=True)
	with np.errstate(over='ignore', under='ignore', invalid='ignore'):
		sum_exp = np.sum(np.exp(scores - s_max), axis=-1)  # (Ny, Nx)
		logZ = (s_max[..., 0] + np.log(sum_exp + 1e-12))   # (Ny, Nx)
	Z = np.exp(logZ)

	data_to_plot = logZ if as_log else Z

	# Dark background styling
	plt.style.use('dark_background')
	fig = plt.figure(figsize=(10, 8), facecolor='black')
	ax = fig.add_subplot(111, projection='3d')
	
	# Remove panes & grids for clean dark look
	ax.grid(False)
	ax.xaxis.pane.set_visible(False)
	ax.yaxis.pane.set_visible(False)
	ax.zaxis.pane.set_visible(False)
	ax.set_facecolor('black')
	
	ax.plot_surface(XX, YY, data_to_plot, cmap='magma', alpha=0.9, linewidth=0)
	ax.set_xlabel('x1', color='white')
	ax.set_ylabel('x2', color='white')
	ax.set_zlabel('log Z(x)' if as_log else 'Z(x)', color='white')
	ax.set_title('Surface of ' + (r'$\log Z(x)$' if as_log else r'$Z(x)$'), color='white')
	plt.tight_layout()

	if save_png:
		out_dir = Path(__file__).resolve().parent / EXPORT_DIR
		out_dir.mkdir(parents=True, exist_ok=True)
		fname = out_dir / f"Z_surface{'_log' if as_log else ''}.png"
		plt.savefig(fname, dpi=1000, facecolor='black', bbox_inches='tight')
		print(f"Saved Z 3D surface to: {fname}")

	if show:
		plt.show()
	else:
		plt.close(fig)


if PLOT_Z2D:
	_plot_Z_heatmap_from_exports(as_log=Z_AS_LOG, save_png=SAVE_Z, show=SHOW_Z)

if PLOT_Z3D:
	_plot_Z_surface_from_exports(as_log=Z_AS_LOG, save_png=SAVE_Z, show=SHOW_Z)


def _plot_P_heatmap_from_exports(
	classes_to_plot: list[int] = P_CLASSES_TO_PLOT,
	num_use_bias: bool = PROB_NUM_USE_BIAS,
	den_use_bias: bool = PROB_DEN_USE_BIAS,
	save_png: bool = SAVE_P,
	show: bool = SHOW_P,
):
	"""Plot probability heatmaps P(n|x) = exp(s_n(x)) / sum_k exp(s_k(x)).

	- Numerator uses s_n(x) = w_n^T x (+ b_n if num_use_bias)
	- Denominator Z uses scores with or without bias depending on den_use_bias
	"""
	try:
		import matplotlib.pyplot as plt
	except Exception:
		print("matplotlib not available, skipping P(n|x) heatmaps.")
		return

	bundle = _compute_grid_and_scores_from_exports()
	if bundle is None:
		return

	XX, YY = bundle["XX"], bundle["YY"]
	scores_b = bundle["scores"]
	scores_nb = bundle["scores_no_bias"]
	class_labels = bundle["class_labels"]
	X_loaded, y_loaded = bundle["X_loaded"], bundle["y_loaded"]
	extent = bundle["extent"]

	# Choose score tensors for numerator and denominator
	S_num = scores_b if num_use_bias else scores_nb
	S_den = scores_b if den_use_bias else scores_nb

	# Precompute logZ via log-sum-exp for denominator
	m = np.max(S_den, axis=-1, keepdims=True)  # (Ny, Nx, 1)
	with np.errstate(over='ignore', under='ignore', invalid='ignore'):
		sum_exp = np.sum(np.exp(S_den - m), axis=-1)  # (Ny, Nx)
		logZ = m[..., 0] + np.log(sum_exp + 1e-12)    # (Ny, Nx)

	# Map requested labels to indices
	# Keep only classes that exist
	labels_to_plot = [int(c) for c in classes_to_plot if int(c) in set(map(int, class_labels))]
	if not labels_to_plot:
		print("No valid classes to plot.")
		return

	# Create a separate figure for each requested class
	for j, lab in enumerate(labels_to_plot):
		# Dark background styling
		plt.style.use('dark_background')
		fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor='black')
		ax.set_facecolor('black')
		
		# class index
		k = int(np.where(class_labels == lab)[0][0])
		# logP = s_k - logZ
		logP = S_num[..., k] - logZ
		with np.errstate(over='ignore', under='ignore', invalid='ignore'):
			P = np.exp(np.clip(logP, -50, 50))  # clip for safety
		hm = ax.imshow(P, origin='lower', extent=extent, aspect='auto', vmin=0.0, vmax=1.0, cmap='magma')
		cbar = plt.colorbar(hm, ax=ax)
		cbar.set_label(f"P(n={lab} | x)", color='white')

		# Overlay dataset points
		colors = {-1: 'tab:red', 0: 'tab:blue', 1: 'tab:green'}
		for yl in np.unique(y_loaded):
			mask = (y_loaded == yl)
			ax.scatter(X_loaded[mask, 0], X_loaded[mask, 1], s=8, c=colors.get(int(yl), 'gray'), alpha=0.6, label=f"y={int(yl)}")
		ax.set_title(f"P(n={lab} | x)" + (" (num+bias)" if num_use_bias else "") + (" / Z(bias)" if den_use_bias else " / Z(no-bias)"), color='white')
		ax.set_xlabel('x1', color='white')
		ax.set_ylabel('x2', color='white')
		# ax.legend(loc='upper left')

		plt.tight_layout()

		if save_png:
			out_dir = Path(__file__).resolve().parent / EXPORT_DIR
			out_dir.mkdir(parents=True, exist_ok=True)
			bias_tag = ("numb" if num_use_bias else "numb0") + ("_denb" if den_use_bias else "_denb0")
			fname = out_dir / f"P_heatmap_n_{lab}_{bias_tag}.png"
			plt.savefig(fname, dpi=1000, facecolor='black', bbox_inches='tight')
			print(f"Saved probability heatmap to: {fname}")

		if show:
			plt.show()
		else:
			plt.close(fig)


if PLOT_P2D:
	_plot_P_heatmap_from_exports(
		classes_to_plot=P_CLASSES_TO_PLOT,
		num_use_bias=PROB_NUM_USE_BIAS,
		den_use_bias=PROB_DEN_USE_BIAS,
		save_png=SAVE_P,
		show=SHOW_P,
	)



