from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


@dataclass
class ArraySignalOverlayView:
	"""Body-frame whisker layout with signal vectors and deflected geometry."""

	layout_xy: np.ndarray
	mesh_edges: list[tuple[int, int]]
	ellipse_major: float
	ellipse_minor: float
	deflection_gain: float
	max_deflection: float
	show_deflection: bool = False
	view_scale: float = 2.0
	default_max_signal: float = 1.0
	signal_color: str = "tab:green"
	orig_color: str = "tab:grey"
	def_color: str = "tab:orange"

	ax: Axes | None = field(default=None, init=False)
	orig_mesh_lines: list[Line2D] = field(default_factory=list, init=False)
	def_mesh_lines: list[Line2D] = field(default_factory=list, init=False)
	orig_ellipses: list[Ellipse] = field(default_factory=list, init=False)
	def_ellipses: list[Ellipse] = field(default_factory=list, init=False)
	signal_quiver: Any | None = field(default=None, init=False)
	whisker_spacing: float = field(default=0.0, init=False)
	reference_max_signal: float = field(default=1.0, init=False)
	quiver_scale: float = field(default=1.0, init=False)

	def _infer_whisker_spacing(self, pts: np.ndarray) -> float:
		"""Estimate nominal whisker spacing as median nearest-neighbor distance."""
		n = int(pts.shape[0])
		if n < 2:
			return 0.0
		d = pts[:, None, :] - pts[None, :, :]
		dist = np.linalg.norm(d, axis=2)
		np.fill_diagonal(dist, np.inf)
		nn = np.min(dist, axis=1)
		nn = nn[np.isfinite(nn) & (nn > 0.0)]
		if nn.size == 0:
			return 0.0
		return float(np.median(nn))

	def _compute_quiver_scale(self) -> float:
		spacing = float(self.whisker_spacing)
		ref_max = float(self.reference_max_signal)
		if spacing <= 1e-12 or ref_max <= 1e-12:
			return 1.0
		# For angles="xy" and scale_units="xy", arrow length ~= |U| / scale.
		# Set scale so max reference signal maps to one whisker spacing.
		return ref_max / spacing

	def set_signal_reference_max(self, max_signal: float | None) -> None:
		if max_signal is None or not np.isfinite(max_signal) or float(max_signal) <= 0.0:
			self.reference_max_signal = max(1e-9, float(self.default_max_signal))
		else:
			self.reference_max_signal = float(max_signal)
		self.quiver_scale = self._compute_quiver_scale()
		if self.signal_quiver is not None:
			self.signal_quiver.scale = self.quiver_scale

	def create_axes(self, fig: Figure, parent: SubplotSpec) -> None:
		self.ax = fig.add_subplot(parent)
		self.ax.set_aspect("equal")
		self.ax.set_title("signal overlay", fontsize=9)
		self.ax.tick_params(axis="both", labelsize=7)
		self.ax.grid(True, color="0.9", lw=0.6)

		orig = np.asarray(self.layout_xy, dtype=float)
		if orig.ndim != 2 or orig.shape[1] != 2:
			raise ValueError("layout_xy must be an Nx2 array")
		n = orig.shape[0]
		self.whisker_spacing = self._infer_whisker_spacing(orig)
		self.set_signal_reference_max(None)

		for _ in self.mesh_edges:
			ln = Line2D([], [], color=self.orig_color, lw=1.2, alpha=0.85, zorder=2)
			ln.set_animated(True)
			self.ax.add_line(ln)
			self.orig_mesh_lines.append(ln)
		if self.show_deflection:
			for _ in self.mesh_edges:
				ln = Line2D([], [], color=self.def_color, lw=1.0, alpha=0.65, zorder=3)
				ln.set_animated(True)
				self.ax.add_line(ln)
				self.def_mesh_lines.append(ln)

		for _ in range(n):
			e_orig = Ellipse((0.0, 0.0), width=self.ellipse_major, height=self.ellipse_minor)
			e_orig.set_edgecolor(self.orig_color)
			e_orig.set_facecolor("none")
			e_orig.set_linewidth(1.1)
			e_orig.set_zorder(2)
			e_orig.set_animated(True)
			self.ax.add_patch(e_orig)
			self.orig_ellipses.append(e_orig)

			if self.show_deflection:
				e_def = Ellipse((0.0, 0.0), width=self.ellipse_major, height=self.ellipse_minor)
				e_def.set_edgecolor(self.def_color)
				e_def.set_facecolor("none")
				e_def.set_linewidth(1.0)
				e_def.set_zorder(3)
				e_def.set_animated(True)
				self.ax.add_patch(e_def)
				self.def_ellipses.append(e_def)

		zeros = np.zeros(n, dtype=float)
		self.signal_quiver = self.ax.quiver(
			orig[:, 0],
			orig[:, 1],
			zeros,
			zeros,
			color=self.signal_color,
			angles="xy",
			scale_units="xy",
			scale=self.quiver_scale,
			width=0.01,
			headwidth=2.2,
			headlength=2.6,
			headaxislength=2.4,
			zorder=4,
		)
		self.signal_quiver.set_animated(True)

		if self.view_scale <= 0.0:
			raise ValueError("view_scale must be positive")
		x_mid = float(np.mean(orig[:, 0]))
		y_mid = float(np.mean(orig[:, 1]))
		x_span = max(1e-9, float(np.max(orig[:, 0]) - np.min(orig[:, 0])))
		y_span = max(1e-9, float(np.max(orig[:, 1]) - np.min(orig[:, 1])))
		half_span = 0.5 * max(x_span, y_span) * float(self.view_scale)
		half_span = max(half_span, self.max_deflection * 1.2, self.ellipse_major * 1.5, 0.01)
		self.ax.set_xlim(x_mid - half_span, x_mid + half_span)
		self.ax.set_ylim(y_mid - half_span, y_mid + half_span)

		self._update_mesh(self.orig_mesh_lines, orig)
		if self.show_deflection:
			self._update_mesh(self.def_mesh_lines, orig)
		self._update_ellipses(self.orig_ellipses, orig)
		if self.show_deflection:
			self._update_ellipses(self.def_ellipses, orig)

	def _update_mesh(self, lines: list[Line2D], pts: np.ndarray) -> None:
		for ln, (i, j) in zip(lines, self.mesh_edges):
			ln.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])

	def _update_ellipses(self, ellipses: list[Ellipse], centers: np.ndarray) -> None:
		for e, p in zip(ellipses, centers):
			e.center = (float(p[0]), float(p[1]))
			e.angle = 0.0

	def update(self, relative_velocity_xy: np.ndarray) -> None:
		if self.ax is None or self.signal_quiver is None:
			return
		sig = np.asarray(relative_velocity_xy, dtype=float)
		orig = np.asarray(self.layout_xy, dtype=float)
		if sig.ndim != 2 or sig.shape != orig.shape:
			raise ValueError(f"relative_velocity_xy must be shape {orig.shape}")

		deflection = sig * float(self.deflection_gain)
		mag = np.linalg.norm(deflection, axis=1)
		mask = mag > float(self.max_deflection)
		if np.any(mask):
			deflection[mask] *= (float(self.max_deflection) / mag[mask])[:, None]
		deff = orig + deflection

		if self.show_deflection:
			self._update_mesh(self.def_mesh_lines, deff)
			self._update_ellipses(self.def_ellipses, deff)
		self.signal_quiver.set_offsets(orig)
		self.signal_quiver.set_UVC(sig[:, 0], sig[:, 1])

	def animated_artists(self) -> tuple:
		if self.signal_quiver is None:
			return tuple(self.orig_mesh_lines + self.def_mesh_lines + self.orig_ellipses + self.def_ellipses)
		return tuple(
			self.orig_mesh_lines
			+ self.def_mesh_lines
			+ self.orig_ellipses
			+ self.def_ellipses
			+ [self.signal_quiver]
		)
