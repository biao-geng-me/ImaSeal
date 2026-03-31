from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D


def _expand_limits(vmin: float, vmax: float, pad: float = 0.08) -> tuple[float, float]:
	if not np.isfinite(vmin) or not np.isfinite(vmax):
		return -1.0, 1.0
	if abs(vmax - vmin) < 1e-12:
		span = max(1e-3, abs(vmax) * 0.1)
		return vmin - span, vmax + span
	span = vmax - vmin
	return vmin - span * pad, vmax + span * pad


@dataclass
class SignalHistoryView:
	"""Compact per-whisker signal history panel with shared scaling.

	Each row corresponds to one whisker. Left y-axis shows lift (vy), right y-axis
	shows drag (vx). Limits are shared across whiskers per signal channel.
	"""

	num_whiskers: int
	time_label: str = "t (s)"
	left_label: str = "vy (lift)"
	right_label: str = "vx (drag)"
	row_label_prefix: str = "W"
	lift_color: str = "tab:blue"
	drag_color: str = "tab:orange"
	line_width: float = 1.1
	_alpha: float = 0.95

	time_hist: list[float] = field(default_factory=list, init=False)
	lift_hist: list[np.ndarray] = field(default_factory=list, init=False)
	drag_hist: list[np.ndarray] = field(default_factory=list, init=False)

	fig: Figure | None = field(default=None, init=False)
	axes_left: list[Axes] = field(default_factory=list, init=False)
	axes_right: list[Axes] = field(default_factory=list, init=False)
	lift_lines: list[Line2D] = field(default_factory=list, init=False)
	drag_lines: list[Line2D] = field(default_factory=list, init=False)
	row_text: list = field(default_factory=list, init=False)

	def create_axes(self, fig: Figure, parent: SubplotSpec, width_ratio: float = 1.0) -> None:
		if self.num_whiskers < 1:
			raise ValueError("num_whiskers must be >= 1")

		self.fig = fig
		self.axes_left.clear()
		self.axes_right.clear()
		self.lift_lines.clear()
		self.drag_lines.clear()
		self.row_text.clear()

		sub = parent.subgridspec(self.num_whiskers, 1, hspace=0.03)
		shared_x: Axes | None = None
		for idx in range(self.num_whiskers):
			ax_l = fig.add_subplot(sub[idx, 0], sharex=shared_x)
			if shared_x is None:
				shared_x = ax_l
			ax_r = ax_l.twinx()

			(ln_l,) = ax_l.plot([], [], color=self.lift_color, lw=self.line_width, alpha=self._alpha)
			(ln_r,) = ax_r.plot([], [], color=self.drag_color, lw=self.line_width, alpha=self._alpha)
			ln_l.set_animated(True)
			ln_r.set_animated(True)

			ax_l.spines["top"].set_visible(False)
			ax_r.spines["top"].set_visible(False)
			ax_l.spines["right"].set_visible(False)
			ax_r.spines["left"].set_visible(False)
			ax_l.grid(True, which="major", axis="both", color="0.85", lw=0.6, alpha=0.8)

			if idx != self.num_whiskers - 1:
				ax_l.tick_params(axis="x", labelbottom=False)
			ax_l.tick_params(axis="y", labelsize=7, pad=1)
			ax_r.tick_params(axis="y", labelsize=7, pad=1)

			if idx == self.num_whiskers // 2:
				ax_l.set_ylabel(self.left_label, fontsize=8, color=self.lift_color, labelpad=2)
				ax_r.set_ylabel(self.right_label, fontsize=8, color=self.drag_color, labelpad=2)
			else:
				ax_l.set_ylabel("")
				ax_r.set_ylabel("")

			label = ax_l.text(
				0.01,
				0.83,
				f"{self.row_label_prefix}{idx + 1}",
				transform=ax_l.transAxes,
				fontsize=7,
				color="0.25",
			)
			label.set_animated(True)

			self.axes_left.append(ax_l)
			self.axes_right.append(ax_r)
			self.lift_lines.append(ln_l)
			self.drag_lines.append(ln_r)
			self.row_text.append(label)

		self.axes_left[-1].set_xlabel(self.time_label, fontsize=8)

	def reset(self, t0: float, signal_xy: np.ndarray) -> None:
		self.time_hist.clear()
		self.lift_hist.clear()
		self.drag_hist.clear()
		self.append(float(t0), signal_xy)

	def append(self, t: float, signal_xy: np.ndarray) -> None:
		s = np.asarray(signal_xy, dtype=float)
		if s.ndim != 2 or s.shape != (self.num_whiskers, 2):
			raise ValueError(f"signal_xy must be shape ({self.num_whiskers}, 2)")
		self.time_hist.append(float(t))
		self.drag_hist.append(s[:, 0].copy())
		self.lift_hist.append(s[:, 1].copy())
		self._refresh_lines()

	def set_history(self, time_hist: np.ndarray, signal_hist_xy: np.ndarray) -> None:
		t = np.asarray(time_hist, dtype=float)
		s = np.asarray(signal_hist_xy, dtype=float)
		if t.ndim != 1:
			raise ValueError("time_hist must be a 1D array")
		if s.ndim != 3 or s.shape[1:] != (self.num_whiskers, 2):
			raise ValueError(f"signal_hist_xy must be shape (K, {self.num_whiskers}, 2)")
		if s.shape[0] != t.shape[0]:
			raise ValueError("time_hist and signal_hist_xy must have matching lengths")

		self.time_hist = [float(v) for v in t]
		self.drag_hist = [row[:, 0].copy() for row in s]
		self.lift_hist = [row[:, 1].copy() for row in s]
		self._refresh_lines()

	def _refresh_lines(self) -> None:
		if not self.time_hist:
			return
		t = np.asarray(self.time_hist, dtype=float)
		lift = np.asarray(self.lift_hist, dtype=float)
		drag = np.asarray(self.drag_hist, dtype=float)

		lift_lim = _expand_limits(float(np.min(lift)), float(np.max(lift)))
		drag_lim = _expand_limits(float(np.min(drag)), float(np.max(drag)))

		for idx in range(self.num_whiskers):
			self.lift_lines[idx].set_data(t, lift[:, idx])
			self.drag_lines[idx].set_data(t, drag[:, idx])
			self.axes_left[idx].set_xlim(float(t[0]), float(t[-1]) + 1e-9)
			self.axes_left[idx].set_ylim(*lift_lim)
			self.axes_right[idx].set_ylim(*drag_lim)

	def animated_artists(self) -> tuple:
		return tuple(self.lift_lines + self.drag_lines + self.row_text)
