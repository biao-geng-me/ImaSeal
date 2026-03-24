from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from read_plt import TecplotZone, read_plt75
from simple_advection import advect_2d


@dataclass
class FlowFrame2D:
	"""Single 2D frame prepared for visualization."""

	name: str
	x: np.ndarray
	y: np.ndarray
	u: np.ndarray
	v: np.ndarray
	vorticity: np.ndarray


def _extract_2d(zone: TecplotZone, var: str) -> np.ndarray:
	if var not in zone.data:
		raise KeyError(f"Variable '{var}' is missing in zone data")
	arr = zone.data[var]
	if arr.ndim != 3 or arr.shape[2] < 1:
		raise ValueError(f"Variable '{var}' has unsupported shape {arr.shape}")
	return arr[:, :, 0]


def _compute_vorticity(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
	# For data layout u[i, j], x-index is axis 0 and y-index is axis 1.
	edge = 2 if min(len(x), len(y)) > 2 else 1
	dvdx = np.gradient(v, x, axis=0, edge_order=edge)
	dudy = np.gradient(u, y, axis=1, edge_order=edge)
	return dvdx - dudy


def _normalize_quiver_uv(
	x_sub: np.ndarray,
	y_sub: np.ndarray,
	u_sub: np.ndarray,
	v_sub: np.ndarray,
	velocity_ref: float = 0.2,
	frac: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
	"""Scale vectors so velocity_ref maps to frac times sampled grid spacing."""

	if x_sub.size < 2 or y_sub.size < 2:
		return u_sub, v_sub
	if velocity_ref <= 0.0:
		raise ValueError("velocity_ref must be positive")
	if frac <= 0.0:
		raise ValueError("frac must be positive")

	dx = float(np.median(np.diff(x_sub)))
	dy = float(np.median(np.diff(y_sub)))
	cell_spacing = min(abs(dx), abs(dy))
	if cell_spacing <= 0.0:
		return u_sub, v_sub
	max_len_allowed = frac * cell_spacing

	# Keep arrows bounded even if data occasionally exceeds the assumed max.
	mag = np.hypot(u_sub, v_sub)
	clip_factor = np.ones_like(mag)
	nonzero = mag > 0.0
	clip_factor[nonzero] = np.minimum(1.0, velocity_ref / mag[nonzero])

	scale_factor = max_len_allowed / velocity_ref
	return u_sub * clip_factor * scale_factor, v_sub * clip_factor * scale_factor


def load_flow_frames(data_path: str | Path) -> list[FlowFrame2D]:
	"""Load all Q.*.plt files from a directory and prepare 2D velocity/vorticity frames."""

	root = Path(data_path)
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Data path is not a directory: {root}")

	files = sorted(root.glob("Q.*.plt"))
	if not files:
		raise FileNotFoundError(f"No files matching 'Q.*.plt' found in: {root}")

	frames: list[FlowFrame2D] = []
	for fp in files:
		parsed = read_plt75(fp)
		if parsed.ndim != 2:
			raise ValueError(f"Only 2D data is supported for visualization, got ndim={parsed.ndim} in {fp.name}")
		if not parsed.zones:
			raise ValueError(f"No zones found in file: {fp.name}")

		zone = parsed.zones[0]
		xc = _extract_2d(zone, "xc")
		yc = _extract_2d(zone, "yc")
		u = _extract_2d(zone, "u")
		v = _extract_2d(zone, "v")

		x = np.asarray(xc[:, 0], dtype=float)
		y = np.asarray(yc[0, :], dtype=float)
		w = _compute_vorticity(x, y, u, v)

		frames.append(
			FlowFrame2D(
				name=fp.name,
				x=x,
				y=y,
				u=u,
				v=v,
				vorticity=w,
			)
		)

	return frames


def _load_trajectory_xy(traj_path: str | Path) -> np.ndarray:
	"""Load x,y trajectory data from a text file with comma-separated columns."""

	fp = Path(traj_path)
	if not fp.is_file():
		raise FileNotFoundError(f"Trajectory file not found: {fp}")

	data = np.loadtxt(fp, delimiter=",")
	if data.ndim != 2 or data.shape[1] < 2:
		raise ValueError(f"Trajectory file must contain at least two columns: {fp}")
	return np.asarray(data[:, :2], dtype=float)


def _resolve_trajectory_path(data_path: str | Path, traj_data: str | Path | None) -> Path | None:
	"""Resolve an explicit or auto-discovered trajectory file for a flow folder."""

	if traj_data is not None:
		fp = Path(traj_data)
		if not fp.is_file():
			raise FileNotFoundError(f"Trajectory file not found: {fp}")
		return fp

	root = Path(data_path)
	direct_candidates = []
	for name in ("xy_path.dat",):
		fp = root / name
		if fp.is_file():
			direct_candidates.append(fp)

	glob_candidates = sorted(root.glob("xy_path*.dat"))
	seen: set[Path] = set()
	ordered_candidates: list[Path] = []
	for fp in direct_candidates + glob_candidates:
		if fp not in seen:
			seen.add(fp)
			ordered_candidates.append(fp)

	if not ordered_candidates:
		return None
	if len(ordered_candidates) > 1:
		names = ", ".join(p.name for p in ordered_candidates)
		raise ValueError(f"Multiple trajectory files found in {root}; use --traj-data explicitly: {names}")
	return ordered_candidates[0]


def visualize_flow(
	data_path: str | Path,
	dt: float,
	quiver_step: int = 24,
	frac: float = 0.9,
	vort_step: int = 4,
	autoplay: bool = False,
	fps: float = 25.0,
	show_slider: bool = False,
	show_colorbar: bool = False,
	decay: float = 0.0,
	traj_data: str | Path | None = None,
) -> None:
	"""Interactive quiver plot over vorticity for all Q.*.plt files in a folder."""

	if dt <= 0.0:
		raise ValueError("dt must be positive")
	if quiver_step < 1:
		raise ValueError("quiver_step must be >= 1")
	if frac <= 0.0:
		raise ValueError("frac must be positive")
	if vort_step < 1:
		raise ValueError("vort_step must be >= 1")
	if fps <= 0.0:
		raise ValueError("fps must be positive")

	frames = load_flow_frames(data_path)
	if not frames:
		raise ValueError("No frames loaded")

	traj_path = _resolve_trajectory_path(data_path, traj_data)
	traj_xy = _load_trajectory_xy(traj_path) if traj_path is not None else None

	vmin = -0.1
	vmax = 0.1

	fig, ax = plt.subplots(figsize=(11, 6))
	fig.subplots_adjust(bottom=0.18 if show_slider else 0.1)

	first = frames[0]
	qstep = quiver_step
	x_sub = first.x[::qstep]
	y_sub = first.y[::qstep]
	x_img = first.x[::vort_step]
	y_img = first.y[::vort_step]
	u_sub = first.u[::qstep, ::qstep].T
	v_sub = first.v[::qstep, ::qstep].T
	u_sub, v_sub = _normalize_quiver_uv(x_sub, y_sub, u_sub, v_sub, velocity_ref=0.2, frac=frac)

	vort_display: list[np.ndarray] = [
		np.asarray(f.vorticity[::vort_step, ::vort_step].T, dtype=np.float32) for f in frames
	]
	quiver_display: list[tuple[np.ndarray, np.ndarray]] = [
		_normalize_quiver_uv(
			x_sub,
			y_sub,
			f.u[::qstep, ::qstep].T,
			f.v[::qstep, ::qstep].T,
			velocity_ref=0.2,
			frac=frac,
		)
		for f in frames
	]
	frame_names: list[str] = [f.name for f in frames]

	if decay > 0.0:
		n_real = len(frames)
		t_tot = n_real * dt  # half-life
		n_extra = round(decay * n_real)

		# Advect on display grids to keep extension cost manageable.
		last = frames[n_real - 1]
		u_adv_img = np.asarray(last.u[::vort_step, ::vort_step].T, dtype=float)
		v_adv_img = np.asarray(last.v[::vort_step, ::vort_step].T, dtype=float)
		u_adv_quiv = np.asarray(last.u[::qstep, ::qstep].T, dtype=float)
		v_adv_quiv = np.asarray(last.v[::qstep, ::qstep].T, dtype=float)

		xg_img, yg_img = np.meshgrid(x_img, y_img, indexing="xy")
		xg_quiv, yg_quiv = np.meshgrid(x_sub, y_sub, indexing="xy")

		for k in range(1, n_extra + 1):
			# tried to use simple advection but it doesn't look realistic
			# u_adv_img, v_adv_img = advect_2d(xg_img, yg_img, u_adv_img, v_adv_img, dt*1000)
			# u_adv_quiv, v_adv_quiv = advect_2d(xg_quiv, yg_quiv, u_adv_quiv, v_adv_quiv, dt*1000)

			decay_factor = 0.5 ** (k * dt / 6)
			
			# u_adv_img *= decay_factor
			# v_adv_img *= decay_factor
			# u_adv_quiv *= decay_factor
			# v_adv_quiv *= decay_factor

			# _compute_vorticity expects u,v shaped as (x, y), so transpose from (y, x).
			w_adv = _compute_vorticity(x_img, y_img, u_adv_img.T*decay_factor, v_adv_img.T*decay_factor).T
			
			vort_display.append(np.asarray(w_adv, dtype=np.float32))

			uq_adv, vq_adv = _normalize_quiver_uv(
				x_sub,
				y_sub,
				u_adv_quiv*decay_factor,
				v_adv_quiv*decay_factor,
				velocity_ref=0.2,
				frac=frac,
			)
			quiver_display.append((uq_adv, vq_adv))
			frame_names.append(f"decay | t={((n_real - 1 + k) * dt):.4f} s")

	total_n = len(vort_display)

	mesh = ax.imshow(
		vort_display[0],
		origin="lower",
		extent=(float(x_img[0]), float(x_img[-1]), float(y_img[0]), float(y_img[-1])),
		cmap="RdBu_r",
		vmin=vmin,
		vmax=vmax,
		interpolation="nearest",
		aspect="equal",
	)
	qv = ax.quiver(
		x_sub,
		y_sub,
		u_sub,
		v_sub,
		color="k",
		angles="xy",
		scale_units="xy",
		scale=1.0,
		headlength=2.0,
		headwidth=1.5,
		headaxislength=2.0,
	)
	if traj_xy is not None and traj_xy.size > 0:
		ax.plot(
			traj_xy[:, 0],
			traj_xy[:, 1],
			color="tab:red",
			lw=2.0,
			alpha=0.75,
			zorder=2.5,
			label="object trajectory",
		)
	if show_colorbar:
		fig.colorbar(mesh, ax=ax, label="vorticity")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_aspect("equal")
	if traj_xy is not None and traj_xy.size > 0:
		ax.legend(loc="upper right")

	title = ax.set_title("")

	def _set_frame(idx: int, redraw: bool = True) -> None:
		mesh.set_data(vort_display[idx])
		uq, vq = quiver_display[idx]
		qv.set_UVC(uq, vq)
		time_value = idx * dt
		title.set_text(f"{frame_names[idx]} | frame={idx + 1}/{total_n} | t={time_value:.4f} s")
		if redraw:
			fig.canvas.draw_idle()

	_set_frame(0)
	current_idx = 0
	slider = None
	if show_slider:
		slider_ax = fig.add_axes([0.12, 0.06, 0.76, 0.04])
		slider = Slider(
			ax=slider_ax,
			label="Frame",
			valmin=0,
			valmax=max(total_n - 1, 0),
			valinit=0,
			valstep=1,
		)

		def _on_slider(val: float) -> None:
			nonlocal current_idx
			current_idx = int(val)
			_set_frame(current_idx)

		slider.on_changed(_on_slider)

	anim_ref: list = [None]
	is_playing: list[bool] = [False]
	interval_ms = max(1, int(round(1000.0 / fps)))
	play_idx = current_idx

	def _start_animation() -> None:
		nonlocal play_idx
		if anim_ref[0] is not None:
			anim_ref[0].event_source.start()
			is_playing[0] = True
			return

		def _advance(_frame_idx: int):
			nonlocal play_idx
			play_idx = (play_idx + 1) % total_n
			if slider is not None:
				slider.set_val(play_idx)
			else:
				_set_frame(play_idx)

		anim = FuncAnimation(fig, _advance, interval=interval_ms, cache_frame_data=False, blit=False)
		anim_ref[0] = anim
		fig._anim = anim  # type: ignore[attr-defined]
		is_playing[0] = True

	def _on_key(event) -> None:
		nonlocal current_idx
		if slider is not None:
			idx = int(slider.val)
		else:
			idx = current_idx
		if event.key == "right":
			next_idx = min(idx + 1, total_n - 1)
			current_idx = next_idx
			if slider is not None:
				slider.set_val(next_idx)
			else:
				_set_frame(next_idx)
		elif event.key == "left":
			next_idx = max(idx - 1, 0)
			current_idx = next_idx
			if slider is not None:
				slider.set_val(next_idx)
			else:
				_set_frame(next_idx)
		elif event.key == "pagedown":
			next_idx = min(idx + 4, total_n - 1)
			current_idx = next_idx
			if slider is not None:
				slider.set_val(next_idx)
			else:
				_set_frame(next_idx)
		elif event.key == "pageup":
			next_idx = max(idx - 4, 0)
			current_idx = next_idx
			if slider is not None:
				slider.set_val(next_idx)
			else:
				_set_frame(next_idx)
		elif event.key == " ":
			if is_playing[0]:
				anim_ref[0].event_source.stop()
				is_playing[0] = False
			else:
				_start_animation()
		elif event.key == "home":
			current_idx = 0
			if slider is not None:
				slider.set_val(0)
			else:
				_set_frame(0)
		elif event.key == "end":
			current_idx = total_n - 1
			if slider is not None:
				slider.set_val(total_n - 1)
			else:
				_set_frame(total_n - 1)

	fig.canvas.mpl_connect("key_press_event", _on_key)

	if autoplay:
		_start_animation()

	plt.show()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Visualize flow frames from q2plt-generated Tecplot files")
	parser.add_argument(
		"--data-path",
		type=Path,
		required=True,
		help="Directory containing Q.*.plt files for interactive visualization",
	)
	parser.add_argument(
		"--dt",
		type=float,
		default=1.0,
		help="Time increment between consecutive frames (default: 1.0)",
	)
	parser.add_argument(
		"--quiver-step",
		type=int,
		default=24,
		help="Subsampling step for quiver arrows (default: 24)",
	)
	parser.add_argument(
		"--frac",
		type=float,
		default=1.414,
		help="Arrow-length factor in grid-spacing units (default: sqrt(2))",
	)
	parser.add_argument(
		"--vort-step",
		type=int,
		default=8,
		help="Subsampling step for vorticity background rendering (default: 8)",
	)
	parser.add_argument(
		"--colorbar",
		action="store_true",
		help="Show the vorticity colorbar",
	)
	parser.add_argument(
		"--slider",
		action="store_true",
		help="Show the frame slider",
	)
	parser.add_argument(
		"--autoplay",
		action="store_true",
		help="Automatically play frames in a loop",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=25.0,
		help="Target playback FPS for --autoplay mode (default: 25)",
	)
	parser.add_argument(
		"--no-slider",
		action="store_true",
		help="Deprecated compatibility flag; slider is hidden by default",
	)
	parser.add_argument(
		"--decay",
		type=float,
		default=0.0,
		metavar="MULT",
		help="Append MULT*N extra frames where velocity decays from the last real frame with half-life = total real duration",
	)
	parser.add_argument(
		"--traj-data",
		type=Path,
		default=None,
		help="Optional trajectory file to overlay (x,y columns). If omitted, auto-load xy_path.dat or a single path_xy*.dat from --data-path",
	)
	args = parser.parse_args()

	visualize_flow(
		args.data_path,
		dt=args.dt,
		quiver_step=args.quiver_step,
		frac=args.frac,
		vort_step=args.vort_step,
		autoplay=args.autoplay,
		fps=args.fps,
		show_slider=args.slider and not args.no_slider,
		show_colorbar=args.colorbar,
		decay=args.decay,
		traj_data=args.traj_data,
	)
