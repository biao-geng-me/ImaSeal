
"""Interactive whisker-array simulator for a seal moving in flow.

Core behavior:
1) Whiskers are mounted on a rigid 2D array (default: 3x3, 20 mm spacing).
2) The array is piloted by gamepad left-stick direction and right-trigger speed.
3) Commanded velocity is acceleration-limited in x and y.
4) Body heading updates instantly to face forward (zero angle-of-attack assumption).
5) Each whisker senses local flow velocity.
6) Deflection is proportional to local flow velocity and shown as displaced whisker centers.

The visualization shows:
- Original and deflected whisker ellipses.
- Arrows from original to deflected whisker centers.
- Mesh wiring among original centers and deflected centers in different colors.

Controls:
- Gamepad (if pygame + joystick available):
  - Left stick: direction
  - Right trigger: speed command (0 to max speed)
- Keyboard fallback:
  - Space: pause/resume
  - Page Up / X: accelerate
  - Page Down / Z: brake
	- Arrow keys: turn heading by 5 deg toward key direction (minimal-turn)
	- F1: show help window
	- W: toggle base whisker layer (original mesh/whiskers)
	- D: toggle deflected layer (deflected mesh/whiskers)
	- A: toggle deflection arrows
  - R: restart (random y position)
	- V: toggle flow field (vorticity + quiver)
  - T: toggle trajectory paths
	- Tab: reset with a new randomly selected flow path
	- Enter/Return: open flow path selector and reset with chosen path
  - Esc: close
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from read_plt import read_plt75
from visualize_flow import FlowFrame2D, _compute_vorticity, _extract_2d, load_flow_frames


_pygame_spec = importlib.util.find_spec("pygame")
if _pygame_spec is not None:
	pygame = importlib.import_module("pygame")
	PYGAME_AVAILABLE = True
else:
	pygame = None
	PYGAME_AVAILABLE = False


Vec2 = np.ndarray


@dataclass
class TimingSummary:
	"""Aggregate timing statistics for one labeled code region."""

	total: float = 0.0
	count: int = 0
	minimum: float = float("inf")
	maximum: float = 0.0

	def add(self, dt: float) -> None:
		self.total += dt
		self.count += 1
		self.minimum = min(self.minimum, dt)
		self.maximum = max(self.maximum, dt)

	@property
	def mean(self) -> float:
		return self.total / self.count if self.count > 0 else 0.0


@dataclass
class StepProfiler:
	"""Collect labeled timing statistics across simulation steps."""

	stats: dict[str, TimingSummary] = field(default_factory=dict)

	def record(self, label: str, dt: float) -> None:
		self.stats.setdefault(label, TimingSummary()).add(float(dt))

	def report(self, header: str = "[profile]") -> str:
		lines = [header]
		for label in sorted(self.stats.keys()):
			item = self.stats[label]
			lines.append(
				f"  {label:<24} total={item.total * 1e3:9.3f} ms | "
				f"mean={item.mean * 1e3:8.3f} ms | "
				f"min={item.minimum * 1e3:8.3f} ms | "
				f"max={item.maximum * 1e3:8.3f} ms | n={item.count}"
			)

		# Tick residual = time in _tick not covered by explicitly timed blocks.
		# This is NOT pure matplotlib draw/blit time.
		if "tick_total" in self.stats:
			tick_stat = self.stats["tick_total"]
			top_level_labels = (
				"control_read",
				"simulation_step_total",
				"trajectory_record",
				"render_flow_update",
				"render_mesh_update",
				"render_shape_update",
				"render_arrow_update",
				"text_update",
			)
			measured_total = 0.0
			for label in top_level_labels:
				item = self.stats.get(label)
				if item is not None:
					measured_total += item.total
			residual_total = max(0.0, tick_stat.total - measured_total)
			if tick_stat.count > 0:
				lines.append("")
				lines.append("  [tick breakdown]")
				lines.append(
					f"  {'tick_unprofiled_residual':<24} total={residual_total * 1e3:9.3f} ms | "
					f"mean={residual_total / tick_stat.count * 1e3:8.3f} ms | "
					f"n={tick_stat.count}"
				)
		return "\n".join(lines)

	def print_report(self, header: str = "[profile]") -> None:
		print(self.report(header=header))


def _build_grid_mesh_edges(rows: int, cols: int) -> list[tuple[int, int]]:
	edges: list[tuple[int, int]] = []

	def _idx(r: int, c: int) -> int:
		return r * cols + c

	for r in range(rows):
		for c in range(cols):
			if c + 1 < cols:
				edges.append((_idx(r, c), _idx(r, c + 1)))
			if r + 1 < rows:
				edges.append((_idx(r, c), _idx(r + 1, c)))
	return edges


def _build_knn_mesh_edges(layout_xy: np.ndarray, k: int = 2) -> list[tuple[int, int]]:
	n = int(layout_xy.shape[0])
	if n <= 1:
		return []
	k = int(np.clip(k, 1, max(1, n - 1)))
	d = layout_xy[:, None, :] - layout_xy[None, :, :]
	dist = np.linalg.norm(d, axis=2)
	np.fill_diagonal(dist, np.inf)
	edges: set[tuple[int, int]] = set()
	for i in range(n):
		nn = np.argpartition(dist[i], k)[:k]
		for j in nn:
			a = min(i, int(j))
			b = max(i, int(j))
			if a != b:
				edges.add((a, b))
	return sorted(edges)


@dataclass
class ControllerCommand:
	"""Normalized control command from pilot input."""

	direction: Vec2
	throttle: float


class InputProvider(Protocol):
	"""Interface for providing pilot commands."""

	def read(self) -> ControllerCommand:
		...

	def shutdown(self) -> None:
		...


@dataclass
class KeyboardInput:
	"""Throttle-and-direction keyboard control.

	Page Up / X increases throttle, Page Down / Z decreases throttle.
	Arrow keys rotate direction in 5 degree increments toward the key direction.
	"""

	max_speed: float = 0.4
	accel_per_frame: float = 0.008
	turn_step_deg: float = 5.0
	pressed: set[str] = field(default_factory=set)
	_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=float))
	_throttle: float = 0.0

	@staticmethod
	def _wrap_angle(angle: float) -> float:
		"""Wrap angle to (-pi, pi]."""
		return float(np.arctan2(np.sin(angle), np.cos(angle)))

	def _step_turn_toward_key(self, key: str) -> None:
		"""Apply one fixed turn step toward key direction using minimal turning path."""
		current = float(np.arctan2(self._direction[1], self._direction[0]))
		target_by_key = {
			"right": 0.0,
			"up": np.pi * 0.5,
			"left": np.pi,
			"down": -np.pi * 0.5,
		}
		target = target_by_key.get(key)
		if target is None:
			return
		delta = self._wrap_angle(target - current)

		step = np.deg2rad(self.turn_step_deg)
		if abs(delta) <= step:
			new_heading = target
		else:
			new_heading = current + np.sign(delta) * step

		self._direction = np.array([np.cos(new_heading), np.sin(new_heading)], dtype=float)

	def on_key_press(self, key: str) -> None:
		self.pressed.add(key)
		if key in ("left", "right", "up", "down"):
			self._step_turn_toward_key(key)

	def on_key_release(self, key: str) -> None:
		self.pressed.discard(key)

	def reset(self) -> None:
		"""Clear throttle and direction back to defaults."""
		self.pressed.clear()
		self._throttle = 0.0
		self._direction = np.array([1.0, 0.0], dtype=float)

	def set_command(self, direction: np.ndarray, throttle: float) -> None:
		"""Set keyboard command state explicitly (used for launch/reset presets)."""
		d = np.asarray(direction, dtype=float)
		norm = float(np.linalg.norm(d))
		if norm > 1e-9:
			self._direction = d / norm
		self._throttle = float(np.clip(throttle, 0.0, 1.0))

	def read(self) -> ControllerCommand:
		throttle_step = self.accel_per_frame / max(self.max_speed, 1e-9)
		if "pageup" in self.pressed or "x" in self.pressed:
			self._throttle = min(1.0, self._throttle + throttle_step)
		if "pagedown" in self.pressed or "z" in self.pressed:
			self._throttle = max(0.0, self._throttle - throttle_step)

		return ControllerCommand(direction=self._direction.copy(), throttle=self._throttle)

	def shutdown(self) -> None:
		return


@dataclass
class PygameGamepadInput:
	"""Gamepad input (left stick direction, right trigger throttle)."""

	deadzone: float = 0.12
	joystick: object | None = None

	def __post_init__(self) -> None:
		if not PYGAME_AVAILABLE:
			raise RuntimeError("pygame not available")
		pygame.init()
		pygame.joystick.init()
		if pygame.joystick.get_count() < 1:
			raise RuntimeError("No joystick found")
		self.joystick = pygame.joystick.Joystick(0)
		self.joystick.init()

	def _filter_axis(self, value: float) -> float:
		if abs(value) < self.deadzone:
			return 0.0
		return float(value)

	def _read_trigger(self) -> float:
		# Common mappings: axis 5 (Xbox/DirectInput) or axis 4.
		for axis_id in (5, 4):
			if axis_id < self.joystick.get_numaxes():
				raw = float(self.joystick.get_axis(axis_id))
				# Triggers are often in [-1, 1], convert to [0, 1].
				return float(np.clip((raw + 1.0) * 0.5, 0.0, 1.0))
		return 0.0

	def read(self) -> ControllerCommand:
		pygame.event.pump()
		x = self._filter_axis(float(self.joystick.get_axis(0)))
		y_raw = self._filter_axis(float(self.joystick.get_axis(1)))
		y = -y_raw
		dir_vec = np.array([x, y], dtype=float)
		norm = float(np.linalg.norm(dir_vec))
		if norm > 1e-9:
			dir_vec /= norm
		throttle = self._read_trigger() if norm > 1e-9 else 0.0
		return ControllerCommand(direction=dir_vec, throttle=throttle)

	def shutdown(self) -> None:
		if self.joystick is not None:
			self.joystick.quit()
		if PYGAME_AVAILABLE:
			pygame.joystick.quit()
			pygame.quit()


@dataclass
class WhiskerArrayGeometry:
	"""Rigid whisker array geometry in body-frame coordinates."""

	layout_xy: np.ndarray
	ellipse_major: float
	ellipse_minor: float
	mesh_edges: list[tuple[int, int]]

	@staticmethod
	def regular_grid(rows: int = 3, cols: int = 3, spacing: float = 0.02) -> "WhiskerArrayGeometry":
		if rows < 1 or cols < 1:
			raise ValueError("rows and cols must be >= 1")
		if spacing <= 0.0:
			raise ValueError("spacing must be positive")

		xs = (np.arange(cols) - (cols - 1) * 0.5) * spacing
		ys = (np.arange(rows) - (rows - 1) * 0.5) * spacing
		xy = np.array([(x, y) for y in ys for x in xs], dtype=float)

		edges = _build_grid_mesh_edges(rows=rows, cols=cols)

		# Ellipse axis ratio 1:2 (minor:major) per spec.
		minor = 0.006
		major = minor * 2.0
		return WhiskerArrayGeometry(layout_xy=xy, ellipse_major=major, ellipse_minor=minor, mesh_edges=edges)

	@staticmethod
	def from_layout(
		layout_xy: np.ndarray,
		*,
		ellipse_major: float = 0.012,
		ellipse_minor: float = 0.006,
		mesh_edges: list[tuple[int, int]] | None = None,
		mesh_knn_k: int = 2,
	) -> "WhiskerArrayGeometry":
		xy = np.asarray(layout_xy, dtype=float)
		if xy.ndim != 2 or xy.shape[1] != 2:
			raise ValueError("layout_xy must be an Nx2 numeric array")
		if xy.shape[0] < 1:
			raise ValueError("layout_xy must contain at least one point")
		if not np.all(np.isfinite(xy)):
			raise ValueError("layout_xy contains non-finite values")
		if ellipse_major <= 0.0 or ellipse_minor <= 0.0:
			raise ValueError("ellipse_major and ellipse_minor must be positive")

		n = int(xy.shape[0])
		if mesh_edges is None:
			edges = _build_knn_mesh_edges(xy, k=mesh_knn_k)
		else:
			edges = []
			for pair in mesh_edges:
				if len(pair) != 2:
					raise ValueError("each mesh edge must contain exactly two indices")
				i = int(pair[0])
				j = int(pair[1])
				if i < 0 or j < 0 or i >= n or j >= n:
					raise ValueError(f"mesh edge {(i, j)} out of bounds for {n} points")
				if i != j:
					edges.append((min(i, j), max(i, j)))
			edges = sorted(set(edges))

		return WhiskerArrayGeometry(
			layout_xy=xy,
			ellipse_major=float(ellipse_major),
			ellipse_minor=float(ellipse_minor),
			mesh_edges=edges,
		)


def _load_array_geometry_from_yaml(config_path: str | Path) -> WhiskerArrayGeometry:
	config_fp = Path(config_path)
	if not config_fp.is_file():
		raise ValueError(f"array config file not found: {config_fp}")

	try:
		import yaml
	except Exception as exc:
		raise RuntimeError("PyYAML is required for --array-config. Install with: pip install pyyaml") from exc

	with config_fp.open("r", encoding="utf-8") as fh:
		data = yaml.safe_load(fh)
	if not isinstance(data, dict):
		raise ValueError("array config must be a YAML mapping/object")

	units = str(data.get("units", "m")).strip().lower()
	if units not in {"m", "mm"}:
		raise ValueError("array config 'units' must be 'm' or 'mm'")
	unit_scale = 1e-3 if units == "mm" else 1.0

	def _parse_layout_xy(raw: object) -> np.ndarray:
		xy = np.asarray(raw, dtype=float)
		if xy.ndim != 2 or xy.shape[1] != 2:
			raise ValueError("layout_xy must be an Nx2 numeric array")
		if xy.shape[0] < 1:
			raise ValueError("layout_xy must contain at least one point")
		if not np.all(np.isfinite(xy)):
			raise ValueError("layout_xy contains non-finite values")
		return xy * unit_scale

	def _parse_mesh_edges(raw_edges: object, n_pts: int) -> list[tuple[int, int]]:
		if not isinstance(raw_edges, list):
			raise ValueError("mesh_edges must be a list of [i, j] pairs")
		edges: list[tuple[int, int]] = []
		for edge in raw_edges:
			if not isinstance(edge, (list, tuple)) or len(edge) != 2:
				raise ValueError("each mesh edge must contain exactly two indices")
			i = int(edge[0])
			j = int(edge[1])
			if i < 0 or j < 0 or i >= n_pts or j >= n_pts:
				raise ValueError(f"mesh edge {(i, j)} out of bounds for {n_pts} points")
			if i != j:
				edges.append((min(i, j), max(i, j)))
		return sorted(set(edges))

	def _local_layout_from_cfg(cfg: dict, *, cfg_name: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
		is_grid_cfg = False
		grid_rows = 0
		grid_cols = 0
		layout_raw = cfg.get("layout_xy")
		if layout_raw is not None:
			xy = _parse_layout_xy(layout_raw)
		else:
			rows = cfg.get("rows")
			cols = cfg.get("cols")
			spacing = cfg.get("spacing")
			if rows is None or cols is None or spacing is None:
				raise ValueError(
					f"{cfg_name} must define either 'layout_xy' or all of 'rows', 'cols', and 'spacing'"
				)
			grid_rows = int(rows)
			grid_cols = int(cols)
			is_grid_cfg = True
			xy = WhiskerArrayGeometry.regular_grid(
				rows=grid_rows, cols=grid_cols, spacing=float(spacing) * unit_scale
			).layout_xy

		mesh_edges_raw = cfg.get("mesh_edges")
		if mesh_edges_raw is not None:
			edges = _parse_mesh_edges(mesh_edges_raw, n_pts=int(xy.shape[0]))
		elif is_grid_cfg:
			edges = _build_grid_mesh_edges(rows=grid_rows, cols=grid_cols)
		else:
			mesh_knn_k = int(cfg.get("mesh_knn_k", data.get("mesh_knn_k", 2)))
			edges = _build_knn_mesh_edges(xy, k=mesh_knn_k)
		return xy, edges

	ellipse_minor = float(data.get("ellipse_minor", 0.006)) * unit_scale
	if "ellipse_major" in data:
		ellipse_major = float(data["ellipse_major"]) * unit_scale
	else:
		ratio = float(data.get("major_to_minor_ratio", 2.0))
		if ratio <= 0.0:
			raise ValueError("major_to_minor_ratio must be positive")
		ellipse_major = ellipse_minor * ratio

	unit_arrays_raw = data.get("unit_arrays")
	if unit_arrays_raw is not None:
		if not isinstance(unit_arrays_raw, dict) or not unit_arrays_raw:
			raise ValueError("unit_arrays must be a non-empty mapping of unit definitions")

		unit_defs: dict[str, tuple[np.ndarray, list[tuple[int, int]]]] = {}
		for unit_name, unit_cfg in unit_arrays_raw.items():
			if not isinstance(unit_cfg, dict):
				raise ValueError(f"unit_arrays.{unit_name} must be a mapping/object")
			unit_defs[str(unit_name)] = _local_layout_from_cfg(unit_cfg, cfg_name=f"unit_arrays.{unit_name}")

		instances_raw = data.get("instances")
		if instances_raw is None:
			instances = [
				{"id": name, "unit": name, "center": [0.0, 0.0], "heading_deg": 0.0, "scale": 1.0}
				for name in unit_defs.keys()
			]
		else:
			if not isinstance(instances_raw, list) or not instances_raw:
				raise ValueError("instances must be a non-empty list when provided")
			instances = instances_raw

		all_xy: list[np.ndarray] = []
		all_edges: list[tuple[int, int]] = []
		idx_offset = 0
		for idx, inst in enumerate(instances):
			if not isinstance(inst, dict):
				raise ValueError(f"instances[{idx}] must be a mapping/object")
			unit_name = str(inst.get("unit", "")).strip()
			if not unit_name:
				raise ValueError(f"instances[{idx}] missing required 'unit'")
			if unit_name not in unit_defs:
				raise ValueError(f"instances[{idx}] references unknown unit '{unit_name}'")

			center_raw = inst.get("center", [0.0, 0.0])
			center = np.asarray(center_raw, dtype=float)
			if center.shape != (2,):
				raise ValueError(f"instances[{idx}].center must be [x, y]")
			center = center * unit_scale

			heading_deg = float(inst.get("heading_deg", 0.0))
			theta = np.deg2rad(heading_deg)
			scale = float(inst.get("scale", 1.0))
			if scale <= 0.0:
				raise ValueError(f"instances[{idx}].scale must be positive")

			local_xy, local_edges = unit_defs[unit_name]
			rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
			world_xy = (local_xy * scale) @ rot.T + center

			all_xy.append(world_xy)
			all_edges.extend((i + idx_offset, j + idx_offset) for i, j in local_edges)
			idx_offset += int(world_xy.shape[0])

		merged_xy = np.vstack(all_xy)
		return WhiskerArrayGeometry.from_layout(
			merged_xy,
			ellipse_major=ellipse_major,
			ellipse_minor=ellipse_minor,
			mesh_edges=all_edges,
		)

	layout_raw = data.get("layout_xy")
	if layout_raw is None:
		rows = data.get("rows")
		cols = data.get("cols")
		spacing = data.get("spacing")
		if rows is None or cols is None or spacing is None:
			raise ValueError(
				"array config must define one of: unit_arrays (+ optional instances), layout_xy, or rows/cols/spacing"
			)
		return WhiskerArrayGeometry.regular_grid(rows=int(rows), cols=int(cols), spacing=float(spacing) * unit_scale)

	layout_xy = _parse_layout_xy(layout_raw)
	mesh_edges_raw = data.get("mesh_edges")
	if mesh_edges_raw is not None:
		mesh_edges = _parse_mesh_edges(mesh_edges_raw, n_pts=int(layout_xy.shape[0]))
		return WhiskerArrayGeometry.from_layout(
			layout_xy,
			ellipse_major=ellipse_major,
			ellipse_minor=ellipse_minor,
			mesh_edges=mesh_edges,
		)

	mesh_knn_k = int(data.get("mesh_knn_k", 2))
	return WhiskerArrayGeometry.from_layout(
		layout_xy,
		ellipse_major=ellipse_major,
		ellipse_minor=ellipse_minor,
		mesh_knn_k=mesh_knn_k,
	)


@dataclass
class ArrayState:
	"""Rigid-body state of the whisker array center and orientation."""

	position: Vec2
	velocity: Vec2
	heading: float


@dataclass
class SimulatorConfig:
	"""All physical and numerical parameters for the simulator."""

	max_speed: float = 0.4
	max_accel: float = 0.4
	deflection_gain: float = 0.03
	max_deflection: float = 0.03
	dt: float = 0.1
	flow_dt: float = 0.04
	flow_time_delay: float = 0.0
	finish_x: float = 2.0  # meters (default 2000 mm)


@dataclass
class FlowFieldSequence:
	"""Time-indexed flow field frames with bilinear point sampling.

	``coord_scale`` converts stored grid coordinates to simulation world units.
	Default 1e-3 converts mm (flow data) → m (simulator SI).
	Velocity values need no conversion because mm/ms == m/s numerically.
	"""

	frames: list[FlowFrame2D]
	coord_scale: float = 1e-3  # flow grid mm → simulator m
	decay_half_life: float = 0.0  # 0 = no decay; >0 = half-life in seconds for flow past last real frame

	@staticmethod
	def from_last_frame(data_path: str | Path, coord_scale: float = 1e-3) -> "FlowFieldSequence":
		"""Load only the last Q.*.plt file for a static flow field (fast startup)."""
		root = Path(data_path)
		files = sorted(root.glob("Q.*.plt"))
		if not files:
			raise ValueError(f"No Q.*.plt files found in {root}")
		fp = files[-1]
		parsed = read_plt75(fp)
		zone = parsed.zones[0]
		xc = _extract_2d(zone, "xc")
		yc = _extract_2d(zone, "yc")
		u = _extract_2d(zone, "u")
		v = _extract_2d(zone, "v")
		x = np.asarray(xc[:, 0], dtype=float)
		y = np.asarray(yc[0, :], dtype=float)
		w = _compute_vorticity(x, y, u, v)
		frame = FlowFrame2D(name=fp.name, x=x, y=y, u=u, v=v, vorticity=w)
		print(f"[flow] loaded single frame: {fp.name}")
		return FlowFieldSequence(frames=[frame], coord_scale=coord_scale)

	@staticmethod
	def from_path(
		data_path: str | Path,
		coord_scale: float = 1e-3,
		*,
		flow_dt: float = 0.04,
		decay_half_life: float = 0.0,
	) -> "FlowFieldSequence":
		if flow_dt <= 0.0:
			raise ValueError("flow_dt must be positive")
		if decay_half_life < 0.0:
			raise ValueError("decay_half_life must be >= 0")

		frames = load_flow_frames(data_path)
		if not frames:
			raise ValueError("No flow frames found")

		return FlowFieldSequence(frames=frames, coord_scale=coord_scale, decay_half_life=decay_half_life)

	def _frame_index(self, t: float, flow_dt: float) -> int:
		n = len(self.frames)
		idx = int(np.floor(t / flow_dt))
		return min(max(idx, 0), n - 1)

	def _frame_blend_indices(self, t: float, flow_dt: float) -> tuple[int, int, float]:
		n = len(self.frames)
		if n <= 1:
			return 0, 0, 0.0
		if flow_dt <= 0.0:
			raise ValueError("flow_dt must be positive")

		phase = t / flow_dt
		if phase <= 0.0:
			return 0, 0, 0.0
		if phase >= (n - 1):
			return n - 1, n - 1, 0.0

		i0 = int(np.floor(phase))
		i1 = i0 + 1
		alpha = float(phase - i0)
		return i0, i1, alpha

	def _decay_factor(self, t: float, flow_dt: float) -> float:
		n = len(self.frames)
		if self.decay_half_life <= 0.0 or n <= 0:
			return 1.0
		t_end = (n - 1) * flow_dt
		if t <= t_end:
			return 1.0
		return float(0.5 ** (float(t - t_end) / self.decay_half_life))

	def frame_signature(self, t: float, flow_dt: float) -> tuple[int, float]:
		"""Return (frame_idx, 1.0) normally, or (n-1, -1.0) as stable sentinel during decay."""
		if self._decay_factor(t=t, flow_dt=flow_dt) < 1.0:
			return len(self.frames) - 1, -1.0
		return self._frame_index(t=t, flow_dt=flow_dt), 1.0

	def _sample_grid_uv(self, t: float, flow_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		n = len(self.frames)
		if self.decay_half_life > 0.0 and n > 0:
			t_end = (n - 1) * flow_dt
			if t > t_end:
				decay_factor = float(0.5 ** (float(t - t_end) / self.decay_half_life))
				f = self.frames[-1]
				u = f.u * decay_factor
				v = f.v * decay_factor
				w = _compute_vorticity(f.x, f.y, u, v)
				return u, v, w
		i0, i1, alpha = self._frame_blend_indices(t=t, flow_dt=flow_dt)
		f0 = self.frames[i0]
		f1 = self.frames[i1]
		if i0 == i1 or alpha <= 0.0:
			return f0.u, f0.v, f0.vorticity
		u = (1.0 - alpha) * f0.u + alpha * f1.u
		v = (1.0 - alpha) * f0.v + alpha * f1.v
		w = _compute_vorticity(f0.x, f0.y, u, v)
		return u, v, w

	def _sample_grid_uv_with_profiler(
		self, t: float, flow_dt: float, profiler: StepProfiler | None
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Sample flow grid with optional step-by-step profiling of vorticity computation."""
		n = len(self.frames)
		if self.decay_half_life > 0.0 and n > 0:
			t_end = (n - 1) * flow_dt
			if t > t_end:
				decay_factor = float(0.5 ** (float(t - t_end) / self.decay_half_life))
				f = self.frames[-1]
				u = f.u * decay_factor
				v = f.v * decay_factor
				vort_start = perf_counter()
				w = _compute_vorticity(f.x, f.y, u, v)
				if profiler is not None:
					profiler.record("flow_vorticity_compute", perf_counter() - vort_start)
				return u, v, w

		interp_start = perf_counter()
		i0, i1, alpha = self._frame_blend_indices(t=t, flow_dt=flow_dt)
		f0 = self.frames[i0]
		f1 = self.frames[i1]
		if i0 == i1 or alpha <= 0.0:
			if profiler is not None:
				profiler.record("flow_grid_interp", perf_counter() - interp_start)
			return f0.u, f0.v, f0.vorticity
		u = (1.0 - alpha) * f0.u + alpha * f1.u
		v = (1.0 - alpha) * f0.v + alpha * f1.v
		if profiler is not None:
			profiler.record("flow_grid_interp", perf_counter() - interp_start)

		vort_start = perf_counter()
		w = _compute_vorticity(f0.x, f0.y, u, v)
		if profiler is not None:
			profiler.record("flow_vorticity_compute", perf_counter() - vort_start)
		return u, v, w

	def sample_frame(self, t: float, flow_dt: float) -> FlowFrame2D:
		"""Return the current raw frame (clamped to last when data ends; no decay applied)."""
		idx, _ = self.frame_signature(t=t, flow_dt=flow_dt)
		return self.frames[idx]

	def _fractional_indices_from_world(
		self,
		world_xy: np.ndarray,
		x: np.ndarray,
		y: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""Convert world coordinates (m) to fractional grid indices and bilinear weights."""
		inv = 1.0 / self.coord_scale
		xi = np.interp(world_xy[:, 0] * inv, x, np.arange(x.size))
		yi = np.interp(world_xy[:, 1] * inv, y, np.arange(y.size))

		x0 = np.clip(np.floor(xi).astype(int), 0, x.size - 1)
		y0 = np.clip(np.floor(yi).astype(int), 0, y.size - 1)
		x1 = np.clip(x0 + 1, 0, x.size - 1)
		y1 = np.clip(y0 + 1, 0, y.size - 1)

		sx = xi - x0
		sy = yi - y0
		return xi, yi, x0, x1, y0, y1, sx, sy

	def sample_velocity_from_frame(self, world_xy: np.ndarray, frame: FlowFrame2D) -> np.ndarray:
		x = frame.x
		y = frame.y
		u_grid = frame.u
		v_grid = frame.v

		_, _, x0, x1, y0, y1, sx, sy = self._fractional_indices_from_world(world_xy, x=x, y=y)

		u00 = u_grid[x0, y0]
		u10 = u_grid[x1, y0]
		u01 = u_grid[x0, y1]
		u11 = u_grid[x1, y1]

		v00 = v_grid[x0, y0]
		v10 = v_grid[x1, y0]
		v01 = v_grid[x0, y1]
		v11 = v_grid[x1, y1]

		u = (1 - sx) * (1 - sy) * u00 + sx * (1 - sy) * u10 + (1 - sx) * sy * u01 + sx * sy * u11
		v = (1 - sx) * (1 - sy) * v00 + sx * (1 - sy) * v10 + (1 - sx) * sy * v01 + sx * sy * v11

		return np.column_stack([u, v])

	def _sample_velocity_from_indices(
		self,
		frame: FlowFrame2D,
		x0: np.ndarray,
		x1: np.ndarray,
		y0: np.ndarray,
		y1: np.ndarray,
		sx: np.ndarray,
		sy: np.ndarray,
	) -> np.ndarray:
		u_grid = frame.u
		v_grid = frame.v

		u00 = u_grid[x0, y0]
		u10 = u_grid[x1, y0]
		u01 = u_grid[x0, y1]
		u11 = u_grid[x1, y1]

		v00 = v_grid[x0, y0]
		v10 = v_grid[x1, y0]
		v01 = v_grid[x0, y1]
		v11 = v_grid[x1, y1]

		u = (1 - sx) * (1 - sy) * u00 + sx * (1 - sy) * u10 + (1 - sx) * sy * u01 + sx * sy * u11
		v = (1 - sx) * (1 - sy) * v00 + sx * (1 - sy) * v10 + (1 - sx) * sy * v01 + sx * sy * v11
		return np.column_stack([u, v])

	def sample_velocity(self, world_xy: np.ndarray, t: float, flow_dt: float) -> np.ndarray:
		# Temporal interpolation applied only to sampled whisker points (not the full grid).
		x = self.frames[0].x
		y = self.frames[0].y
		_, _, x0, x1, y0, y1, sx, sy = self._fractional_indices_from_world(world_xy, x=x, y=y)

		decay = self._decay_factor(t=t, flow_dt=flow_dt)
		if decay < 1.0:
			v_last = self._sample_velocity_from_indices(self.frames[-1], x0, x1, y0, y1, sx, sy)
			return v_last * decay

		i0, i1, alpha = self._frame_blend_indices(t=t, flow_dt=flow_dt)
		f0 = self.frames[i0]
		if i0 == i1 or alpha <= 0.0:
			return self._sample_velocity_from_indices(f0, x0, x1, y0, y1, sx, sy)

		f1 = self.frames[i1]
		v0 = self._sample_velocity_from_indices(f0, x0, x1, y0, y1, sx, sy)
		v1 = self._sample_velocity_from_indices(f1, x0, x1, y0, y1, sx, sy)
		return (1.0 - alpha) * v0 + alpha * v1

	def sample_background(self, t: float, flow_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return (u, v, vorticity) grids at time t (clamped; no decay applied)."""
		frame = self.sample_frame(t=t, flow_dt=flow_dt)
		return frame.u, frame.v, frame.vorticity

	def extent(self) -> tuple[float, float, float, float]:
		"""Return (xmin, xmax, ymin, ymax) in simulator units (m)."""
		f0 = self.frames[0]
		s = self.coord_scale
		return float(f0.x[0]) * s, float(f0.x[-1]) * s, float(f0.y[0]) * s, float(f0.y[-1]) * s


@dataclass
class FlowCase:
	"""One selectable simulation case: flow path + optional matching trajectory."""

	label: str
	data_path: Path
	traj_xy: np.ndarray | None


def _rotation_matrix(theta: float) -> np.ndarray:
	c = float(np.cos(theta))
	s = float(np.sin(theta))
	return np.array([[c, -s], [s, c]], dtype=float)


@dataclass
class WhiskerArraySimulator:
	"""Simulation core: kinematics + whisker sensing and deflection."""

	geometry: WhiskerArrayGeometry
	flow: FlowFieldSequence
	config: SimulatorConfig
	state: ArrayState
	time: float = 0.0
	current_flow_frame: FlowFrame2D | None = None
	current_flow_signature: tuple[int, float] | None = None
	profiler: StepProfiler | None = None

	start_x: float = 0.05  # spawn x in simulator units (m); 50 mm default

	def _random_start_state(self) -> ArrayState:
		ext = self.flow.extent()
		y = float(np.random.uniform(ext[2], ext[3]))
		return ArrayState(
			position=np.array([self.start_x, y], dtype=float),
			velocity=np.zeros(2, dtype=float),
			heading=0.0,
		)

	def reset(self) -> None:
		"""Restart simulation: random y at start_x, zero velocity."""
		self.state = self._random_start_state()
		self.time = 0.0
		self.current_flow_frame = None
		self.current_flow_signature = None

	def _flow_sample_time(self) -> float:
		return self.time + self.config.flow_time_delay

	def _record_profile(self, label: str, start_time: float) -> None:
		if self.profiler is not None:
			self.profiler.record(label, perf_counter() - start_time)

	def _update_current_flow_frame(self) -> FlowFrame2D:
		started = perf_counter()
		t_sample = self._flow_sample_time()
		signature = self.flow.frame_signature(t=t_sample, flow_dt=self.config.flow_dt)
		if self.current_flow_frame is not None and self.current_flow_signature == signature:
			self._record_profile("flow_frame_update", started)
			return self.current_flow_frame

		self.current_flow_frame = self.flow.sample_frame(
			t=t_sample,
			flow_dt=self.config.flow_dt,
		)
		self.current_flow_signature = signature
		self._record_profile("flow_frame_update", started)
		return self.current_flow_frame

	def _get_current_flow_frame(self) -> FlowFrame2D:
		if self.current_flow_frame is None:
			return self._update_current_flow_frame()
		return self.current_flow_frame

	def world_whisker_centers(self) -> np.ndarray:
		rot = _rotation_matrix(self.state.heading)
		return self.state.position + self.geometry.layout_xy @ rot.T

	def _apply_accel_limit(self, target_vel: Vec2) -> None:
		dv = target_vel - self.state.velocity
		max_dv = self.config.max_accel * self.config.dt
		dv = np.clip(dv, -max_dv, max_dv)
		self.state.velocity = self.state.velocity + dv

	def _sample_deflected_centers(self, orig_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		sample_started = perf_counter()
		flow_vel = self.flow.sample_velocity(
			orig_centers,
			t=self._flow_sample_time(),
			flow_dt=self.config.flow_dt,
		)
		self._record_profile("whisker_sample", sample_started)

		deflect_started = perf_counter()
		deflection = flow_vel * self.config.deflection_gain
		def_mag = np.linalg.norm(deflection, axis=1)
		mask = def_mag > self.config.max_deflection
		if np.any(mask):
			deflection[mask] *= (self.config.max_deflection / def_mag[mask])[:, None]

		deflected_centers = orig_centers + deflection
		self._record_profile("deflection_compute", deflect_started)
		return deflected_centers, flow_vel

	def step(self, command: ControllerCommand) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		step_started = perf_counter()
		kinematics_started = perf_counter()
		target_speed = np.clip(command.throttle, 0.0, 1.0) * self.config.max_speed
		target_vel = command.direction * target_speed
		self._apply_accel_limit(target_vel)

		speed = float(np.linalg.norm(self.state.velocity))
		if np.linalg.norm(command.direction) > 1e-9:
			self.state.heading = float(np.arctan2(command.direction[1], command.direction[0]))
		elif speed > 1e-9:
			self.state.heading = float(np.arctan2(self.state.velocity[1], self.state.velocity[0]))

		self.state.position = self.state.position + self.state.velocity * self.config.dt
		self.time += self.config.dt
		self._record_profile("kinematics_update", kinematics_started)
		self._update_current_flow_frame()

		center_started = perf_counter()
		orig_centers = self.world_whisker_centers()
		self._record_profile("whisker_centers", center_started)
		deflected_centers, flow_vel = self._sample_deflected_centers(orig_centers)
		self._record_profile("simulation_step_total", step_started)
		return orig_centers, deflected_centers, flow_vel

	def step_with_velocity(self, velocity_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Advance exactly one step using a fixed world-frame velocity command."""
		step_started = perf_counter()
		kinematics_started = perf_counter()
		vel = np.asarray(velocity_xy, dtype=float)
		if vel.shape != (2,):
			raise ValueError("velocity_xy must be a 2-vector")

		speed = float(np.linalg.norm(vel))
		self.state.velocity = vel
		if speed > 1e-9:
			self.state.heading = float(np.arctan2(vel[1], vel[0]))

		self.state.position = self.state.position + self.state.velocity * self.config.dt
		self.time += self.config.dt
		self._record_profile("kinematics_update", kinematics_started)
		self._update_current_flow_frame()

		center_started = perf_counter()
		orig_centers = self.world_whisker_centers()
		self._record_profile("whisker_centers", center_started)
		deflected_centers, flow_vel = self._sample_deflected_centers(orig_centers)
		self._record_profile("simulation_step_total", step_started)
		return orig_centers, deflected_centers, flow_vel


def _save_headless_animation(
	sim: WhiskerArraySimulator,
	sampled_frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]],
	output_path: str | Path,
	fps: float,
	object_traj_xy: np.ndarray | None = None,
	save_dpi: int = 90,
) -> None:
	"""Save a compact animation of the headless run.

	Each sampled frame tuple is ``(t, orig_centers, deflected_centers, array_center, heading_rad)``.
	"""
	if fps <= 0.0:
		raise ValueError("fps must be positive")
	if not sampled_frames:
		raise ValueError("No sampled frames available for export")

	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)

	ext = output.suffix.lower()
	if ext not in {".mp4", ".m4v", ".mov", ".gif"}:
		output = output.with_suffix(".mp4")
		ext = ".mp4"

	fig, ax = plt.subplots(figsize=(6.4, 4.2))
	extent = sim.flow.extent()
	ax.set_xlim(extent[0], extent[1])
	ax.set_ylim(extent[2], extent[3])
	ax.set_aspect("equal")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")

	f0 = sim.flow.frames[0]
	vs = 8
	qstep = 24
	s = sim.flow.coord_scale
	x_img = f0.x[::vs] * s
	y_img = f0.y[::vs] * s
	x_q = f0.x[::qstep] * s
	y_q = f0.y[::qstep] * s
	bg = ax.imshow(
		f0.vorticity[::vs, ::vs].T,
		origin="lower",
		extent=(float(x_img[0]), float(x_img[-1]), float(y_img[0]), float(y_img[-1])),
		cmap="RdBu_r",
		vmin=-0.1,
		vmax=0.1,
		interpolation="nearest",
		alpha=0.5,
	)
	flow_q = ax.quiver(
		x_q,
		y_q,
		f0.u[::qstep, ::qstep].T,
		f0.v[::qstep, ::qstep].T,
		color="0.25",
		angles="xy",
		scale_units="xy",
		scale=6.0,
		width=0.0015,
		alpha=0.55,
	)
	finish_line = ax.axvline(sim.config.finish_x, color="tab:red", lw=1.1, ls="--", alpha=0.9)
	(center_dot,) = ax.plot([], [], marker="o", ms=3.0, color="tab:green", lw=0.0)
	orig_mesh_lines = _create_mesh_lines(
		ax,
		sim.geometry.mesh_edges,
		color="tab:grey",
		lw=1.5,
		alpha=0.85,
		zorder=2,
	)
	orig_whiskers: list[Ellipse] = []
	def_whiskers: list[Ellipse] = []
	for _ in range(sim.geometry.layout_xy.shape[0]):
		e_orig = Ellipse((0.0, 0.0), width=sim.geometry.ellipse_major, height=sim.geometry.ellipse_minor)
		e_orig.set_edgecolor("tab:grey")
		e_orig.set_facecolor("none")
		e_orig.set_linewidth(1.2)
		e_orig.set_zorder(2)
		ax.add_patch(e_orig)
		orig_whiskers.append(e_orig)

		e_def = Ellipse((0.0, 0.0), width=sim.geometry.ellipse_major, height=sim.geometry.ellipse_minor)
		e_def.set_edgecolor("tab:orange")
		e_def.set_facecolor("none")
		e_def.set_linewidth(1.0)
		e_def.set_zorder(4)
		e_def.set_visible(False)
		ax.add_patch(e_def)
		def_whiskers.append(e_def)

	init_orig = sampled_frames[0][1]
	_zeros = np.zeros(len(init_orig))
	deflection_quiver = ax.quiver(
		init_orig[:, 0],
		init_orig[:, 1],
		_zeros,
		_zeros,
		color="tab:red",
		angles="xy",
		scale_units="xy",
		scale=1.0,
		width=0.002,
		headwidth=1.4,
		headlength=1.6,
		headaxislength=1.6,
		zorder=3,
	)

	if object_traj_xy is not None and object_traj_xy.size > 0:
		ax.plot(
			object_traj_xy[:, 0],
			object_traj_xy[:, 1],
			color="tab:red",
			lw=1.8,
			alpha=0.7,
			zorder=2.5,
			label="object trajectory",
		)

	center_hist = np.vstack([frame[3] for frame in sampled_frames])
	whisker_hist = np.stack([frame[1] for frame in sampled_frames], axis=0)
	(center_traj_line,) = ax.plot([], [], color="tab:green", lw=1.6, alpha=0.8, zorder=2.6, label="array center")
	whisker_traj_lines: list[Line2D] = []
	for whisker_idx in range(whisker_hist.shape[1]):
		(ln,) = ax.plot([], [], color="tab:purple", lw=0.9, alpha=0.25, zorder=2.55)
		whisker_traj_lines.append(ln)

	if object_traj_xy is not None and object_traj_xy.size > 0:
		ax.legend(loc="upper right", fontsize=8)
	title = ax.set_title("", fontsize=10)

	def _update(frame_idx: int):
		t, orig, deff, center, heading_rad = sampled_frames[frame_idx]
		flow_idx = sim.flow._frame_index(t=t, flow_dt=sim.config.flow_dt)
		f = sim.flow.frames[flow_idx]
		_decay = sim.flow._decay_factor(t, sim.config.flow_dt)
		bg.set_data(f.vorticity[::vs, ::vs].T * _decay)
		flow_q.set_UVC(f.u[::qstep, ::qstep].T * _decay, f.v[::qstep, ::qstep].T * _decay)
		angle_deg = float(np.degrees(heading_rad))
		_update_mesh_lines_common(orig_mesh_lines, sim.geometry.mesh_edges, orig)
		_update_whisker_ellipses_common(orig_whiskers, orig, angle_deg)
		_update_whisker_ellipses_common(def_whiskers, deff, angle_deg)
		delta = deff - orig
		deflection_quiver.set_offsets(orig)
		deflection_quiver.set_UVC(delta[:, 0], delta[:, 1])
		center_dot.set_data([float(center[0])], [float(center[1])])
		center_traj_line.set_data(center_hist[: frame_idx + 1, 0], center_hist[: frame_idx + 1, 1])
		for whisker_idx, ln in enumerate(whisker_traj_lines):
			ln.set_data(
				whisker_hist[: frame_idx + 1, whisker_idx, 0],
				whisker_hist[: frame_idx + 1, whisker_idx, 1],
			)
		title.set_text(f"t={t:.1f}s")
		return (
			bg,
			flow_q,
			finish_line,
			*orig_mesh_lines,
			*orig_whiskers,
			*def_whiskers,
			deflection_quiver,
			center_dot,
			center_traj_line,
			*whisker_traj_lines,
			title,
		)

	interval_ms = max(1, int(round(1000.0 / fps)))
	anim = FuncAnimation(
		fig,
		_update,
		frames=len(sampled_frames),
		interval=interval_ms,
		blit=False,
		cache_frame_data=False,
	)

	if ext == ".gif":
		anim.save(str(output), writer="pillow", fps=int(round(fps)), dpi=save_dpi)
	else:
		try:
			writer = FFMpegWriter(
				fps=int(round(fps)),
				codec="libx264",
				bitrate=600,
				extra_args=["-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p"],
			)
			anim.save(str(output), writer=writer, dpi=save_dpi)
		except Exception:
			fallback = output.with_suffix(".gif")
			anim.save(str(fallback), writer="pillow", fps=int(round(fps)), dpi=save_dpi)
			print(f"[headless] ffmpeg unavailable; wrote GIF instead: {fallback}")

	plt.close(fig)
	print(f"[headless] saved animation: {output}")


def _run_headless_mode(
	sim: WhiskerArraySimulator,
	*,
	sample_rate_hz: float = 80.0,
	save_file: str | Path | None = None,
	save_fps: float = 10.0,
	save_dpi: int = 90,
	start_xy_m: tuple[float, float] = (0.150, 0.500),
	constant_speed_mps: float = 0.2,
	object_traj_xy: np.ndarray | None = None,
) -> None:
	"""Deterministic, non-interactive simulation runner for testing and export."""
	if sample_rate_hz <= 0.0:
		raise ValueError("sample_rate_hz must be positive")
	if save_fps <= 0.0:
		raise ValueError("save_fps must be positive")
	if constant_speed_mps <= 0.0:
		raise ValueError("constant_speed_mps must be positive")

	sim.config.dt = 1.0 / sample_rate_hz
	sim.time = 0.0
	sim.state.position = np.array([float(start_xy_m[0]), float(start_xy_m[1])], dtype=float)
	sim.state.velocity = np.array([constant_speed_mps, 0.0], dtype=float)
	sim.state.heading = 0.0

	sampled_frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
	next_save_t = 0.0
	save_dt = 1.0 / save_fps

	initial_orig = sim.world_whisker_centers()
	initial_def, _ = sim._sample_deflected_centers(initial_orig)
	if save_file is not None:
		sampled_frames.append((sim.time, initial_orig.copy(), initial_def.copy(), sim.state.position.copy(), float(sim.state.heading)))
		next_save_t = save_dt

	fixed_vel = np.array([constant_speed_mps, 0.0], dtype=float)
	while True:
		orig, deff, _ = sim.step_with_velocity(fixed_vel)
		if save_file is not None and sim.time + 1e-12 >= next_save_t:
			sampled_frames.append((sim.time, orig.copy(), deff.copy(), sim.state.position.copy(), float(sim.state.heading)))
			next_save_t += save_dt

		if float(sim.state.position[0]) >= sim.config.finish_x:
			break

	# Ensure final frame is included in export.
	if save_file is not None:
		if not sampled_frames or sampled_frames[-1][0] < sim.time - 1e-12:
			sampled_frames.append((sim.time, orig.copy(), deff.copy(), sim.state.position.copy(), float(sim.state.heading)))
		_save_headless_animation(
			sim,
			sampled_frames,
			output_path=save_file,
			fps=save_fps,
			object_traj_xy=object_traj_xy,
			save_dpi=save_dpi,
		)

	print(
		f"[headless] finished at t={sim.time:.3f}s, x={sim.state.position[0]:.3f}m, y={sim.state.position[1]:.3f}m, "
		f"dt={sim.config.dt:.5f}s ({sample_rate_hz:.1f} Hz)"
	)
	if sim.profiler is not None:
		sim.profiler.print_report(header="[profile] finish-line summary")
	return


def _load_replay_data(replay_path: str | Path) -> np.ndarray:
	"""Load replay table with 5 columns: t, x, y, x_vel, y_vel (SI units)."""
	fp = Path(replay_path)
	if not fp.is_file():
		raise ValueError(f"replay file not found: {fp}")

	try:
		data = np.loadtxt(fp, delimiter=",")
	except Exception:
		try:
			data = np.loadtxt(fp, delimiter=",", skiprows=1)
		except Exception:
			try:
				data = np.loadtxt(fp)
			except Exception:
				data = np.loadtxt(fp, skiprows=1)

	arr = np.asarray(data, dtype=float)
	if arr.ndim == 1:
		if arr.size < 5:
			raise ValueError("replay file must contain at least 5 columns: t,x,y,x_vel,y_vel")
		arr = arr[None, :]
	if arr.ndim != 2 or arr.shape[1] < 5:
		raise ValueError("replay file must be a 2D table with at least 5 columns: t,x,y,x_vel,y_vel")

	arr = arr[:, :5]
	if arr.shape[0] < 1:
		raise ValueError("replay file contains no rows")
	if not np.all(np.isfinite(arr)):
		raise ValueError("replay file contains NaN/Inf values")
	if np.any(np.diff(arr[:, 0]) < 0.0):
		raise ValueError("replay time column must be nondecreasing")
	return arr


def _save_replay_data(replay_txyvv: np.ndarray, output_path: str | Path) -> None:
	"""Save replay table as CSV with columns: t, x, y, x_vel, y_vel."""
	arr = np.asarray(replay_txyvv, dtype=float)
	if arr.ndim != 2 or arr.shape[1] < 5:
		raise ValueError("replay array must be shape (N,5+) with t,x,y,x_vel,y_vel")
	if arr.shape[0] < 1:
		print("[replay] no replay rows recorded; skipping save")
		return

	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	np.savetxt(
		out,
		arr[:, :5],
		delimiter=",",
		header="t,x,y,x_vel,y_vel",
		comments="",
		fmt="%.9f",
	)
	print(f"[replay] saved: {out} ({arr.shape[0]} rows)")


def _apply_replay_row_to_sim(sim: WhiskerArraySimulator, row_txyvv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Apply one replay row to simulator state and sample whisker deflection."""
	t = float(row_txyvv[0])
	x = float(row_txyvv[1])
	y = float(row_txyvv[2])
	vx = float(row_txyvv[3])
	vy = float(row_txyvv[4])

	sim.time = t
	sim.state.position = np.array([x, y], dtype=float)
	sim.state.velocity = np.array([vx, vy], dtype=float)
	v_mag = float(np.hypot(vx, vy))
	if v_mag > 1e-9:
		sim.state.heading = float(np.arctan2(vy, vx))
	sim.current_flow_frame = None
	sim.current_flow_signature = None
	sim._update_current_flow_frame()
	orig = sim.world_whisker_centers()
	deff, _ = sim._sample_deflected_centers(orig)
	return orig, deff


def _run_headless_replay_mode(
	sim: WhiskerArraySimulator,
	*,
	replay_txyvv: np.ndarray,
	save_file: str | Path | None = None,
	save_fps: float = 10.0,
	save_dpi: int = 90,
	object_traj_xy: np.ndarray | None = None,
) -> None:
	"""Headless non-interactive runner driven by replay rows until replay end-time."""
	if replay_txyvv.shape[0] < 1:
		raise ValueError("replay_txyvv must contain at least one row")
	if save_fps <= 0.0:
		raise ValueError("save_fps must be positive")

	sampled_frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
	next_save_t = float(replay_txyvv[0, 0])
	save_dt = 1.0 / save_fps
	last_orig: np.ndarray | None = None
	last_deff: np.ndarray | None = None
	last_heading: float | None = None

	for row in replay_txyvv:
		orig, deff = _apply_replay_row_to_sim(sim, row)
		last_orig = orig
		last_deff = deff
		last_heading = float(sim.state.heading)
		t = float(sim.time)
		if save_file is not None and t + 1e-12 >= next_save_t:
			sampled_frames.append((t, orig.copy(), deff.copy(), sim.state.position.copy(), float(sim.state.heading)))
			next_save_t += save_dt

	if save_file is not None and last_orig is not None and last_deff is not None and last_heading is not None:
		if not sampled_frames or sampled_frames[-1][0] < sim.time - 1e-12:
			sampled_frames.append((sim.time, last_orig.copy(), last_deff.copy(), sim.state.position.copy(), last_heading))
		_save_headless_animation(
			sim,
			sampled_frames,
			output_path=save_file,
			fps=save_fps,
			object_traj_xy=object_traj_xy,
			save_dpi=save_dpi,
		)

	print(
		f"[replay] finished at t={sim.time:.3f}s, x={sim.state.position[0]:.3f}m, y={sim.state.position[1]:.3f}m, "
		f"rows={replay_txyvv.shape[0]}"
	)


def _load_matching_trajectory(
	data_path: str | Path,
	traj_path: str | Path,
	coord_scale: float = 1e-3,
) -> np.ndarray | None:
	"""Load the single trajectory file whose numeric index matches *data_path*.

	For example ``data_path='…/path38'`` looks for ``path_xy38.dat`` inside *traj_path*.
	Returns an (N, 2) float array in world units (m), or *None* if not found.
	"""
	m = re.search(r"(\d+)", Path(data_path).name)
	if m is None:
		return None
	idx = m.group(1)
	fp = Path(traj_path) / f"path_xy{idx}.dat"
	if not fp.is_file():
		return None
	data = np.loadtxt(fp, delimiter=",")
	if data.ndim == 2 and data.shape[1] >= 2:
		return data[:, :2] * coord_scale
	return None


def _numeric_suffix(name: str) -> int | None:
	"""Return trailing integer from names like 'path38' or None if missing."""
	m = re.fullmatch(r"[A-Za-z_]+(\d+)", name)
	if m is None:
		return None
	return int(m.group(1))


def _load_excluded_path_ids(exclude_file: Path | None) -> set[int]:
	"""Load excluded path indices from a text file (one integer per line)."""
	if exclude_file is None or (not exclude_file.is_file()):
		return set()

	excluded: set[int] = set()
	for raw in exclude_file.read_text(encoding="utf-8").splitlines():
		line = raw.strip()
		if not line or line.startswith("#"):
			continue
		try:
			excluded.add(int(line))
		except ValueError:
			print(f"[flow] ignoring invalid exclude entry: {line!r}")
	return excluded


def _discover_numbered_flow_paths(root: Path, excluded_ids: set[int] | None = None) -> list[Path]:
	"""Find and numerically sort child folders matching path<number> with Q.*.plt files."""
	if not root.is_dir():
		return []
	excluded_ids = excluded_ids if excluded_ids is not None else set()
	items: list[tuple[int, Path]] = []
	for child in root.iterdir():
		if not child.is_dir():
			continue
		if not child.name.lower().startswith("path"):
			continue
		idx = _numeric_suffix(child.name)
		if idx is None:
			continue
		if idx in excluded_ids:
			continue
		if any(child.glob("Q.*.plt")):
			items.append((idx, child))
	items.sort(key=lambda it: it[0])
	return [p for _, p in items]


def _choose_flow_index(cases: list[FlowCase], current_idx: int) -> int | None:
	"""Open a small listbox dialog and return selected case index, or None on cancel."""
	if not cases:
		return None
	try:
		import tkinter as tk
	except Exception as exc:
		print(f"[select] tkinter unavailable ({exc})")
		return None

	selection: dict[str, int | None] = {"idx": None}
	root = tk.Tk()
	root.title("Select flow path")
	root.resizable(False, False)

	tk.Label(root, text="Choose one flow path and press Select:").pack(padx=10, pady=(10, 4), anchor="w")
	height = min(24, max(8, len(cases)))
	listbox = tk.Listbox(root, width=24, height=height, exportselection=False)
	for case in cases:
		listbox.insert(tk.END, case.label)
	listbox.pack(padx=10, pady=4, fill="both", expand=True)

	cur = int(np.clip(current_idx, 0, len(cases) - 1))
	listbox.selection_set(cur)
	listbox.see(cur)

	def _confirm(_event=None):
		sel = listbox.curselection()
		if sel:
			selection["idx"] = int(sel[0])
		root.destroy()

	def _cancel(_event=None):
		root.destroy()

	btn_row = tk.Frame(root)
	btn_row.pack(padx=10, pady=(6, 10), fill="x")
	tk.Button(btn_row, text="Select", width=10, command=_confirm).pack(side="left")
	tk.Button(btn_row, text="Cancel", width=10, command=_cancel).pack(side="right")

	listbox.bind("<Double-Button-1>", _confirm)
	root.bind("<Return>", _confirm)
	root.bind("<Escape>", _cancel)
	root.protocol("WM_DELETE_WINDOW", _cancel)
	root.mainloop()
	return selection["idx"]


def _build_flow_cases(
	data_path: str | Path | None,
	data_root: str | Path | None,
	traj_path: str | Path | None,
	exclude_list: str | Path | None,
) -> tuple[list[FlowCase], int]:
	"""Build selectable flow cases and return (cases, initial_index)."""
	selected_path = Path(data_path) if data_path is not None else None

	# Resolve trajectory directory first: explicit arg wins, otherwise sibling xy_data.
	traj_dir: Path | None = None
	if traj_path is not None:
		cand = Path(traj_path)
		if cand.is_dir():
			traj_dir = cand

	if selected_path is not None:
		if not selected_path.is_dir():
			raise ValueError(f"data_path does not exist: {selected_path}")
		parent = selected_path.parent
		exclude_file = Path(exclude_list) if exclude_list is not None else (parent / "exclude_list.txt")
		excluded_ids = _load_excluded_path_ids(exclude_file)
		numbered = _discover_numbered_flow_paths(parent, excluded_ids=excluded_ids)
		if traj_dir is None:
			cand = parent / "xy_data"
			if cand.is_dir():
				traj_dir = cand

		if numbered and selected_path in numbered:
			paths = numbered
			initial_idx = paths.index(selected_path)
		else:
			paths = [selected_path]
			initial_idx = 0
	else:
		root = Path(data_root) if data_root is not None else Path("flowdata") / "env1"
		exclude_file = Path(exclude_list) if exclude_list is not None else (root / "exclude_list.txt")
		excluded_ids = _load_excluded_path_ids(exclude_file)
		paths = _discover_numbered_flow_paths(root, excluded_ids=excluded_ids)
		if not paths:
			raise ValueError(f"No eligible path<number> directories with Q.*.plt found in {root}")
		if traj_dir is None:
			cand = root / "xy_data"
			if cand.is_dir():
				traj_dir = cand
		initial_idx = int(np.random.randint(0, len(paths)))

	cases: list[FlowCase] = []
	for path in paths:
		traj_xy = _load_matching_trajectory(path, traj_dir) if traj_dir is not None else None
		cases.append(FlowCase(label=path.name, data_path=path, traj_xy=traj_xy))

	return cases, initial_idx


def _create_mesh_lines(
	ax: Axes,
	mesh_edges: list[tuple[int, int]],
	*,
	color: str,
	lw: float,
	alpha: float,
	zorder: int = 2,
) -> list[Line2D]:
	lines: list[Line2D] = []
	for _ in mesh_edges:
		ln = Line2D([], [], color=color, lw=lw, alpha=alpha, zorder=zorder)
		ax.add_line(ln)
		lines.append(ln)
	return lines


def _update_mesh_lines_common(lines: list[Line2D], mesh_edges: list[tuple[int, int]], pts: np.ndarray) -> None:
	for ln, (i, j) in zip(lines, mesh_edges):
		ln.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])


def _update_whisker_ellipses_common(ellipses: list[Ellipse], centers: np.ndarray, angle_deg: float) -> None:
	for e, p in zip(ellipses, centers):
		e.center = (float(p[0]), float(p[1]))
		e.angle = angle_deg


@dataclass
class SimulationRenderer:
	"""Matplotlib renderer for whisker array state and flow backdrop."""

	sim: WhiskerArraySimulator
	show_flow_quiver: bool = True
	flow_quiver_step: int = 24
	vort_step: int = 8  # subsample vorticity imshow for faster rendering
	flow_cases: list[FlowCase] = field(default_factory=list)
	current_case_idx: int = 0
	use_dynamic_flow: bool = False
	flow_decay_half_life: float = 0.0
	array_trajectory_txy: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float), init=False)
	array_replay_txyvv: np.ndarray = field(default_factory=lambda: np.empty((0, 5), dtype=float), init=False)
	last_flow_signature: tuple[int, float] | None = field(default=None, init=False)
	_last_decay_visual_t: float = field(default=-1e18, init=False)

	def _build_mesh_lines(self, color: str, lw: float, alpha: float, zorder: int = 2) -> list[Line2D]:
		return _create_mesh_lines(
			self.ax,
			self.sim.geometry.mesh_edges,
			color=color,
			lw=lw,
			alpha=alpha,
			zorder=zorder,
		)

	def _update_mesh_lines(self, lines: list[Line2D], pts: np.ndarray) -> None:
		_update_mesh_lines_common(lines, self.sim.geometry.mesh_edges, pts)

	def setup(self) -> None:
		ext = self.sim.flow.extent()
		self.fig, self.ax = plt.subplots(figsize=(10.5, 7.0))
		manager = getattr(self.fig.canvas, "manager", None)
		handler_id = getattr(manager, "key_press_handler_id", None)
		if handler_id is not None:
			self.fig.canvas.mpl_disconnect(handler_id)
		self.fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
		self.ax.set_xlim(ext[0], ext[1])
		self.ax.set_ylim(ext[2], ext[3])
		self.ax.set_aspect("equal")
		self.ax.set_axis_off()
		self.finish_line = self.ax.axvline(
			self.sim.config.finish_x,
			color="tab:red",
			lw=1.2,
			ls="--",
			alpha=0.9,
			zorder=1.5,
		)

		f0 = self.sim.flow.frames[0]
		vs = self.vort_step
		s = self.sim.flow.coord_scale
		x_img = f0.x[::vs] * s
		y_img = f0.y[::vs] * s
		self.bg = self.ax.imshow(
			f0.vorticity[::vs, ::vs].T,
			origin="lower",
			extent=(float(x_img[0]), float(x_img[-1]), float(y_img[0]), float(y_img[-1])),
			cmap="RdBu_r",
			vmin=-0.1,
			vmax=0.1,
			interpolation="nearest",
			alpha=0.55,
		)
		self.bg.set_animated(True)

		self.flow_q = None
		if self.show_flow_quiver:
			s = self.sim.flow.coord_scale
			xsub = f0.x[:: self.flow_quiver_step] * s  # mm → m
			ysub = f0.y[:: self.flow_quiver_step] * s
			u = f0.u[:: self.flow_quiver_step, :: self.flow_quiver_step].T
			v = f0.v[:: self.flow_quiver_step, :: self.flow_quiver_step].T
			self.flow_q = self.ax.quiver(
				xsub,
				ysub,
				u,
				v,
				color="0.25",
				angles="xy",
				scale_units="xy",
				scale=6.0,
				width=0.0015,
				alpha=0.55,
			)
			self.flow_q.set_animated(True)

		# Trajectory sits beneath everything else (zorder=1), hidden by default.
		(self.traj_line,) = self.ax.plot([], [], color="tab:red", lw=4, alpha=0.5, zorder=1, visible=False)
		(self.array_center_traj_line,) = self.ax.plot([], [], color="tab:green", lw=1.8, alpha=0.7, zorder=1.1, visible=False)
		if self.flow_cases:
			xy = self.flow_cases[self.current_case_idx].traj_xy
			if xy is not None and xy.size > 0:
				self.traj_line.set_data(xy[:, 0], xy[:, 1])

		# Layer order: original (zorder=2) → arrows (3) → deflected (4)
		self.orig_mesh_lines = self._build_mesh_lines(color="tab:grey", lw=1.5, alpha=0.85, zorder=2)

		n = self.sim.geometry.layout_xy.shape[0]
		self.whisker_traj_lines: list[Line2D] = []
		for _ in range(n):
			(ln_hist,) = self.ax.plot([], [], color="tab:purple", lw=1.0, alpha=0.35, zorder=1.05, visible=False)
			self.whisker_traj_lines.append(ln_hist)

		self.whisker_orig: list[Ellipse] = []
		self.whisker_def: list[Ellipse] = []
		for _ in range(n):
			e_orig = Ellipse((0.0, 0.0), width=self.sim.geometry.ellipse_major, height=self.sim.geometry.ellipse_minor)
			e_orig.set_edgecolor("tab:grey")
			e_orig.set_facecolor("none")
			e_orig.set_linewidth(1.5)
			e_orig.set_zorder(2)
			self.ax.add_patch(e_orig)
			self.whisker_orig.append(e_orig)

		init_centers = self.sim.world_whisker_centers()
		_zeros = np.zeros(len(init_centers))
		self.deflection_quiver = self.ax.quiver(
			init_centers[:, 0], init_centers[:, 1], _zeros, _zeros,
			color="tab:red", angles="xy", scale_units="xy", scale=1.0,width=0.002,
			headwidth=1.4, headlength=1.6, headaxislength=1.6,
			zorder=3, animated=True,
		)

		self.def_mesh_lines = self._build_mesh_lines(color="tab:orange", lw=1.2, alpha=0.5, zorder=4)
		for ln in self.def_mesh_lines:
			ln.set_visible(False)
		for _ in range(n):
			e_def = Ellipse((0.0, 0.0), width=self.sim.geometry.ellipse_major, height=self.sim.geometry.ellipse_minor)
			e_def.set_edgecolor("tab:orange")
			e_def.set_facecolor("none")
			e_def.set_linewidth(1.0)
			e_def.set_zorder(4)
			e_def.set_visible(False)
			self.ax.add_patch(e_def)
			self.whisker_def.append(e_def)
		xmin, xmax, ymin, ymax = ext
		dx = xmax - xmin
		dy = ymax - ymin
		self.text = self.ax.text(
			xmin + 0.5 * dx,
			ymax - 0.015 * dy,
			"",
			ha="center",
			va="top",
			fontsize=16,
			color="red",
			zorder=10,
		)
		self.text.set_animated(True)

		# Mark all per-frame artists as animated so blit skips them in the background.
		for ln in self.orig_mesh_lines + self.def_mesh_lines:
			ln.set_animated(True)
		for e in self.whisker_orig + self.whisker_def:
			e.set_animated(True)

	def _update_flow_background(self, t: float) -> None:
		t_flow = t + self.sim.config.flow_time_delay
		decay = self.sim.flow._decay_factor(t_flow, self.sim.config.flow_dt)
		vs = self.vort_step
		qs = self.flow_quiver_step

		if decay < 1.0:
			# Decay period: rate-limit visual updates to 1 FPS; apply decay to subsampled slices.
			if t - self._last_decay_visual_t < 1.0:
				return
			frame = self.sim.flow.frames[-1]
			self.bg.set_data(frame.vorticity[::vs, ::vs].T * decay)
			if self.flow_q is not None:
				self.flow_q.set_UVC(
					frame.u[::qs, ::qs].T * decay,
					frame.v[::qs, ::qs].T * decay,
				)
			self._last_decay_visual_t = t
			return

		# Normal period: only update when the discrete frame changes.
		if abs(t - self.sim.time) <= 1e-12:
			signature = self.sim.current_flow_signature
			frame = self.sim._get_current_flow_frame()
		else:
			signature = self.sim.flow.frame_signature(t=t_flow, flow_dt=self.sim.config.flow_dt)
			frame = self.sim.flow.sample_frame(t=t_flow, flow_dt=self.sim.config.flow_dt)
		if signature is not None and signature == self.last_flow_signature:
			return
		self.bg.set_data(frame.vorticity[::vs, ::vs].T)
		if self.flow_q is not None:
			self.flow_q.set_UVC(frame.u[::qs, ::qs].T, frame.v[::qs, ::qs].T)
		self.last_flow_signature = signature

	def _update_whisker_shapes(self, orig: np.ndarray, deff: np.ndarray) -> None:
		angle_deg = np.degrees(self.sim.state.heading)
		_update_whisker_ellipses_common(self.whisker_orig, orig, angle_deg)
		_update_whisker_ellipses_common(self.whisker_def, deff, angle_deg)

	def _animated_artists(self) -> tuple:
		artists: list = [
			self.bg,
			*self.orig_mesh_lines,
			*self.def_mesh_lines,
			*self.whisker_orig,
			*self.whisker_def,
			self.deflection_quiver,
			self.text,
		]
		if self.flow_q is not None:
			artists.append(self.flow_q)
		return tuple(artists)

	def animate(self, input_provider: InputProvider | None, replay_txyvv: np.ndarray | None = None, save_replay: str | Path | None = None) -> None:
		self.setup()
		paused = {"value": True}
		finished = {"value": False, "x_mm": None, "t": None}
		flow_visible = {"value": False}
		traj_visible = {"value": False}
		base_visible = {"value": True}
		deflected_visible = {"value": False}
		arrows_visible = {"value": True}
		replay_mode = replay_txyvv is not None
		replay_idx = {"value": 0}
		save_replay_path = Path(save_replay) if save_replay is not None else None
		save_replay_run = {"value": 0}  # counts completed runs for numbered filenames
		keyboard = input_provider if isinstance(input_provider, KeyboardInput) else None
		self.text.set_text("Press space to start")
		self.bg.set_visible(False)
		if self.flow_q is not None:
			self.flow_q.set_visible(False)
		profile_reported = {"value": False}

		def _set_idle_text() -> None:
			self.text.set_text("Press space to start | F1: Help")

		def _set_small_initial_velocity() -> None:
			if replay_mode:
				return
			init_vx = float(min(0.1, self.sim.config.max_speed * 0.3))
			self.sim.state.velocity = np.array([init_vx, 0.0], dtype=float)
			if keyboard is not None:
				keyboard.set_command(np.array([1.0, 0.0], dtype=float), init_vx / max(keyboard.max_speed, 1e-9))

		def _refresh_preview_pose() -> None:
			orig = self.sim.world_whisker_centers()
			deff = orig.copy()
			self._update_mesh_lines(self.orig_mesh_lines, orig)
			self._update_mesh_lines(self.def_mesh_lines, deff)
			self._update_whisker_shapes(orig, deff)
			self.deflection_quiver.set_offsets(orig)
			zeros = np.zeros(orig.shape[0], dtype=float)
			self.deflection_quiver.set_UVC(zeros, zeros)

		center_time_hist: list[float] = []
		center_xy_hist: list[np.ndarray] = []
		center_vel_hist: list[np.ndarray] = []
		whisker_xy_hist: list[np.ndarray] = []

		def _reset_recorded_trajectory() -> None:
			center_time_hist.clear()
			center_xy_hist.clear()
			center_vel_hist.clear()
			whisker_xy_hist.clear()

			pos = self.sim.state.position.copy()
			vel = self.sim.state.velocity.copy()
			orig = self.sim.world_whisker_centers().copy()
			center_time_hist.append(float(self.sim.time))
			center_xy_hist.append(np.array([float(pos[0]), float(pos[1])], dtype=float))
			center_vel_hist.append(np.array([float(vel[0]), float(vel[1])], dtype=float))
			whisker_xy_hist.append(orig)
			self.array_trajectory_txy = np.array([[float(self.sim.time), float(pos[0]), float(pos[1])]], dtype=float)
			self.array_replay_txyvv = np.array(
				[[float(self.sim.time), float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])]],
				dtype=float,
			)

			changed = self.array_center_traj_line.get_visible() or any(ln.get_visible() for ln in self.whisker_traj_lines)
			self.array_center_traj_line.set_data([], [])
			self.array_center_traj_line.set_visible(False)
			for ln in self.whisker_traj_lines:
				ln.set_data([], [])
				ln.set_visible(False)
			if changed:
				_redraw_static_after_toggle()

		def _append_recorded_trajectory(orig: np.ndarray) -> None:
			pos = self.sim.state.position
			vel = self.sim.state.velocity
			center_time_hist.append(float(self.sim.time))
			center_xy_hist.append(np.array([float(pos[0]), float(pos[1])], dtype=float))
			center_vel_hist.append(np.array([float(vel[0]), float(vel[1])], dtype=float))
			whisker_xy_hist.append(orig.copy())

		def _reveal_recorded_trajectory() -> None:
			if not center_xy_hist:
				return
			centers = np.vstack(center_xy_hist)
			times = np.asarray(center_time_hist, dtype=float)
			self.array_trajectory_txy = np.column_stack((times, centers))
			vels = np.vstack(center_vel_hist)
			self.array_replay_txyvv = np.column_stack((times, centers, vels))

			if save_replay_path is not None:
				save_replay_run["value"] += 1
				stem = save_replay_path.stem
				suffix = save_replay_path.suffix or ".csv"
				out_path = save_replay_path.with_name(f"{stem}_{save_replay_run['value']:03d}{suffix}")
				_save_replay_data(self.array_replay_txyvv, out_path)

			self.array_center_traj_line.set_data(centers[:, 0], centers[:, 1])
			self.array_center_traj_line.set_visible(True)

			if whisker_xy_hist:
				hist = np.stack(whisker_xy_hist, axis=0)
				for i, ln in enumerate(self.whisker_traj_lines):
					ln.set_data(hist[:, i, 0], hist[:, i, 1])
					ln.set_visible(True)
			_redraw_static_after_toggle()

		if replay_mode and replay_txyvv is not None and replay_txyvv.shape[0] > 0:
			_apply_replay_row_to_sim(self.sim, replay_txyvv[0])
		_set_small_initial_velocity()
		_refresh_preview_pose()
		_reset_recorded_trajectory()
		_set_idle_text()

		def _redraw_static_after_toggle() -> None:
			# With blit=True, static background is cached; clear it so visibility changes stick.
			anim = getattr(self.fig, "_anim", None)
			blit_cache = getattr(anim, "_blit_cache", None)
			if isinstance(blit_cache, dict):
				blit_cache.clear()
			self.fig.canvas.draw()

		def _toggle_traj() -> None:
			traj_visible["value"] = not traj_visible["value"]
			self.traj_line.set_visible(traj_visible["value"])
			_redraw_static_after_toggle()

		def _switch_to_case(case_idx: int) -> None:
			if not self.flow_cases:
				return
			case_idx = int(np.clip(case_idx, 0, len(self.flow_cases) - 1))
			case_changed = case_idx != self.current_case_idx
			self.current_case_idx = case_idx
			case = self.flow_cases[self.current_case_idx]
			if case_changed:
				if self.use_dynamic_flow:
					self.sim.flow = FlowFieldSequence.from_path(
						case.data_path,
						flow_dt=self.sim.config.flow_dt,
						decay_half_life=self.flow_decay_half_life,
					)
				else:
					self.sim.flow = FlowFieldSequence.from_last_frame(case.data_path)
				self.last_flow_signature = None
				self._last_decay_visual_t = -1e18
			self.sim.reset()
			if keyboard is not None:
				keyboard.reset()
			_set_small_initial_velocity()
			paused["value"] = True
			finished["value"] = False
			finished["x_mm"] = None
			finished["t"] = None

			# Update map extent and backdrop artists for the newly loaded flow grid.
			ext = self.sim.flow.extent()
			self.ax.set_xlim(ext[0], ext[1])
			self.ax.set_ylim(ext[2], ext[3])
			xmin, xmax, ymin, ymax = ext
			dx = xmax - xmin
			dy = ymax - ymin
			self.text.set_position((xmin + 0.5 * dx, ymax - 0.015 * dy))

			f0 = self.sim.flow.frames[0]
			vs = self.vort_step
			s = self.sim.flow.coord_scale
			x_img = f0.x[::vs] * s
			y_img = f0.y[::vs] * s
			self.bg.set_extent((float(x_img[0]), float(x_img[-1]), float(y_img[0]), float(y_img[-1])))
			self.bg.set_data(f0.vorticity[::vs, ::vs].T)

			flow_visible["value"] = False
			self.bg.set_visible(False)
			traj_visible["value"] = False
			self.traj_line.set_visible(False)
			if self.flow_q is not None:
				self.flow_q.remove()
				self.flow_q = None
			if self.show_flow_quiver:
				xsub = f0.x[:: self.flow_quiver_step] * s
				ysub = f0.y[:: self.flow_quiver_step] * s
				u = f0.u[:: self.flow_quiver_step, :: self.flow_quiver_step].T
				v = f0.v[:: self.flow_quiver_step, :: self.flow_quiver_step].T
				self.flow_q = self.ax.quiver(
					xsub,
					ysub,
					u,
					v,
					color="0.25",
					angles="xy",
					scale_units="xy",
					scale=6.0,
					width=0.0015,
					alpha=0.55,
				)
				self.flow_q.set_visible(flow_visible["value"])

			xy = case.traj_xy
			if xy is not None and xy.size > 0:
				self.traj_line.set_data(xy[:, 0], xy[:, 1])
			else:
				self.traj_line.set_data([], [])

			_set_idle_text()
			_refresh_preview_pose()
			_reset_recorded_trajectory()
			_redraw_static_after_toggle()

		def _toggle_flow() -> None:
			flow_visible["value"] = not flow_visible["value"]
			self.bg.set_visible(flow_visible["value"])
			if self.flow_q is not None:
				self.flow_q.set_visible(flow_visible["value"])
			if flow_visible["value"]:
				self.last_flow_signature = None
				self._last_decay_visual_t = -1e18
			_redraw_static_after_toggle()

		def _show_flow_and_traj() -> None:
			changed = False
			if not flow_visible["value"]:
				flow_visible["value"] = True
				self.bg.set_visible(True)
				if self.flow_q is not None:
					self.flow_q.set_visible(True)
				changed = True
			if not traj_visible["value"]:
				traj_visible["value"] = True
				self.traj_line.set_visible(True)
				changed = True
			if changed:
				_redraw_static_after_toggle()

		def _hide_flow_and_traj() -> None:
			changed = False
			if flow_visible["value"]:
				flow_visible["value"] = False
				self.bg.set_visible(False)
				if self.flow_q is not None:
					self.flow_q.set_visible(False)
				changed = True
			if traj_visible["value"]:
				traj_visible["value"] = False
				self.traj_line.set_visible(False)
				changed = True
			if changed:
				_redraw_static_after_toggle()

		def _toggle_base_layer() -> None:
			base_visible["value"] = not base_visible["value"]
			for ln in self.orig_mesh_lines:
				ln.set_visible(base_visible["value"])
			for e in self.whisker_orig:
				e.set_visible(base_visible["value"])

		def _toggle_deflected_layer() -> None:
			deflected_visible["value"] = not deflected_visible["value"]
			for ln in self.def_mesh_lines:
				ln.set_visible(deflected_visible["value"])
			for e in self.whisker_def:
				e.set_visible(deflected_visible["value"])

		def _toggle_deflection_arrows() -> None:
			arrows_visible["value"] = not arrows_visible["value"]
			self.deflection_quiver.set_visible(arrows_visible["value"])

		def _show_help_popup() -> None:
			help_text = (
				"Whisker Simulator Controls\n"
				"\n"
				"Movement\n"
				"  Space           Start/Pause\n"
				"  Page Up / X     Accelerate\n"
				"  Page Down / Z   Brake\n"
				"  Arrow Keys      Turn heading by 5 deg toward key direction\n"
				"\n"
				"Layers\n"
				"  W  Toggle base whisker layer (original mesh/whiskers)\n"
				"  D  Toggle deflected layer (deflected mesh/whiskers)\n"
				"  A  Toggle deflection arrows\n"
				"  V  Toggle flow field (vorticity + quiver)\n"
				"  T  Toggle trajectory path\n"
				"\n"
				"Session\n"
				"  R  Reset to pre-start state\n"
				"  Tab  Reset with randomly selected flow path\n"
				"  Enter/Return  Open flow selector and reset\n"
				"  Esc  Close\n"
			)
			try:
				import tkinter as tk
				from tkinter import messagebox

				root = tk.Tk()
				root.withdraw()
				messagebox.showinfo("Simulator Help", help_text, parent=root)
				root.destroy()
			except Exception:
				print("[help]\n" + help_text)

		def _on_press(event) -> None:
			if keyboard is not None and event.key is not None:
				keyboard.on_key_press(event.key)
			if event.key is not None and event.key.lower() == "f1":
				_show_help_popup()
			if event.key == " ":
				paused["value"] = not paused["value"]
			if event.key in ("w", "W"):
				_toggle_base_layer()
			if event.key in ("d", "D"):
				_toggle_deflected_layer()
			if event.key in ("a", "A"):
				_toggle_deflection_arrows()
			if event.key in ("v", "V"):
				_toggle_flow()
			if event.key in ("t", "T"):
				_toggle_traj()
			if event.key in ("r", "R"):
				self.sim.reset()
				replay_idx["value"] = 0
				if replay_mode and replay_txyvv is not None and replay_txyvv.shape[0] > 0:
					_apply_replay_row_to_sim(self.sim, replay_txyvv[0])
				if keyboard is not None:
					keyboard.reset()
				_set_small_initial_velocity()
				paused["value"] = True
				finished["value"] = False
				finished["x_mm"] = None
				finished["t"] = None
				_set_idle_text()
				_refresh_preview_pose()
				_reset_recorded_trajectory()
				_hide_flow_and_traj()
			if event.key == "tab" and len(self.flow_cases) > 1:
				next_idx = int(np.random.randint(0, len(self.flow_cases)))
				if next_idx == self.current_case_idx:
					next_idx = (next_idx + 1) % len(self.flow_cases)
				_switch_to_case(next_idx)
			if event.key in ("enter", "return") and len(self.flow_cases) > 0:
				selected = _choose_flow_index(self.flow_cases, self.current_case_idx)
				if selected is not None:
					_switch_to_case(selected)
			if event.key == "escape":
				plt.close(self.fig)

		def _on_release(event) -> None:
			if keyboard is not None and event.key is not None:
				keyboard.on_key_release(event.key)

		self.fig.canvas.mpl_connect("key_press_event", _on_press)
		self.fig.canvas.mpl_connect("key_release_event", _on_release)

		interval_ms = max(1, int(round(self.sim.config.dt * 1000.0)))

		def _tick(_frame_idx: int):
			tick_started = perf_counter()
			if paused["value"]:
				if self.sim.profiler is not None:
					self.sim.profiler.record("tick_total", perf_counter() - tick_started)
				return self._animated_artists()

			if not finished["value"]:
				if replay_mode and replay_txyvv is not None:
					if replay_idx["value"] >= replay_txyvv.shape[0]:
						finished["value"] = True
						finished["x_mm"] = float(self.sim.state.position[0] * 1000.0)
						finished["t"] = float(self.sim.time)
						self.sim.state.velocity[:] = 0.0
						_reveal_recorded_trajectory()
						_show_flow_and_traj()
						orig = self.sim.world_whisker_centers()
						deff = orig.copy()
					else:
						row = replay_txyvv[replay_idx["value"]]
						replay_idx["value"] += 1
						orig, deff = _apply_replay_row_to_sim(self.sim, row)
				else:
					if input_provider is None:
						raise RuntimeError("input_provider is required for non-replay interactive mode")
					control_started = perf_counter()
					cmd = input_provider.read()
					if self.sim.profiler is not None:
						self.sim.profiler.record("control_read", perf_counter() - control_started)
					orig, deff, _ = self.sim.step(cmd)
				traj_started = perf_counter()
				_append_recorded_trajectory(orig)
				if self.sim.profiler is not None:
					self.sim.profiler.record("trajectory_record", perf_counter() - traj_started)
			else:
				orig = self.sim.world_whisker_centers()
				deff = orig.copy()

			if flow_visible["value"]:
				flow_started = perf_counter()
				self._update_flow_background(self.sim.time)
				if self.sim.profiler is not None:
					self.sim.profiler.record("render_flow_update", perf_counter() - flow_started)

			mesh_started = perf_counter()
			self._update_mesh_lines(self.orig_mesh_lines, orig)
			self._update_mesh_lines(self.def_mesh_lines, deff)
			if self.sim.profiler is not None:
				self.sim.profiler.record("render_mesh_update", perf_counter() - mesh_started)

			shape_started = perf_counter()
			self._update_whisker_shapes(orig, deff)
			if self.sim.profiler is not None:
				self.sim.profiler.record("render_shape_update", perf_counter() - shape_started)

			arrow_started = perf_counter()
			delta = deff - orig
			self.deflection_quiver.set_offsets(orig)
			self.deflection_quiver.set_UVC(delta[:, 0], delta[:, 1])
			if self.sim.profiler is not None:
				self.sim.profiler.record("render_arrow_update", perf_counter() - arrow_started)

			x_center = float(self.sim.state.position[0])
			if (not replay_mode) and (not finished["value"]) and x_center >= self.sim.config.finish_x:
				finished["value"] = True
				finished["x_mm"] = x_center * 1000.0
				finished["t"] = self.sim.time
				self.sim.state.velocity[:] = 0.0
				_reveal_recorded_trajectory()
				_show_flow_and_traj()
			if finished["value"]:
				text_started = perf_counter()
				pos = self.sim.state.position
				vel = self.sim.state.velocity
				v_mag = float(np.linalg.norm(vel))
				v_deg = float(np.degrees(np.arctan2(vel[1], vel[0]))) if v_mag > 1e-9 else 0.0
				if replay_mode:
					self.text.set_text(
						f"Replay finished at t={float(finished['t']):.2f} s | "
						f"pos=({pos[0]:.3f}, {pos[1]:.3f}) m | vel={v_mag:.3f} m/s @ {v_deg:.1f} deg | Press R to replay | F1: Help"
					)
				else:
					self.text.set_text(
						f"Finished at x={float(finished['x_mm']):.1f} mm, t={float(finished['t']):.2f} s | "
						f"pos=({pos[0]:.3f}, {pos[1]:.3f}) m | vel={v_mag:.3f} m/s @ {v_deg:.1f} deg | Press R to restart | F1: Help"
					)
				if self.sim.profiler is not None:
					self.sim.profiler.record("text_update", perf_counter() - text_started)
					self.sim.profiler.record("tick_total", perf_counter() - tick_started)
					if not profile_reported["value"]:
						self.sim.profiler.print_report(header="[profile] finish-line summary")
						profile_reported["value"] = True
				return self._animated_artists()
			text_started = perf_counter()
			elapsed_sec = int(self.sim.time)
			pos = self.sim.state.position
			vel = self.sim.state.velocity
			v_mag = float(np.linalg.norm(vel))
			v_deg = float(np.degrees(np.arctan2(vel[1], vel[0]))) if v_mag > 1e-9 else 0.0
			self.text.set_text(
				f"t={elapsed_sec:.1f} s | pos=({pos[0]:.3f}, {pos[1]:.3f}) m | "
				f"vel={v_mag:.3f} m/s @ {v_deg:.1f} deg | F1: Help"
			)
			if self.sim.profiler is not None:
				self.sim.profiler.record("text_update", perf_counter() - text_started)
				self.sim.profiler.record("tick_total", perf_counter() - tick_started)
			return self._animated_artists()

		anim = FuncAnimation(
			self.fig, _tick,
			init_func=self._animated_artists,
			interval=interval_ms,
			blit=True,
			cache_frame_data=False,
		)
		self.fig._anim = anim  # type: ignore[attr-defined]
		try:
			plt.show()
		finally:
			if center_xy_hist and center_vel_hist:
				times = np.asarray(center_time_hist, dtype=float)
				centers = np.vstack(center_xy_hist)
				vels = np.vstack(center_vel_hist)
				self.array_trajectory_txy = np.column_stack((times, centers))
				self.array_replay_txyvv = np.column_stack((times, centers, vels))
			if input_provider is not None:
				input_provider.shutdown()


def _build_input_provider(
	prefer_gamepad: bool,
	max_speed: float = 0.4,
	accel_per_frame: float = 0.008,
) -> tuple[InputProvider, str]:
	if prefer_gamepad:
		try:
			provider = PygameGamepadInput()
			name = provider.joystick.get_name() if provider.joystick is not None else "gamepad"
			return provider, f"gamepad: {name}"
		except Exception as exc:
			print(f"[input] gamepad unavailable ({exc}); falling back to keyboard.")
	return KeyboardInput(max_speed=max_speed, accel_per_frame=accel_per_frame), "keyboard"


def run_simulation(
	data_path: str | Path | None = None,
	*,
	array_config: str | Path | None = None,
	replay: str | Path | None = None,
	save_replay: str | Path | None = None,
	data_root: str | Path | None = None,
	traj_path: str | Path | None = None,
	exclude_list: str | Path | None = None,
	fps: float = 10.0,
	flow_dt: float = 0.04,
	rows: int = 3,
	cols: int = 3,
	spacing: float = 0.02,
	max_speed: float = 0.4,
	max_accel: float = 0.4,
	deflection_gain: float = 0.03,
	max_deflection: float = 0.03,
	finish_x_mm: float = 2000.0,
	prefer_gamepad: bool = True,
	dynamic_flow: bool = False,
	flow_decay: float = 0.0,
	flow_time_delay: float = 0.0,
	profile_step: bool = False,
	headless: bool = False,
	save_file: str | Path | None = None,
	save_fps: float = 10.0,
	save_dpi: int = 90,
) -> None:
	"""Entry point for whisker-array simulation."""

	if fps <= 0.0:
		raise ValueError("fps must be positive")
	if flow_dt <= 0.0:
		raise ValueError("flow_dt must be positive")
	if flow_decay < 0.0:
		raise ValueError("flow_decay must be >= 0")
	if flow_time_delay < 0.0:
		raise ValueError("flow_time_delay must be >= 0")
	if finish_x_mm <= 0.0:
		raise ValueError("finish_x_mm must be positive")
	if save_dpi <= 0:
		raise ValueError("save_dpi must be positive")

	replay_txyvv: np.ndarray | None = None
	if replay is not None:
		replay_txyvv = _load_replay_data(replay)
		print(f"[replay] loaded {replay_txyvv.shape[0]} rows from {Path(replay)}")

	flow_cases, initial_case_idx = _build_flow_cases(
		data_path=data_path,
		data_root=data_root,
		traj_path=traj_path,
		exclude_list=exclude_list,
	)
	if not flow_cases:
		raise ValueError("No valid flow cases found")
	initial_case = flow_cases[initial_case_idx]
	if dynamic_flow and data_path is not None:
		flow_cases = [initial_case]
		initial_case_idx = 0
	print(f"[flow] launch case: {initial_case.label} ({initial_case.data_path})")

	if dynamic_flow:
		flow = FlowFieldSequence.from_path(initial_case.data_path, flow_dt=flow_dt, decay_half_life=flow_decay)
		print(f"[flow] dynamic time series enabled ({len(flow.frames)} real frames)")
	else:
		flow = FlowFieldSequence.from_last_frame(initial_case.data_path)
	if array_config is not None:
		geometry = _load_array_geometry_from_yaml(array_config)
		print(f"[array] loaded layout from {Path(array_config)} ({geometry.layout_xy.shape[0]} whiskers)")
	else:
		geometry = WhiskerArrayGeometry.regular_grid(rows=rows, cols=cols, spacing=spacing)
	ext = flow.extent()
	if replay_txyvv is not None and replay_txyvv.shape[0] > 0:
		first = replay_txyvv[0]
		init_pos = np.array([float(first[1]), float(first[2])], dtype=float)
		init_vel = np.array([float(first[3]), float(first[4])], dtype=float)
		init_speed = float(np.linalg.norm(init_vel))
		init_heading = float(np.arctan2(init_vel[1], init_vel[0])) if init_speed > 1e-9 else 0.0
		state = ArrayState(position=init_pos, velocity=init_vel, heading=init_heading)
	else:
		init_pos = np.array([0.05, float(np.random.uniform(ext[2], ext[3]))], dtype=float)
		init_vx = float(min(0.03, max_speed * 0.1))
		state = ArrayState(position=init_pos, velocity=np.array([init_vx, 0.0], dtype=float), heading=0.0)
	dt = 1.0 / fps
	config = SimulatorConfig(
		max_speed=max_speed,
		max_accel=max_accel,
		deflection_gain=deflection_gain,
		max_deflection=max_deflection,
		dt=dt,
		flow_dt=flow_dt,
		flow_time_delay=flow_time_delay,
		finish_x=finish_x_mm * 1e-3,
	)
	sim = WhiskerArraySimulator(
		geometry=geometry,
		flow=flow,
		config=config,
		state=state,
		profiler=StepProfiler() if profile_step else None,
	)
	if replay_txyvv is not None and replay_txyvv.shape[0] > 0:
		sim.time = float(replay_txyvv[0, 0])
	if save_file is not None and not headless:
		print("[headless] --save-file provided; enabling headless mode")
		headless = True

	if headless:
		import matplotlib
		matplotlib.use("Agg")
		start = perf_counter()
		if replay_txyvv is not None:
			_run_headless_replay_mode(
				sim,
				replay_txyvv=replay_txyvv,
				save_file=save_file,
				save_fps=save_fps,
				save_dpi=save_dpi,
				object_traj_xy=initial_case.traj_xy,
			)
		else:
			_run_headless_mode(
				sim,
				sample_rate_hz=fps,
				save_file=save_file,
				save_fps=save_fps,
				save_dpi=save_dpi,
				object_traj_xy=initial_case.traj_xy,
			)
		print(f'Headless mode done in {perf_counter() - start:.3f} s')
		return

	renderer = SimulationRenderer(
		sim=sim,
		flow_cases=flow_cases,
		current_case_idx=initial_case_idx,
		use_dynamic_flow=dynamic_flow,
		flow_decay_half_life=flow_decay,
	)
	if replay_txyvv is not None:
		print("[replay] interactive replay mode enabled")
		renderer.animate(None, replay_txyvv=replay_txyvv, save_replay=save_replay)
	else:
		accel_per_frame = max_accel * dt * 0.25
		provider, provider_name = _build_input_provider(
			prefer_gamepad=prefer_gamepad,
			max_speed=max_speed,
			accel_per_frame=accel_per_frame,
		)
		print(f"[input] using {provider_name}")
		renderer.animate(provider, save_replay=save_replay)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Interactive whisker-array simulator")
	parser.add_argument(
		"--data-path",
		type=Path,
		default=None,
		help="Specific directory containing Q.*.plt files (if omitted, pick random from --data-root)",
	)
	parser.add_argument(
		"--data-root",
		type=Path,
		default=Path("flowdata") / "env1",
		help="Root containing path<number> flow folders (default: flowdata/env1)",
	)
	parser.add_argument(
		"--exclude-list",
		type=Path,
		default=None,
		help="Text file listing path indices to exclude (default: <data-root>/exclude_list.txt)",
	)
	parser.add_argument("--fps", type=float, default=10.0, help="Simulation step rate in Hz for both interactive and headless modes (default: 10)")
	parser.add_argument("--flow-dt", type=float, default=0.04, help="Time spacing of flow frames (s)")
	parser.add_argument(
		"--replay",
		type=Path,
		default=None,
		help="Replay file with 5 columns: t,x,y,x_vel,y_vel (SI units)",
	)
	parser.add_argument(
		"--save-replay",
		type=Path,
		default=None,
		help="Save played interactive trajectory to CSV with columns t,x,y,x_vel,y_vel",
	)
	parser.add_argument(
		"--dynamic-flow",
		action="store_true",
		help="Load and use all Q.*.plt frames as a time series with temporal interpolation",
	)
	parser.add_argument(
		"--flow-decay",
		type=float,
		default=0.0,
		help="Half-life in seconds for exponential flow decay past the last real frame (0 = no decay, wraps/clamps to last frame)",
	)
	parser.add_argument(
		"--flow-time-delay",
		type=float,
		default=0.0,
		help="Delay (s) before array time maps into flow time for dynamic sampling",
	)
	parser.add_argument(
		"--profile-step",
		action="store_true",
		help="Collect timing statistics for simulation-step sub-parts and print them when the run finishes",
	)
	parser.add_argument("--rows", type=int, default=3, help="Whisker rows")
	parser.add_argument("--cols", type=int, default=3, help="Whisker columns")
	parser.add_argument("--spacing", type=float, default=0.02, help="Whisker spacing (m)")
	parser.add_argument(
		"--array-config",
		type=Path,
		default=None,
		help="Single YAML file defining unit_arrays (+ optional instances), or layout_xy/grid params; overrides --rows/--cols/--spacing",
	)
	parser.add_argument("--max-speed", type=float, default=0.2, help="Maximum commanded speed (m/s)")
	parser.add_argument("--max-accel", type=float, default=0.2, help="Per-axis acceleration limit (m/s^2)")
	parser.add_argument(
		"--deflection-gain",
		type=float,
		default=0.2,
		help="Deflection gain (m per m/s of local flow velocity)",
	)
	parser.add_argument("--max-deflection", type=float, default=0.2, help="Maximum whisker tip displacement (m)")
	parser.add_argument(
		"--finish-x-mm",
		type=float,
		default=1920.0,
		help="Finish line x-position (mm); simulation stops when center reaches it",
	)
	parser.add_argument("--traj-path", type=Path, default=None, help="Directory containing *.dat trajectory files (x,y columns in mm)")
	parser.add_argument(
		"--keyboard-only",
		action="store_true",
		help="Disable gamepad probing and use keyboard controls only",
	)
	parser.add_argument(
		"--headless",
		action="store_true",
		help="Run deterministic non-interactive mode (no live visualization)",
	)
	parser.add_argument(
		"--save-file",
		type=Path,
		default=None,
		help="Optional animation output path in headless mode (.mp4/.m4v/.mov/.gif)",
	)
	parser.add_argument(
		"--save-fps",
		type=float,
		default=10.0,
		help="Output frame rate when saving headless animation (default: 10 FPS)",
	)
	parser.add_argument(
		"--save-dpi",
		type=int,
		default=90,
		help="DPI for saved animation (default: 90; 6.4x4.2 in figure gives 576x378 px at 90 dpi)",
	)
	args = parser.parse_args()

	run_simulation(
		args.data_path,
		array_config=args.array_config,
		replay=args.replay,
		save_replay=args.save_replay,
		data_root=args.data_root,
		traj_path=args.traj_path,
		exclude_list=args.exclude_list,
		fps=args.fps,
		flow_dt=args.flow_dt,
		rows=args.rows,
		cols=args.cols,
		spacing=args.spacing,
		max_speed=args.max_speed,
		max_accel=args.max_accel,
		deflection_gain=args.deflection_gain,
		max_deflection=args.max_deflection,
		finish_x_mm=args.finish_x_mm,
		prefer_gamepad=not args.keyboard_only,
		dynamic_flow=args.dynamic_flow,
		flow_decay=args.flow_decay,
		flow_time_delay=args.flow_time_delay,
		profile_step=args.profile_step,
		headless=args.headless,
		save_file=args.save_file,
		save_fps=args.save_fps,
		save_dpi=args.save_dpi,
	)