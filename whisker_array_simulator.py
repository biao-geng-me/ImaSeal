
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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
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
	flow_loop: bool = True
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
	def from_path(data_path: str | Path, coord_scale: float = 1e-3) -> "FlowFieldSequence":
		frames = load_flow_frames(data_path)
		if not frames:
			raise ValueError("No flow frames found")
		return FlowFieldSequence(frames=frames, coord_scale=coord_scale)

	def _frame_index(self, t: float, flow_dt: float, loop: bool) -> int:
		n = len(self.frames)
		idx = int(np.floor(t / flow_dt))
		if loop:
			return idx % n
		return min(max(idx, 0), n - 1)

	def sample_velocity(self, world_xy: np.ndarray, t: float, flow_dt: float, loop: bool) -> np.ndarray:
		frame = self.frames[self._frame_index(t=t, flow_dt=flow_dt, loop=loop)]
		x = frame.x
		y = frame.y
		u_grid = frame.u
		v_grid = frame.v

		# world_xy is in simulator units (m); x/y grid is in flow data units (mm).
		inv = 1.0 / self.coord_scale
		xi = np.interp(world_xy[:, 0] * inv, x, np.arange(x.size))
		yi = np.interp(world_xy[:, 1] * inv, y, np.arange(y.size))

		x0 = np.clip(np.floor(xi).astype(int), 0, x.size - 1)
		y0 = np.clip(np.floor(yi).astype(int), 0, y.size - 1)
		x1 = np.clip(x0 + 1, 0, x.size - 1)
		y1 = np.clip(y0 + 1, 0, y.size - 1)

		sx = xi - x0
		sy = yi - y0

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

	def world_whisker_centers(self) -> np.ndarray:
		rot = _rotation_matrix(self.state.heading)
		return self.state.position + self.geometry.layout_xy @ rot.T

	def _apply_accel_limit(self, target_vel: Vec2) -> None:
		dv = target_vel - self.state.velocity
		max_dv = self.config.max_accel * self.config.dt
		dv = np.clip(dv, -max_dv, max_dv)
		self.state.velocity = self.state.velocity + dv

	def step(self, command: ControllerCommand) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

		orig_centers = self.world_whisker_centers()
		flow_vel = self.flow.sample_velocity(
			orig_centers,
			t=self.time,
			flow_dt=self.config.flow_dt,
			loop=self.config.flow_loop,
		)

		deflection = flow_vel * self.config.deflection_gain
		def_mag = np.linalg.norm(deflection, axis=1)
		mask = def_mag > self.config.max_deflection
		if np.any(mask):
			deflection[mask] *= (self.config.max_deflection / def_mag[mask])[:, None]

		deflected_centers = orig_centers + deflection
		return orig_centers, deflected_centers, flow_vel


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


@dataclass
class SimulationRenderer:
	"""Matplotlib renderer for whisker array state and flow backdrop."""

	sim: WhiskerArraySimulator
	show_flow_quiver: bool = True
	flow_quiver_step: int = 24
	vort_step: int = 8  # subsample vorticity imshow for faster rendering
	flow_cases: list[FlowCase] = field(default_factory=list)
	current_case_idx: int = 0
	array_trajectory_txy: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float), init=False)

	def _build_mesh_lines(self, color: str, lw: float, alpha: float, zorder: int = 2) -> list[Line2D]:
		lines: list[Line2D] = []
		for _ in self.sim.geometry.mesh_edges:
			ln = Line2D([], [], color=color, lw=lw, alpha=alpha, zorder=zorder)
			self.ax.add_line(ln)
			lines.append(ln)
		return lines

	def _update_mesh_lines(self, lines: list[Line2D], pts: np.ndarray) -> None:
		for ln, (i, j) in zip(lines, self.sim.geometry.mesh_edges):
			ln.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])

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

		# Trajectory sits beneath everything else (zorder=1), hidden by default.
		(self.traj_line,) = self.ax.plot([], [], color="tab:red", lw=4, alpha=0.5, zorder=1, visible=False)
		(self.array_center_traj_line,) = self.ax.plot([], [], color="tab:green", lw=1.8, alpha=0.7, zorder=1.1, visible=False)
		if self.flow_cases:
			xy = self.flow_cases[self.current_case_idx].traj_xy
			if xy is not None and xy.size > 0:
				self.traj_line.set_data(xy[:, 0], xy[:, 1])

		# Layer order: original (zorder=2) → arrows (3) → deflected (4)
		self.orig_mesh_lines = self._build_mesh_lines(color="tab:grey", lw=1.8, alpha=0.85, zorder=2)

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
			e_orig.set_linewidth(2.0)
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

		self.def_mesh_lines = self._build_mesh_lines(color="tab:orange", lw=1.8, alpha=0.5, zorder=4)
		for _ in range(n):
			e_def = Ellipse((0.0, 0.0), width=self.sim.geometry.ellipse_major, height=self.sim.geometry.ellipse_minor)
			e_def.set_edgecolor("tab:orange")
			e_def.set_facecolor("none")
			e_def.set_linewidth(2.0)
			e_def.set_zorder(4)
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
		idx = self.sim.flow._frame_index(t=t, flow_dt=self.sim.config.flow_dt, loop=self.sim.config.flow_loop)
		f = self.sim.flow.frames[idx]
		vs = self.vort_step
		self.bg.set_data(f.vorticity[::vs, ::vs].T)
		if self.flow_q is not None:
			u = f.u[:: self.flow_quiver_step, :: self.flow_quiver_step].T
			v = f.v[:: self.flow_quiver_step, :: self.flow_quiver_step].T
			self.flow_q.set_UVC(u, v)

	def _update_whisker_shapes(self, orig: np.ndarray, deff: np.ndarray) -> None:
		angle_deg = np.degrees(self.sim.state.heading)
		for e, p in zip(self.whisker_orig, orig):
			e.center = (float(p[0]), float(p[1]))
			e.angle = angle_deg
		for e, p in zip(self.whisker_def, deff):
			e.center = (float(p[0]), float(p[1]))
			e.angle = angle_deg

	def _animated_artists(self) -> tuple:
		return (
			*self.orig_mesh_lines,
			*self.def_mesh_lines,
			*self.whisker_orig,
			*self.whisker_def,
			self.deflection_quiver,
			self.text,
		)

	def animate(self, input_provider: InputProvider) -> None:
		self.setup()
		paused = {"value": True}
		finished = {"value": False, "x_mm": None, "t": None}
		flow_visible = {"value": False}
		traj_visible = {"value": False}
		base_visible = {"value": True}
		deflected_visible = {"value": True}
		arrows_visible = {"value": True}
		keyboard = input_provider if isinstance(input_provider, KeyboardInput) else None
		self.text.set_text("Press space to start")
		self.bg.set_visible(False)
		if self.flow_q is not None:
			self.flow_q.set_visible(False)

		def _set_idle_text() -> None:
			self.text.set_text("Press space to start | F1: Help")

		def _set_small_initial_velocity() -> None:
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
		whisker_xy_hist: list[np.ndarray] = []

		def _reset_recorded_trajectory() -> None:
			center_time_hist.clear()
			center_xy_hist.clear()
			whisker_xy_hist.clear()

			pos = self.sim.state.position.copy()
			orig = self.sim.world_whisker_centers().copy()
			center_time_hist.append(float(self.sim.time))
			center_xy_hist.append(np.array([float(pos[0]), float(pos[1])], dtype=float))
			whisker_xy_hist.append(orig)
			self.array_trajectory_txy = np.array([[float(self.sim.time), float(pos[0]), float(pos[1])]], dtype=float)

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
			center_time_hist.append(float(self.sim.time))
			center_xy_hist.append(np.array([float(pos[0]), float(pos[1])], dtype=float))
			whisker_xy_hist.append(orig.copy())

		def _reveal_recorded_trajectory() -> None:
			if not center_xy_hist:
				return
			centers = np.vstack(center_xy_hist)
			times = np.asarray(center_time_hist, dtype=float)
			self.array_trajectory_txy = np.column_stack((times, centers))

			self.array_center_traj_line.set_data(centers[:, 0], centers[:, 1])
			self.array_center_traj_line.set_visible(True)

			if whisker_xy_hist:
				hist = np.stack(whisker_xy_hist, axis=0)
				for i, ln in enumerate(self.whisker_traj_lines):
					ln.set_data(hist[:, i, 0], hist[:, i, 1])
					ln.set_visible(True)
			_redraw_static_after_toggle()

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
				self.sim.flow = FlowFieldSequence.from_last_frame(case.data_path)
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
			if paused["value"]:
				return self._animated_artists()

			if not finished["value"]:
				cmd = input_provider.read()
				orig, deff, _ = self.sim.step(cmd)
				_append_recorded_trajectory(orig)
			else:
				orig = self.sim.world_whisker_centers()
				deff = orig.copy()

			self._update_mesh_lines(self.orig_mesh_lines, orig)
			self._update_mesh_lines(self.def_mesh_lines, deff)
			self._update_whisker_shapes(orig, deff)

			delta = deff - orig
			self.deflection_quiver.set_offsets(orig)
			self.deflection_quiver.set_UVC(delta[:, 0], delta[:, 1])

			x_center = float(self.sim.state.position[0])
			if (not finished["value"]) and x_center >= self.sim.config.finish_x:
				finished["value"] = True
				finished["x_mm"] = x_center * 1000.0
				finished["t"] = self.sim.time
				self.sim.state.velocity[:] = 0.0
				_reveal_recorded_trajectory()
				_show_flow_and_traj()
			if finished["value"]:
				pos = self.sim.state.position
				vel = self.sim.state.velocity
				v_mag = float(np.linalg.norm(vel))
				v_deg = float(np.degrees(np.arctan2(vel[1], vel[0]))) if v_mag > 1e-9 else 0.0
				self.text.set_text(
					f"Finished at x={float(finished['x_mm']):.1f} mm, t={float(finished['t']):.2f} s | "
					f"pos=({pos[0]:.3f}, {pos[1]:.3f}) m | vel={v_mag:.3f} m/s @ {v_deg:.1f} deg | Press R to restart | F1: Help"
				)
				return self._animated_artists()
			elapsed_sec = int(self.sim.time)
			pos = self.sim.state.position
			vel = self.sim.state.velocity
			v_mag = float(np.linalg.norm(vel))
			v_deg = float(np.degrees(np.arctan2(vel[1], vel[0]))) if v_mag > 1e-9 else 0.0
			self.text.set_text(
				f"t={elapsed_sec:.1f} s | pos=({pos[0]:.3f}, {pos[1]:.3f}) m | "
				f"vel={v_mag:.3f} m/s @ {v_deg:.1f} deg | F1: Help"
			)
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
	data_root: str | Path | None = None,
	traj_path: str | Path | None = None,
	exclude_list: str | Path | None = None,
	dt: float = 0.1,
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
) -> None:
	"""Entry point for whisker-array simulation."""

	if dt <= 0.0:
		raise ValueError("dt must be positive")
	if flow_dt <= 0.0:
		raise ValueError("flow_dt must be positive")
	if finish_x_mm <= 0.0:
		raise ValueError("finish_x_mm must be positive")

	flow_cases, initial_case_idx = _build_flow_cases(
		data_path=data_path,
		data_root=data_root,
		traj_path=traj_path,
		exclude_list=exclude_list,
	)
	if not flow_cases:
		raise ValueError("No valid flow cases found")
	initial_case = flow_cases[initial_case_idx]
	print(f"[flow] launch case: {initial_case.label} ({initial_case.data_path})")

	flow = FlowFieldSequence.from_last_frame(initial_case.data_path)
	if array_config is not None:
		geometry = _load_array_geometry_from_yaml(array_config)
		print(f"[array] loaded layout from {Path(array_config)} ({geometry.layout_xy.shape[0]} whiskers)")
	else:
		geometry = WhiskerArrayGeometry.regular_grid(rows=rows, cols=cols, spacing=spacing)
	ext = flow.extent()
	init_pos = np.array([0.05, float(np.random.uniform(ext[2], ext[3]))], dtype=float)
	init_vx = float(min(0.03, max_speed * 0.1))
	state = ArrayState(position=init_pos, velocity=np.array([init_vx, 0.0], dtype=float), heading=0.0)
	config = SimulatorConfig(
		max_speed=max_speed,
		max_accel=max_accel,
		deflection_gain=deflection_gain,
		max_deflection=max_deflection,
		dt=dt,
		flow_dt=flow_dt,
		finish_x=finish_x_mm * 1e-3,
	)
	sim = WhiskerArraySimulator(geometry=geometry, flow=flow, config=config, state=state)
	accel_per_frame = max_accel * dt * 0.25
	provider, provider_name = _build_input_provider(
		prefer_gamepad=prefer_gamepad,
		max_speed=max_speed,
		accel_per_frame=accel_per_frame,
	)
	print(f"[input] using {provider_name}")
	renderer = SimulationRenderer(sim=sim, flow_cases=flow_cases, current_case_idx=initial_case_idx)
	renderer.animate(provider)


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
	parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step (s), default 10 Hz")
	parser.add_argument("--flow-dt", type=float, default=0.04, help="Time spacing of flow frames (s)")
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
	args = parser.parse_args()

	run_simulation(
		args.data_path,
		array_config=args.array_config,
		data_root=args.data_root,
		traj_path=args.traj_path,
		exclude_list=args.exclude_list,
		dt=args.dt,
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
	)