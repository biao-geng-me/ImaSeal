"""Synthetic Karman vortex street generator and viewer.

This module builds a vectorized Lamb-Oseen point-vortex model along a provided
2D trajectory. Vortices are emitted alternately on each side of the trajectory,
with spacing based on Strouhal scaling:

St = f D / U ~= 0.21  ->  lambda = U/f = D/St

For alternating shedding, emission spacing is lambda/2 along trajectory arclength.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

from visualize_flow import _compute_vorticity


@dataclass
class VortexBatch:
    """Packed vortex parameters for vectorized flow evaluation."""

    birth_t: np.ndarray      # (M,)
    birth_xy: np.ndarray     # (M, 2)
    conv_vel: np.ndarray     # (M, 2)
    circulation: np.ndarray  # (M,)
    side: np.ndarray         # (M,) +/- 1


@dataclass
class SyntheticKarmanVortexStreet:
    """Karman vortex street represented by Lamb-Oseen point vortices."""

    grid_x: np.ndarray
    grid_y: np.ndarray
    traj_t: np.ndarray
    traj_xy: np.ndarray
    D: float = 0.05
    strouhal: float = 0.21
    nu: float = 1e-6
    vortex_convect_factor: float = 0.05
    convection_spread_deg: float = 75.0
    total_time: float = float("inf")
    circulation_half_life: float = 20.0
    circulation_coeff: float = 4.5
    core_age_eps: float = 2e-3

    def __post_init__(self) -> None:
        self.grid_x = np.asarray(self.grid_x, dtype=float)
        self.grid_y = np.asarray(self.grid_y, dtype=float)
        self.traj_t = np.asarray(self.traj_t, dtype=float)
        self.traj_xy = np.asarray(self.traj_xy, dtype=float)

        if self.grid_x.ndim != 1 or self.grid_y.ndim != 1:
            raise ValueError("grid_x and grid_y must be 1D arrays")
        if self.traj_t.ndim != 1:
            raise ValueError("traj_t must be 1D")
        if self.traj_xy.ndim != 2 or self.traj_xy.shape[1] != 2:
            raise ValueError("traj_xy must be Nx2")
        if self.traj_xy.shape[0] != self.traj_t.size:
            raise ValueError("traj_t and traj_xy length mismatch")
        if self.traj_t.size < 3:
            raise ValueError("trajectory must have at least 3 samples")
        if not np.all(np.diff(self.traj_t) > 0.0):
            raise ValueError("traj_t must be strictly increasing")
        if self.D <= 0.0:
            raise ValueError("D must be positive")
        if self.strouhal <= 0.0:
            raise ValueError("strouhal must be positive")
        if self.nu <= 0.0:
            raise ValueError("nu must be positive")
        if self.total_time <= 0.0:
            raise ValueError("total_time must be positive (or inf)")
        if self.circulation_half_life <= 0.0:
            raise ValueError("circulation_half_life must be positive")

        # Precompute trajectory differential geometry.
        dxy_dt = np.gradient(self.traj_xy, self.traj_t, axis=0, edge_order=2)
        speed = np.linalg.norm(dxy_dt, axis=1)
        speed = np.clip(speed, 1e-9, None)
        tan = dxy_dt / speed[:, None]
        norm = np.column_stack([-tan[:, 1], tan[:, 0]])

        ds = np.linalg.norm(np.diff(self.traj_xy, axis=0), axis=1)
        self.traj_s = np.concatenate([[0.0], np.cumsum(ds)])
        self.traj_speed = speed
        self.traj_tangent = tan
        self.traj_normal = norm

        self.lambda_shed = self.D / self.strouhal
        self.delta_emit = 0.5 * self.lambda_shed
        self.offset_dist = 0.32 * self.lambda_shed * 0.5 # theoretical: 0.281 

        # Estimate circulation scale from reference speed and body diameter.
        u_ref = float(np.median(self.traj_speed))
        gamma0 = self.circulation_coeff * u_ref * self.D

        self.vortices = self._build_vortices(gamma0=gamma0)
        self._xg, self._yg = np.meshgrid(self.grid_x, self.grid_y, indexing="xy")
        self._u_buf = np.zeros((self.grid_x.size, self.grid_y.size), dtype=float)
        self._v_buf = np.zeros((self.grid_x.size, self.grid_y.size), dtype=float)

    def _effective_time(self, t: float) -> float:
        """Clip query time to total_time horizon if finite."""
        if np.isfinite(self.total_time):
            return float(min(float(t), float(self.total_time)))
        return float(t)

    def _interp_scalar_over_s(self, values: np.ndarray, s_query: np.ndarray) -> np.ndarray:
        return np.interp(s_query, self.traj_s, values)

    def _interp_xy_over_s(self, s_query: np.ndarray) -> np.ndarray:
        x = np.interp(s_query, self.traj_s, self.traj_xy[:, 0])
        y = np.interp(s_query, self.traj_s, self.traj_xy[:, 1])
        return np.column_stack([x, y])

    def _interp_vec_over_s(self, vec: np.ndarray, s_query: np.ndarray) -> np.ndarray:
        vx = np.interp(s_query, self.traj_s, vec[:, 0])
        vy = np.interp(s_query, self.traj_s, vec[:, 1])
        v = np.column_stack([vx, vy])
        n = np.linalg.norm(v, axis=1)
        n = np.clip(n, 1e-9, None)
        return v / n[:, None]

    def _build_vortices(self, gamma0: float) -> VortexBatch:
        s_end = float(self.traj_s[-1])
        if s_end <= 0.0:
            raise ValueError("trajectory arclength is zero")

        s_birth = np.arange(0.0, s_end + 1e-12, self.delta_emit, dtype=float)
        t_birth = self._interp_scalar_over_s(self.traj_t, s_birth)
        xy_center = self._interp_xy_over_s(s_birth)
        tan = self._interp_vec_over_s(self.traj_tangent, s_birth)
        norm = self._interp_vec_over_s(self.traj_normal, s_birth)
        u_local = self._interp_scalar_over_s(self.traj_speed, s_birth)

        idx = np.arange(s_birth.size, dtype=int)
        side = np.where((idx % 2) == 0, 1.0, -1.0)
        circulation = gamma0 * side
        birth_xy = xy_center + norm * (self.offset_dist * side)[:, None]

        # Rotate convection direction away from trajectory by side-dependent angle.
        spread = np.deg2rad(float(self.convection_spread_deg))
        conv_dir = np.cos(spread) * tan + np.sin(spread) * (norm * side[:, None])
        conv_vel = conv_dir * (self.vortex_convect_factor * u_local)[:, None]

        return VortexBatch(
            birth_t=t_birth,
            birth_xy=birth_xy,
            conv_vel=conv_vel,
            circulation=circulation,
            side=side,
        )

    def active_vortex_positions(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return active vortex positions and side (+1/-1) at time t."""
        t_eff = self._effective_time(t)
        age = t_eff - self.vortices.birth_t
        mask = age > self.core_age_eps
        if not np.any(mask):
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
        age_m = age[mask]
        xy = self.vortices.birth_xy[mask] + self.vortices.conv_vel[mask] * age_m[:, None]
        return xy, self.vortices.side[mask]

    def active_vortex_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return active vortex centers, side (+1/-1), and Lamb-Oseen core radius."""
        t_eff = self._effective_time(t)
        age = t_eff - self.vortices.birth_t
        mask = age > self.core_age_eps
        if not np.any(mask):
            return (
                np.empty((0, 2), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
            )
        age_m = age[mask]
        xy = self.vortices.birth_xy[mask] + self.vortices.conv_vel[mask] * age_m[:, None]
        radius = np.sqrt(4.0 * self.nu * age_m)
        return xy, self.vortices.side[mask], radius

    def eval_velocity(self, t: float, xq: np.ndarray, yq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate flow velocity at query arrays xq,yq (same shape, vectorized)."""
        if xq.shape != yq.shape:
            raise ValueError("xq and yq must have identical shape")

        t_eff = self._effective_time(t)
        age = t_eff - self.vortices.birth_t
        mask = age > self.core_age_eps
        if not np.any(mask):
            return np.zeros_like(xq, dtype=float), np.zeros_like(yq, dtype=float)

        age_m = age[mask]
        gamma = self.vortices.circulation[mask] * (0.5 ** (age_m / self.circulation_half_life))
        xy = self.vortices.birth_xy[mask] + self.vortices.conv_vel[mask] * age_m[:, None]

        dx = xq[None, ...] - xy[:, 0][:, None, None]
        dy = yq[None, ...] - xy[:, 1][:, None, None]
        r2 = dx * dx + dy * dy
        r2 = np.maximum(r2, 1e-12)
        den = 4.0 * self.nu * age_m[:, None, None]
        factor = gamma[:, None, None] / (2.0 * np.pi * r2)
        core = 1.0 - np.exp(-r2 / den)

        u = np.sum(-factor * core * dy, axis=0)
        v = np.sum(factor * core * dx, axis=0)
        return u, v

    def eval_on_grid(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate u,v,w on configured Cartesian grid.

        Returns arrays shaped (Nx, Ny) consistent with the rest of this codebase.
        """
        u_xy, v_xy = self.eval_velocity(float(t), self._xg, self._yg)
        # Convert to codebase layout (x index first, y index second).
        self._u_buf[:, :] = u_xy.T
        self._v_buf[:, :] = v_xy.T
        w = _compute_vorticity(self.grid_x, self.grid_y, self._u_buf, self._v_buf)
        return self._u_buf, self._v_buf, w


def _apply_quiver_speed_threshold(
    uq: np.ndarray,
    vq: np.ndarray,
    speed_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide low-speed arrows by setting them to NaN so quiver skips drawing them."""
    if speed_threshold <= 0.0:
        return uq, vq

    out_u = np.array(uq, copy=True)
    out_v = np.array(vq, copy=True)
    mag = np.hypot(out_u, out_v)
    mask = mag < speed_threshold
    out_u[mask] = np.nan
    out_v[mask] = np.nan
    return out_u, out_v


def make_straight_trajectory(
    *,
    start_xy: tuple[float, float] = (0.2, 0.5),
    direction_xy: tuple[float, float] = (1.0, 0.0),
    speed: float = 0.25,
    duration: float = 8.0,
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a simple constant-speed straight-line trajectory."""
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    t = np.arange(0.0, duration + 0.5 * dt, dt, dtype=float)
    dir_vec = np.asarray(direction_xy, dtype=float)
    n = float(np.linalg.norm(dir_vec))
    if n <= 1e-12:
        raise ValueError("direction_xy must be non-zero")
    dir_vec = dir_vec / n
    start = np.asarray(start_xy, dtype=float)
    xy = start[None, :] + (speed * t)[:, None] * dir_vec[None, :]
    return t, xy


def _normalize_quiver_uv(
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    u_sub: np.ndarray,
    v_sub: np.ndarray,
    velocity_ref: float = 0.3,
    frac: float = 1.1,
) -> tuple[np.ndarray, np.ndarray]:
    if x_sub.size < 2 or y_sub.size < 2:
        return u_sub, v_sub
    dx = float(np.median(np.diff(x_sub)))
    dy = float(np.median(np.diff(y_sub)))
    cell_spacing = min(abs(dx), abs(dy))
    if cell_spacing <= 0.0 or velocity_ref <= 0.0:
        return u_sub, v_sub
    max_len_allowed = frac * cell_spacing
    mag = np.hypot(u_sub, v_sub)
    clip = np.ones_like(mag)
    nz = mag > 0.0
    clip[nz] = np.minimum(1.0, velocity_ref / mag[nz])
    scale = max_len_allowed / velocity_ref
    return u_sub * clip * scale, v_sub * clip * scale


def visualize_synthetic_kvs(
    street: SyntheticKarmanVortexStreet,
    *,
    fps: float = 20.0,
    vort_step: int = 8,
    quiver_step: int = 20,
    quiver_speed_threshold: float = 0.0,
    vort_vlim: float = 8.0,
    autoplay: bool = False,
    show_slider: bool = True,
) -> None:
    """Interactive viewer similar to visualize_flow.py for synthetic KVS."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if vort_step < 1:
        raise ValueError("vort_step must be >= 1")
    if quiver_step < 1:
        raise ValueError("quiver_step must be >= 1")
    if quiver_speed_threshold < 0.0:
        raise ValueError("quiver_speed_threshold must be >= 0")

    t0 = float(street.traj_t[0])
    t1 = float(street.total_time) if np.isfinite(street.total_time) else float(street.traj_t[-1])
    if t1 < t0:
        raise ValueError("total_time is earlier than trajectory start")
    dt = 1.0 / fps
    timeline = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(bottom=0.18 if show_slider else 0.1)

    x_img = street.grid_x
    y_img = street.grid_y
    x_vort = x_img[::vort_step]
    y_vort = y_img[::vort_step]
    xsub = x_img[::quiver_step]
    ysub = y_img[::quiver_step]

    xg_vort, yg_vort = np.meshgrid(x_vort, y_vort, indexing="xy")
    xg_quiv, yg_quiv = np.meshgrid(xsub, ysub, indexing="xy")

    def _eval_display_fields(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute only on display grids for speed (instead of full-resolution grid).
        u_vort_xy, v_vort_xy = street.eval_velocity(t, xg_vort, yg_vort)
        u_vort = u_vort_xy.T
        v_vort = v_vort_xy.T
        w = _compute_vorticity(x_vort, y_vort, u_vort, v_vort)

        uq_raw, vq_raw = street.eval_velocity(t, xg_quiv, yg_quiv)
        uq, vq = _normalize_quiver_uv(xsub, ysub, uq_raw, vq_raw, velocity_ref=0.3, frac=1.2)
        uq, vq = _apply_quiver_speed_threshold(uq, vq, quiver_speed_threshold)
        return uq, vq, w

    uq0, vq0, w0 = _eval_display_fields(float(timeline[0]))

    mesh = ax.imshow(
        w0.T,
        origin="lower",
        extent=(float(x_vort[0]), float(x_vort[-1]), float(y_vort[0]), float(y_vort[-1])),
        cmap="RdBu_r",
        vmin=-vort_vlim,
        vmax=vort_vlim,
        interpolation="nearest",
        aspect="equal",
    )
    qv = ax.quiver(
        xsub,
        ysub,
        uq0,
        vq0,
        color="k",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        headlength=2.0,
        headwidth=1.5,
        headaxislength=2.0,
    )

    (traj_line,) = ax.plot(street.traj_xy[:, 0], street.traj_xy[:, 1], color="tab:red", lw=2.0, alpha=0.8, label="trajectory")
    active_vortex_circles: list[Circle] = []

    legend_handles = [
        traj_line,
        Line2D([], [], marker="o", markersize=6, markerfacecolor="none", markeredgecolor="tab:blue", linestyle="None", label="+ vortex radius"),
        Line2D([], [], marker="o", markersize=6, markerfacecolor="none", markeredgecolor="tab:orange", linestyle="None", label="- vortex radius"),
    ]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(handles=legend_handles, loc="upper right")
    title = ax.set_title("")

    def _set_frame(idx: int, redraw: bool = True) -> None:
        t = float(timeline[idx])
        uq_i, vq_i, w = _eval_display_fields(t)
        mesh.set_data(w.T)
        qv.set_UVC(uq_i, vq_i)

        for cir in active_vortex_circles:
            cir.remove()
        active_vortex_circles.clear()

        xy_v, side, radius = street.active_vortex_state(t)
        for (xc, yc), sgn, r in zip(xy_v, side, radius):
            edge = "tab:blue" if sgn > 0.0 else "tab:orange"
            cir = Circle((float(xc), float(yc)), radius=float(r), fill=False, edgecolor=edge, linewidth=1.2, alpha=0.9, zorder=3.2)
            ax.add_patch(cir)
            active_vortex_circles.append(cir)

        title.set_text(f"Synthetic KVS | frame={idx + 1}/{timeline.size} | t={t:.3f} s")
        if redraw:
            fig.canvas.draw_idle()

    current_idx = 0
    _set_frame(0)

    slider = None
    if show_slider:
        slider_ax = fig.add_axes([0.12, 0.06, 0.76, 0.04])
        slider = Slider(
            ax=slider_ax,
            label="Frame",
            valmin=0,
            valmax=max(timeline.size - 1, 0),
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
            play_idx = (play_idx + 1) % timeline.size
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
        idx = int(slider.val) if slider is not None else current_idx
        if event.key == "right":
            next_idx = min(idx + 1, timeline.size - 1)
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
        elif event.key == " ":
            if is_playing[0]:
                anim_ref[0].event_source.stop()
                is_playing[0] = False
            else:
                _start_animation()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    if autoplay:
        _start_animation()
    plt.show()


def _make_demo_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Smooth curved path for local testing."""
    t = np.linspace(0.0, 9.0, 900)
    x = 0.15 + 0.22 * t
    y = 0.50 + 0.09 * np.sin(0.65 * t)
    xy = np.column_stack([x, y])
    return t, xy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Karman vortex street visualization")
    parser.add_argument("--fps", type=float, default=5.0, help="Viewer FPS")
    parser.add_argument("--D", type=float, default=0.06, help="Body diameter")
    parser.add_argument("--strouhal", type=float, default=0.21, help="Strouhal number")
    parser.add_argument("--nu", type=float, default=1e-5, help="Kinematic viscosity for Lamb-Oseen core growth")
    parser.add_argument("--vortex-convect-factor", type=float, default=0.01, help="Vortex convection speed factor")
    parser.add_argument(
        "--convection-spread-deg",
        type=float,
        default=75.0,
        help="Convection direction angle (deg) rotated away from trajectory by vortex side",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=float("100"),
        help="Total simulation time horizon in seconds (default: 100)",
    )
    parser.add_argument(
        "--circulation-half-life",
        type=float,
        default=20.0,
        help="Circulation half-life in seconds — circulation halves every this many seconds (default: 10.0)",
    )
    parser.add_argument("--circulation-coeff", type=float, default=4.5, help="Gamma scale coefficient: gamma0=coeff*U_ref*D")
    parser.add_argument("--quiver-step", type=int, default=24, help="Subsampling step for quiver")
    parser.add_argument("--vort-step", type=int, default=4, help="Subsampling step for vorticity background")
    parser.add_argument(
        "--quiver-speed-threshold",
        type=float,
        default=0.001,
        help="Hide quiver arrows with |u,v| below this threshold after normalization (0 = disabled)",
    )
    parser.add_argument("--vort-vlim", type=float, default=8.0, help="Symmetric vorticity color limit")
    parser.add_argument("--autoplay", action="store_true", help="Start playback automatically")
    parser.add_argument("--no-slider", action="store_true", help="Disable frame slider")
    args = parser.parse_args()

    t, xy = _make_demo_trajectory()
    x_grid = np.linspace(0.0, 2.4, 2001)
    y_grid = np.linspace(0.0, 1.2, 1001)

    street = SyntheticKarmanVortexStreet(
        grid_x=x_grid,
        grid_y=y_grid,
        traj_t=t,
        traj_xy=xy,
        D=args.D,
        strouhal=args.strouhal,
        nu=args.nu,
        vortex_convect_factor=args.vortex_convect_factor,
        convection_spread_deg=args.convection_spread_deg,
        total_time=args.total_time,
        circulation_half_life=args.circulation_half_life,
        circulation_coeff=args.circulation_coeff,
    )

    visualize_synthetic_kvs(
        street,
        fps=args.fps,
        vort_step=args.vort_step,
        quiver_step=args.quiver_step,
        quiver_speed_threshold=args.quiver_speed_threshold,
        vort_vlim=args.vort_vlim,
        autoplay=args.autoplay,
        show_slider=not args.no_slider,
    )
