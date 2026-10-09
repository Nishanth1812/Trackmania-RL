"""Recorded-route projection, telemetry road features, episode rules, and the simple progress reward.

Coordinates follow the game: x/z is the horizontal plane, y is up. Route points are
resampled at a fixed spacing, so index * spacing_m is arc length in metres.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

SMOKE_NO_ROUTE = "smoke-no-route"
FEATURE_DIM = 15
FEATURE_NAMES = (
    "speed", "gear", "rpm", "tangent_velocity", "lateral_velocity", "lateral_offset",
    "motion_err_sin", "motion_err_cos",
    "turn_5m_sin", "turn_5m_cos", "turn_15m_sin", "turn_15m_cos", "turn_30m_sin", "turn_30m_cos",
    "motion_valid",
)
ACTION_NAMES = ("gas", "brake", "steer")
CONTROL_HZ = 20

# Plausibility limits; the speed cap matches the /100 feature normalization.
MAX_PLAUSIBLE_SPEED_MPS = 100.0
MIN_AHEAD_M = 10.0
BACK_WINDOW_M = 10.0
JUMP_FLOOR_M = 5.0
MOTION_MIN_MPS = 1.0
LOOKAHEAD_M = (5.0, 15.0, 30.0)

SPEED_SCALE_MPS = 100.0
GEAR_SCALE = 6.0
VELOCITY_SCALE_MPS = 100.0
LATERAL_SCALE_M = 10.0

PROGRESS_WEIGHT = 0.1
BACKWARD_WEIGHT = 0.1
FINISH_BONUS = 10.0  # PLAN Phase I: one +10 award on the first valid game finish

# Episode rules (PLAN Phase I).
CORRIDOR_M = 10.0
OFF_ROUTE_S = 1.0
STUCK_GRACE_S = 5.0
STUCK_WINDOW_S = 3.0
STUCK_MIN_GAIN_M = 1.0
FINISH_RADIUS_M = 10.0
FINISH_PROGRESS_FRACTION = 0.95
TIME_EPS = 1e-9


class EpisodeFault(RuntimeError):
    """Invalid experience: the episode must be aborted and its buffer discarded."""


class RouteIdentityError(RuntimeError):
    """The route file does not match its declared identity; production must refuse it."""


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_route(path, expected_sha256) -> str:
    """Return the route file's sha256 if it matches ``expected_sha256``; raise otherwise."""
    if not expected_sha256:
        raise RouteIdentityError(f"route_sha256 missing for {path}; a route needs a declared identity")
    actual = file_sha256(path)
    if actual != str(expected_sha256).lower():
        raise RouteIdentityError(f"route sha256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return actual


def reward_identity() -> str:
    """sha256 over every constant that defines the reward, feature, and episode-rule contract."""
    params = {
        "progress_weight": PROGRESS_WEIGHT,
        "backward_weight": BACKWARD_WEIGHT,
        "finish_bonus": FINISH_BONUS,
        "jump_floor_m": JUMP_FLOOR_M,
        "max_plausible_speed_mps": MAX_PLAUSIBLE_SPEED_MPS,
        "back_window_m": BACK_WINDOW_M,
        "min_ahead_m": MIN_AHEAD_M,
        "motion_min_mps": MOTION_MIN_MPS,
        "lookahead_m": list(LOOKAHEAD_M),
        "speed_scale_mps": SPEED_SCALE_MPS,
        "gear_scale": GEAR_SCALE,
        "velocity_scale_mps": VELOCITY_SCALE_MPS,
        "lateral_scale_m": LATERAL_SCALE_M,
        "corridor_m": CORRIDOR_M,
        "off_route_s": OFF_ROUTE_S,
        "stuck_grace_s": STUCK_GRACE_S,
        "stuck_window_s": STUCK_WINDOW_S,
        "stuck_min_gain_m": STUCK_MIN_GAIN_M,
        "finish_radius_m": FINISH_RADIUS_M,
        "finish_progress_fraction": FINISH_PROGRESS_FRACTION,
        "feature_names": list(FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "control_hz": CONTROL_HZ,
    }
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RouteState:
    index: int
    progress_m: float
    lateral_m: float
    done: bool


class Route:
    """Recorded route loaded from NPZ, or a dummy route for pipeline smoke tests.

    NPZ keys: ``points`` float32[N,3] (x, y, z) and ``spacing_m`` scalar.
    Telemetry mapping keys for ``features``: ``speed_mps``, ``gear``, ``rpm``,
    ``vx_mps``, ``vz_mps`` (horizontal velocity).
    """

    def __init__(self, path: str, rpm_scale: float = 1.0, expected_sha256: Optional[str] = None):
        self.path = path
        self.rpm_scale = float(rpm_scale)
        self._dummy = path == SMOKE_NO_ROUTE
        if self._dummy:
            self.spacing_m = 1.0
            self._points = None
            self._xz = None
            self._tangents = None
        else:
            if expected_sha256 is not None:
                verify_route(path, expected_sha256)
            # allow_pickle=False: route files come from trusted-but-external sources.
            with np.load(path, allow_pickle=False) as data:
                points = np.asarray(data["points"], dtype=np.float64)
                spacing = float(data["spacing_m"])
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
                raise ValueError(f"route points must be (N>=2, 3), got {points.shape}")
            if not np.all(np.isfinite(points)) or not (spacing > 0.0):
                raise ValueError("route points must be finite and spacing_m must be > 0")
            self.spacing_m = spacing
            self._points = points
            self._xz = points[:, [0, 2]]
            self._tangents = self._build_tangents(self._xz)
        self.reset()

    @property
    def is_dummy(self) -> bool:
        return self._dummy

    @property
    def total_length_m(self) -> float:
        return (self._xz.shape[0] - 1) * self.spacing_m

    @property
    def end_xyz(self) -> np.ndarray:
        return self._points[-1].copy()

    @staticmethod
    def _build_tangents(xz: np.ndarray) -> np.ndarray:
        # Unit tangent of segment i (point i -> i+1); zero-length segments reuse the previous tangent.
        tangents = np.zeros((xz.shape[0], 2), dtype=np.float64)
        current = np.array([1.0, 0.0])
        for i in range(xz.shape[0] - 1):
            delta = xz[i + 1] - xz[i]
            length = math.hypot(delta[0], delta[1])
            if length > 1e-9:
                current = delta / length
            tangents[i] = current
        tangents[-1] = current
        return tangents

    def reset(self) -> None:
        """Clear the projection index and motion-direction state (call on every episode reset)."""
        self._index = 0
        if self._dummy:
            self._motion_dir = np.array([1.0, 0.0])
        else:
            self._motion_dir = self._tangents[0].copy()

    def project(self, position, dt: float) -> RouteState:
        """Nearest route segment within a window around the previous projection."""
        if self._dummy:
            return RouteState(index=0, progress_m=0.0, lateral_m=0.0, done=False)

        xz = self._xz
        last = xz.shape[0] - 2  # last valid segment start
        point = np.asarray(position, dtype=np.float64)[[0, 2]]

        ahead_m = max(MIN_AHEAD_M, 2.0 * MAX_PLAUSIBLE_SPEED_MPS * max(dt, 0.0))
        lo = max(0, self._index - int(math.ceil(BACK_WINDOW_M / self.spacing_m)))
        hi = min(last, self._index + int(math.ceil(ahead_m / self.spacing_m)))

        starts = xz[lo : hi + 1]
        ends = xz[lo + 1 : hi + 2]
        seg = ends - starts
        seg_len2 = np.maximum(np.sum(seg * seg, axis=1), 1e-12)
        t = np.clip(np.sum((point - starts) * seg, axis=1) / seg_len2, 0.0, 1.0)
        closest = starts + t[:, None] * seg
        dist2 = np.sum((point - closest) ** 2, axis=1)

        k = int(np.argmin(dist2))
        index = lo + k
        tangent = self._tangents[index]
        normal = np.array([-tangent[1], tangent[0]])  # positive lateral side (x/z rotated 90 degrees)
        lateral = float(np.dot(point - closest[k], normal))
        progress = (index + t[k]) * self.spacing_m
        total = self.total_length_m

        self._index = index
        return RouteState(
            index=index,
            progress_m=float(progress),
            lateral_m=lateral,
            done=bool(progress >= total - 1e-6),
        )

    def features(self, telemetry: Optional[Mapping], route_state: Optional[RouteState]) -> np.ndarray:
        """Return float32[15]: [0:3] in [0,1], [3:14] in [-1,1], [14] validity flag in [0,1]."""
        if self._dummy:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        i = route_state.index
        tangent = self._tangents[i]
        normal = np.array([-tangent[1], tangent[0]])

        velocity = np.array([telemetry["vx_mps"], telemetry["vz_mps"]], dtype=np.float64)
        speed_xz = float(np.hypot(velocity[0], velocity[1]))
        heading_valid = speed_xz >= MOTION_MIN_MPS
        if heading_valid:
            self._motion_dir = velocity / speed_xz  # below threshold, keep the last valid direction
        motion = self._motion_dir

        sin_err = tangent[0] * motion[1] - tangent[1] * motion[0]
        cos_err = float(np.dot(tangent, motion))

        turns = []
        for distance in LOOKAHEAD_M:
            # Clamp at the route end: never wrap to the start.
            j = min(i + int(round(distance / self.spacing_m)), self._tangents.shape[0] - 1)
            ahead = self._tangents[j]
            turns.append(tangent[0] * ahead[1] - tangent[1] * ahead[0])
            turns.append(float(np.dot(tangent, ahead)))

        out = np.empty(FEATURE_DIM, dtype=np.float64)
        out[0] = float(telemetry["speed_mps"]) / SPEED_SCALE_MPS
        out[1] = float(telemetry["gear"]) / GEAR_SCALE
        out[2] = float(telemetry["rpm"]) / self.rpm_scale
        out[3] = float(np.dot(velocity, tangent)) / VELOCITY_SCALE_MPS
        out[4] = float(np.dot(velocity, normal)) / VELOCITY_SCALE_MPS
        out[5] = route_state.lateral_m / LATERAL_SCALE_M
        out[6] = sin_err
        out[7] = cos_err
        out[8:14] = turns
        out[14] = 1.0 if heading_valid else 0.0

        out[0:3] = np.clip(out[0:3], 0.0, 1.0)
        out[3:14] = np.clip(out[3:14], -1.0, 1.0)
        return out.astype(np.float32)


class EpisodeMonitor:
    """Episode rules from PLAN Phase I: teleport, finish gate, off-route, and stuck.

    ``observe`` returns a terminal reason ("finished", "off_route", "stuck") or None.
    It raises EpisodeFault for invalid experience: a teleport jump, a finish flag that
    fails the gate, a finish flag at episode start, or telemetry time going backwards.
    """

    def __init__(self, route: Route):
        self._route = route
        self.reset()

    def reset(self) -> None:
        self._t0: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._prev_pos: Optional[np.ndarray] = None
        self._elapsed = 0.0
        self._max_progress = 0.0
        self._history: deque = deque()  # (t, max_progress) pairs kept for the stuck window
        self._off_since: Optional[float] = None
        self._finished = False
        self._ended = False  # stuck or off-route already reported

    @property
    def elapsed_s(self) -> float:
        return self._elapsed

    def observe(self, t: float, position, route_state: RouteState, finished: bool) -> Optional[str]:
        pos = np.asarray(position, dtype=np.float64)
        if not (math.isfinite(t) and np.all(np.isfinite(pos))):
            raise EpisodeFault("non-finite telemetry time or position")

        if self._prev_t is None:
            self._t0 = t
            if finished:
                raise EpisodeFault("finish flag set at episode start (integration fault)")
        else:
            dt = t - self._prev_t
            if dt < 0.0:
                raise EpisodeFault(f"telemetry time went backwards by {-dt:.3f} s")
            jump = float(np.linalg.norm(pos - self._prev_pos))
            limit = max(JUMP_FLOOR_M, 2.0 * MAX_PLAUSIBLE_SPEED_MPS * dt)
            if jump > limit:
                raise EpisodeFault(f"teleport: {jump:.2f} m in {dt:.3f} s (limit {limit:.2f} m)")
        self._prev_t = t
        self._prev_pos = pos
        self._elapsed = t - self._t0

        progress = route_state.progress_m
        self._max_progress = max(self._max_progress, progress)

        if finished and not self._finished:
            total = self._route.total_length_m
            distance = float(np.linalg.norm(pos - self._route.end_xyz))
            if progress < FINISH_PROGRESS_FRACTION * total or distance > FINISH_RADIUS_M:
                raise EpisodeFault(
                    f"early finish flag: progress {progress:.1f}/{total:.1f} m, "
                    f"{distance:.1f} m from finish"
                )
            self._finished = True
            return "finished"

        if self._finished or self._ended:
            return None

        if abs(route_state.lateral_m) > CORRIDOR_M:
            if self._off_since is None:
                self._off_since = t
            if t - self._off_since >= OFF_ROUTE_S - TIME_EPS:
                self._ended = True
                return "off_route"
        else:
            self._off_since = None

        # Stuck: after the starting grace, max progress must gain STUCK_MIN_GAIN_M within the window.
        self._history.append((t, self._max_progress))
        window_start = t - STUCK_WINDOW_S
        while len(self._history) >= 2 and self._history[1][0] <= window_start:
            self._history.popleft()
        grace_done = self._elapsed - STUCK_WINDOW_S >= STUCK_GRACE_S - TIME_EPS
        baseline_t, baseline_progress = self._history[0]
        if grace_done and baseline_t <= window_start + TIME_EPS:
            if self._max_progress - baseline_progress < STUCK_MIN_GAIN_M:
                self._ended = True
                return "stuck"
        return None


class ProgressReward:
    """Reward forward high-water progress, penalize backward movement, and pay the finish once."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._best_m = 0.0
        self._last_m: Optional[float] = None
        self._finish_latched = False

    def step(self, route_state: RouteState, finished: bool, dt: float) -> Tuple[float, Optional[str]]:
        progress = route_state.progress_m
        reward = 0.0

        if self._last_m is not None:
            jump_limit = max(JUMP_FLOOR_M, 2.0 * MAX_PLAUSIBLE_SPEED_MPS * max(dt, 0.0))
            if abs(progress - self._last_m) <= jump_limit:
                gain = max(0.0, progress - self._best_m)
                backward = max(0.0, self._last_m - progress)
                reward += PROGRESS_WEIGHT * gain - BACKWARD_WEIGHT * backward
                self._best_m = max(self._best_m, progress)
                self._last_m = progress
            # Implausible jumps earn nothing and leave the reference unchanged.
        else:
            self._last_m = progress
            self._best_m = max(self._best_m, progress)

        reason = None
        if finished and not self._finish_latched:
            self._finish_latched = True
            reward += FINISH_BONUS
            reason = "finished"

        return float(reward), reason
