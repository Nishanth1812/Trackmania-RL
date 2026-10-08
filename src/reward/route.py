"""Recorded-route projection, telemetry road features, and the simple progress reward.

Coordinates follow the game: x/z is the horizontal plane, y is up. Route points are
resampled at a fixed spacing, so index * spacing_m is arc length in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

SMOKE_NO_ROUTE = "smoke-no-route"
FEATURE_DIM = 15

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
FINISH_BONUS = 100.0


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

    def __init__(self, path: str, rpm_scale: float = 1.0):
        self.path = path
        self.rpm_scale = float(rpm_scale)
        self._dummy = path == SMOKE_NO_ROUTE
        if self._dummy:
            self.spacing_m = 1.0
            self._xz = None
            self._tangents = None
        else:
            # allow_pickle=False: route files come from trusted-but-external sources.
            with np.load(path, allow_pickle=False) as data:
                points = np.asarray(data["points"], dtype=np.float64)
                spacing = float(data["spacing_m"])
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
                raise ValueError(f"route points must be (N>=2, 3), got {points.shape}")
            if not np.all(np.isfinite(points)) or not (spacing > 0.0):
                raise ValueError("route points must be finite and spacing_m must be > 0")
            self.spacing_m = spacing
            self._xz = points[:, [0, 2]]
            self._tangents = self._build_tangents(self._xz)
        self.reset()

    @property
    def is_dummy(self) -> bool:
        return self._dummy

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
        total = (xz.shape[0] - 1) * self.spacing_m

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
