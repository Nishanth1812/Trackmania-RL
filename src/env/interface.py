"""rtgym interface for TrackMania driven by the TMRL_GrabData telemetry stream.

Observations and rewards come from the recorded route (src/reward/route.py); controls are
sent through a virtual Xbox 360 pad. Windows-only dependencies (vgamepad, tmrl's gamepad
helpers) are imported lazily inside methods so this module imports on any platform.
"""
import time

import numpy as np
from gymnasium import spaces
from rtgym import RealTimeGymInterface

from src.env.telemetry import TelemetryClient
from src.reward.route import ProgressReward, Route

OBS_DIM = 15
ACTION_DIM = 3
ACTION_OVERSHOOT = 1.05
CONTROL_HZ = 20
RESET_WAIT_S = 2.0
FALLBACK_DT_S = 1.0 / CONTROL_HZ


class TelemetryInterface(RealTimeGymInterface):
    """Maps telemetry and the recorded route to an rtgym observation; sends [gas, brake, steer]."""

    def __init__(self, smoke=False, route_path="smoke-no-route"):
        self.smoke = smoke
        self.route_path = route_path
        self._route = Route(route_path)
        self._reward = ProgressReward()
        self._client = None
        self._gamepad = None
        self._prev_pos = None
        self._prev_time = None
        if not smoke:
            self._client = TelemetryClient()

    # --- helpers -----------------------------------------------------------

    def _get_gamepad(self):
        if self._gamepad is None:
            import vgamepad as vg  # Windows-only; lazy on purpose
            self._gamepad = vg.VX360Gamepad()
        return self._gamepad

    @staticmethod
    def _zero_obs():
        return [np.zeros(OBS_DIM, dtype=np.float32)]

    def _reset_episode_state(self):
        self._route.reset()
        self._reward.reset()
        self._prev_pos = None
        self._prev_time = None

    def _velocity_and_dt(self, t):
        # Horizontal velocity by finite difference between consecutive telemetry samples.
        pos = np.array([t.pos_x, t.pos_z], dtype=np.float64)
        dt = FALLBACK_DT_S
        velocity = np.zeros(2, dtype=np.float64)
        if self._prev_pos is not None:
            dt_measured = t.received_monotonic - self._prev_time
            if dt_measured > 1e-3:
                dt = dt_measured
                velocity = (pos - self._prev_pos) / dt
        self._prev_pos = pos
        self._prev_time = t.received_monotonic
        return velocity, dt

    @staticmethod
    def _validate_action(control):
        action = np.asarray(control, dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"action contains NaN/Inf: {action}")
        if np.any(np.abs(action) > ACTION_OVERSHOOT):
            raise ValueError(f"action outside [-{ACTION_OVERSHOOT}, {ACTION_OVERSHOOT}]: {action}")
        return np.clip(action, -1.0, 1.0)

    # --- rtgym API ---------------------------------------------------------

    def send_control(self, control):
        action = self._validate_action(control)
        if self.smoke:
            return
        from tmrl.custom.tm.utils.control_gamepad import control_gamepad
        control_gamepad(self._get_gamepad(), action)

    def neutralize(self):
        self.send_control(self.get_default_action())

    def reset(self, seed=None, options=None):
        if not self.smoke:
            from tmrl.custom.tm.utils.control_gamepad import gamepad_reset
            self.neutralize()
            gamepad_reset(self._get_gamepad())
            time.sleep(RESET_WAIT_S)
        self._reset_episode_state()
        return self._zero_obs(), {}

    def wait(self):
        time.sleep(FALLBACK_DT_S)

    def get_obs_rew_terminated_info(self):
        if self.smoke:
            return self._zero_obs(), 0.0, False, {}
        t = self._client.latest()
        velocity, dt = self._velocity_and_dt(t)
        position = np.array([t.pos_x, t.pos_y, t.pos_z], dtype=np.float64)
        route_state = self._route.project(position, dt)
        telemetry = {
            "speed_mps": t.speed,
            "gear": t.gear,
            "rpm": t.rpm,
            "vx_mps": velocity[0],
            "vz_mps": velocity[1],
        }
        obs = self._route.features(telemetry, route_state)
        finished = bool(t.finish > 0.5)
        reward, _reason = self._reward.step(route_state, finished, dt)
        info = {"seq": t.seq, "progress_m": route_state.progress_m, "lateral_m": route_state.lateral_m}
        return [obs], reward, finished, info

    def get_observation_space(self):
        low = np.array([0.0] * 3 + [-1.0] * 11 + [0.0], dtype=np.float32)
        high = np.array([1.0] * 3 + [1.0] * 11 + [1.0], dtype=np.float32)
        return spaces.Tuple((spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32),))

    def get_action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def get_default_action(self):
        return np.zeros(ACTION_DIM, dtype=np.float32)

    def close(self):
        if self._gamepad is not None:
            try:
                self.neutralize()
            finally:
                self._gamepad = None
        if self._client is not None:
            self._client.close()
            self._client = None
