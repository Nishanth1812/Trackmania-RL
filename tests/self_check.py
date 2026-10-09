"""Offline checks: packet framing, reconnect, actions, spaces, route projection, reward.

No game, no network beyond localhost, no Modal. Run:
    .\\.venv\\Scripts\\python.exe tests\\self_check.py
"""
import math
import socket
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.telemetry import PACKET, TelemetryClient  # noqa: E402
from src.reward.route import ProgressReward, Route, RouteState  # noqa: E402


def _close(got: tuple, want: tuple) -> bool:
    # PACKET is float32, so decoded values are float32-rounded; compare with tolerance.
    return len(got) == len(want) and all(
        math.isclose(g, w, rel_tol=1e-6, abs_tol=1e-6) for g, w in zip(got, want)
    )


def check_packets() -> None:
    vals = (12.5, 100.0, 1.0, 2.0, 3.0, 0.1, 0.8, 0.0, 0.0, 3.0, 6000.0)
    raw = PACKET.pack(*vals)
    assert _close(PACKET.unpack(raw), vals), "round-trip failed"

    # Real fragmentation: feed 3 concatenated packets in 7-byte chunks through a
    # bytearray buffer exactly like the reader loop, then decode all.
    stream = raw + raw + raw
    buf = bytearray()
    got = []
    for i in range(0, len(stream), 7):
        buf.extend(stream[i : i + 7])
        while len(buf) >= PACKET.size:
            got.append(PACKET.unpack(bytes(buf[: PACKET.size])))
            del buf[: PACKET.size]
    assert len(got) == 3 and all(_close(g, vals) for g in got), "fragmented feed failed"
    assert len(buf) == 0, "buffer not drained"

    # A packet split across two reads decodes once complete.
    buf = bytearray(raw[:20])
    assert len(buf) < PACKET.size
    buf.extend(raw[20:])
    assert _close(PACKET.unpack(bytes(buf[: PACKET.size])), vals), "split failed"

    # NaN rejected.
    bad = PACKET.pack(*((float("nan"),) + vals[1:]))
    assert any(not math.isfinite(x) for x in PACKET.unpack(bad)), "NaN check failed"
    print("self_check: packet framing OK", flush=True)


def check_reconnect() -> None:
    """Fake server: send packets, drop the connection (EOF), accept again.

    The client thread must survive the drop and resume with fresh packets.
    """
    vals = (10.0, 50.0, 7.0, 8.0, 9.0, 0.0, 1.0, 0.0, 0.0, 3.0, 5000.0)
    raw = PACKET.pack(*vals)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    srv.settimeout(5.0)
    port = srv.getsockname()[1]

    def serve(rounds: int):
        for _ in range(rounds):
            conn, _ = srv.accept()
            with conn:
                for _ in range(3):
                    conn.sendall(raw)
                    time.sleep(0.02)
            # exiting `with` closes -> client sees EOF, must reconnect
        srv.close()

    t = threading.Thread(target=serve, args=(2,), daemon=True)
    t.start()
    client = TelemetryClient(port=port, reconnect_delay=0.05, max_reconnects=20)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            time.sleep(0.1)
            try:
                if client.latest(max_age_s=1.0).seq >= 5:
                    break
            except Exception:  # noqa: BLE001
                pass
        last = client.latest(max_age_s=1.0)
        assert last.seq >= 5, f"client did not survive EOF (seq={last.seq})"
        assert client._thread.is_alive(), "reader thread died on EOF"
    finally:
        client.close()
    print("self_check: EOF reconnect OK", flush=True)


def check_actions_and_spaces() -> None:
    from src.env.interface import ACTION_DIM, OBS_DIM, TelemetryInterface

    itf = TelemetryInterface(smoke=True)
    ok = itf._validate_action([0.5, -0.2, 0.9])
    assert ok.shape == (3,)
    boundary = itf._validate_action([1.0, -1.0, 1.0])
    assert np.all(np.abs(boundary) <= 1.0)
    clipped = itf._validate_action([1.04, 0.0, 0.0])
    assert float(clipped[0]) == 1.0, "small overshoot must clip, not raise"
    for bad in (
        [float("nan"), 0.0, 0.0],
        [0.0, float("inf"), 0.0],
        [0.0, 0.0],
        [1.5, 0.0, 0.0],
    ):
        try:
            itf._validate_action(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"action {bad} must raise ValueError")
    dflt = itf.get_default_action()
    assert dflt.shape == (3,) and dflt.dtype == np.float32 and bool(np.all(dflt == 0))

    obs_space, act_space = itf.get_observation_space(), itf.get_action_space()
    assert obs_space[0].shape == (OBS_DIM,) == (15,)
    assert obs_space[0].dtype == np.float32 and act_space.dtype == np.float32
    assert np.all(obs_space[0].low == np.array([0.0] * 3 + [-1.0] * 11 + [0.0], dtype=np.float32))
    assert np.all(obs_space[0].high == np.ones(15, dtype=np.float32))

    from src.pipeline import build_spaces

    actor_obs, actor_act = build_spaces()
    assert actor_obs[0].shape == (15,) and actor_act.shape == (ACTION_DIM,)
    assert np.all(actor_obs[0].low == obs_space[0].low) and np.all(actor_obs[0].high == obs_space[0].high)
    assert len(actor_obs) == 3, "actor observation must be (base, act-hist, act-hist)"
    print("self_check: actions + spaces OK", flush=True)


def _state(progress: float) -> RouteState:
    return RouteState(index=0, progress_m=progress, lateral_m=0.0, done=False)


def check_reward() -> None:
    r = ProgressReward()
    dt = 0.05
    assert r.step(_state(0.0), False, dt) == (0.0, None)  # first step sets the reference
    rew, _ = r.step(_state(2.0), False, dt)
    assert rew == 0.2, f"forward +2m must pay +0.2, got {rew}"
    rew, _ = r.step(_state(2.0), False, dt)
    assert rew == 0.0, "stationary must pay 0"
    rew, _ = r.step(_state(1.0), False, dt)
    assert rew == -0.1, f"backward -1m must cost -0.1, got {rew}"
    rew, _ = r.step(_state(2.0), False, dt)
    assert rew == 0.0, "re-gaining old ground must pay 0 (no oscillation farming)"
    rew, _ = r.step(_state(200.0), False, dt)
    assert rew == 0.0, "teleport jump must pay 0"
    rew, reason = r.step(_state(3.0), True, dt)
    assert rew == 10.1 and reason == "finished", f"finish must pay gain+10 once, got {rew}"
    rew, reason = r.step(_state(4.0), True, dt)
    assert rew == 0.1 and reason is None, "repeated finish flag must not re-pay the bonus"
    r.reset()
    rew, reason = r.step(_state(0.0), True, dt)
    assert rew == 10.0 and reason == "finished", "reset must clear the finish latch"
    print("self_check: reward fixtures OK", flush=True)


def check_projection() -> None:
    xs = np.arange(10, dtype=np.float64)
    points = np.stack([xs, np.zeros(10), np.zeros(10)], axis=1).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "route.npz")
        np.savez(path, points=points, spacing_m=np.float32(1.0))
        route = Route(path)
        st = route.project(np.array([4.0, 0.0, 0.2]), 0.05)
        assert abs(st.progress_m - 4.0) < 0.05, f"along-track progress wrong: {st.progress_m}"
        assert abs(st.lateral_m - 0.2) < 0.05, f"lateral offset wrong: {st.lateral_m}"
        st2 = route.project(np.array([6.5, 0.0, -0.3]), 0.05)
        assert st2.progress_m > st.progress_m and abs(st2.lateral_m + 0.3) < 0.05
        route.reset()
        st3 = route.project(np.array([0.5, 0.0, 0.0]), 0.05)
        assert st3.index <= 1 and st3.progress_m < 1.0, "reset must clear the projection index"
    print("self_check: route projection OK", flush=True)


def main() -> int:
    check_packets()
    check_reconnect()
    check_actions_and_spaces()
    check_reward()
    check_projection()
    print("self_check: ALL OFFLINE CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
