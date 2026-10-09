"""Offline checks: packet framing, reconnect, actions, spaces, route projection, reward.

No game, no network beyond localhost, no Modal. Run:
    .\\.venv\\Scripts\\python.exe tests\\self_check.py
"""
import json
import math
import socket
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.env.telemetry import PACKET, TelemetryClient  # noqa: E402
from src.reward.route import (  # noqa: E402
    SMOKE_NO_ROUTE,
    EpisodeFault,
    EpisodeMonitor,
    ProgressReward,
    Route,
    RouteState,
    reward_identity,
)


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


def check_stall_reconnect() -> None:
    """Server accepts but goes silent: the client must reconnect on its own."""
    vals = (10.0, 50.0, 7.0, 8.0, 9.0, 0.0, 1.0, 0.0, 0.0, 3.0, 5000.0)
    raw = PACKET.pack(*vals)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    srv.settimeout(5.0)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()  # first connection: silence, never a byte
        time.sleep(2.0)
        conn.close()
        conn2, _ = srv.accept()  # reconnected client gets data here
        with conn2:
            for _ in range(3):
                conn2.sendall(raw)
                time.sleep(0.02)
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    client = TelemetryClient(port=port, reconnect_delay=0.05, max_reconnects=20,
                             stale_reconnect_s=0.5)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            time.sleep(0.1)
            try:
                if client.latest(max_age_s=1.0).seq >= 1:
                    break
            except Exception:  # noqa: BLE001
                pass
        last = client.latest(max_age_s=1.0)
        assert last.seq >= 1, "client never recovered from a silent stall"
        assert client.reconnects >= 1, "silent stall did not trigger a reconnect"
    finally:
        client.close()
    print("self_check: stall reconnect OK", flush=True)


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


ROUTE_LEN_M = 100
DT = 0.05


def _line_route(tmp: str) -> Route:
    """Straight 100 m route along x at z=0, 1 m spacing, so lateral offset equals z."""
    xs = np.arange(ROUTE_LEN_M + 1, dtype=np.float64)
    points = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1).astype(np.float32)
    path = str(Path(tmp) / "line.npz")
    np.savez(path, points=points, spacing_m=np.float32(1.0))
    return Route(path)


def _path(xs, z=0.0):
    return [np.array([x, 0.0, z], dtype=np.float64) for x in xs]


def _drive(route, monitor, reward, positions, finished=None):
    """Run positions through projection, episode monitor, and reward, as the live step does.

    Returns one (reward, reason) per position. An EpisodeFault from the monitor propagates.
    """
    route.reset()  # projection is windowed around the previous index, so each episode starts from index 0
    out = []
    for k, pos in enumerate(positions):
        fin = bool(finished[k]) if finished is not None else False
        st = route.project(pos, DT)
        reason = monitor.observe(k * DT, pos, st, fin)
        rew, _ = reward.step(st, fin, DT)
        out.append((rew, reason))
    return out


def check_episode_rules() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        route = _line_route(tmp)
        total = route.total_length_m
        assert abs(total - ROUTE_LEN_M) < 1e-3, f"route length wrong: {total}"

        # Forward: 0.5 m per 0.05 s step (10 m/s) for 20 m pays 0.1 per metre, no terminal reason.
        res = _drive(route, EpisodeMonitor(route), ProgressReward(), _path(np.arange(41) * 0.5))
        assert all(reason is None for _, reason in res), "forward run must not end the episode"
        assert abs(sum(r for r, _ in res) - 2.0) < 1e-6, "forward 20 m must pay 2.0"

        # Stationary: no progress. Stuck fires only after the 5 s grace plus the 3 s window (t = 8 s).
        res = _drive(route, EpisodeMonitor(route), ProgressReward(), _path([10.0] * 200))
        fired = [(k * DT, reason) for k, (_, reason) in enumerate(res) if reason is not None]
        assert fired and fired[0][1] == "stuck", "stationary car must end as stuck"
        assert 7.95 <= fired[0][0] <= 8.1, f"stuck must wait for the grace, fired at {fired[0][0]:.2f} s"
        assert all(abs(r) < 1e-9 for r, _ in res), "stationary must pay 0"

        # Oscillation: forward 2 m, back 2 m, forward 2 m, then 2 m of new ground.
        xs = [0, 0.5, 1, 1.5, 2, 1.5, 1, 0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        res = _drive(route, EpisodeMonitor(route), ProgressReward(), _path(xs))
        assert all(reason is None for _, reason in res), "oscillation must not end the episode"
        assert abs(sum(r for r, _ in res) - 0.2) < 1e-6, "oscillation must pay only net new ground"
        assert abs(sum(r for r, _ in res[9:13])) < 1e-9, "re-gaining old ground must pay 0"

        # Backward: positions pulled back by 1 m are penalised, and the penalty is recorded.
        res = _drive(route, EpisodeMonitor(route), ProgressReward(), _path([10.0, 9.0]))
        assert abs(res[1][0] + 0.1) < 1e-9, "backward 1 m must cost 0.1"

        # Teleport: a 190 m jump in one step is an EpisodeFault, not a reward.
        try:
            _drive(route, EpisodeMonitor(route), ProgressReward(), _path([10.0, 200.0]))
        except EpisodeFault as exc:
            assert "teleport" in str(exc), f"wrong fault: {exc}"
        else:
            raise AssertionError("teleport must raise EpisodeFault")

        # Off-route: lateral ramps 0 -> 12 m (ends at t = 0.55 s when |lateral| first exceeds 10 m).
        # The episode must end exactly 1 s later, not before.
        ramp = [(0.5 * k, min(12.0, float(k))) for k in range(60)]
        res = _drive(route, EpisodeMonitor(route), ProgressReward(), [np.array([x, 0.0, z]) for x, z in ramp])
        fired = [(k * DT, reason) for k, (_, reason) in enumerate(res) if reason is not None]
        assert fired and fired[0][1] == "off_route", "holding 12 m off-route must end the episode"
        t_exceed = 11 * DT
        assert 0.95 <= fired[0][0] - t_exceed <= 1.1, f"off_route after {fired[0][0] - t_exceed:.2f} s, want 1 s"

        # Corridor: holding 9 m lateral (inside the 10 m corridor) is never terminal.
        ramp = [(0.5 * k, min(9.0, float(k))) for k in range(120)]
        res = _drive(route, EpisodeMonitor(route), ProgressReward(), [np.array([x, 0.0, z]) for x, z in ramp])
        assert all(reason is None for _, reason in res), "9 m lateral is inside the corridor"

        # Finish gate: finish flag at 104.5 m (4.5 m past the route end) is accepted; the projection
        # clamps progress to the route total, and the bonus pays once.
        route.reset()
        for x in [0.5 * k for k in range(201)] + [104.5]:
            st_end = route.project(np.array([x, 0.0, 0.0]), DT)
        assert abs(st_end.progress_m - total) < 1e-4 and st_end.done, "progress must clamp at route end"
        xs = [0.5 * k for k in range(201)] + [104.5]
        fin = [False] * 201 + [True]
        res = _drive(route, EpisodeMonitor(route), ProgressReward(), _path(xs), finished=fin)
        assert res[-1][1] == "finished", "finish within 10 m of the end must be accepted"
        assert abs(res[-1][0] - 10.0) < 1e-3, f"finish step must pay the 10 bonus, got {res[-1][0]}"

        # Repeated finish flag: no second reason and no second bonus.
        xs = [0.5 * k for k in range(201)] + [104.5, 104.5]
        fin = [False] * 201 + [True, True]
        res = _drive(route, EpisodeMonitor(route), ProgressReward(), _path(xs), finished=fin)
        assert res[-1][1] is None and abs(res[-1][0]) < 1e-9, "repeated finish flag must not re-pay"

        # Early finish is a fault: finish flag at 50 m of 100 m.
        try:
            _drive(route, EpisodeMonitor(route), ProgressReward(), _path([0.5 * k for k in range(101)]),
                   finished=[False] * 100 + [True])
        except EpisodeFault as exc:
            assert "early finish" in str(exc), f"wrong fault: {exc}"
        else:
            raise AssertionError("early finish flag must raise EpisodeFault")

        # Finish flag at episode start is a fault.
        try:
            _drive(route, EpisodeMonitor(route), ProgressReward(), _path([0.0]), finished=[True])
        except EpisodeFault as exc:
            assert "start" in str(exc), f"wrong fault: {exc}"
        else:
            raise AssertionError("finish at start must raise EpisodeFault")

        # Reset: after a finished episode, a cleared route, monitor, and reward start fresh.
        route.reset()
        mon, rew = EpisodeMonitor(route), ProgressReward()
        _drive(route, mon, rew, _path([0.5 * k for k in range(201)] + [104.5]),
               finished=[False] * 201 + [True])
        route.reset()
        mon.reset()
        rew.reset()
        st0 = route.project(np.array([0.0, 0.0, 0.0]), DT)
        assert st0.progress_m < 1.0, "route reset must clear the projection index"
        res = _drive(route, mon, rew, _path([0.0, 0.5]))
        assert all(reason is None for _, reason in res), "new episode after reset must start clean"
    print("self_check: episode rule fixtures OK", flush=True)


def check_identity() -> None:
    import importlib

    sys.path.insert(0, str(REPO / "scripts"))
    bc = importlib.import_module("bootstrap_config")

    # Manifest: a genuine manifest passes; any edit to its identity fields is rejected.
    project = {"observation_schema": "telemetry-route-v1", "hidden_sizes": [256, 256], "route_path": SMOKE_NO_ROUTE}
    manifest = bc.build_manifest(project, smoke=True)
    bc.check_manifest(manifest)
    for field, value in (("reward_sha256", "0" * 64), ("route_sha256", "f" * 64), ("action_order", ["x"])):
        tampered = dict(manifest)
        tampered[field] = value
        try:
            bc.check_manifest(tampered)
        except bc.IdentityError:
            pass
        else:
            raise AssertionError(f"manifest with edited {field} must be rejected")

    # Production validation refuses a declared route hash that does not match the file.
    prod = {
        "route_path": "tracks/v1/route.npz",
        "route_sha256": "0" * 64,
        "reward_sha256": reward_identity(),
    }
    try:
        bc.validate_identity(prod, smoke=False)
    except bc.IdentityError:
        pass
    else:
        raise AssertionError("route hash mismatch must be refused for production")

    # Config reward identity matches the code constants.
    common = json.loads((REPO / "configs" / "common.json").read_text(encoding="utf-8"))
    declared = common["PROJECT"]["reward_sha256"]
    assert declared == reward_identity(), f"configs reward_sha256 {declared} != reward_identity() {reward_identity()}"
    print("self_check: manifest and config identity OK", flush=True)


def main() -> int:
    check_packets()
    check_reconnect()
    check_stall_reconnect()
    check_actions_and_spaces()
    check_reward()
    check_projection()
    check_episode_rules()
    check_identity()
    print("self_check: ALL OFFLINE CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
