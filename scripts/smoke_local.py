"""Minimal local smoke checks: telemetry view and gamepad controls test.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe scripts\\smoke_local.py telemetry --seconds 15
    .\\.venv\\Scripts\\python.exe scripts\\smoke_local.py controls

Controls test sends [gas, brake, steer] in [-1,1] via vgamepad at 20 Hz and
verifies the inputs appear in telemetry fields 5-7. Keep the Trackmania
window focused and hands off other controllers during the test.
"""
import argparse
import os
import socket
import struct
import sys
import time
from pathlib import Path

PACKET = struct.Struct("<11f")
_COUNTDOWN = int(os.environ.get("SMOKE_COUNTDOWN", "5"))  # run_all_checks.py sets 0 after its own countdown

# Telemetry field indices (TMRL_GrabData 11-float packet)
IDX_SPEED = 0
IDX_STEER = 5
IDX_GAS = 6
IDX_BRAKE = 7


def read_packets(host: str, port: int, seconds: float):
    """Print one telemetry line per second for `seconds`."""
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.settimeout(5)
        print("Telemetry flowing. Drive now (read-only, no inputs sent).", flush=True)
        start = time.monotonic()
        nxt = start
        n = 0
        first_pos = None
        moved = False
        while time.monotonic() - start < seconds:
            raw = bytearray()
            while len(raw) < PACKET.size:
                chunk = sock.recv(PACKET.size - len(raw))
                if not chunk:
                    raise RuntimeError("Telemetry connection closed.")
                raw.extend(chunk)
            v = PACKET.unpack(raw)
            n += 1
            pos = tuple(round(x, 3) for x in v[2:5])
            if first_pos is None:
                first_pos = pos
            moved = moved or (pos != first_pos)
            now = time.monotonic()
            if now >= nxt:
                print(
                    f"+{now-start:4.1f}s speed={v[0]:8.3f} pos={pos} "
                    f"steer={v[IDX_STEER]:6.2f} gas={v[IDX_GAS]:5.2f} brake={v[IDX_BRAKE]:.0f}",
                    flush=True,
                )
                nxt = now + 1.0
        print(f"Captured {n} packets; position changed={moved}.", flush=True)


def cmd_controls() -> int:
    try:
        import vgamepad
        from tmrl.custom.tm.utils.control_gamepad import control_gamepad
    except Exception as e:  # noqa: BLE001
        print(f"Missing gamepad deps: {e}", flush=True)
        return 2

    # Fresh socket for verification (fails fast if plugin not listening).
    try:
        sock = socket.create_connection(("127.0.0.1", 9000), timeout=5)
    except OSError as e:
        print(
            "Cannot connect to telemetry on 127.0.0.1:9000. "
            "In Trackmania: F3 -> Developer -> Reload plugin 'TMRL_GrabData', "
            f"then retry. ({e})",
            flush=True,
        )
        return 2
    sock.settimeout(0.01)

    def hold(action, seconds: float, label: str):
        """Refresh `action` at 20 Hz; sample telemetry DURING the hold.

        Returns (min_steer, max_gas, max_brake, max_speed) peaks seen.
        """
        buf = bytearray()
        min_steer = 0.0
        max_steer = 0.0
        max_gas = 0.0
        max_brake = 0.0
        max_speed = 0.0
        v0 = None
        vlast = None
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            control_gamepad(pad, action)
            try:
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
            except (BlockingIOError, TimeoutError, socket.timeout):
                pass
            while len(buf) >= PACKET.size:
                v = PACKET.unpack(buf[: PACKET.size])
                del buf[: PACKET.size]
                if v0 is None:
                    v0 = v
                vlast = v
                min_steer = min(min_steer, v[IDX_STEER])
                max_steer = max(max_steer, v[IDX_STEER])
                max_gas = max(max_gas, v[IDX_GAS])
                max_brake = max(max_brake, v[IDX_BRAKE])
                max_speed = max(max_speed, v[IDX_SPEED])
            time.sleep(0.05)
        dist = "-"
        if v0 is not None and vlast is not None:
            dist = f"dx={vlast[2]-v0[2]:.1f} dz={vlast[4]-v0[4]:.1f}"
        print(
            f"{label}: cmd gas={action[0]:+.1f} brake={action[1]:+.1f} steer={action[2]:+.1f} "
            f"-> tel steer=[{min_steer:+.2f},{max_steer:+.2f}] gasmax={max_gas:.2f} brakemax={max_brake:.0f} "
            f"spdmax={max_speed:.1f} {dist}",
            flush=True,
        )
        return min_steer, max_steer, max_gas, max_brake, max_speed

    pad = vgamepad.VX360Gamepad()
    for i in range(_COUNTDOWN, 0, -1):
        print(f"Focus Trackmania now... {i}", flush=True)
        time.sleep(1)
    print(
        "Controls test: keep Trackmania focused, car on track, hands off other inputs.",
        flush=True,
    )
    print("Sequence: full throttle 2s -> neutral -> steer left -> steer right -> brake tap.", flush=True)
    print("NOTE: partial gas (~0.6) reads as 0.00 in telemetry (trigger deadzone);", flush=True)
    print("full deflection is required for the gas check.", flush=True)
    try:
        hold([0.0, 0.0, 0.0], 0.5, "neutral0")
        _, _, gas_thr, _, spd_thr = hold([1.0, 0.0, 0.0], 2.0, "throttle")
        hold([0.0, 0.0, 0.0], 0.5, "coast")
        smin_l, _, gas_l, _, _ = hold([1.0, 0.0, -1.0], 1.5, "steer-left")
        _, smax_r, gas_r, _, _ = hold([1.0, 0.0, 1.0], 1.5, "steer-right")
        _, _, _, brake_max, _ = hold([0.0, 1.0, 0.0], 1.0, "brake")
        fails = []
        if gas_thr < 0.5:
            fails.append(f"throttle gasmax={gas_thr:.2f} (<0.5)")
        if spd_thr < 5.0:
            fails.append(f"throttle spdmax={spd_thr:.1f} (<5: car did not move)")
        if smin_l > -0.5:
            fails.append(f"steer-left min={smin_l:+.2f} (>-0.5)")
        if smax_r < 0.5:
            fails.append(f"steer-right max={smax_r:+.2f} (<+0.5)")
        if brake_max < 0.5:
            fails.append("brake flag never set")
        if fails:
            print("FAIL: " + "; ".join(fails), flush=True)
            return 1
        print("PASS: gas/steer-left/steer-right/brake all tracked in telemetry.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
        return 130
    finally:
        try:
            control_gamepad(pad, [0.0, 0.0, 0.0])
        except Exception:  # noqa: BLE001
            pass
        try:
            sock.close()
        except Exception:  # noqa: BLE001
            pass
        print("Controls neutralized.", flush=True)


def cmd_reset(cycles: int) -> int:
    import vgamepad
    from tmrl.custom.tm.utils.control_gamepad import (
        control_gamepad,
        gamepad_close_finish_pop_up_tm20,
        gamepad_reset,
    )

    try:
        from tmrl.custom.tm.utils.control_keyboard import keyres as keyboard_reset
    except Exception:  # noqa: BLE001
        keyboard_reset = None

    try:
        sock = socket.create_connection(("127.0.0.1", 9000), timeout=5)
    except OSError as e:
        print(f"No telemetry on 9000 ({e}). Reload TMRL_GrabData, then retry.", flush=True)
        return 2
    sock.settimeout(0.01)
    pad = vgamepad.VX360Gamepad()
    buf = bytearray()

    def drain_latest():
        nonlocal buf
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
        except (BlockingIOError, TimeoutError, socket.timeout):
            pass
        latest = None
        while len(buf) >= PACKET.size:
            latest = PACKET.unpack(buf[: PACKET.size])
            del buf[: PACKET.size]
        return latest

    def fresh_sample():
        # Block briefly for one fresh packet.
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            v = drain_latest()
            if v is not None:
                return v
            control_gamepad(pad, [0.0, 0.0, 0.0])
            time.sleep(0.05)
        return None

    def drive_away(seconds=2.5):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            control_gamepad(pad, [0.8, 0.0, 0.0])
            drain_latest()
            time.sleep(0.05)

    def do_reset_b():
        gamepad_close_finish_pop_up_tm20(pad)
        time.sleep(0.3)
        gamepad_reset(pad)
        time.sleep(0.3)
        control_gamepad(pad, [0.0, 0.0, 0.0])

    def do_reset_del():
        if keyboard_reset is None:
            return False
        gamepad_close_finish_pop_up_tm20(pad)
        time.sleep(0.3)
        keyboard_reset()
        time.sleep(0.3)
        return True

    for i in range(_COUNTDOWN, 0, -1):
        print(f"Focus Trackmania now... {i}", flush=True)
        time.sleep(1)

    ok = 0
    spawn = None
    try:
        # The car may be parked anywhere (e.g. after the controls test); restart once so the
        # spawn reference is the real full-race start, not the current position.
        print("Initial restart to capture the real spawn...", flush=True)
        do_reset_b()
        time.sleep(3.0)
        for cycle in range(1, cycles + 1):
            v = fresh_sample()
            if v is None:
                print(f"[{cycle}] FAIL: no telemetry", flush=True)
                continue
            if spawn is None:
                spawn = (v[2], v[3], v[4])
                print(f"Spawn ref {tuple(round(x,1) for x in spawn)} speed={v[0]:.2f}", flush=True)
            print(f"[{cycle}] driving away...", flush=True)
            drive_away()
            away = fresh_sample()
            if away is not None:
                import math

                d = math.dist((away[2], away[3], away[4]), spawn)
                print(f"[{cycle}] away +{d:.1f}m, resetting with B...", flush=True)
            do_reset_b()
            time.sleep(2.0)  # countdown guess; tuned after measurement
            v = fresh_sample()
            if v is None:
                print(f"[{cycle}] FAIL: no packet after reset", flush=True)
                continue
            import math

            d = math.dist((v[2], v[3], v[4]), spawn)
            finished = v[8]
            passed = d <= 2.0 and v[0] < 0.5 and finished == 0
            print(
                f"[{cycle}] after B: d={d:.2f}m speed={v[0]:.2f} finish={finished:.0f} "
                f"-> {'PASS' if passed else 'CHECKPOINT-RESPAWN?'}",
                flush=True,
            )
            if not passed and keyboard_reset is not None:
                print(f"[{cycle}] B failed, trying keyboard Delete...", flush=True)
                do_reset_del()
                time.sleep(2.0)
                v = fresh_sample()
                if v is not None:
                    d = math.dist((v[2], v[3], v[4]), spawn)
                    passed = d <= 2.0 and v[0] < 0.5 and v[8] == 0
                    print(f"[{cycle}] after Del: d={d:.2f}m -> {'PASS' if passed else 'FAIL'}", flush=True)
            if passed:
                ok += 1
        print(f"Reset: {ok}/{cycles} returned to spawn (<=2m, speed<0.5, finish cleared).", flush=True)
        return 0 if ok == cycles else 1
    finally:
        try:
            control_gamepad(pad, [0.0, 0.0, 0.0])
        except Exception:  # noqa: BLE001
            pass
        sock.close()


def cmd_environment(episodes: int) -> int:
    import math

    import numpy as np

    from src.env.interface import TelemetryInterface
    from src.env.telemetry import TelemetryClient, TelemetryError

    # Part 1: offline shape/dtype/default-action contract (no game needed).
    itf = TelemetryInterface(smoke=True)
    obs_space, act_space = itf.get_observation_space(), itf.get_action_space()
    assert obs_space[0].shape == (15,) and obs_space[0].dtype == np.float32
    assert act_space.shape == (3,) and act_space.dtype == np.float32
    dflt = itf.get_default_action()
    assert dflt.shape == (3,) and bool(np.all(np.isfinite(dflt)))
    (obs0, info0), (obs1, rew1, term1, info1) = itf.reset(), itf.get_obs_rew_terminated_info()
    assert obs0[0].shape == (15,) and obs1[0].shape == (15,)
    assert isinstance(rew1, float) and isinstance(term1, bool)
    print("offline contract: spaces (15,)+(3,) float32, default action, reset/step shapes OK", flush=True)

    # Part 2: live episodes against the game.
    try:
        client = TelemetryClient()
    except Exception as e:  # noqa: BLE001
        print(f"No telemetry on 9000 ({e}). Reload TMRL_GrabData, then retry.", flush=True)
        return 2
    import vgamepad
    from tmrl.custom.tm.utils.control_gamepad import control_gamepad, gamepad_reset

    pad = vgamepad.VX360Gamepad()
    for i in range(_COUNTDOWN, 0, -1):
        print(f"Focus Trackmania now... {i}", flush=True)
        time.sleep(1)
    ok = 0
    try:
        for ep in range(1, episodes + 1):
            control_gamepad(pad, [0.0, 0.0, 0.0])
            try:
                pre = client.latest()
            except TelemetryError as e:
                print(f"[{ep}] FAIL: pre-reset {e}", flush=True)
                continue
            gamepad_reset(pad)
            # Post-reset re-sync: map reload/countdown can stall the stream for a
            # bit; the 250 ms freshness rule applies DURING control, not across
            # a reset boundary (PLAN reset_timeout_s = 10 s).
            t0 = None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                control_gamepad(pad, [0.0, 0.0, 0.0])
                try:
                    t0 = client.latest()
                    break
                except TelemetryError:
                    time.sleep(0.1)
            if t0 is None:
                print(f"[{ep}] FAIL: no fresh telemetry within 10 s of reset", flush=True)
                continue
            # Require the stream to be settled (3 fresh reads in a row) before driving.
            settled = True
            for _ in range(3):
                time.sleep(0.1)
                try:
                    client.latest()
                except TelemetryError:
                    settled = False
                    break
            if not settled:
                print(f"[{ep}] FAIL: stream unsettled after reset", flush=True)
                continue
            reset_moved = math.dist((t0.pos_x, t0.pos_y, t0.pos_z), (pre.pos_x, pre.pos_y, pre.pos_z))
            start, max_spd, max_gas, n = (t0.pos_x, t0.pos_y, t0.pos_z), 0.0, 0.0, 0
            end = time.monotonic() + 3.0
            live_ok = True
            while time.monotonic() < end:
                control_gamepad(pad, [0.8, 0.0, 0.0])
                try:
                    t = client.latest()
                except TelemetryError:
                    live_ok = False
                    break
                if not all(math.isfinite(x) for x in (t.speed, t.pos_x, t.pos_y, t.pos_z)):
                    live_ok = False
                    break
                max_spd = max(max_spd, t.speed)
                max_gas = max(max_gas, t.gas)
                n += 1
                time.sleep(0.05)
            control_gamepad(pad, [0.0, 0.0, 0.0])
            try:
                t1 = client.latest()
            except TelemetryError:
                print(f"[{ep}] FAIL: stream died at episode end (live_ok={live_ok} steps={n})", flush=True)
                continue
            dist = math.dist((t1.pos_x, t1.pos_y, t1.pos_z), start)
            # ~100 ms per loop iteration (pad update + sleep granularity), so ~30
            # steps per 3 s window; the gate is sustained fresh control, not a count.
            passed = live_ok and n >= 20 and dist > 1.0
            ok += passed
            print(f"[{ep}] {'PASS' if passed else 'FAIL'}: reset_moved={reset_moved:.1f}m gas_seen={max_gas:.2f} max_speed={max_spd:.1f} moved={dist:.1f}m steps={n} live_ok={live_ok}", flush=True)
        print(f"Environment: {ok}/{episodes} live episodes OK.", flush=True)
        return 0 if ok == episodes else 1
    finally:
        try:
            control_gamepad(pad, [0.0, 0.0, 0.0])
        except Exception:  # noqa: BLE001
            pass
        client.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("telemetry")
    t.add_argument("--seconds", type=float, default=15.0)
    t.add_argument("--host", default="127.0.0.1")
    t.add_argument("--port", type=int, default=9000)
    sub.add_parser("controls")
    r = sub.add_parser("reset")
    r.add_argument("--cycles", type=int, default=5)
    e = sub.add_parser("environment")
    e.add_argument("--episodes", type=int, default=10)
    a = p.parse_args(argv)
    if a.cmd == "telemetry":
        read_packets(a.host, a.port, a.seconds)
        return 0
    if a.cmd == "reset":
        return cmd_reset(a.cycles)
    if a.cmd == "environment":
        return cmd_environment(a.episodes)
    return cmd_controls()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
