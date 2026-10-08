"""Minimal local smoke checks: telemetry view and gamepad controls test.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe scripts\\smoke_local.py telemetry --seconds 15
    .\\.venv\\Scripts\\python.exe scripts\\smoke_local.py controls

Controls test sends [gas, brake, steer] in [-1,1] via vgamepad at 20 Hz and
verifies the inputs appear in telemetry fields 5-7. Keep the Trackmania
window focused and hands off other controllers during the test.
"""
import argparse
import socket
import struct
import sys
import time
from pathlib import Path

PACKET = struct.Struct("<11f")

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
        """Refresh `action` at 20 Hz; sample telemetry DURING the hold."""
        buf = bytearray()
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
                max_steer = v[IDX_STEER]
                max_gas = max(max_gas, v[IDX_GAS])
                max_brake = max(max_brake, v[IDX_BRAKE])
                max_speed = max(max_speed, v[IDX_SPEED])
            time.sleep(0.05)
        dist = "-"
        if v0 is not None and vlast is not None:
            dist = f"dx={vlast[2]-v0[2]:.1f} dz={vlast[4]-v0[4]:.1f}"
        print(
            f"{label}: cmd gas={action[0]:+.1f} brake={action[1]:+.1f} steer={action[2]:+.1f} "
            f"-> tel steer={max_steer:+.2f} gasmax={max_gas:.2f} brakemax={max_brake:.0f} "
            f"spdmax={max_speed:.1f} {dist}",
            flush=True,
        )
        return vlast

    pad = vgamepad.VX360Gamepad()
    for i in (5, 4, 3, 2, 1):
        print(f"Focus Trackmania now... {i}", flush=True)
        time.sleep(1)
    print(
        "Controls test: keep Trackmania focused, car on track, hands off other inputs.",
        flush=True,
    )
    print("Sequence: throttle 2s -> neutral -> steer left -> steer right -> brake tap.", flush=True)
    try:
        hold([0.0, 0.0, 0.0], 0.5, "neutral0")
        hold([0.6, 0.0, 0.0], 2.0, "throttle")
        hold([0.0, 0.0, 0.0], 0.5, "coast")
        hold([0.3, 0.0, -0.7], 1.5, "steer-left")
        hold([0.3, 0.0, 0.7], 1.5, "steer-right")
        hold([0.0, 0.8, 0.0], 1.0, "brake")
        print("PASS: inputs sent; check that telemetry steer/gas/brake tracked commands.", flush=True)
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

    for i in (5, 4, 3, 2, 1):
        print(f"Focus Trackmania now... {i}", flush=True)
        time.sleep(1)

    ok = 0
    spawn = None
    try:
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
    a = p.parse_args(argv)
    if a.cmd == "telemetry":
        read_packets(a.host, a.port, a.seconds)
        return 0
    if a.cmd == "reset":
        return cmd_reset(a.cycles)
    return cmd_controls()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
