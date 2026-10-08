"""Run every available pre-training check in one go and print a pass/fail table.

Usage (from repo root), with Trackmania focused on a practice track:
    .\\.venv\\Scripts\\python.exe scripts\\run_all_checks.py                # everything available
    .\\.venv\\Scripts\\python.exe scripts\\run_all_checks.py --skip live    # offline + remote only
    .\\.venv\\Scripts\\python.exe scripts\\run_all_checks.py --only offline

Stages: offline (no game), live (needs game focused, ~3-6 min), remote (Modal; needs deployment/modal_app.py).
A stage whose script does not exist yet reports NOT IMPLEMENTED, which counts as not passed.
Writes logs/checks-<UTC time>.json. Exit code 0 only if every selected check passed.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SMOKE_LOCAL = ["scripts/smoke_local.py"]


def checks(reset_cycles: int, episodes: int):
    """(stage, milestone, name, argv or None if not implemented)."""
    modal_app = (ROOT / "deployment" / "modal_app.py").exists()
    modal = lambda fn, *extra: [  # noqa: E731
        "-m", "modal", "run", f"deployment/modal_app.py::{fn}", *extra
    ] if modal_app else None
    return [
        ("offline", "M1", "packet decode self-check", ["tests/self_check.py"]),
        ("offline", "M4", "synthetic spaces + SAC update", ["scripts/smoke_pipeline.py", "local", "--updates", "1"]),
        ("live", "M1", "telemetry stream (15 s)", SMOKE_LOCAL + ["telemetry", "--seconds", "15"]),
        ("live", "M2", "gamepad controls", SMOKE_LOCAL + ["controls"]),
        ("live", "M3", f"reset x{reset_cycles}", SMOKE_LOCAL + ["reset", "--cycles", str(reset_cycles)]),
        ("live", "M4", f"environment x{episodes} episodes", SMOKE_LOCAL + ["environment", "--episodes", str(episodes)]),
        ("remote", "M5", "Modal TLS echo through tunnel", modal("echo_server")),
        ("remote", "M6", "Modal L4 GPU check", modal("gpu_check")),
        ("remote", "M6", "Modal 1,000 SAC updates + checkpoint restart", modal("benchmark", "--updates", "1000")),
        ("remote", "M7", "weights round-trip Windows <-> Modal", ["scripts/smoke_pipeline.py", "remote", "--updates", "10"]
            if modal_app else None),
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["offline", "live", "remote"])
    p.add_argument("--skip", choices=["offline", "live", "remote"], action="append", default=[])
    p.add_argument("--reset-cycles", type=int, default=100, help="M3 requires 100/100")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--countdown", type=int, default=8, help="seconds to focus the game before live checks")
    p.add_argument("--stop-on-fail", action="store_true")
    a = p.parse_args(argv)

    selected = [c for c in checks(a.reset_cycles, a.episodes)
                if (a.only is None or c[0] == a.only) and c[0] not in a.skip]
    results = []
    counted_down = False
    for stage, ms, name, argv_ in selected:
        if argv_ is None:
            results.append({"stage": stage, "milestone": ms, "name": name, "status": "NOT IMPLEMENTED", "seconds": 0})
            print(f"[{ms}] {name}: NOT IMPLEMENTED", flush=True)
            continue
        env = dict(os.environ)
        if stage == "live":
            if not counted_down:
                for i in range(a.countdown, 0, -1):
                    print(f"Focus Trackmania now (live checks start in {i}s)...", flush=True)
                    time.sleep(1)
                counted_down = True
            env["SMOKE_COUNTDOWN"] = "0"
        print(f"\n=== [{ms}] {name} ===", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run([PY, *argv_], cwd=ROOT, env=env)
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results.append({"stage": stage, "milestone": ms, "name": name, "status": status,
                        "seconds": round(time.monotonic() - t0, 1), "returncode": proc.returncode})
        if status == "FAIL" and a.stop_on_fail:
            break

    print("\n" + "=" * 64)
    for r in results:
        print(f"{r['status']:<16} {r['milestone']:<3} {r['name']} ({r['seconds']}s)")
    passed = sum(r["status"] == "PASS" for r in results)
    print(f"{passed}/{len(results)} checks passed")

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    out = log_dir / f"checks-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
