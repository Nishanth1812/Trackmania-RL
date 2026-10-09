"""Offline replay of a recorded raw lap through the REAL TelemetryInterface.

Feeds the raw npz samples (t, x, y, z, speed, finish) through SequenceTelemetry into
TelemetryInterface, so observation, progress projection, reward, and episode rules run exactly
as in live use. No game, no socket, no gamepad. Gear and rpm are not in the raw file and are
fed as 0; the rpm and gear observation features are therefore constant in this replay.

    .\\.venv\\Scripts\\python.exe scripts\\replay_raw_lap.py
"""
import argparse
import datetime as dt_mod
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.env.interface import TelemetryInterface  # noqa: E402
from src.env.telemetry import SequenceTelemetry, Telemetry  # noqa: E402
from src.reward.route import (  # noqa: E402
    BACKWARD_WEIGHT,
    CORRIDOR_M,
    JUMP_FLOOR_M,
    MAX_PLAUSIBLE_SPEED_MPS,
    EpisodeFault,
    Route,
)

DEFAULT_ROUTE = REPO_ROOT / "tracks" / "v1" / "route.npz"
DEFAULT_RAW = REPO_ROOT / "tracks" / "v1" / "raw" / "raw_20261009T170848Z.npz"
DEFAULT_META = REPO_ROOT / "tracks" / "v1" / "metadata.json"
STATIONARY_MPS = 0.5


def load_raw(path: Path):
    with np.load(path, allow_pickle=False) as raw:
        keys = ("t", "x", "y", "z", "speed", "finish")
        missing = [k for k in keys if k not in raw.files]
        if missing:
            raise KeyError(f"raw npz missing keys: {missing}")
        return {k: np.asarray(raw[k], dtype=np.float64) for k in keys}


def to_samples(raw):
    n = raw["t"].shape[0]
    return [
        Telemetry(
            speed=float(raw["speed"][i]), distance=0.0,
            pos_x=float(raw["x"][i]), pos_y=float(raw["y"][i]), pos_z=float(raw["z"][i]),
            steer=0.0, gas=0.0, brake=0.0, finish=float(raw["finish"][i]),
            gear=0.0, rpm=0.0, seq=i + 1, received_monotonic=float(raw["t"][i]),
        )
        for i in range(n)
    ]


def run_replay(route_path: Path, raw_path: Path, expected_sha: str):
    raw = load_raw(raw_path)
    samples = to_samples(raw)
    source = SequenceTelemetry(samples)
    itf = TelemetryInterface(route_path=str(route_path), route_sha256=expected_sha, telemetry_source=source)
    route_ref = Route(str(route_path), expected_sha256=expected_sha)
    space = itf.get_observation_space()[0]
    itf.reset()

    rows = []
    fault = None
    for i in range(len(samples)):
        try:
            obs_list, reward, terminated, info = itf.get_obs_rew_terminated_info()
        except EpisodeFault as exc:
            fault = {"index": i, "message": str(exc)}
            break
        obs = obs_list[0]
        in_space = bool(np.all(np.isfinite(obs)) and np.all(obs >= space.low) and np.all(obs <= space.high))
        rows.append({
            "i": i, "obs": obs, "in_space": in_space, "reward": reward, "terminated": terminated,
            "reason": info["terminal_reason"], "progress": info["progress_m"],
            "lateral": info["lateral_m"], "elapsed": info["elapsed_s"],
        })
    return raw, route_ref, rows, fault


def analyse(raw, route_ref, rows, fault):
    t = raw["t"]
    n = t.shape[0]
    total = route_ref.total_length_m
    prog = np.array([r["progress"] for r in rows])
    lat = np.array([r["lateral"] for r in rows])
    rew = np.array([r["reward"] for r in rows])
    speed = raw["speed"][: len(rows)]
    dts = np.diff(t)
    d_prog = np.diff(prog)

    # Teleport gate, recomputed independently from raw 3D positions (same rule as EpisodeMonitor).
    pos = np.stack([raw["x"], raw["y"], raw["z"]], axis=1)
    jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    limits = np.maximum(JUMP_FLOOR_M, 2.0 * MAX_PLAUSIBLE_SPEED_MPS * np.maximum(dts, 0.0))
    tele_trips = np.nonzero(jumps > limits)[0]
    zero_dt = np.nonzero(dts <= 1e-9)[0]
    zero_dt_jump_max = float(jumps[zero_dt].max()) if zero_dt.size else 0.0

    gap_idx = int(np.argmax(dts))
    gap = {
        "index": gap_idx, "dt_s": float(dts[gap_idx]),
        "pos_change_m": float(jumps[gap_idx]),
        "horizontal_change_m": float(math.hypot(raw["x"][gap_idx + 1] - raw["x"][gap_idx], raw["z"][gap_idx + 1] - raw["z"][gap_idx])),
        "speed_before": float(raw["speed"][gap_idx]), "speed_after": float(raw["speed"][gap_idx + 1]),
    }
    if gap_idx + 1 < len(rows):
        gap["progress_before_m"] = float(prog[gap_idx])
        gap["progress_after_m"] = float(prog[gap_idx + 1])
        gap["progress_delta_m"] = float(prog[gap_idx + 1] - prog[gap_idx])
        gap["reward_at_gap_end"] = float(rew[gap_idx + 1])

    stationary = (speed < STATIONARY_MPS) & (np.abs(np.concatenate([[0.0], d_prog])) < 1e-6)
    finished_idx = [r["i"] for r in rows if r["reason"] == "finished"]
    reasons = {}
    for r in rows:
        if r["reason"] is not None:
            reasons.setdefault(r["reason"], []).append(r["i"])

    backward = d_prog < -1e-6
    report = {
        "samples_total": int(n),
        "samples_processed": len(rows),
        "fault": fault,
        "route_total_length_m": float(total),
        "route_end_xyz": [float(v) for v in route_ref.end_xyz],
        "in_space_all": bool(all(r["in_space"] for r in rows)),
        "progress_start_m": float(prog[0]),
        "progress_final_m": float(prog[-1]),
        "progress_max_fraction": float(prog.max() / total),
        "progress_final_fraction": float(prog[-1] / total),
        "progress_backward_steps": int(backward.sum()),
        "progress_backward_worst_m": float(-d_prog.min()) if d_prog.size else 0.0,
        "progress_largest_forward_step_m": float(d_prog.max()) if d_prog.size else 0.0,
        "progress_largest_forward_step_index": int(np.argmax(d_prog)) if d_prog.size else -1,
        "stationary_samples": int(stationary.sum()),
        "stationary_reward_abs_max": float(np.abs(rew[stationary[: len(rew)]]).max()) if stationary.any() else 0.0,
        "reward_total": float(rew.sum()),
        "finish_samples_reward_events": finished_idx,
        "finish_bonus_count": len(finished_idx),
        "terminal_reasons": {k: {"count": len(v), "first_index": v[0]} for k, v in reasons.items()},
        "max_abs_corridor_deviation_m": float(np.abs(lat).max()),
        "corridor_limit_m": CORRIDOR_M,
        "corridor_exceed_samples": int((np.abs(lat) > CORRIDOR_M).sum()),
        "teleport_gate_trips": int(tele_trips.size),
        "teleport_gate_first_trip_index": int(tele_trips[0]) if tele_trips.size else None,
        "zero_dt_samples": int(zero_dt.size),
        "zero_dt_max_jump_m": zero_dt_jump_max,
        "max_raw_step_m": float(jumps.max()),
        "max_raw_step_index": int(np.argmax(jumps)),
        "gap_largest": gap,
        "backward_weight": BACKWARD_WEIGHT,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", default=str(DEFAULT_ROUTE))
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--metadata", default=str(DEFAULT_META))
    parser.add_argument("--out", default=None, help="JSON report path (default logs/replay_raw_lap-<UTC>.json)")
    args = parser.parse_args(argv)

    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    expected_sha = meta["route_sha256"]
    raw, route_ref, rows, fault = run_replay(Path(args.route), Path(args.raw), expected_sha)
    report = analyse(raw, route_ref, rows, fault)

    stamp = dt_mod.datetime.now(dt_mod.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else REPO_ROOT / "logs" / f"replay_raw_lap-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"replay report written to {out}", flush=True)

    ok = (
        fault is None
        and report["in_space_all"]
        and report["finish_bonus_count"] == 1
        and report["teleport_gate_trips"] == 0
        and report["corridor_exceed_samples"] == 0
        and "stuck" not in report["terminal_reasons"]
        and "off_route" not in report["terminal_reasons"]
        and report["stationary_reward_abs_max"] < 1e-9
    )
    print("replay: PASS" if ok else "replay: FLAGGED (see report)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
