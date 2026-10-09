"""Record the human reference lap and write the Phase I route file.

Read-only: only TelemetryClient packets are read. No controls are ever sent to the game.

  .venv\\Scripts\\python.exe scripts\\record_route.py [--track NAME] [--out-dir tracks/v1]
  .venv\\Scripts\\python.exe scripts\\record_route.py --selftest

Flow: beep (armed) -> wait for a stationary car with finish 0 (spawn reference) ->
record from the first packet with speed > 0.5 m/s until finish has been seen for 1 s ->
trim at the first finish sample, resample to 1 m, validate, write route.npz and metadata.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.telemetry import Telemetry, TelemetryClient, TelemetryError  # noqa: E402
from src.reward.route import Route  # noqa: E402

try:
    import winsound
except ImportError:  # not Windows: beeps are skipped
    winsound = None

MOVING_MPS = 0.5
FINISH_FLAG = 0.5
MAX_PLAUSIBLE_SPEED_MPS = 100.0
JUMP_FLOOR_M = 5.0
ARM_TIMEOUT_S = 15 * 60
RECORD_MAX_S = 120.0
POST_FINISH_S = 1.0
STALE_S = 0.5
POLL_S = 0.001
RESAMPLE_M = 1.0
DEDUPE_M = 1e-3
MIN_POINTS = 20
MIN_LENGTH_M = 100.0
MAX_LENGTH_M = 3000.0
FINISH_TOL_M = 30.0
LATERAL_LIMIT_M = 10.0

BEEPS = {"armed": [(1200, 150), (1200, 150)], "success": [(1500, 1000)], "reject": [(400, 600)]}

ARMING, READY, RECORDING, DONE = "arming", "ready", "recording", "done"


class RouteRejected(Exception):
    pass


def beep(kind):
    if winsound is None:
        return
    for freq, ms in BEEPS[kind]:
        winsound.Beep(freq, ms)
        time.sleep(0.15)


class RouteRecorder:
    """One recording attempt at a time: arm -> spawn -> record -> finish. Feed it telemetry."""

    def __init__(self, now, arm_timeout_s=ARM_TIMEOUT_S, record_max_s=RECORD_MAX_S,
                 post_finish_s=POST_FINISH_S):
        self.arm_deadline = now + arm_timeout_s
        self.record_max_s = record_max_s
        self.post_finish_s = post_finish_s
        self.discarded = 0
        self.missed = 0
        self.last_seq = None
        self._reset_attempt()

    def _reset_attempt(self):
        self.state = ARMING
        self.spawn_row = None
        self.rows = []
        self.start_t = None
        self.finish_idx = None
        self.finish_t = None
        self.reconnects0 = 0

    def _discard(self, reason):
        self.discarded += 1
        self._reset_attempt()
        return "discard", reason

    def _clock(self, now, reconnects):
        """Checks that do not need a fresh packet."""
        if self.state == RECORDING:
            if reconnects != self.reconnects0:
                return self._discard("stream reconnect")
            if now - self.start_t > self.record_max_s:
                self.state = DONE
                return "record_timeout", f"no finish within {self.record_max_s:.0f} s of movement"
            if self.finish_t is not None and now - self.finish_t >= self.post_finish_s:
                self.state = DONE
                return "finished", None
        elif now > self.arm_deadline:
            self.state = DONE
            return "arm_timeout", f"no stationary restart within {ARM_TIMEOUT_S / 60:.0f} min"
        return None

    def feed(self, tel, reconnects, now):
        """Returns (event, detail). tel is None when the stream is stale or empty."""
        if self.state == DONE:
            return "none", None
        event = self._clock(now, reconnects)
        if event is not None:
            return event
        if tel is None:
            return self._discard("stream stale") if self.state == RECORDING else ("none", None)
        if tel.seq == self.last_seq:
            return "none", None
        if self.last_seq is not None and tel.seq > self.last_seq + 1:
            self.missed += tel.seq - self.last_seq - 1  # packets that arrived between polls
        self.last_seq = tel.seq

        fields = (tel.speed, tel.pos_x, tel.pos_y, tel.pos_z, tel.finish)
        if not all(math.isfinite(v) for v in fields):
            return self._discard("non-finite packet") if self.state == RECORDING else ("none", None)
        row = (tel.received_monotonic, tel.pos_x, tel.pos_y, tel.pos_z, tel.speed, tel.finish)

        if self.state == ARMING:
            if tel.speed < MOVING_MPS and tel.finish < FINISH_FLAG:
                self.spawn_row = row
                self.state = READY
                return "spawn", None
            return "none", None

        if self.state == READY:
            if tel.finish >= FINISH_FLAG:  # previous lap's finish still showing: wait for a restart
                self._reset_attempt()
                return "none", None
            if tel.speed > MOVING_MPS:
                self.state = RECORDING
                self.start_t = tel.received_monotonic
                self.reconnects0 = reconnects
                self.rows = [self.spawn_row, row]
                return "started", None
            return "none", None

        prev = self.rows[-1]
        jump = math.dist(row[1:4], prev[1:4])
        limit = max(JUMP_FLOOR_M, 2.0 * MAX_PLAUSIBLE_SPEED_MPS * max(tel.received_monotonic - prev[0], 0.0))
        if jump > limit:
            return self._discard(f"position jump {jump:.1f} m")
        self.rows.append(row)
        if tel.finish >= FINISH_FLAG and self.finish_idx is None:
            self.finish_idx = len(self.rows) - 1
            self.finish_t = tel.received_monotonic
            return "finish_seen", None
        return "sample", None

    def raw_array(self):
        """Columns: t, x, y, z, speed, finish. Row 0 is the spawn reference."""
        return np.array(self.rows, dtype=np.float64).reshape(-1, 6)


def speed_ratio(raw, finish_idx):
    """Median position-derived speed over median packet speed (about 1.0 when packet speed is m/s)."""
    rows = raw[: finish_idx + 1]
    dt = np.diff(rows[:, 0])
    dist = np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1)
    packet = rows[1:, 4]
    ok = (dt > 0) & (packet > MOVING_MPS)
    if not np.any(ok):
        return None
    return float(np.median(dist[ok] / dt[ok]) / np.median(packet[ok]))


def build_outputs(raw, finish_idx, track, git_rev, diagnostics):
    """Trim at the first finish sample, drop duplicates, resample to 1 m, validate. Raises RouteRejected."""
    if raw.ndim != 2 or raw.shape[0] < 2 or not np.all(np.isfinite(raw)):
        raise RouteRejected("recording is empty or has non-finite values")
    pts = raw[: finish_idx + 1, 1:4]
    kept = [0]
    for i in range(1, len(pts)):
        if math.dist(pts[i], pts[kept[-1]]) > DEDUPE_M:
            kept.append(i)
    pts = pts[kept]
    if len(pts) < 2:
        raise RouteRejected("fewer than 2 distinct points")

    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = float(s[-1])
    n = max(2, int(round(total / RESAMPLE_M)) + 1)
    arc = np.linspace(0.0, total, n)
    positions = np.stack([np.interp(arc, s, pts[:, k]) for k in range(3)], axis=1).astype(np.float32)
    arc32 = arc.astype(np.float32)
    spacing = total / (n - 1)
    spawn = raw[0, 1:4]
    finish = raw[finish_idx, 1:4]

    reasons = []
    if len(positions) < MIN_POINTS:
        reasons.append(f"only {len(positions)} points, need {MIN_POINTS}")
    if not MIN_LENGTH_M <= total <= MAX_LENGTH_M:
        reasons.append(f"length {total:.1f} m outside {MIN_LENGTH_M:.0f}-{MAX_LENGTH_M:.0f} m")
    if not (np.all(np.isfinite(positions)) and np.all(np.isfinite(arc32))):
        reasons.append("non-finite route value")
    gap = math.dist(positions[-1].astype(np.float64), finish)
    if gap > FINISH_TOL_M:
        reasons.append(f"last point {gap:.1f} m from finish")
    if not np.all(np.diff(arc32) > 0):
        reasons.append("arc length not strictly increasing")
    if reasons:
        raise RouteRejected("; ".join(reasons))

    meta = {
        "track": track,
        "total_length_m": total,
        "point_count": int(len(positions)),
        "spacing_m": float(spacing),
        "spawn_xyz": [float(v) for v in spawn],
        "finish_xyz": [float(v) for v in finish],
        "lap_time_s": float(raw[finish_idx, 0] - raw[1, 0]),
        "speed_ratio_position_to_packet": speed_ratio(raw, finish_idx),
        "discarded_attempts": diagnostics["discarded_attempts"],
        "packets_missed_between_polls": diagnostics["packets_missed"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_rev,
    }
    return {"positions": positions, "arc_length": arc32, "spacing_m": float(spacing), "meta": meta}


def write_route(out_dir, result):
    out_dir.mkdir(parents=True, exist_ok=True)
    route_path = out_dir / "route.npz"
    # positions/arc_length per the Phase I spec; points/spacing_m for src/reward/route.py.
    np.savez(route_path, positions=result["positions"], arc_length=result["arc_length"],
             points=result["positions"], spacing_m=np.float32(result["spacing_m"]))
    meta = dict(result["meta"], route_sha256=hashlib.sha256(route_path.read_bytes()).hexdigest())
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def save_raw(out_dir, raw, stamp):
    path = out_dir / "raw" / f"raw_{stamp}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, t=raw[:, 0], x=raw[:, 1], y=raw[:, 2], z=raw[:, 3], speed=raw[:, 4], finish=raw[:, 5])
    return path


def git_revision():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def record_live(out_dir, track):
    client = TelemetryClient()
    try:
        deadline = time.monotonic() + 10.0
        while True:
            try:
                client.latest(max_age_s=STALE_S)
                break
            except TelemetryError:
                if time.monotonic() > deadline:
                    print("no telemetry on 127.0.0.1:9000; is the game running with the plugin?", flush=True)
                    return 2
                time.sleep(0.05)

        rec = RouteRecorder(time.monotonic())
        beep("armed")
        print("ARMED: restart the map, then drive one clean lap.", flush=True)
        while True:
            try:
                tel = client.latest(max_age_s=STALE_S)
            except TelemetryError:
                tel = None
            event, detail = rec.feed(tel, client.reconnects, time.monotonic())
            if event in ("spawn", "started", "finish_seen"):
                print(event, flush=True)
            elif event == "discard":
                print(f"attempt discarded: {detail}; restart the map to re-arm", flush=True)
                beep("armed")
            elif event in ("arm_timeout", "record_timeout"):
                print(f"{event}: {detail}", flush=True)
                beep("reject")
                return 3 if event == "record_timeout" else 2
            elif event == "finished":
                break
            time.sleep(POLL_S)
    finally:
        client.close()

    raw = rec.raw_array()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = save_raw(out_dir, raw, stamp)
    diagnostics = {"discarded_attempts": rec.discarded, "packets_missed": rec.missed}
    try:
        result = build_outputs(raw, rec.finish_idx, track, git_revision(), diagnostics)
    except RouteRejected as exc:
        print(f"REJECTED: {exc}. Raw recording kept at {raw_path}", flush=True)
        beep("reject")
        return 1
    result["meta"]["raw_recording"] = raw_path.relative_to(out_dir).as_posix()
    meta = write_route(out_dir, result)
    beep("success")
    print(f"route written to {out_dir / 'route.npz'}", flush=True)
    print(f"lap {meta['lap_time_s']:.2f} s, length {meta['total_length_m']:.1f} m, "
          f"{meta['point_count']} points, speed ratio {meta['speed_ratio_position_to_packet']}, "
          f"sha256 {meta['route_sha256']}", flush=True)
    return 0


# ---------------------------------------------------------------- selftest (offline)

def make_packet(seq, t, x, z, speed, finish):
    return Telemetry(speed=speed, distance=0.0, pos_x=x, pos_y=0.0, pos_z=z, steer=0.0, gas=0.0,
                     brake=0.0, finish=finish, gear=3.0, rpm=5000.0, seq=seq, received_monotonic=t)


def synthetic_stream(segments, speed=25.0, hz=100.0, t0=1000.0, post_s=1.2):
    """Stationary spawn, a driven path of (length_m, curvature) segments, then finish flagged for post_s."""
    dt = 1.0 / hz
    frames = []
    x, z, heading = 100.0, 200.0, 0.0
    frames += [(x, z, 0.0, 0.0)] * int(round(hz))  # 1 s stationary at spawn
    for length, curvature in segments:
        for _ in range(int(round(length / (speed * dt)))):
            heading += curvature * speed * dt
            x += speed * dt * math.sin(heading)
            z += speed * dt * math.cos(heading)
            frames.append((x, z, speed, 0.0))
    for _ in range(int(round(post_s * hz))):
        x += speed * dt * math.sin(heading)
        z += speed * dt * math.cos(heading)
        frames.append((x, z, speed, 1.0))
    return [make_packet(i + 1, t0 + i * dt, fx, fz, fs, ff) for i, (fx, fz, fs, ff) in enumerate(frames)]


def feed_all(rec, packets, reconnects=0):
    """Feed packets through the recorder; return the notable (event, detail) pairs."""
    out = []
    for tel in packets:
        event, detail = rec.feed(tel, reconnects, tel.received_monotonic)
        if event not in ("none", "sample"):
            out.append((event, detail))
        if event in ("finished", "record_timeout", "arm_timeout"):
            break
    return out


def selftest() -> int:
    failures = []

    def expect(ok, label):
        print(("PASS " if ok else "FAIL ") + label, flush=True)
        if not ok:
            failures.append(label)

    lap = synthetic_stream([(200.0, 0.0), (math.pi * 50.0, 1.0 / 50.0), (300.0, 0.0)])
    finish_i = next(i for i, p in enumerate(lap) if p.finish >= FINISH_FLAG)
    first_moving = next(i for i, p in enumerate(lap) if p.speed > MOVING_MPS)
    fin, spawn = lap[finish_i], lap[0]
    poly = np.array([[p.pos_x, p.pos_z] for p in lap[: finish_i + 1]])
    expected_len = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))

    rec = RouteRecorder(lap[0].received_monotonic)
    names = [e for e, _ in feed_all(rec, lap)]
    expect(names == ["spawn", "started", "finish_seen", "finished"], f"nominal lap events {names}")
    raw = rec.raw_array()
    result = build_outputs(raw, rec.finish_idx, "selftest", "selftest",
                           {"discarded_attempts": rec.discarded, "packets_missed": rec.missed})
    pos, arc, meta = result["positions"], result["arc_length"], result["meta"]
    step = np.diff(arc.astype(np.float64))
    expect(pos.dtype == np.float32 and arc.dtype == np.float32 and pos.shape[1] == 3,
           "positions and arc_length are float32, positions (N,3)")
    expect(pos.shape[0] >= MIN_POINTS and np.all(np.isfinite(pos)), "point count >= 20, all finite")
    expect(bool(np.all(step > 0)) and np.allclose(step, result["spacing_m"], atol=1e-3)
           and abs(result["spacing_m"] - 1.0) < 0.01, "arc length strictly increasing at ~1 m steps")
    expect(np.allclose(pos[0], [spawn.pos_x, 0.0, spawn.pos_z], atol=1e-3), "first point is the spawn")
    expect(np.linalg.norm(pos[-1] - np.array([fin.pos_x, 0.0, fin.pos_z])) < 1e-3, "last point is the finish")
    expect(abs(meta["total_length_m"] - expected_len) < 1e-3, f"length {meta['total_length_m']:.2f} m matches path")
    expect(abs(meta["lap_time_s"] - (fin.received_monotonic - lap[first_moving].received_monotonic)) < 1e-6,
           "lap time matches packet timestamps")
    expect(abs(meta["speed_ratio_position_to_packet"] - 1.0) < 0.01, "speed ratio ~1.0 on synthetic m/s stream")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_route(tmp, result)
        with np.load(tmp / "route.npz", allow_pickle=False) as data:
            expect(set(data.files) == {"positions", "arc_length", "points", "spacing_m"}, "route.npz keys")
            expect(data["positions"].dtype == np.float32 and data["arc_length"].dtype == np.float32,
                   "route.npz stored as float32, loads with allow_pickle=False")
        save_raw(tmp, raw, "selftest")
        with np.load(tmp / "raw" / "raw_selftest.npz", allow_pickle=False) as data:
            expect(set(data.files) == {"t", "x", "y", "z", "speed", "finish"}, "raw npz keys, no pickle")
        route = Route(str(tmp / "route.npz"))
        states = [route.project(p, 0.01) for p in raw[:, 1:4]]
        progress = np.array([s.progress_m for s in states])
        lateral = np.array([s.lateral_m for s in states])
        total = float(arc[-1])
        expect(bool(np.all(np.diff(progress) > -0.5)) and progress[-1] >= total - 0.5,
               "route replay: progress rises from 0 to the finish")
        expect(float(np.max(np.abs(lateral))) < LATERAL_LIMIT_M, "route replay: lateral within 10 m")

    rec = RouteRecorder(lap[0].received_monotonic)
    feed_all(rec, lap[:500])
    fresh = synthetic_stream([(200.0, 0.0), (math.pi * 50.0, 1.0 / 50.0), (300.0, 0.0)],
                             t0=lap[499].received_monotonic + 1.0)
    second = feed_all(rec, fresh, reconnects=1)
    expect(second[:1] == [("discard", "stream reconnect")] and second[-1][0] == "finished"
           and rec.discarded == 1, "reconnect mid-lap discards, fresh start completes")

    bad = list(lap)
    bad[500] = dataclasses.replace(lap[500], pos_x=float("nan"))
    out = feed_all(RouteRecorder(lap[0].received_monotonic), bad)
    expect(("discard", "non-finite packet") in out, "non-finite packet discards the attempt")

    jump = list(lap)
    jump[500] = dataclasses.replace(lap[500], pos_x=lap[500].pos_x + 50.0)
    out = feed_all(RouteRecorder(lap[0].received_monotonic), jump)
    expect(any(e == "discard" and str(d).startswith("position jump") for e, d in out),
           "50 m teleport mid-lap discards the attempt")

    rec = RouteRecorder(lap[0].received_monotonic)
    feed_all(rec, lap[:500])
    expect(rec.feed(None, 0, lap[499].received_monotonic + 0.6) == ("discard", "stream stale"),
           "stale stream mid-lap discards the attempt")

    short = synthetic_stream([(50.0, 0.0)])
    rec = RouteRecorder(short[0].received_monotonic)
    feed_all(rec, short)
    try:
        build_outputs(rec.raw_array(), rec.finish_idx, "selftest", "selftest",
                      {"discarded_attempts": 0, "packets_missed": 0})
        expect(False, "50 m lap rejected")
    except RouteRejected as exc:
        expect("length" in str(exc), f"50 m lap rejected: {exc}")

    expect(RouteRecorder(0.0).feed(None, 0, ARM_TIMEOUT_S + 1.0)[0] == "arm_timeout",
           "no stationary restart within 15 min -> arm timeout")
    stream = [make_packet(1, 1000.0, 100.0, 200.0, 0.0, 0.0)]
    stream += [make_packet(i + 2, 1000.01 + i * 0.01, 100.0 + 0.25 * (i + 1), 200.0, 25.0, 0.0)
               for i in range(13000)]
    out = feed_all(RouteRecorder(1000.0), stream)
    expect(bool(out) and out[-1][0] == "record_timeout", "no finish within 120 s -> record timeout")

    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s)")
        return 1
    print("SELFTEST PASSED")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record the human reference lap into tracks/v1/route.npz.")
    parser.add_argument("--selftest", action="store_true", help="offline check on a synthetic stream")
    parser.add_argument("--track", default="unknown", help="track name for metadata.json")
    parser.add_argument("--out-dir", default="tracks/v1", help="output directory (relative to repo root)")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    try:
        return record_live(out_dir, args.track)
    except KeyboardInterrupt:
        print("interrupted; nothing written", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
