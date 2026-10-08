"""Synthetic pipeline check: spaces, random transitions, and SAC updates on CPU. No game needed.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe scripts\\smoke_pipeline.py local --updates 1
"""
import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.pipeline import build_agent_cls, build_spaces  # noqa: E402

TRANSITIONS = 128


def synthetic_batch(observation_space, action_space, rng):
    """Build one SAC batch (o, a, r, o2, d, info) of TRANSITIONS random transitions."""
    obs, act, rew, obs2, done = [], [], [], [], []
    for _ in range(TRANSITIONS):
        o = observation_space.sample()
        o2 = observation_space.sample()
        assert observation_space.contains(o) and observation_space.contains(o2)
        obs.append(o)
        obs2.append(o2)
        act.append(action_space.sample())
        rew.append(rng.standard_normal())
        done.append(float(rng.random() < 0.05))

    def stack_tuple(items):
        return tuple(torch.as_tensor(np.stack([item[k] for item in items]), dtype=torch.float32) for k in range(3))

    return (
        stack_tuple(obs),
        torch.as_tensor(np.stack(act), dtype=torch.float32),
        torch.as_tensor(np.asarray(rew), dtype=torch.float32),
        stack_tuple(obs2),
        torch.as_tensor(np.asarray(done), dtype=torch.float32),
        None,
    )


def cmd_local(updates):
    observation_space, action_space = build_spaces()
    rng = np.random.default_rng(0)
    agent = build_agent_cls()(observation_space=observation_space, action_space=action_space, device="cpu")
    batch = synthetic_batch(observation_space, action_space, rng)

    for step in range(updates):
        stats = agent.train(batch)
        values = [float(v) for v in (stats.values() if isinstance(stats, dict) else stats) if v is not None]
        assert values and all(math.isfinite(v) for v in values), f"non-finite update stats at step {step}: {values}"

    params = list(agent.model.parameters())
    assert all(torch.isfinite(p).all() for p in params), "non-finite SAC parameters after update"
    print(f"PASS local: {TRANSITIONS} transitions, {updates} SAC update(s), stats={values}", flush=True)
    return 0


def _echo(host, port, cert, payload, tls=True):
    """One echo exchange; returns the reply bytes or the exception class name."""
    import socket
    import ssl

    try:
        raw = socket.create_connection((host, port), timeout=10)
        raw.settimeout(10)
        if tls:
            ctx = ssl.create_default_context(cafile=cert)
            conn = ctx.wrap_socket(raw, server_hostname="default")  # certificate is pinned
        else:
            conn = raw
        with conn:
            conn.sendall(payload)
            return conn.recv(64)
    except (OSError, ssl.SSLError) as e:
        return type(e).__name__


def cmd_echo_client(host, port, cert):
    """M5: TLS echo through the Modal tunnel plus the rejection cases."""
    import os
    import tempfile
    import time

    from make_tls_cert import generate

    password = os.environ.get("TMRL_PASSWORD", "")
    if not password:
        print("TMRL_PASSWORD is not set.", flush=True)
        return 2
    t0 = time.perf_counter()
    reply = _echo(host, port, cert, f"ping {password}".encode())
    latency_ms = (time.perf_counter() - t0) * 1000
    with tempfile.TemporaryDirectory() as other:
        generate(other)
        wrong_cert = _echo(host, port, str(Path(other) / "certificate.pem"), b"ping x")
    cases = [
        ("pinned cert + password -> pong", reply == b"pong"),
        ("wrong password rejected", _echo(host, port, cert, b"ping wrong") == b"denied"),
        ("plaintext client rejected", _echo(host, port, cert, b"ping x", tls=False) != b"pong"),
        ("wrong pinned cert rejected", isinstance(wrong_cert, str)),
    ]
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}: {name}", flush=True)
    print(f"round-trip latency (incl. TLS handshake): {latency_ms:.0f} ms", flush=True)
    return 0 if all(ok for _, ok in cases) else 1


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    echo = sub.add_parser("echo-client", help="TLS echo check against the Modal tunnel")
    echo.add_argument("--host", required=True)
    echo.add_argument("--port", type=int, required=True)
    echo.add_argument("--cert", default="secrets_local/certificate.pem")
    local = sub.add_parser("local", help="synthetic CPU spaces + SAC update check")
    local.add_argument("--updates", type=int, default=1)
    a = p.parse_args(argv)
    if a.mode == "local":
        return cmd_local(a.updates)
    if a.mode == "echo-client":
        return cmd_echo_client(a.host, a.port, a.cert)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
