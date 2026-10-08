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


class SyntheticEnv:
    """Game-free environment with the production spaces: deterministic obs, reward = 0.01*step."""

    EPISODE_STEPS = 100

    def __init__(self):
        self.observation_space, self.action_space = build_spaces()
        self.default_action = np.zeros(3, dtype=np.float32)
        self.unwrapped = self
        self._t = 0

    def _obs(self, t):
        rng = np.random.default_rng(t)
        base = rng.uniform(-1.0, 1.0, size=15).astype(np.float32)
        base[:3] = np.abs(base[:3])
        base[14] = abs(base[14])
        return (base, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

    def reset(self, **_):
        self._t = 0
        return self._obs(0), {}

    def step(self, action):
        self._t += 1
        return self._obs(self._t), 0.01 * self._t, self._t >= self.EPISODE_STEPS, False, {}


def actor_digest(actor):
    import hashlib

    h = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


FIXED_OBS_SEED = 123456


def cmd_remote(updates, timeout_s):
    """M7: synthetic episodes out to the Modal trainer, updated actor weights back to Windows (no game)."""
    import time

    from tmrl.config import config_constants as cfg
    from tmrl.custom.custom_models import SquashedGaussianMLPActor
    from tmrl.networking import RolloutWorker
    from tmrl.util import partial

    from src.pipeline import HIDDEN_SIZES, MAX_SAMPLES_PER_EPISODE, worker_model_path

    assert cfg.SECURITY == "TLS", "remote check requires TLS"
    worker = RolloutWorker(
        env_cls=SyntheticEnv,
        actor_module_cls=partial(SquashedGaussianMLPActor, hidden_sizes=HIDDEN_SIZES),
        sample_compressor=None,
        obs_preprocessor=None,
        device="cpu",
        max_samples_per_episode=MAX_SAMPLES_PER_EPISODE,
        model_path=str(worker_model_path()),
        standalone=False,
        server_ip=cfg.PUBLIC_IP_SERVER,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_WORKER,
        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME,
    )
    fixed_obs = SyntheticEnv()._obs(FIXED_OBS_SEED)
    initial_digest = actor_digest(worker.actor)
    received = 0
    digests = []
    start = time.monotonic()
    episodes = 0
    while time.monotonic() - start < timeout_s:
        if episodes < 8 or episodes % 5 == 0:  # keep the trainer fed without flooding it
            worker.collect_train_episode(max_samples=SyntheticEnv.EPISODE_STEPS)
            assert len(worker.buffer.memory) == SyntheticEnv.EPISODE_STEPS + 1, len(worker.buffer.memory)
            worker.send_and_clear_buffer()
        episodes += 1
        got = worker.update_actor_weights(verbose=False)
        if got:
            received += got
            digests.append(actor_digest(worker.actor))
            if len(set(digests)) >= 2:
                break
        time.sleep(1.0)

    ok_received = len(set(digests)) >= 2
    print(f"{'PASS' if ok_received else 'FAIL'}: sent {episodes} synthetic episodes; "
          f"received {received} actor publications with {len(set(digests))} distinct weight sets", flush=True)
    if not ok_received:
        return 1

    import torch

    saved = worker.actor.load(str(worker_model_path()), device="cpu")
    same_file = actor_digest(saved) == actor_digest(worker.actor)
    with torch.no_grad():
        a1 = worker.actor.act_(fixed_obs, test=True)
        a2 = saved.act_(fixed_obs, test=True)
    finite = bool(np.all(np.isfinite(a1))) and bool(np.all(np.abs(a1) <= 1.0)) and a1.shape == (3,)
    deterministic = bool(np.allclose(a1, a2, atol=1e-5, rtol=1e-5))
    changed = actor_digest(worker.actor) != initial_digest
    checks = [
        ("saved weights reload to the identical actor", same_file),
        ("fixed-observation action finite, in [-1,1], shape (3,)", finite),
        ("action reproducible across reloads (1e-5)", deterministic),
        ("returned actor differs from the local initial actor", changed),
    ]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}", flush=True)
    print(f"actor sha256 {actor_digest(worker.actor)[:16]}  fixed-obs action {np.round(a1, 4).tolist()}", flush=True)
    return 0 if all(ok for _, ok in checks) else 1


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    remote = sub.add_parser("remote", help="synthetic episodes to the Modal trainer and weights back")
    remote.add_argument("--updates", type=int, default=10, help="kept for the runner; smoke epochs are 10 updates")
    remote.add_argument("--timeout", type=float, default=300.0)
    echo = sub.add_parser("echo-client", help="TLS echo check against the Modal tunnel")
    echo.add_argument("--host", required=True)
    echo.add_argument("--port", type=int, required=True)
    echo.add_argument("--cert", default="secrets_local/certificate.pem")
    local = sub.add_parser("local", help="synthetic CPU spaces + SAC update check")
    local.add_argument("--updates", type=int, default=1)
    a = p.parse_args(argv)
    if a.mode == "local":
        return cmd_local(a.updates)
    if a.mode == "remote":
        return cmd_remote(a.updates, a.timeout)
    if a.mode == "echo-client":
        return cmd_echo_client(a.host, a.port, a.cert)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
