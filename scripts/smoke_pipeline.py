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


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    local = sub.add_parser("local", help="synthetic CPU spaces + SAC update check")
    local.add_argument("--updates", type=int, default=1)
    a = p.parse_args(argv)
    if a.mode == "local":
        return cmd_local(a.updates)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
