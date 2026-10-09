"""Merge common + machine overlay into the effective TMRL config.

Usage:
    TMRL_PASSWORD=... python scripts/bootstrap_config.py --profile local|windows|modal \
        --server 127.0.0.1 --run-name NAME [--smoke]

Writes the effective ~/TmrlData/config/config.json atomically and a redacted
copy in logs/<run>/effective_config.json plus a shared manifest.

Production (non-smoke) runs validate the route and reward identities before anything is written:
the route file sha256 must match PROJECT.route_sha256, the route metadata must agree, and
PROJECT.reward_sha256 must equal reward_identity(). The manifest is fingerprinted over its full
body, so any later edit to its identity fields is detected by check_manifest().
"""
import argparse
import copy
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.reward.route import (  # noqa: E402
    ACTION_NAMES,
    CONTROL_HZ,
    FEATURE_NAMES,
    SMOKE_NO_ROUTE,
    file_sha256,
    reward_identity,
)

DEPENDENCIES = ("tmrl", "rtgym", "gymnasium", "numpy", "torch")


class IdentityError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def route_identity(route_path: str):
    """sha256 of the route file on disk, or None for the smoke dummy route."""
    if route_path == SMOKE_NO_ROUTE:
        return None
    return file_sha256(str((REPO / route_path).resolve()))


def code_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"],
            cwd=REPO, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def dependency_versions() -> dict:
    out = {}
    for name in DEPENDENCIES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "absent"
    return out


def build_identity(project: dict, smoke: bool) -> dict:
    return {
        "route_path": project.get("route_path"),
        "route_sha256": None if smoke else route_identity(project["route_path"]),
        "reward_sha256": reward_identity(),
        "feature_names": list(FEATURE_NAMES),
        "action_order": list(ACTION_NAMES),
        "control_period_s": 1.0 / CONTROL_HZ,
        "code_revision": code_revision(),
        "dependency_versions": dependency_versions(),
    }


def fingerprint(body: dict) -> str:
    payload = {k: v for k, v in body.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_manifest(project: dict, smoke: bool) -> dict:
    body = {
        "observation_schema": project.get("observation_schema"),
        "hidden_sizes": project.get("hidden_sizes"),
        "smoke": smoke,
        **build_identity(project, smoke),
    }
    body["fingerprint"] = fingerprint(body)
    return body


def check_manifest(manifest: dict) -> None:
    """Reject a manifest whose fingerprint does not match its contents."""
    if manifest.get("fingerprint") != fingerprint(manifest):
        raise IdentityError("manifest fingerprint does not match its contents; manifest rejected")


def validate_identity(project: dict, smoke: bool) -> None:
    """Production gate: declared route and reward identities must match the files and code."""
    if smoke:
        return
    route_path = project.get("route_path")
    if not route_path or route_path == SMOKE_NO_ROUTE:
        raise IdentityError("production run must use a real route, not the smoke dummy route")
    declared = project.get("route_sha256")
    if not declared:
        raise IdentityError("production run needs PROJECT.route_sha256 declared")
    actual = route_identity(route_path)
    if actual != declared:
        raise IdentityError(f"route file sha256 {actual} does not match declared {declared}")
    meta_path = (REPO / route_path).parent / "metadata.json"
    if not meta_path.exists():
        raise IdentityError(f"missing route metadata {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("route_sha256") != declared:
        raise IdentityError("route metadata route_sha256 does not match the declared route")
    if project.get("reward_sha256") != reward_identity():
        raise IdentityError(
            f"PROJECT.reward_sha256 {project.get('reward_sha256')} does not match reward_identity() {reward_identity()}"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["local", "windows", "modal"], required=True)
    p.add_argument("--server", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tls-dir", default=None, help="folder holding the pinned certificate.pem (and key.pem on the server)")
    a = p.parse_args(argv)

    password = os.environ.get("TMRL_PASSWORD", "")
    if not password:
        print("TMRL_PASSWORD is not set.", flush=True)
        return 2

    tmrl_cfg = Path(os.path.expanduser("~/TmrlData/config/config.json"))
    if not tmrl_cfg.exists():
        print(f"Missing installed TMRL config at {tmrl_cfg}. Run `tmrl --install` first.", flush=True)
        return 2
    installed = json.loads(tmrl_cfg.read_text())
    common = json.loads((REPO / "configs/common.json").read_text())
    overlay = json.loads((REPO / f"configs/{a.profile}.json").read_text())

    eff = deep_merge(installed, common)
    eff = deep_merge(eff, overlay)
    eff["WANDB_KEY"] = ""
    eff["PASSWORD"] = password
    if a.profile == "windows":
        eff["PUBLIC_IP_SERVER"] = a.server
        if a.port:
            eff["PORT"] = a.port
    if a.tls_dir:
        eff["TLS_CREDENTIALS_DIRECTORY"] = str(Path(a.tls_dir).resolve())
    if a.run_name:
        eff["RUN_NAME"] = a.run_name
    if a.smoke:
        eff["RUN_NAME"] = "pipeline_smoke"
        eff["ENVIRONMENT_STEPS_BEFORE_TRAINING"] = 64
        eff["UPDATE_MODEL_INTERVAL"] = 1
        eff.setdefault("PROJECT", {})["route_path"] = SMOKE_NO_ROUTE

    # Validation.
    ports = [eff.get("PORT"), eff.get("LOCAL_PORT_SERVER"), eff.get("LOCAL_PORT_TRAINER"), eff.get("LOCAL_PORT_WORKER")]
    assert len(set(ports)) == 4, f"ports must be unique: {ports}"
    assert eff.get("BUFFER_SIZE", 0) > 0 and eff.get("BUFFERS_MAXLEN", 0) > 0
    if a.profile == "modal":
        assert eff.get("CUDA_TRAINING") is True
    if a.profile in ("windows", "modal"):
        assert eff.get("TLS") is True

    project = eff.get("PROJECT", {})
    try:
        validate_identity(project, a.smoke)
        manifest = build_manifest(project, a.smoke)
        check_manifest(manifest)
    except IdentityError as e:
        print(f"identity check failed: {e}", flush=True)
        return 2

    run = eff["RUN_NAME"]
    logdir = REPO / "logs" / run
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    tmp = tmrl_cfg.with_suffix(".tmp")
    tmp.write_text(json.dumps(eff, indent=2))
    tmp.replace(tmrl_cfg)
    redacted = {k: ("***" if k == "PASSWORD" else v) for k, v in eff.items()}
    (logdir / "effective_config.json").write_text(json.dumps(redacted, indent=2))
    print(f"Effective config written for run={run} profile={a.profile} manifest={manifest['fingerprint']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
