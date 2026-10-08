"""Merge common + machine overlay into the effective TMRL config.

Usage:
    TMRL_PASSWORD=... python scripts/bootstrap_config.py --profile local|windows|modal \
        --server 127.0.0.1 --run-name NAME [--smoke]

Writes the effective ~/TmrlData/config/config.json atomically and a redacted
copy in logs/<run>/effective_config.json plus a shared manifest.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


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
        eff.setdefault("PROJECT", {})["route_path"] = "smoke-no-route"

    # Validation.
    ports = [eff.get("PORT"), eff.get("LOCAL_PORT_SERVER"), eff.get("LOCAL_PORT_TRAINER"), eff.get("LOCAL_PORT_WORKER")]
    assert len(set(ports)) == 4, f"ports must be unique: {ports}"
    assert eff.get("BUFFER_SIZE", 0) > 0 and eff.get("BUFFERS_MAXLEN", 0) > 0
    if a.profile == "modal":
        assert eff.get("CUDA_TRAINING") is True
    if a.profile in ("windows", "modal"):
        assert eff.get("TLS") is True

    run = eff["RUN_NAME"]
    logdir = REPO / "logs" / run
    logdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "observation_schema": eff.get("PROJECT", {}).get("observation_schema"),
        "hidden_sizes": eff.get("PROJECT", {}).get("hidden_sizes"),
        "route_path": eff.get("PROJECT", {}).get("route_path"),
        "smoke": a.smoke,
    }
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
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
