"""Entry point for the TMRL server, trainer, and Windows worker roles.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe train.py --role worker --profile windows
    .\\.venv\\Scripts\\python.exe train.py --role trainer --profile modal

The effective config must be prepared first by scripts/bootstrap_config.py; this script
validates it and never rewrites it or clears a checkpoint.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from tmrl.config import config_constants as cfg  # noqa: E402
from tmrl.networking import Server, Trainer  # noqa: E402

from src.pipeline import build_training_cls, build_worker, run_worker, trainer_paths  # noqa: E402
from src.reward.route import SMOKE_NO_ROUTE  # noqa: E402


def validate_effective_config(profile):
    eff = cfg.TMRL_CONFIG
    if "PROJECT" not in eff:
        raise SystemExit("effective config has no PROJECT block; run scripts/bootstrap_config.py first")
    ports =[eff["PORT"], eff["LOCAL_PORT_SERVER"], eff["LOCAL_PORT_TRAINER"], eff["LOCAL_PORT_WORKER"]]
    if len(set(ports)) != len(ports):
        raise SystemExit(f"ports must be unique: {ports}")
    if profile in ("windows", "modal") and eff["TLS"] is not True:
        raise SystemExit(f"profile {profile} requires TLS=true")
    if profile == "modal" and eff["CUDA_TRAINING"] is not True:
        raise SystemExit("profile modal requires CUDA_TRAINING=true")

    route = eff["PROJECT"]["route_path"]
    if route == SMOKE_NO_ROUTE:
        if eff["RUN_NAME"] != "pipeline_smoke":
            raise SystemExit("smoke route is only allowed with RUN_NAME=pipeline_smoke")
    elif profile == "windows" and not (REPO_ROOT / route).is_file():
        raise SystemExit(f"route file not found: {REPO_ROOT / route}")


def _block_forever():
    while True:
        time.sleep(60)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["server", "trainer", "worker"], required=True)
    p.add_argument("--profile", choices=["windows", "modal"], required=True)
    a = p.parse_args(argv)
    if a.role == "worker" and a.profile != "windows":
        p.error("the worker role needs the windows profile (it drives the game)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_effective_config(a.profile)
    logging.info("run=%s role=%s profile=%s", cfg.RUN_NAME, a.role, a.profile)

    if a.role == "server":
        server = Server(  # keep the reference: the relay shuts down when the object is garbage-collected
            port=cfg.PORT,
            password=cfg.PASSWORD,
            local_port=cfg.LOCAL_PORT_SERVER,
            header_size=cfg.HEADER_SIZE,
            security=cfg.SECURITY,
            keys_dir=cfg.CREDENTIALS_DIRECTORY,
            max_workers=cfg.NB_WORKERS,
        )
        try:
            _block_forever()
        finally:
            del server
    elif a.role == "trainer":
        weights, checkpoint = trainer_paths()
        Trainer(
            training_cls=build_training_cls(),
            server_ip="127.0.0.1",
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            local_com_port=cfg.LOCAL_PORT_TRAINER,
            header_size=cfg.HEADER_SIZE,
            max_buf_len=cfg.BUFFER_SIZE,
            security=cfg.SECURITY,
            keys_dir=cfg.CREDENTIALS_DIRECTORY,
            hostname=cfg.HOSTNAME,
            model_path=str(weights),
            checkpoint_path=str(checkpoint),
        ).run()
    else:
        run_worker(build_worker(standalone=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
