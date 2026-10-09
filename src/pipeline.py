"""Shared TMRL spaces, worker/trainer construction, and the worker episode loop.

Imported by train.py and scripts/smoke_pipeline.py after the repository root is on sys.path.
Every constructor argument that matters for this project is bound explicitly here; the
effective config is read from ~/TmrlData/config/config.json (via tmrl's config constants).
"""
import logging
import time
from pathlib import Path

import numpy as np
from gymnasium import spaces

from tmrl.config import config_constants as cfg
from tmrl.custom.custom_algorithms import SpinupSacAgent
from tmrl.custom.custom_memories import GenericTorchMemory
from tmrl.custom.custom_models import MLPActorCritic, SquashedGaussianMLPActor
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker
from tmrl.training_offline import TorchTrainingOffline
from tmrl.util import partial

from src.env.interface import ACTION_DIM, OBS_DIM, TelemetryInterface
from src.env.telemetry import TelemetryError
from src.reward.route import SMOKE_NO_ROUTE

REPO_ROOT = Path(__file__).resolve().parents[1]
HIDDEN_SIZES = (256, 256)
MAX_SAMPLES_PER_EPISODE = 1200
MEMORY_SIZE = 1000000
BATCH_SIZE = 256
TRAINER_DEVICE = "cuda"
WORKER_DEVICE = "cpu"
RECOVERY_WAIT_S = 1.0
STALE_POLL_S = 1.0


def build_spaces():
    """Actor-side spaces: base telemetry Box plus the two previous-action Boxes rtgym appends."""
    # Bounds must match TelemetryInterface.get_observation_space / get_action_space in src/env/interface.py.
    low = np.array([0.0] * 3 + [-1.0] * 11 + [0.0], dtype=np.float32)
    high = np.array([1.0] * 3 + [1.0] * 11 + [1.0], dtype=np.float32)
    base = spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32)
    action_history = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
    observation = spaces.Tuple((base, action_history, action_history))
    action = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
    return observation, action


def _route_path():
    """Smoke mode keeps the dummy route; production resolves the route relative to the repository."""
    route = cfg.TMRL_CONFIG["PROJECT"]["route_path"]
    if route == SMOKE_NO_ROUTE:
        return route
    return str((REPO_ROOT / route).resolve())


def rtgym_config():
    smoke = cfg.TMRL_CONFIG["PROJECT"]["route_path"] == SMOKE_NO_ROUTE
    return {
        "interface": TelemetryInterface,
        "interface_kwargs": {"smoke": smoke, "route_path": _route_path()},
        "time_step_duration": 0.05,
        "start_obs_capture": 0.05,
        "time_step_timeout_factor": 1.0,
        "act_in_obs": True,
        "act_buf_len": 2,
        "reset_act_buf": True,
        "wait_on_done": True,
        "ep_max_length": 1200,
        "last_act_on_reset": False,
    }


def worker_model_path():
    path = REPO_ROOT / "weights" / f"{cfg.RUN_NAME}.tmod"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def trainer_paths():
    """Return (actor weights path, checkpoint path) for the trainer, creating parent directories."""
    weights = REPO_ROOT / "weights" / f"{cfg.RUN_NAME}_t.tmod"
    checkpoint = REPO_ROOT / "checkpoints" / f"{cfg.RUN_NAME}_t.tcpt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return weights, checkpoint


def build_agent_cls():
    # Bound explicitly: the installed TMRL defaults in ~/TmrlData differ from this project's SAC settings.
    return partial(
        SpinupSacAgent,
        model_cls=partial(MLPActorCritic, hidden_sizes=HIDDEN_SIZES),
        gamma=0.99,
        polyak=0.995,
        alpha=0.1,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_entropy=3e-4,
        learn_entropy_coef=True,
        target_entropy=-3.0,
        optimizer_actor="adam",
        optimizer_critic="adam",
    )


def build_worker(standalone=False):
    """Windows rollout worker. Set standalone=True to run without connecting to a server."""
    env_cls = partial(GenericGymEnv, id="real-time-gym-v1", gym_kwargs={"config": rtgym_config()})
    return RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=partial(SquashedGaussianMLPActor, hidden_sizes=HIDDEN_SIZES),
        sample_compressor=None,
        obs_preprocessor=None,
        device=WORKER_DEVICE,
        max_samples_per_episode=MAX_SAMPLES_PER_EPISODE,
        model_path=str(worker_model_path()),
        standalone=standalone,
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


class ProjectTrainingOffline(TorchTrainingOffline):
    """TorchTrainingOffline that publishes the current actor at the start of every epoch.

    This gives newly connected workers an initial actor before warmup and after checkpoint resume.
    """

    def run_epoch(self, interface):
        interface.broadcast_model(self.agent.get_actor())
        stats = super().run_epoch(interface)
        logging.info("epoch %s finished with %d stat records", self.epoch, len(stats))
        return stats


def build_training_cls():
    observation_space, action_space = build_spaces()
    smoke = cfg.TMRL_CONFIG["PROJECT"]["route_path"] == SMOKE_NO_ROUTE
    # Smoke runs publish after every short epoch so a round-trip cannot wait on production thresholds.
    schedule = (
        {"rounds": 1, "steps": 10, "start_training": 64, "update_model_interval": 1}
        if smoke
        else {"rounds": 10, "steps": 100, "start_training": 2000, "update_model_interval": 100}
    )
    return partial(
        ProjectTrainingOffline,
        env_cls=(observation_space, action_space),
        memory_cls=partial(GenericTorchMemory, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE),
        training_agent_cls=build_agent_cls(),
        epochs=10000,
        max_training_steps_per_env_step=1.0,
        update_buffer_interval=1,
        sleep_between_buffer_retrieval_attempts=1.0,
        profiling=False,
        device=TRAINER_DEVICE,
        **schedule,
    )


def _neutralize(worker):
    worker.env.unwrapped.interface.neutralize()


def _policy_is_stale(last_policy, started, project, now):
    if last_policy is None:
        return now - started > project["initial_policy_grace_s"]
    return now - last_policy > project["policy_stale_pause_s"]


def run_worker(worker):
    """Episode loop: keep the actor fresh, collect one episode, and send it; pause while the policy is stale."""
    project = cfg.TMRL_CONFIG["PROJECT"]
    started = time.monotonic()
    last_policy = None  # monotonic time of the last positive update_actor_weights() count
    faults = 0
    while True:
        if worker.update_actor_weights(verbose=False) > 0:
            last_policy = time.monotonic()

        if _policy_is_stale(last_policy, started, project, time.monotonic()):
            _neutralize(worker)
            logging.warning("policy is stale; waiting for a new model")
            time.sleep(STALE_POLL_S)
            continue

        try:
            worker.collect_train_episode(max_samples=MAX_SAMPLES_PER_EPISODE)
        except TelemetryError as e:
            # Drop the whole incomplete episode: never send a partial record stream.
            _neutralize(worker)
            worker.buffer.clear()
            faults += 1
            logging.error("episode fault %d/%d: %s", faults, project["reset_attempts"], e)
            if faults >= project["reset_attempts"]:
                raise
            time.sleep(RECOVERY_WAIT_S)
            continue

        faults = 0
        _neutralize(worker)
        worker.send_and_clear_buffer()
