"""Modal side of the pipeline: TLS echo, GPU check, SAC benchmark, and the TMRL trainer service.

From the repo root (Modal secret `tmrl-secrets` must exist: TMRL_PASSWORD, TLS_CERT, TLS_KEY):
    .\\.venv\\Scripts\\python.exe -m modal run deployment/modal_app.py::echo_check
    .\\.venv\\Scripts\\python.exe -m modal run deployment/modal_app.py::gpu_check
    .\\.venv\\Scripts\\python.exe -m modal run deployment/modal_app.py::benchmark --updates 1000
    .\\.venv\\Scripts\\python.exe -m modal deploy deployment/modal_app.py   # then trainer_service.spawn()

Stop a running trainer with `modal app stop trackmania-rl`; a running L4 bills continuously.
Each run is capped at MAX_RUN_S (the $5 budget); the workspace spend limit in Modal billing is the total cap.
"""
import os
import signal
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "trackmania-rl"
TUNNEL_PORT = 55555
APP_DIR = "/app"
STATE_DIR = "/state"
HOSTNAME = "default"  # matches the certificate and TMRL HOSTNAME; the cert is pinned, not the Modal host

# Spend guard: Modal bills L4 at $0.000222/s (~$0.80/h). CPU and memory bill on top, so assume +25%.
# Modal enforces `timeout` per call, so no single run can cost more than BUDGET_USD even if it hangs.
BUDGET_USD = 5.0
L4_USD_PER_S = 0.000222
MAX_RUN_S = int(BUDGET_USD / (L4_USD_PER_S * 1.25))  # 18018 s, about 5.0 h

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("tmrl-state", create_if_missing=True)
endpoint = modal.Dict.from_name("tmrl-endpoint", create_if_missing=True)
secrets = [modal.Secret.from_name("tmrl-secrets")]

# Versions match requirements-windows.lock.txt; the Linux torch wheel from PyPI ships CUDA.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.14.1", "numpy==2.5.3", "gymnasium==1.4.0", "rtgym==0.16", "tlspyo==0.3.0",
        "pandas", "pyyaml", "wandb", "requests", "opencv-python-headless", "mss", "pyinstrument", "chardet", "packaging",
    )
    # tmrl hard-requires pywin32 and vgamepad (Windows only); the trainer container needs neither.
    .pip_install("tmrl==0.7.1", extra_options="--no-deps")
    .run_commands("python -c \"import tmrl.config.config_constants\"")  # creates ~/TmrlData (what `tmrl --install` does)
    .add_local_dir(
        ".", APP_DIR, ignore=["logs", "weights", "checkpoints", "secrets_local", ".venv", ".git", "**/__pycache__"]
    )
)


def _write_tls(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "certificate.pem").write_text(os.environ["TLS_CERT"])
    (folder / "key.pem").write_text(os.environ["TLS_KEY"])
    return folder


def _publish(host: str, port: int):
    endpoint["address"] = {"host": host, "port": port, "started": time.time()}
    print(f"tunnel address published: {host}:{port}", flush=True)


# ---------------------------------------------------------------- M5: TLS echo through the tunnel
@app.function(image=image, secrets=secrets, timeout=900)
def echo_server(minutes: float = 3.0):
    """TLS echo: replies `pong` to `ping <password>` and `denied` to anything else."""
    creds = _write_tls(Path("/tmp/creds"))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(creds / "certificate.pem", creds / "key.pem")
    password = os.environ["TMRL_PASSWORD"]

    srv = socket.create_server(("0.0.0.0", TUNNEL_PORT))
    srv.settimeout(1.0)
    with modal.forward(TUNNEL_PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        _publish(host, port)
        deadline = time.monotonic() + minutes * 60
        while time.monotonic() < deadline:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            conn.settimeout(5)
            try:
                with ctx.wrap_socket(conn, server_side=True) as tls:
                    data = tls.recv(256)
                    tls.sendall(b"pong" if data == f"ping {password}".encode() else b"denied")
            except (ssl.SSLError, OSError) as e:
                print(f"connection rejected: {type(e).__name__}: {e}", flush=True)
    srv.close()


@app.local_entrypoint()
def echo_check():
    """Start the echo server, wait for its address, run the Windows-side client checks, stop it."""
    t0 = time.time()
    call = echo_server.spawn(minutes=3)
    try:
        addr = None
        end = time.monotonic() + 240
        while time.monotonic() < end:
            addr = endpoint.get("address", None)
            if addr and addr["started"] > t0:
                break
            addr = None
            time.sleep(2)
        if addr is None:
            print("FAIL: echo server never published an address", flush=True)
            sys.exit(1)
        root = Path(__file__).resolve().parents[1]
        rc = subprocess.run(
            [sys.executable, str(root / "scripts" / "smoke_pipeline.py"), "echo-client",
             "--host", addr["host"], "--port", str(addr["port"]),
             "--cert", str(root / "secrets_local" / "certificate.pem")],
            cwd=root,
        ).returncode
    finally:
        call.cancel()
    sys.exit(rc)


# ---------------------------------------------------------------- M6: GPU, benchmark, checkpoint restart
@app.function(image=image, gpu="L4", timeout=600)
def gpu_check():
    import torch

    assert torch.cuda.is_available(), "CUDA is not available"
    name = torch.cuda.get_device_name(0)
    assert "L4" in name, f"expected an L4, got {name}"
    from tmrl.networking import Server, Trainer  # noqa: F401  (headless import check)

    x = torch.randn(1024, 1024, device="cuda")
    assert torch.isfinite(x @ x).all()
    print(f"PASS gpu_check: torch {torch.__version__} cuda {torch.version.cuda} device {name}", flush=True)


def _make_agent(device):
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from src.pipeline import build_agent_cls, build_spaces

    observation_space, action_space = build_spaces()
    agent = build_agent_cls()(observation_space=observation_space, action_space=action_space, device=device)
    return agent, observation_space, action_space


def _batch(observation_space, action_space, size, seed):
    import numpy as np

    sys.path.insert(0, APP_DIR)
    from scripts import smoke_pipeline as sp

    sp.TRANSITIONS = size
    return sp.synthetic_batch(observation_space, action_space, np.random.default_rng(seed))


def _to_device(batch, device):
    import torch

    def move(x):
        if isinstance(x, tuple):
            return tuple(move(i) for i in x)
        return x.to(device) if isinstance(x, torch.Tensor) else x

    return move(batch)


def _run_updates(device, updates):
    import math

    import torch

    agent, obs_space, act_space = _make_agent(device)
    batch = _to_device(_batch(obs_space, act_space, 256, 0), device)
    before = [p.detach().clone() for p in agent.model.actor.parameters()]
    t0 = time.perf_counter()
    for i in range(updates):
        stats = agent.train(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    values = [float(v) for v in (stats.values() if isinstance(stats, dict) else stats) if v is not None]
    assert values and all(math.isfinite(v) for v in values), f"non-finite stats: {values}"
    changed = any(not torch.equal(a, b) for a, b in zip(before, agent.model.actor.parameters()))
    assert changed, "actor parameters did not change"
    return agent, updates / elapsed


@app.function(image=image, gpu="L4", volumes={STATE_DIR: vol}, timeout=1800)
def resume_check():
    """Runs in a fresh container: reload the committed checkpoint and perform another update."""
    import math

    import torch

    vol.reload()
    path = Path(STATE_DIR) / "checkpoints" / "benchmark.pt"
    assert path.is_file(), f"checkpoint missing from the Volume: {path}"
    agent, obs_space, act_space = _make_agent("cuda")
    state = torch.load(path, map_location="cuda")
    agent.model.load_state_dict(state["model"])
    batch = _to_device(_batch(obs_space, act_space, 256, 1), "cuda")
    stats = agent.train(batch)
    values = [float(v) for v in (stats.values() if isinstance(stats, dict) else stats) if v is not None]
    assert values and all(math.isfinite(v) for v in values), f"non-finite stats after restart: {values}"
    print(f"PASS resume_check: restored checkpoint (updates={state['updates']}) and updated again", flush=True)


@app.function(image=image, gpu="L4", volumes={STATE_DIR: vol}, timeout=3600)
def benchmark_remote(updates: int = 1000):
    import torch

    _, cpu_rate = _run_updates("cpu", max(10, updates // 20))
    agent, gpu_rate = _run_updates("cuda", updates)
    print(f"GPU {gpu_rate:.1f} updates/s vs CPU {cpu_rate:.1f} updates/s ({gpu_rate / cpu_rate:.1f}x)", flush=True)

    ckpt_dir = Path(STATE_DIR) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": agent.model.state_dict(), "updates": updates}, ckpt_dir / "benchmark.pt")
    vol.commit()
    print(f"PASS benchmark: {updates} finite GPU updates, actor changed, checkpoint committed", flush=True)
    return gpu_rate


@app.local_entrypoint()
def benchmark(updates: int = 1000):
    benchmark_remote.remote(updates)
    resume_check.remote()  # separate function call => separate container
    print("PASS: checkpoint restored in a fresh container", flush=True)


# ---------------------------------------------------------------- production service
def _wait_port(port, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"port {port} never started listening")


@app.function(
    image=image, gpu="L4", volumes={STATE_DIR: vol}, secrets=secrets,
    timeout=MAX_RUN_S, max_containers=1,
)
def trainer_service(run_name: str = "pipeline_smoke", smoke: bool = True):
    """TMRL server + trainer in one container, tunneled on 55555 with TLS and the shared password."""
    os.chdir(APP_DIR)
    for name in ("weights", "checkpoints"):  # persist under the Volume
        target = Path(STATE_DIR) / name
        target.mkdir(parents=True, exist_ok=True)
        link = Path(APP_DIR) / name
        if not link.exists():
            link.symlink_to(target)

    from tlspyo.credentials import get_default_keys_folder

    _write_tls(Path(get_default_keys_folder()))
    cmd = [sys.executable, "scripts/bootstrap_config.py", "--profile", "modal", "--server", "127.0.0.1",
           "--run-name", run_name]
    subprocess.run(cmd + (["--smoke"] if smoke else []), check=True)

    server = subprocess.Popen([sys.executable, "train.py", "--role", "server", "--profile", "modal"])
    _wait_port(TUNNEL_PORT)
    trainer = subprocess.Popen([sys.executable, "train.py", "--role", "trainer", "--profile", "modal"])

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    try:
        with modal.forward(TUNNEL_PORT, unencrypted=True) as tunnel:
            host, port = tunnel.tcp_socket
            _publish(host, port)
            last_commit = time.monotonic()
            while not stop["flag"] and server.poll() is None and trainer.poll() is None:
                time.sleep(2)
                if time.monotonic() - last_commit > 60:
                    vol.commit()
                    last_commit = time.monotonic()
    finally:
        for proc in (trainer, server):
            if proc.poll() is None:
                proc.terminate()
        for proc in (trainer, server):
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        vol.commit()


@app.local_entrypoint()
def roundtrip_check():
    """M7: start the real server+trainer, run the game-free Windows worker against it, then stop everything."""
    import shutil

    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    t0 = time.time()
    call = trainer_service.spawn(run_name="pipeline_smoke", smoke=True)
    cfg_path = Path.home() / "TmrlData" / "config" / "config.json"
    backup = cfg_path.with_suffix(".json.bak")
    rc = 1
    try:
        addr = None
        end = time.monotonic() + 420
        while time.monotonic() < end:
            addr = endpoint.get("address", None)
            if addr and addr["started"] > t0:
                break
            addr = None
            time.sleep(3)
        if addr is None:
            print("FAIL: trainer service never published an address", flush=True)
            sys.exit(1)
        shutil.copy2(cfg_path, backup)  # the game smoke tests share this effective config; restore it afterwards
        subprocess.run(
            [py, "scripts/bootstrap_config.py", "--profile", "windows", "--server", addr["host"],
             "--port", str(addr["port"]), "--run-name", "pipeline_smoke", "--smoke",
             "--tls-dir", str(root / "secrets_local")],
            cwd=root, check=True,
        )
        rc = subprocess.run([py, "scripts/smoke_pipeline.py", "remote", "--updates", "10"], cwd=root).returncode
    finally:
        if backup.exists():
            shutil.move(str(backup), str(cfg_path))
        call.cancel()
    sys.exit(rc)
