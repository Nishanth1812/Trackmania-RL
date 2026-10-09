"""Packet reader for the TMRL_GrabData 44-byte 11-float telemetry stream."""
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass

PACKET = struct.Struct("<11f")

FIELD_NAMES = (
    "speed", "distance", "pos_x", "pos_y", "pos_z",
    "steer", "gas", "brake", "finish", "gear", "rpm",
)


@dataclass
class Telemetry:
    speed: float
    distance: float
    pos_x: float
    pos_y: float
    pos_z: float
    steer: float
    gas: float
    brake: float
    finish: float
    gear: float
    rpm: float
    seq: int
    received_monotonic: float


class TelemetryError(RuntimeError):
    pass


class TelemetryClient:
    def __init__(self, host="127.0.0.1", port=9000, reconnect_delay=0.5, max_reconnects=10,
                 stale_reconnect_s=1.0):
        self._host = host
        self._port = port
        self._reconnect_delay = reconnect_delay
        self._max_reconnects = max_reconnects
        # Silence on an open socket (no EOF/RST, just no packets) also triggers a
        # reconnect: the plugin sometimes stops sending on a live connection and only
        # serves fresh ones. This must stay well above the 0.25 s control freshness
        # gate so normal jitter never causes a reconnect.
        self._stale_reconnect_s = stale_reconnect_s
        self._lock = threading.Lock()
        self._latest: Telemetry | None = None
        self._seq = 0
        self._reconnects = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _connect(self) -> socket.socket:
        last = None
        for _ in range(self._max_reconnects):
            try:
                return socket.create_connection((self._host, self._port), timeout=5)
            except OSError as e:
                last = e
                time.sleep(self._reconnect_delay)
        raise TelemetryError(f"cannot connect to {self._host}:{self._port}: {last}")

    def _run(self):
        while not self._stop.is_set():
            try:
                sock = self._connect()
            except TelemetryError:
                time.sleep(self._reconnect_delay)
                continue
            buf = bytearray()
            sock.settimeout(1.0)
            last_data = time.monotonic()
            try:
                while not self._stop.is_set():
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        if time.monotonic() - last_data > self._stale_reconnect_s:
                            break  # silent stall on an open socket: fresh connection only
                        continue
                    except OSError:
                        break  # RST / abort: reconnect below, do not kill the thread
                    if chunk == b"":
                        break  # EOF: reconnect with bounded delay
                    buf.extend(chunk)
                    while len(buf) >= PACKET.size:
                        raw = bytes(buf[: PACKET.size])
                        del buf[: PACKET.size]
                        try:
                            values = PACKET.unpack(raw)
                        except struct.error:
                            continue
                        last_data = time.monotonic()
                        if any(not math.isfinite(x) for x in values):
                            continue
                        with self._lock:
                            self._seq += 1
                            self._latest = Telemetry(*values, seq=self._seq, received_monotonic=time.monotonic())
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
                with self._lock:
                    self._reconnects += 1
                buf = bytearray()  # reset accumulation on reconnect
                if not self._stop.is_set():
                    time.sleep(self._reconnect_delay)

    @property
    def reconnects(self) -> int:
        with self._lock:
            return self._reconnects

    def latest(self, max_age_s: float = 0.25) -> Telemetry:
        with self._lock:
            t = self._latest
        if t is None:
            raise TelemetryError("no telemetry received yet")
        if time.monotonic() - t.received_monotonic > max_age_s:
            raise TelemetryError("telemetry stale")
        return t

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
