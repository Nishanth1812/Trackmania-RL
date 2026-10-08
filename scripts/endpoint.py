"""Read the Modal tunnel address that the trainer container publishes.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe scripts\\endpoint.py get
    .\\.venv\\Scripts\\python.exe scripts\\endpoint.py wait --newer-than 1760000000 --timeout 300
"""
import argparse
import json
import sys
import time

DICT_NAME = "tmrl-endpoint"


def read_endpoint():
    import modal

    d = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    return d.get("address", None)


def wait_endpoint(newer_than: float, timeout: float):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        addr = read_endpoint()
        if addr and addr.get("started", 0) > newer_than:
            return addr
        time.sleep(2)
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("get")
    w = sub.add_parser("wait")
    w.add_argument("--newer-than", type=float, default=0.0, help="epoch seconds; ignore older addresses")
    w.add_argument("--timeout", type=float, default=300.0)
    a = p.parse_args(argv)

    addr = read_endpoint() if a.cmd == "get" else wait_endpoint(a.newer_than, a.timeout)
    if not addr:
        print("no endpoint published", file=sys.stderr)
        return 1
    print(json.dumps(addr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
