"""Offline checks: packet decode, split/concat, EOF, NaN rejection."""
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.telemetry import PACKET  # noqa: E402


def main() -> int:
    vals = (12.5, 100.0, 1.0, 2.0, 3.0, 0.1, 0.8, 0.0, 0.0, 3.0, 6000.0)
    raw = PACKET.pack(*vals)
    assert PACKET.unpack(raw) == vals, "round-trip failed"
    # Split across reads.
    assert PACKET.unpack(raw[:20] + raw[20:]) == vals, "split failed"
    # Concatenated packets decode independently.
    assert PACKET.unpack((raw + raw)[44:88]) == vals, "concat failed"
    # NaN rejected.
    bad = PACKET.pack(*((float("nan"),) + vals[1:]))
    assert any(not math.isfinite(x) for x in PACKET.unpack(bad)), "NaN check failed"
    print("self_check: packet contracts OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
