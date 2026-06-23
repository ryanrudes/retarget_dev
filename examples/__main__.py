"""Dev shim for ``python3 -m examples`` (prefer ``retarget examples``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `cli` importable

from cli import main

if __name__ == "__main__":
    raise SystemExit(main(prog="python -m examples"))
