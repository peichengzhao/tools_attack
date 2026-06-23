#!/usr/bin/env python3
"""Validate cross-eval setup and emit 128-cell manifest (no API calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cross_eval.setup_check import run_setup_check  # noqa: E402


def main() -> None:
    report = run_setup_check(write_manifest=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
