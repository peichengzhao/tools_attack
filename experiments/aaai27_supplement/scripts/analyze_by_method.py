#!/usr/bin/env python3
"""Aggregate fixed-denominator per-case results by a normalized method label."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def normalize_method(value: str) -> str:
    value = value.split("/", 1)[-1]
    if "|" in value:
        parts = value.split("|")
        if len(parts) == 3:
            return parts[-1]
    value = re.sub(r"^protect_agent_", "", value)
    value = re.sub(r"^(GPT|OPENROUTER)_", "", value)
    return value


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: analyze_by_method.py per_case.csv output.csv")
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["normalized_method", "cases", "attack_success", "asr_all_pct"])
        by_method: dict[str, list[dict]] = {}
        for row in rows:
            key = normalize_method(row["method"])
            by_method.setdefault(key, []).append(row)
        for key in sorted(by_method):
            items = by_method[key]
            success = sum(item["attack_success_final"] == "True" for item in items)
            writer.writerow(
                [key, len(items), success, round(success / len(items) * 100, 2)]
            )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
