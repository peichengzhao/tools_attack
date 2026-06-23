#!/usr/bin/env python3
"""Validate vendored baseline source, provenance, and entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from aaai27_supplement.baselines import load_registry, validate_baseline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default=str(EXPERIMENT_ROOT / "configs" / "baselines.json"),
    )
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))
    reports = [
        validate_baseline(Path(args.root), spec)
        for _, spec in sorted(registry.items())
    ]
    ok = all(report["ok"] for report in reports)

    if args.json:
        print(json.dumps({"ok": ok, "baselines": reports}, indent=2, ensure_ascii=False))
        return 0 if ok else 1

    print("| Baseline | Provenance | Commit | Entrypoints | Status |")
    print("|---|---|---|---|---|")
    for report in reports:
        entrypoints = sum(report["entrypoints"].values())
        total = len(report["entrypoints"])
        status = "OK" if report["ok"] else "; ".join(report["errors"])
        print(
            f"| {report['display_name']} | {report['provenance']} | "
            f"`{report['commit'][:12]}` | {entrypoints}/{total} | {status} |"
        )
    representative = [
        report
        for report in reports
        if report.get("representative_defense")
    ]
    if representative:
        print()
        for report in representative:
            print(
                f"{report['display_name']} representative defense: "
                f"`{report['representative_defense']}`"
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
