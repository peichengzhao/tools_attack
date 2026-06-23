#!/usr/bin/env python3
"""Build the separate Task Shield/MELON/CaMeL interactive-defense table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "interactive_baselines.json"
DEFAULT_RESULTS = REPO_ROOT / "results" / "aaai27_supplement" / "interactive_baselines"


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def load_cells(
    config: dict[str, Any],
    results_root: Path,
    models: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for suite in config["suites"]:
            for defense in config["defenses"]:
                directory = (
                    results_root / safe_name(model) / suite / defense["id"]
                )
                cell_path = directory / "cell.json"
                if cell_path.is_file():
                    payload = json.loads(cell_path.read_text(encoding="utf-8"))
                else:
                    payload = {
                        "model": model,
                        "suite": suite,
                        "defense": defense["id"],
                        "display_name": defense["display_name"],
                        "provenance": defense["provenance"],
                        "attack": config["attack"],
                        "status": "pending",
                        "n": None,
                        "utility_pct": None,
                        "asr_pct": None,
                        "exit_code": None,
                    }
                rows.append(payload)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--models", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    models = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else list(config["models"])
    )
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_cells(config, results_root, models)

    lines = [
        "# Interactive Defense Baselines",
        "",
        (
            f"Benchmark: AgentDojo `{config['benchmark_version']}` · "
            f"Attack: `{config['attack']}`"
        ),
        "",
        "| Defense | Provenance | Model | Suite | N | Utility (%) ↑ | ASR (%) ↓ | Status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['provenance']} | {row['model']} | "
            f"{row['suite']} | {row.get('n') or '—'} | "
            f"{fmt(row.get('utility_pct'))} | {fmt(row.get('asr_pct'))} | "
            f"{row.get('status', 'pending')} |"
        )
    lines.extend(
        [
            "",
            "Task Shield is a third-party reproduction. The methods in this table "
            "change the agent execution loop and are therefore not mixed into the "
            "single-pass static-defense Table 2.",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "interactive_baseline_table.md").write_text(
        markdown,
        encoding="utf-8",
    )
    with (output_dir / "interactive_baseline_table.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "display_name",
                "provenance",
                "model",
                "suite",
                "attack",
                "n",
                "utility_pct",
                "asr_pct",
                "status",
                "exit_code",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "interactive_baseline_table.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
