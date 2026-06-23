"""Cross-eval cell manifest and run-command generation (no API calls)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.cross_eval.constants import (
    ATTACK_LABELS,
    CROSS_EVAL_ATTACKS,
    CROSS_EVAL_DEFENSES,
    CROSS_EVAL_MODELS,
    DEFENSE_IMPLEMENTATION_NOTES,
    DEFENSE_LABELS,
    OFFLINE_POISON_ATTACKS,
)


def iter_cells(
    models: tuple[str, ...] | None = None,
    defenses: tuple[str, ...] | None = None,
    attacks: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    models = models or CROSS_EVAL_MODELS
    defenses = defenses or CROSS_EVAL_DEFENSES
    attacks = attacks or CROSS_EVAL_ATTACKS
    cells: list[dict[str, str]] = []
    for model in models:
        for defense in defenses:
            for attack in attacks:
                cells.append(
                    {
                        "model": model,
                        "defense": defense,
                        "attack": attack,
                        "cell_id": f"{model}|{defense}|{attack}",
                        "attack_label": ATTACK_LABELS.get(attack, attack),
                        "defense_label": DEFENSE_LABELS.get(defense, defense),
                        "defense_implementation_note": DEFENSE_IMPLEMENTATION_NOTES.get(
                            defense, ""
                        ),
                        "output_file": f"results/cross_eval/full/{model}|{defense}|{attack}.jsonl",
                        "needs_offline_poison": attack in OFFLINE_POISON_ATTACKS,
                    }
                )
    return cells


def cross_eval_command(
    model: str,
    defense: str,
    attack: str,
    *,
    smoke: bool = False,
    limit: int = 0,
    use_cache: bool = True,
) -> str:
    parts = [
        "python src/evaluate_cross_eval.py",
        f"--model {model}",
        f"--defense {defense}",
        f"--attack {attack}",
        "--setting base --prompt_type InjecAgent",
    ]
    if smoke:
        parts.append("--smoke")
    if limit > 0:
        parts.append(f"--limit {limit}")
    if use_cache:
        parts.append("--use_cache")
    return " ".join(parts)


def write_manifest_csv(path: Path, cells: list[dict[str, Any]] | None = None) -> Path:
    cells = cells or iter_cells()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_id",
        "model",
        "defense",
        "attack",
        "attack_label",
        "defense_label",
        "defense_implementation_note",
        "needs_offline_poison",
        "output_file",
        "run_command",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            row = dict(cell)
            row["run_command"] = cross_eval_command(
                cell["model"], cell["defense"], cell["attack"]
            )
            writer.writerow(row)
    return path


def write_manifest_json(path: Path, cells: list[dict[str, Any]] | None = None) -> Path:
    cells = cells or iter_cells()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_cells": len(cells),
        "defense_labels": DEFENSE_LABELS,
        "defense_implementation_notes": DEFENSE_IMPLEMENTATION_NOTES,
        "cells": cells,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
