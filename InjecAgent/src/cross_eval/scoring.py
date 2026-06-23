"""Scoring and CSV aggregation for cross-evaluation."""

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
    DEFENSE_LABELS,
)
from src.utility_parsing import attach_user_task_utility


def score_cross_eval_file(output_file: str | Path) -> dict[str, Any]:
    records: list[dict] = []
    path = Path(output_file)
    if not path.exists():
        return {"error": "file_not_found", "path": str(path)}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if not total:
        return {"error": "empty_file", "path": str(path)}

    asr_succ = utility_succ = invalid = unsucc = 0
    for item in records:
        ev = item.get("eval")
        if ev == "succ":
            asr_succ += 1
        elif ev == "invalid":
            invalid += 1
        elif ev == "unsucc":
            unsucc += 1

        if not item.get("utility_eval"):
            attach_user_task_utility(item)
        if item.get("utility_eval") == "succ":
            utility_succ += 1

    valid = total - invalid
    return {
        "n": total,
        "asr_all_pct": round(asr_succ / total * 100, 2),
        "asr_valid_pct": round(asr_succ / valid * 100, 2) if valid else None,
        "utility_pct": round(utility_succ / total * 100, 2),
        "valid_rate_pct": round(valid / total * 100, 2),
        "invalid_n": invalid,
        "unsucc_n": unsucc,
        "asr_succ_n": asr_succ,
        "utility_succ_n": utility_succ,
    }


def cell_label(asr_pct: float, utility_pct: float) -> str:
    return f"{asr_pct:.1f} ({utility_pct:.1f})"


def collect_cross_eval_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        stem = path.stem
        if "|" not in stem:
            continue
        model, defense, attack = stem.split("|", 2)
        scores = score_cross_eval_file(path)
        if scores.get("error"):
            continue
        rows.append(
            {
                "model": model,
                "defense": defense,
                "attack": attack,
                "attack_label": ATTACK_LABELS.get(attack, attack),
                "defense_label": DEFENSE_LABELS.get(defense, defense),
                "asr_pct": scores["asr_all_pct"],
                "utility_pct": scores["utility_pct"],
                "cell": cell_label(scores["asr_all_pct"], scores["utility_pct"]),
                "path": str(path),
                **scores,
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, Any]], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "defense",
        "attack",
        "attack_label",
        "defense_label",
        "asr_pct",
        "utility_pct",
        "cell",
        "n",
        "valid_rate_pct",
        "path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def write_pivot_markdown(
    rows: list[dict[str, Any]],
    md_path: Path,
    model: str,
) -> Path:
    """One model: rows=attacks, cols=defenses."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    model_rows = [r for r in rows if r["model"] == model]
    attacks = [a for a in CROSS_EVAL_ATTACKS if any(r["attack"] == a for r in model_rows)]
    defenses = list(CROSS_EVAL_DEFENSES)
    lookup = {(r["attack"], r["defense"]): r["cell"] for r in model_rows}

    lines = [f"# Cross-Eval — {model}", ""]
    header = "| Attack | " + " | ".join(DEFENSE_LABELS[d] for d in defenses) + " |"
    sep = "|---|" + "|".join(["---"] * len(defenses)) + "|"
    lines.extend([header, sep])
    for attack in attacks:
        label = ATTACK_LABELS.get(attack, attack)
        cells = [lookup.get((attack, d), "—") for d in defenses]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path
