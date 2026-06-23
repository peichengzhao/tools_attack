#!/usr/bin/env python3
"""Summarize the 20-case pilot including fixed-denominator metrics and runtime."""

from __future__ import annotations

import csv
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
INJECAGENT_ROOT = REPO_ROOT / "InjecAgent"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from aaai27_supplement.metrics import records_from_file, summarize  # noqa: E402


ENTRIES = [
    (
        "TopicAttack",
        INJECAGENT_ROOT
        / "results/topicattack_baseline_GPT_deepseek-v4.1-flash-1017_InjecAgent/test_cases_conv_attack_baseline.jsonl",
        REPO_ROOT / "results/aaai27_supplement/logs/t2_topicattack_deepseek_41017-*",
    ),
    (
        "Adaptive Attack",
        INJECAGENT_ROOT
        / "results/adaptive_agent_GPT_deepseek-v4.1-flash-1017_deepseek-v4.1-flash-1017_InjecAgent_base/test_cases_ds_base.json",
        REPO_ROOT / "results/aaai27_supplement/logs/t2_adaptive_deepseek_41017-*",
    ),
    (
        "Ours-Attack",
        REPO_ROOT
        / "results/aaai27_supplement/table2/ours_attack_deepseek-v4.1-flash-1017.jsonl",
        REPO_ROOT / "results/aaai27_supplement/logs/t2_ours_attack_deepseek_41017-*",
    ),
    (
        "T3 Fixed",
        REPO_ROOT
        / "results/aaai27_supplement/table3/fixed_deepseek-v4.1-flash-1017.jsonl",
        REPO_ROOT / "results/aaai27_supplement/logs/t3_fixed_deepseek_41017-*",
    ),
    (
        "T3 Attack-only",
        REPO_ROOT
        / "results/aaai27_supplement/table3/attack_only_deepseek-v4.1-flash-1017.jsonl",
        REPO_ROOT / "results/aaai27_supplement/logs/t3_attack_only_deepseek_41017-*",
    ),
    (
        "T3 Defense-only",
        REPO_ROOT
        / "results/aaai27_supplement/table3/defense_only_deepseek-v4.1-flash-1017.jsonl",
        REPO_ROOT / "results/aaai27_supplement/logs/t3_defense_only_deepseek_41017-*",
    ),
    (
        "T3 Independent",
        REPO_ROOT
        / "results/aaai27_supplement/table3/independent_deepseek-v4.1-flash-1017.jsonl",
        REPO_ROOT / "results/aaai27_supplement/logs/t3_independent_deepseek_41017-*",
    ),
    (
        "T3 Full alternating",
        REPO_ROOT
        / "results/aaai27_supplement/table3/full_deepseek-v4.1-flash-1017.jsonl",
        REPO_ROOT / "results/aaai27_supplement/logs/t3_full_deepseek_41017-*",
    ),
]


def parse_runtime(log_glob: str) -> float | None:
    paths = sorted(glob.glob(log_glob))
    if not paths:
        return None
    lines = Path(paths[-1]).read_text(encoding="utf-8", errors="ignore").splitlines()
    started = None
    finished = None
    for line in lines:
        if line.startswith("# started "):
            started = datetime.fromisoformat(line.removeprefix("# started ").replace("Z", "+00:00"))
        elif "# exit_code" in line and "finished " in line:
            stamp = line.split("finished ", 1)[1].strip()
            finished = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if started and finished:
        return round((finished - started).total_seconds(), 1)
    return None


def main() -> int:
    output_dir = REPO_ROOT / "results/aaai27_supplement/pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, path, log_glob in ENTRIES:
        if not path.exists():
            continue
        raw, records, duplicates = records_from_file(
            path, source=str(path), method=method
        )
        row = {
            "method": method,
            **summarize(records, raw_count=len(raw), duplicates=duplicates),
            "runtime_sec": parse_runtime(str(log_glob)),
            "path": str(path),
        }
        rows.append(row)

    fieldnames = [
        "method",
        "total_cases",
        "valid",
        "invalid",
        "invalid_rate_pct",
        "attack_success_final",
        "asr_all_pct",
        "asr_valid_pct",
        "utility_known",
        "utility_success",
        "utility_pct_all",
        "svu",
        "svu_pct",
        "runtime_sec",
        "path",
    ]
    with (output_dir / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Method | Cases | Valid | Invalid | ASR-all | ASR-valid | Utility | SVU | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        runtime = row["runtime_sec"]
        runtime_text = "—" if runtime is None else f"{runtime / 60:.1f} min"
        lines.append(
            "| {method} | {total_cases} | {valid} | {invalid} | {asr_all_pct} | "
            "{asr_valid_pct} | {utility_pct_all} | {svu_pct} | {runtime} |".format(
                runtime=runtime_text,
                **row,
            )
        )
    (output_dir / "pilot_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
