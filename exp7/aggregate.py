from __future__ import annotations

import json
from pathlib import Path

from exp7.config import (
    BENCHMARK_LABELS,
    EXP7_BENCHMARKS,
    EXP7_MODELS,
    EXP7_RESULTS,
    HARNESS_BASELINE,
    HARNESS_MAX_DEFENSE,
    METRIC_LABELS,
)
from exp7.runners._common import load_result


def delta_utility(baseline: float | None, defended: float | None) -> float | None:
    if baseline is None or defended is None:
        return None
    return round(defended - baseline, 2)


def build_table_rows() -> list[dict]:
    rows: list[dict] = []
    for benchmark in EXP7_BENCHMARKS:
        for model in EXP7_MODELS:
            base = load_result(benchmark, model, HARNESS_BASELINE)
            defended = load_result(benchmark, model, HARNESS_MAX_DEFENSE)
            base_u = base.get("utility") if base else None
            def_u = defended.get("utility") if defended else None
            rows.append(
                {
                    "benchmark": benchmark,
                    "benchmark_label": BENCHMARK_LABELS[benchmark],
                    "model": model,
                    "metric_name": METRIC_LABELS[benchmark],
                    "utility_baseline": base_u,
                    "utility_max_defense": def_u,
                    "delta_utility": delta_utility(base_u, def_u),
                    "baseline_status": (base or {}).get("status"),
                    "max_defense_status": (defended or {}).get("status"),
                }
            )
    return rows


def format_markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Benchmark | Model | 原始 harness (%) | 强化 harness (%) | ΔUtility |",
        "|-----------|-------|------------------|------------------|----------|",
    ]
    for row in rows:
        b = _fmt(row["utility_baseline"], row.get("baseline_status"))
        d = _fmt(row["utility_max_defense"], row.get("max_defense_status"))
        delta = row["delta_utility"]
        delta_s = "-" if delta is None else f"{delta:+.2f}"
        lines.append(
            f"| {row['benchmark_label']} | {row['model']} | {b} | {d} | {delta_s} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value, status: str | None) -> str:
    if value is None:
        return "pending" if status in (None, "pending") else "err"
    return f"{float(value):.1f}"


def aggregate_results(output_dir: Path | None = None) -> dict:
    output_dir = output_dir or EXP7_RESULTS
    rows = build_table_rows()
    payload = {
        "rows": rows,
        "markdown_table": format_markdown_table(rows),
        "notes": {
            "delta_utility": "ΔUtility = 强化 harness 任务成功率 − 原始 harness 任务成功率（单位：百分点）",
            "negative_delta": "负值表示 max_defense 降低了正常任务成功率（可能存在 over-refusal 或 tool 误伤）",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exp7_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "exp7_table.md").write_text(payload["markdown_table"], encoding="utf-8")
    return payload


if __name__ == "__main__":
    summary = aggregate_results()
    print(summary["markdown_table"])
