#!/usr/bin/env python3
"""Build fixed-denominator summaries from existing per-case result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from aaai27_supplement.metrics import (  # noqa: E402
    CaseRecord,
    records_from_file,
    summarize,
    write_case_csv,
    write_summary_csv,
    write_summary_markdown,
)


SKIP_PARTS = {
    "__pycache__",
    ".git",
    "offline_poison",
}
SKIP_NAMES = {
    "summary_cross_eval.csv",
    "run_state.json",
    "cell_manifest.json",
    "experiment_matrix.json",
    "chatinject_case_goal_map.json",
    "attacker_simulated_responses.json",
}


def source_method(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    stem = path.stem
    if "|" in stem:
        return stem
    if stem.startswith("test_cases_"):
        return path.parent.name
    parts = list(rel.parts)
    if len(parts) >= 2:
        return f"{parts[-2]}/{path.name}"
    return path.name


def iter_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name in SKIP_NAMES:
            continue
        candidates.append(path)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Result directory to scan")
    parser.add_argument("--output-dir", required=True, help="Directory for generated tables")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only include paths containing this substring; may be repeated",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=0,
        help="Drop sources with fewer unique parsed cases than this threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Result root does not exist: {root}")

    all_records: list[CaseRecord] = []
    summary_rows: list[dict] = []
    manifest_files: list[dict] = []

    for path in iter_candidates(root):
        rel = str(path.relative_to(root))
        if args.include and not any(token in rel for token in args.include):
            continue
        method = source_method(path, root)
        try:
            raw, records, duplicates = records_from_file(
                path, source=rel, method=method
            )
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            manifest_files.append({"path": rel, "status": "skipped", "error": str(exc)})
            continue
        if not records or len(records) < args.min_cases:
            continue
        row = {
            "source": rel,
            "method": method,
            **summarize(records, raw_count=len(raw), duplicates=duplicates),
        }
        summary_rows.append(row)
        all_records.extend(records)
        manifest_files.append(
            {
                "path": rel,
                "status": "parsed",
                "raw_records": len(raw),
                "unique_cases": len(records),
                "duplicates": duplicates,
            }
        )

    summary_rows.sort(key=lambda row: (row["method"], row["source"]))
    write_case_csv(output_dir / "per_case.csv", all_records)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_markdown(output_dir / "summary.md", summary_rows)
    manifest = {
        "root": str(root),
        "files": manifest_files,
        "parsed_files": len(summary_rows),
        "cases": len(all_records),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Wrote {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
