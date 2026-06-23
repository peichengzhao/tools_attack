#!/usr/bin/env python3
"""Filter historical per-case results to the frozen AAAI27-S1-v1 split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from aaai27_supplement.metrics import (  # noqa: E402
    canonical_content_sha256,
    load_split_lookup,
)


SKIP_PARTS = {"__pycache__", ".git", "offline_poison"}


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".json":
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        try:
            value = json.loads(text)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [value]
        except json.JSONDecodeError:
            return [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    lookup = load_split_lookup(args.manifest)
    written_files = 0
    written_records = 0
    unmatched_records = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        records = load_records(path)
        selected = []
        for record in records:
            official = lookup.get(canonical_content_sha256(record))
            if official is None:
                unmatched_records += 1
                continue
            if official["split"] != args.split:
                continue
            item = dict(record)
            item["official_case_id"] = official["case_id"]
            item["split"] = official["split"]
            item["attack_type"] = official["attack_type"]
            selected.append(item)
        if not selected:
            continue
        rel = path.relative_to(root)
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for item in selected:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        written_files += 1
        written_records += len(selected)

    summary = {
        "root": str(root),
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split,
        "files": written_files,
        "records": written_records,
        "unmatched_records": unmatched_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "filter_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
