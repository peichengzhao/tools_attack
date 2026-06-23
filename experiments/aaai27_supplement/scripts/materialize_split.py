#!/usr/bin/env python3
"""Materialize the frozen AAAI27-S1-v1 split without modifying source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits/split_manifest.csv")
    parser.add_argument("--output-dir", default="data/splits")
    return_code = 0
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    if len(rows) != 1054:
        raise SystemExit(f"expected 1054 manifest rows, got {len(rows)}")

    cache: dict[str, list[dict]] = {}
    materialized: dict[str, list[dict]] = {split: [] for split in SPLITS}
    by_attack_split: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in sorted(rows, key=lambda item: item["case_id"]):
        source_file = row["source_file"]
        if source_file not in cache:
            cache[source_file] = json.loads(
                Path(source_file).read_text(encoding="utf-8")
            )
        index = int(row["source_index"]) - 1
        case = dict(cache[source_file][index])
        case["case_id"] = row["case_id"]
        case["split"] = row["split"]
        case["attack_type"] = row["attack_type"]
        case["source_file"] = source_file
        case["source_index"] = int(row["source_index"])
        materialized[row["split"]].append(case)
        by_attack_split[(row["attack_type"], row["split"])].append(case)

    counts = {split: len(materialized[split]) for split in SPLITS}
    expected = {"train": 632, "validation": 211, "test": 211}
    if counts != expected:
        raise SystemExit(f"split count mismatch: {counts}")

    names = {
        "train": "train_cases_632.json",
        "validation": "validation_cases_211.json",
        "test": "test_cases_211.json",
    }
    attack_names = {
        ("data-stealing", "train"): "train_cases_ds_326.json",
        ("data-stealing", "validation"): "validation_cases_ds_109.json",
        ("data-stealing", "test"): "test_cases_ds_109.json",
        ("direct-harm", "train"): "train_cases_dh_306.json",
        ("direct-harm", "validation"): "validation_cases_dh_102.json",
        ("direct-harm", "test"): "test_cases_dh_102.json",
    }

    hashes = {}
    for split in SPLITS:
        path = output_dir / names[split]
        write_json(path, materialized[split])
        hashes[path.name] = sha256_file(path)
    for key, name in attack_names.items():
        path = output_dir / name
        write_json(path, by_attack_split[key])
        hashes[path.name] = sha256_file(path)

    summary = {
        "manifest_version": "AAAI27-S1-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "counts": counts,
        "by_attack_type": {
            f"{attack_type}/{split}": len(items)
            for (attack_type, split), items in sorted(by_attack_split.items())
        },
        "files": hashes,
    }
    write_json(output_dir / "materialization_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
