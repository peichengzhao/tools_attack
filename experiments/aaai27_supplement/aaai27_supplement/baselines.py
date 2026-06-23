"""Registry and validation helpers for external defense baselines."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    display_name: str
    source_url: str
    commit: str
    provenance: str
    repo_dir: str
    benchmarks: tuple[str, ...]
    entrypoints: tuple[str, ...]
    runner: str
    notes: str
    representative_defense: str | None = None
    selection_rationale: str = ""

    @classmethod
    def from_dict(cls, baseline_id: str, payload: dict[str, Any]) -> "BaselineSpec":
        return cls(
            baseline_id=baseline_id,
            display_name=payload["display_name"],
            source_url=payload["source_url"],
            commit=payload["commit"],
            provenance=payload["provenance"],
            repo_dir=payload["repo_dir"],
            benchmarks=tuple(payload.get("benchmarks", [])),
            entrypoints=tuple(payload.get("entrypoints", [])),
            runner=payload["runner"],
            notes=payload.get("notes", ""),
            representative_defense=payload.get("representative_defense"),
            selection_rationale=payload.get("selection_rationale", ""),
        )


def load_registry(path: Path) -> dict[str, BaselineSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        baseline_id: BaselineSpec.from_dict(baseline_id, item)
        for baseline_id, item in payload["baselines"].items()
    }


def repo_path(repo_root: Path, spec: BaselineSpec) -> Path:
    path = (repo_root / spec.repo_dir).resolve()
    root = repo_root.resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Baseline path escapes repository: {path}")
    return path


def validate_baseline(repo_root: Path, spec: BaselineSpec) -> dict[str, Any]:
    path = repo_path(repo_root, spec)
    commit_file = path / "VENDORED_COMMIT.txt"
    actual_commit = (
        commit_file.read_text(encoding="utf-8").strip()
        if commit_file.is_file()
        else None
    )
    entrypoints = {entry: (path / entry).exists() for entry in spec.entrypoints}
    errors: list[str] = []
    if not path.is_dir():
        errors.append(f"missing repository directory: {path}")
    if actual_commit != spec.commit:
        errors.append(
            f"commit marker mismatch: expected {spec.commit}, got {actual_commit}"
        )
    missing_entries = [entry for entry, exists in entrypoints.items() if not exists]
    if missing_entries:
        errors.append(f"missing entrypoints: {', '.join(missing_entries)}")
    return {
        "id": spec.baseline_id,
        "display_name": spec.display_name,
        "path": str(path),
        "source_url": spec.source_url,
        "commit": spec.commit,
        "provenance": spec.provenance,
        "benchmarks": list(spec.benchmarks),
        "entrypoints": entrypoints,
        "ok": not errors,
        "errors": errors,
        "notes": spec.notes,
        "representative_defense": spec.representative_defense,
        "selection_rationale": spec.selection_rationale,
    }


def read_model_runtime_config(path: Path) -> dict[str, str]:
    """Read literal constants from the model config without importing it."""
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, str):
            values[target.id] = value
    return values
