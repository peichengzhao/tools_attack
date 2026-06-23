"""Validate cross-eval experiment setup without calling the victim API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.cross_eval.attack_payloads import (
    AttackPayloadProvider,
)
from src.cross_eval.chatinject_mapping import (
    CASE_GOAL_MAP_PATH,
    CHATINJECT_MULTITURN_JSON,
    load_case_goal_map,
)
from src.cross_eval.constants import (
    CROSS_EVAL_ATTACKS,
    CROSS_EVAL_DEFENSES,
    CROSS_EVAL_MODELS,
    OFFLINE_POISON_ATTACKS,
)
from src.cross_eval.manifest import iter_cells, write_manifest_csv, write_manifest_json
from src.evaluate_attack_agent import make_case_id
from src.prompts.static_defense_baselines import (
    apply_static_defense_to_prompts,
    get_static_defense_strategy,
    prepare_tool_response,
)
from src.prompts.protect_prompt import apply_defense_strategy, empty_defense_strategy


def _check_file(path: Path, label: str, errors: list[str], warnings: list[str]) -> bool:
    if not path.exists():
        errors.append(f"Missing {label}: {path}")
        return False
    if path.stat().st_size == 0:
        warnings.append(f"Empty {label}: {path}")
        return False
    return True


def validate_poison_generation(
    cases: list[dict],
    ours_attacks: str = "",
    allow_offline_fallback: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    sample = cases[0]
    for attack in CROSS_EVAL_ATTACKS:
        provider = AttackPayloadProvider(
            attack,
            "qwen3.5-27b",
            "base",
            cases,
            ours_attack_path=ours_attacks,
            allow_offline_fallback=allow_offline_fallback,
        )
        failures = 0
        for item in cases:
            try:
                poison, _ = provider.get_poison(item)
                if not poison or not str(poison).strip():
                    failures += 1
            except Exception:
                failures += 1
        try:
            sample_len = len(provider.get_poison(sample)[0])
        except Exception:
            sample_len = 0
        report[attack] = {
            "ok": failures == 0,
            "failures": failures,
            "sample_poison_len": sample_len,
        }
    return report


def validate_defenses(sample_user_instruction: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    base_sys = "BASE_SYSTEM"
    base_user = "## Task Description\nUser Input: {input}"
    for defense in CROSS_EVAL_DEFENSES:
        try:
            if defense == "ours":
                strategy = empty_defense_strategy()
                sys_p, user_p = apply_defense_strategy(
                    base_sys,
                    base_user,
                    strategy,
                )
            else:
                sys_p, user_p = apply_static_defense_to_prompts(
                    base_sys, base_user, defense, sample_user_instruction
                )
                strategy = get_static_defense_strategy(defense, sample_user_instruction)
            obs = prepare_tool_response(
                'template "<Attacker Instruction>" end',
                "POISON",
                defense=defense,
            )
            out[defense] = bool(sys_p.strip()) and bool(user_p.strip()) and bool(obs)
        except Exception:
            out[defense] = False
    return out


def run_setup_check(
    setting: str = "base",
    write_manifest: bool = True,
    ours_attacks: str = "",
    test_data: str = "",
    allow_offline_fallback: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    data_path = (
        Path(test_data)
        if test_data
        else Path("data") / f"test_cases_ds_{setting}.json"
    )
    _check_file(data_path, "benchmark dataset", errors, warnings)
    cases: list[dict] = []
    if data_path.exists():
        cases = json.loads(data_path.read_text(encoding="utf-8"))
        if not test_data and len(cases) != 544:
            warnings.append(f"Expected 544 cases, got {len(cases)} in {data_path}")

    _check_file(CASE_GOAL_MAP_PATH, "ChatInject case map", errors, warnings)
    _check_file(CHATINJECT_MULTITURN_JSON, "ChatInject multiturn JSON", errors, warnings)

    if CASE_GOAL_MAP_PATH.exists() and cases:
        mapping = load_case_goal_map()
        if len(mapping) != len(cases):
            warnings.append(
                f"ChatInject map size {len(mapping)} != cases {len(cases)}; run build_chatinject_case_map.py"
            )

    poison_report = (
        validate_poison_generation(
            cases,
            ours_attacks,
            allow_offline_fallback=allow_offline_fallback,
        )
        if cases
        else {}
    )
    for attack, info in poison_report.items():
        if not info.get("ok"):
            warnings.append(f"Poison generation failures for {attack}: {info.get('failures')}")

    defense_report = validate_defenses(
        cases[0].get("User Instruction", "") if cases else "sample task"
    )
    for defense, ok in defense_report.items():
        if not ok:
            errors.append(f"Defense failed to apply: {defense}")

    offline_report: dict[str, Any] = {}
    for model in CROSS_EVAL_MODELS:
        offline_report[model] = {}
        for attack in OFFLINE_POISON_ATTACKS:
            provider = AttackPayloadProvider(
                attack,
                model,
                setting,
                cases,
            )
            cov = provider.coverage_summary() or {}
            offline_report[model][attack] = cov
            if cov["cached"] < cov["expected"]:
                warnings.append(
                    f"Offline poison incomplete: {model}/{attack} "
                    f"{cov['cached']}/{cov['expected']}"
                )

    cells = iter_cells()
    manifest_dir = Path("data/cross_eval")
    if write_manifest:
        write_manifest_csv(manifest_dir / "cell_manifest.csv", cells)
        write_manifest_json(manifest_dir / "cell_manifest.json", cells)

    status = "ok" if not errors else "error"
    if status == "ok" and warnings:
        status = "ok_with_warnings"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "cases": len(cases),
        "cells": len(cells),
        "poison_generation": poison_report,
        "defenses": defense_report,
        "offline_poison": offline_report,
        "manifest": {
            "csv": str(manifest_dir / "cell_manifest.csv"),
            "json": str(manifest_dir / "cell_manifest.json"),
        },
    }
