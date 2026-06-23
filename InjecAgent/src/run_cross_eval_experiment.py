#!/usr/bin/env python3
"""
Orchestrate the full Cross-Eval experiment pipeline (Experiment 10).

Phases
------
  setup     — ChatInject map + validate manifest (no victim API)
  offline   — Enhanced + Adaptive poison optimization per model, then export cache
  cross_eval — 128 cells (or filtered); ASR (Utility) per cell
  report    — Regenerate CSV / pivot tables from existing jsonl outputs
  all       — setup → offline → cross_eval → report

Examples
--------
  # Dry-run: print commands only
  python src/run_cross_eval_experiment.py --phase all --dry_run

  # Smoke: 32 cases, one model, ready attacks only (no Enhanced/Adaptive rows)
  python src/run_cross_eval_experiment.py --phase cross_eval --smoke --limit 32 \\
    --model qwen3.5-27b --ready_attacks_only

  # Full matrix for all models (resume with --use_cache)
  python src/run_cross_eval_experiment.py --phase cross_eval --use_cache

  # Offline poison only for one victim
  python src/run_cross_eval_experiment.py --phase offline --model qwen3.5-27b --use_cache
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

INJECAGENT_ROOT = Path(__file__).resolve().parents[1]
if str(INJECAGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(INJECAGENT_ROOT))

from src.cross_eval.constants import (  # noqa: E402
    CROSS_EVAL_ATTACKS,
    CROSS_EVAL_DEFENSES,
    CROSS_EVAL_MODELS,
    expand_model_selection,
    OFFLINE_POISON_ATTACKS,
)
from src.cross_eval.scoring import (  # noqa: E402
    collect_cross_eval_results,
    write_pivot_markdown,
    write_summary_csv,
)
from src.cross_eval.setup_check import run_setup_check  # noqa: E402

PYTHON = sys.executable
LOG_DIR = INJECAGENT_ROOT / "results" / "cross_eval" / "logs"
STATE_FILE = INJECAGENT_ROOT / "results" / "cross_eval" / "run_state.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expand_models(selected: str) -> list[str]:
    return expand_model_selection(selected)


def run_command(cmd: list[str], *, dry_run: bool, log_file: Path | None = None) -> int:
    line = " ".join(cmd)
    print(f"\n>>> {line}")
    if dry_run:
        return 0

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"# started {_utc_now()}\n# cmd: {line}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(INJECAGENT_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\n# exit_code {proc.returncode} finished {_utc_now()}\n")
    print(f"    exit={proc.returncode}  log={log_file}")
    return proc.returncode


def phase_setup(
    *,
    dry_run: bool,
    ours_attacks: str = "",
    test_data: str = "",
    allow_offline_fallback: bool = False,
) -> dict:
    steps = []
    rc_map = {}

    cmd_map = [PYTHON, "src/build_chatinject_case_map.py"]
    log = LOG_DIR / "setup_build_chatinject_map.log"
    rc_map["build_chatinject_map"] = run_command(cmd_map, dry_run=dry_run, log_file=log)
    steps.append("build_chatinject_map")

    if not dry_run:
        report = run_setup_check(
            write_manifest=True,
            ours_attacks=ours_attacks,
            test_data=test_data,
            allow_offline_fallback=allow_offline_fallback,
        )
    else:
        report = {"status": "dry_run", "cells": 128}
    rc_map["validate"] = 0 if report.get("status") != "error" else 1
    steps.append("validate_setup")
    return {"phase": "setup", "steps": steps, "returncodes": rc_map, "report": report}


def phase_offline(
    models: list[str],
    *,
    dry_run: bool,
    use_cache: bool,
    limit: int,
) -> dict:
    rc_map: dict[str, int] = {}
    for model in models:
        for script, key in (
            ("src/evaluate_attack_agent.py", "enhanced"),
            ("src/evaluate_adaptive_attack.py", "adaptive"),
        ):
            cmd = [
                PYTHON,
                script,
                "--model_name",
                model,
                "--setting",
                "base",
                "--prompt_type",
                "InjecAgent",
            ]
            if use_cache:
                cmd.append("--use_cache")
            if limit > 0:
                cmd.extend(["--limit", str(limit)])
            log = LOG_DIR / f"offline_{model.replace('.', '_')}_{key}.log"
            rc = run_command(cmd, dry_run=dry_run, log_file=log)
            rc_map[f"{model}/{key}"] = rc

    cmd = [PYTHON, "src/build_offline_poison_cache.py", "--model", "all", "--attack", "all"]
    log = LOG_DIR / "offline_build_cache.log"
    rc_map["build_offline_poison_cache"] = run_command(cmd, dry_run=dry_run, log_file=log)
    return {"phase": "offline", "models": models, "returncodes": rc_map}


def phase_cross_eval(
    models: list[str],
    *,
    dry_run: bool,
    use_cache: bool,
    limit: int,
    smoke: bool,
    ready_attacks_only: bool,
    defense: str,
    attack: str,
    harness_file: str,
    ours_attacks: str,
    test_data: str,
    allow_offline_fallback: bool,
) -> dict:
    rc_map: dict[str, int] = {}
    for model in models:
        cmd = [
            PYTHON,
            "src/evaluate_cross_eval.py",
            "--model",
            model,
            "--defense",
            defense,
            "--attack",
            attack,
            "--setting",
            "base",
            "--prompt_type",
            "InjecAgent",
        ]
        if smoke:
            cmd.append("--smoke")
        if use_cache:
            cmd.append("--use_cache")
        if limit > 0:
            cmd.extend(["--limit", str(limit)])
        if ready_attacks_only:
            cmd.append("--skip_offline_attacks")
        if harness_file:
            cmd.extend(["--harness_file", harness_file])
        if ours_attacks:
            cmd.extend(["--ours_attacks", ours_attacks])
        if test_data:
            cmd.extend(["--test_data", test_data])
        if allow_offline_fallback:
            cmd.append("--allow_offline_fallback")

        tag = "smoke" if smoke else "full"
        log = LOG_DIR / f"cross_eval_{tag}_{model.replace('.', '_')}.log"
        rc_map[model] = run_command(cmd, dry_run=dry_run, log_file=log)
    return {"phase": "cross_eval", "models": models, "returncodes": rc_map}


def phase_report(*, smoke: bool) -> dict:
    subdir = "smoke" if smoke else "full"
    results_dir = INJECAGENT_ROOT / "results" / "cross_eval" / subdir
    rows = collect_cross_eval_results(results_dir)
    if not rows:
        return {"phase": "report", "status": "no_results", "results_dir": str(results_dir)}

    csv_path = write_summary_csv(rows, results_dir / "summary_cross_eval.csv")
    pivots = []
    for model in CROSS_EVAL_MODELS:
        md = write_pivot_markdown(
            rows,
            results_dir / f"pivot_{model.replace('.', '_')}.md",
            model,
        )
        pivots.append(str(md))

    complete = sum(1 for r in rows if r.get("n", 0) > 0)
    return {
        "phase": "report",
        "status": "ok",
        "results_dir": str(results_dir),
        "csv": str(csv_path),
        "pivots": pivots,
        "cells_with_data": complete,
        "cells_expected": len(CROSS_EVAL_MODELS) * len(CROSS_EVAL_DEFENSES) * len(CROSS_EVAL_ATTACKS),
    }


def save_state(payload: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if STATE_FILE.exists():
        try:
            history = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("history", [])
        except json.JSONDecodeError:
            history = []
    history.append(payload)
    STATE_FILE.write_text(
        json.dumps({"updated": _utc_now(), "history": history[-20:]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cross-Eval experiment pipeline")
    parser.add_argument(
        "--phase",
        choices=["setup", "offline", "cross_eval", "report", "all"],
        default="cross_eval",
        help="Pipeline phase to run (default: cross_eval)",
    )
    parser.add_argument(
        "--model",
        default="all",
        help=f"Victim model or 'all' ({', '.join(CROSS_EVAL_MODELS)})",
    )
    parser.add_argument(
        "--defense",
        default="all",
        help="Defense column or 'all' (passed to evaluate_cross_eval.py)",
    )
    parser.add_argument(
        "--attack",
        default="all",
        help="Attack row or 'all' (passed to evaluate_cross_eval.py)",
    )
    parser.add_argument("--smoke", action="store_true", help="Use results/cross_eval/smoke/")
    parser.add_argument(
        "--harness_file",
        default="",
        help="Validation-selected harness JSON for the Ours defense column",
    )
    parser.add_argument(
        "--ours_attacks",
        default="",
        help="Frozen Ours-Attack JSONL for the Ours-Attack row",
    )
    parser.add_argument(
        "--test_data",
        default="",
        help="Custom case JSON, e.g. InjecAgent/data/splits/test_cases_211.json",
    )
    parser.add_argument(
        "--allow_offline_fallback",
        action="store_true",
        help="Use seed attack for missing Adaptive/Enhanced caches (smoke only)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit cases per cell (0=all 544)")
    parser.add_argument("--use_cache", action="store_true", help="Resume from partial outputs")
    parser.add_argument(
        "--ready_attacks_only",
        action="store_true",
        help="Skip Enhanced/Adaptive rows (6 attacks × 4 defenses × N models)",
    )
    parser.add_argument(
        "--skip_setup",
        action="store_true",
        help="With --phase all, skip setup phase",
    )
    parser.add_argument(
        "--skip_offline",
        action="store_true",
        help="With --phase all, skip offline poison phase",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands only; do not invoke scripts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = expand_models(args.model)
    started = time.time()
    results: list[dict] = []

    print(
        json.dumps(
            {
                "phase": args.phase,
                "models": models,
                "smoke": args.smoke,
                "limit": args.limit,
                "use_cache": args.use_cache,
                "ready_attacks_only": args.ready_attacks_only,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    phases_to_run: list[str]
    if args.phase == "all":
        phases_to_run = []
        if not args.skip_setup:
            phases_to_run.append("setup")
        if not args.skip_offline:
            phases_to_run.append("offline")
        phases_to_run.extend(["cross_eval", "report"])
    else:
        phases_to_run = [args.phase]

    exit_code = 0
    for phase in phases_to_run:
        if phase == "setup":
            res = phase_setup(
                dry_run=args.dry_run,
                ours_attacks=args.ours_attacks,
                test_data=args.test_data,
                allow_offline_fallback=args.allow_offline_fallback,
            )
        elif phase == "offline":
            res = phase_offline(
                models,
                dry_run=args.dry_run,
                use_cache=args.use_cache,
                limit=args.limit,
            )
        elif phase == "cross_eval":
            res = phase_cross_eval(
                models,
                dry_run=args.dry_run,
                use_cache=args.use_cache,
                limit=args.limit,
                smoke=args.smoke,
                ready_attacks_only=args.ready_attacks_only,
                defense=args.defense,
                attack=args.attack,
                harness_file=args.harness_file,
                ours_attacks=args.ours_attacks,
                test_data=args.test_data,
                allow_offline_fallback=args.allow_offline_fallback,
            )
        elif phase == "report":
            res = phase_report(smoke=args.smoke)
        else:
            raise ValueError(phase)

        results.append(res)
        rc_values = [v for v in res.get("returncodes", {}).values() if isinstance(v, int)]
        if rc_values and any(v != 0 for v in rc_values):
            exit_code = 1
            print(f"WARNING: phase {phase} had non-zero exit code(s)")

    payload = {
        "started": _utc_now(),
        "elapsed_sec": round(time.time() - started, 1),
        "args": vars(args),
        "results": results,
        "exit_code": exit_code,
    }
    if not args.dry_run:
        save_state(payload)

    print("\n" + json.dumps(payload, indent=2, ensure_ascii=False))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
