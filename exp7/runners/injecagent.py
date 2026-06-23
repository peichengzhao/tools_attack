from __future__ import annotations

import json
import sys
from pathlib import Path

from exp7.config import METRIC_LABELS, TOOLS_ATTACK_ROOT
from exp7.runners._common import make_result, openai_env, run_cmd, write_result

INJEC_ROOT = TOOLS_ATTACK_ROOT / "InjecAgent"


def run(*, model: str, harness: str, force: bool = False, limit: int = 0) -> dict:
    metric = METRIC_LABELS["injecagent"]
    defense = "none" if harness == "baseline" else "max_defense"
    cmd = [
        sys.executable,
        "src/evaluate_injecagent_utility.py",
        "--defense",
        defense,
        "--model_name",
        model,
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if not force:
        cmd.append("--use_cache")

    env = openai_env(model)
    proc = run_cmd(cmd, cwd=INJEC_ROOT, env=env)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        payload = make_result(
            benchmark="injecagent",
            model=model,
            harness=harness,
            utility=None,
            metric_name=metric,
            status="error",
            error=(stderr or stdout)[-2000:],
        )
        write_result(payload)
        return payload

    scores = _parse_json_block(stdout)
    utility = _float_or_none(scores.get("Utility Rate (succ)"))
    payload = make_result(
        benchmark="injecagent",
        model=model,
        harness=harness,
        utility=utility,
        metric_name=metric,
        details={
            "scores": scores,
            "over_refusal": scores.get("Over-Refusal Rate"),
            "utility_invalid": scores.get("Invalid Rate"),
        },
    )
    write_result(payload)
    return payload


def _parse_json_block(text: str) -> dict:
    """Parse score JSON from evaluate_injecagent_utility stdout.

    The script prints `params` as a Python repr (single quotes) before the
    real JSON score blocks — a single json.loads over the whole stdout fails.
    """
    decoder = json.JSONDecoder()
    idx = 0
    scores: dict = {}
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and "Utility Rate (succ)" in obj:
            scores = obj
        idx = end
    return scores


def _float_or_none(value) -> float | None:
    if value is None or value == "-":
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return None
