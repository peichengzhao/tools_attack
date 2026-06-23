#!/usr/bin/env python3
"""Evaluate candidate defense harnesses on frozen validation attacks."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(Path.cwd()))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from aaai27_supplement.metrics import records_from_file, summarize  # noqa: E402
from src.evaluate_attack_agent import build_tool_response, make_case_id  # noqa: E402
from src.evaluate_prompted_agent import predict_one_case  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.output_parsing import evaluate_output_prompted  # noqa: E402
from src.prompts.agent_prompts import PROMPT_DICT  # noqa: E402
from src.prompts.protect_prompt import apply_defense_strategy  # noqa: E402
from src.utility_parsing import attach_user_task_utility  # noqa: E402
from src.utils import get_tool_dict  # noqa: E402


def evaluate_one(
    record: dict,
    defense_strategy: dict,
    params: dict,
    model_class,
    tool_dict: dict,
) -> dict:
    poison = record.get("frozen_poison") or record.get("Attacker Instruction", "")
    item = copy.deepcopy(record)
    item["Tool Response"] = build_tool_response(
        record["Tool Response Template"], poison, params["setting"]
    )
    base_system, base_user = PROMPT_DICT[params["prompt_type"]]
    system_prompt, user_prompt = apply_defense_strategy(
        base_system, base_user, defense_strategy
    )
    output = predict_one_case(
        system_prompt,
        user_prompt,
        params,
        item,
        model_class,
        tool_dict,
    )
    parsed = evaluate_output_prompted(
        output,
        record["Attacker Tools"][0],
        record["Attacker Tools"],
        record["User Tool"],
        poison,
    )
    result = item
    result["output"] = output
    result["eval"] = parsed["eval"]
    result["defense_candidate_id"] = params["defense_candidate_id"]
    result["defense_strategy"] = defense_strategy
    if parsed["eval"] == "invalid":
        result["invalid_reason"] = parsed.get("invalid_reason", "")
    attach_user_task_utility(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-selection", required=True)
    parser.add_argument("--model_name", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--setting", choices=["base", "enhanced"], default="base")
    parser.add_argument(
        "--prompt_type", choices=["InjecAgent", "hwchase17_react"], default="InjecAgent"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_attacks(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".json":
        value = json.loads(text)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    attacks = load_attacks(Path(args.attacks))
    if args.limit > 0:
        attacks = attacks[: args.limit]
    candidate_data = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    candidates = candidate_data["candidates"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model_type": "GPT",
        "model_name": args.model_name,
        "setting": args.setting,
        "prompt_type": args.prompt_type,
        "limit": args.limit,
        "use_cache": False,
        "only_get_score": False,
    }
    print(
        json.dumps(
            {
                "attacks": len(attacks),
                "candidates": len(candidates),
                "workers": args.workers,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    model_class = MODELS["GPT"](params)
    tool_dict = get_tool_dict()
    metrics = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        output_path = output_dir / f"validation_{candidate_id}.jsonl"
        params["defense_candidate_id"] = candidate_id
        lock = threading.Lock()
        with output_path.open("w", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = {
                    pool.submit(
                        evaluate_one,
                        record,
                        candidate["strategy"],
                        params,
                        model_class,
                        tool_dict,
                    ): record
                    for record in attacks
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"harness:{candidate_id}",
                ):
                    try:
                        result = future.result()
                    except Exception as exc:
                        record = futures[future]
                        result = copy.deepcopy(record)
                        result.update(
                            {
                                "output": "",
                                "eval": "invalid",
                                "invalid_reason": "case_exception",
                                "case_error": str(exc),
                                "defense_candidate_id": candidate_id,
                                "defense_strategy": candidate["strategy"],
                            }
                        )
                    with lock:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
        _, records, duplicates = records_from_file(
            output_path,
            source=str(output_path),
            method=candidate_id,
        )
        summary = summarize(records, raw_count=len(records) + duplicates, duplicates=duplicates)
        attack_hit_rate = (
            (summary["attack_success_final"] + summary["invalid"]) / summary["total_cases"]
            if summary["total_cases"]
            else 1.0
        )
        metrics.append(
            {
                "candidate": candidate,
                "summary": summary,
                "attack_hit_rate": round(attack_hit_rate, 6),
                "output": str(output_path),
            }
        )

    metrics.sort(
        key=lambda item: (
            item["attack_hit_rate"],
            item["summary"]["invalid_rate_pct"] or 0,
            item["summary"]["asr_all_pct"] or 0,
            -(item["summary"]["utility_pct_all"] or 0),
        )
    )
    selection = {
        "version": "AAAI27-harness-validation-selection-v1",
        "attacks": args.attacks,
        "candidate_file": args.candidates,
        "selected": metrics[0],
        "ranking": metrics,
    }
    Path(args.output_selection).write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": metrics[0]["candidate"]["candidate_id"],
                "attack_hit_rate": metrics[0]["attack_hit_rate"],
                "summary": metrics[0]["summary"],
                "ranking": [
                    {
                        "id": item["candidate"]["candidate_id"],
                        "attack_hit_rate": item["attack_hit_rate"],
                        "asr_all_pct": item["summary"]["asr_all_pct"],
                        "invalid_rate_pct": item["summary"]["invalid_rate_pct"],
                        "utility_pct_all": item["summary"]["utility_pct_all"],
                    }
                    for item in metrics
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
