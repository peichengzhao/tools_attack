#!/usr/bin/env python3
"""Evaluate an exported co-evolved poison corpus without a defense."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from tqdm import tqdm

# The script is launched from InjecAgent/, but Python otherwise places this
# script's own directory on sys.path instead of the current working directory.
sys.path.insert(0, str(Path.cwd()))

from src.evaluate_attack_agent import build_tool_response, load_cached_results, make_case_id
from src.evaluate_prompted_agent import predict_one_case
from src.models import MODELS
from src.output_parsing import evaluate_output_prompted
from src.prompts.agent_prompts import PROMPT_DICT
from src.utility_parsing import attach_user_task_utility
from src.utils import get_tool_dict


def load_poison_map(path: Path) -> dict[str, str]:
    poison_map: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            case_id = record.get("case_id") or make_case_id(record)
            poison = (
                record.get("best_poison")
                or record.get("final_poison")
                or record.get("poison")
            )
            if case_id and poison:
                poison_map[str(case_id)] = str(poison)
    return poison_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison-source", required=True)
    parser.add_argument("--data", default="data/test_cases_ds_base.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_type", default="GPT")
    parser.add_argument("--model_name", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--setting", choices=["base", "enhanced"], default="base")
    parser.add_argument(
        "--prompt_type", choices=["InjecAgent", "hwchase17_react"], default="InjecAgent"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    poison_source = Path(args.poison_source)
    data_path = Path(args.data)
    output_path = Path(args.output)
    params = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "setting": args.setting,
        "prompt_type": args.prompt_type,
        "limit": args.limit,
        "use_cache": args.use_cache,
        "only_get_score": False,
    }

    print(
        json.dumps(
            {
                "poison_source": str(poison_source),
                "data": str(data_path),
                "output": str(output_path),
                "params": params,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    poison_map = load_poison_map(poison_source)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        data = data[: args.limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cached_results(str(output_path)) if args.use_cache else {}
    model_class = MODELS[params["model_type"]](params)
    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]
    tool_dict = get_tool_dict()

    counts = {"succ": 0, "unsucc": 0, "invalid": 0, "fallback_seed": 0}
    with output_path.open("w", encoding="utf-8") as handle:
        for item in tqdm(data, desc=f"exported-poison:{args.model_name}"):
            case_id = make_case_id(item)
            if case_id in cached and cached[case_id].get("output") is not None:
                result = cached[case_id]
            else:
                poison = poison_map.get(case_id)
                if not poison:
                    counts["fallback_seed"] += 1
                    poison = item["Attacker Instruction"]
                case_item = copy.deepcopy(item)
                case_item["Attacker Instruction"] = poison
                case_item["exported_poison"] = poison
                case_item["exported_poison_case_id"] = case_id
                case_item["exported_poison_source"] = str(poison_source)
                case_item["Tool Response"] = build_tool_response(
                    item["Tool Response Template"], poison, args.setting
                )
                output = predict_one_case(
                    system_prompt,
                    user_prompt,
                    params,
                    case_item,
                    model_class,
                    tool_dict,
                )
                parsed = evaluate_output_prompted(
                    output,
                    item["Attacker Tools"][0],
                    item["Attacker Tools"],
                    item["User Tool"],
                    poison,
                )
                result = case_item
                result["output"] = output
                result["eval"] = parsed["eval"]
                if parsed["eval"] == "invalid":
                    result["invalid_reason"] = parsed.get("invalid_reason")
                attach_user_task_utility(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            counts[result.get("eval", "invalid")] = (
                counts.get(result.get("eval", "invalid"), 0) + 1
            )

    total = len(data)
    print(
        json.dumps(
            {
                "total": total,
                "asr_all_pct": round(counts["succ"] / total * 100, 2) if total else None,
                "asr_valid_pct": round(
                    counts["succ"] / (total - counts["invalid"]) * 100, 2
                )
                if total - counts["invalid"]
                else None,
                "counts": counts,
                "output": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
