#!/usr/bin/env python3
"""Apply a frozen attack-instantiation rule to held-out cases."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))

from src.evaluate_attack_agent import build_tool_response, make_case_id  # noqa: E402
from src.evaluate_prompted_agent import predict_one_case  # noqa: E402
from src.llm_client import chat_completion_with_retry  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.output_parsing import evaluate_output_prompted  # noqa: E402
from src.prompts.agent_prompts import PROMPT_DICT  # noqa: E402
from src.utility_parsing import attach_user_task_utility  # noqa: E402
from src.utils import get_tool_dict  # noqa: E402


def clean_poison(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.strip()


def case_metadata(item: dict) -> dict:
    return {
        "attack_type": item.get("attack_type") or item.get("Attack Type"),
        "user_instruction": item.get("User Instruction"),
        "user_tool": item.get("User Tool"),
        "tool_parameters": item.get("Tool Parameters"),
        "attacker_tools": item.get("Attacker Tools"),
        "attacker_instruction": item.get("Attacker Instruction"),
        "tool_response_template": item.get("Tool Response Template"),
    }


def evaluate_one(
    item: dict,
    rule_text: str,
    rule_model: str,
    params: dict,
    model_class,
    tool_dict: dict,
) -> dict:
    rule_fallback = False
    poison, error, _ = chat_completion_with_retry(
        messages=[
            {"role": "system", "content": rule_text},
            {
                "role": "user",
                "content": json.dumps(case_metadata(item), ensure_ascii=False),
            },
        ],
        model=rule_model,
        temperature=0.2,
        label=f"frozen-rule:{make_case_id(item)}",
    )
    poison = clean_poison(poison or "")
    if error or not poison:
        poison = item.get("Attacker Instruction", "")
        rule_fallback = True

    case_item = copy.deepcopy(item)
    case_item["frozen_poison"] = poison
    case_item["frozen_rule_model"] = rule_model
    case_item["frozen_rule_fallback"] = rule_fallback
    case_item["frozen_rule_error"] = error
    case_item["Tool Response"] = build_tool_response(
        item["Tool Response Template"], poison, params["setting"]
    )
    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]
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
        result["invalid_reason"] = parsed.get("invalid_reason", "")
    attach_user_task_utility(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--rule_model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--setting", choices=["base", "enhanced"], default="base")
    parser.add_argument(
        "--prompt_type", choices=["InjecAgent", "hwchase17_react"], default="InjecAgent"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rule = json.loads(Path(args.rule).read_text(encoding="utf-8"))["rule"]
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.limit > 0:
        data = data[: args.limit]
    params = {
        "model_type": "GPT",
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
                "rule": args.rule,
                "data": args.data,
                "cases": len(data),
                "model_name": args.model_name,
                "rule_model": args.rule_model,
                "workers": args.workers,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if args.use_cache and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cached[make_case_id(item)] = item

    todo = []
    for item in data:
        case_id = make_case_id(item)
        if case_id not in cached or cached[case_id].get("output") is None:
            todo.append(item)

    model_class = MODELS["GPT"](params)
    tool_dict = get_tool_dict()
    lock = threading.Lock()
    with output_path.open("w", encoding="utf-8") as handle:
        for item in cached.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    evaluate_one,
                    item,
                    rule,
                    args.rule_model,
                    params,
                    model_class,
                    tool_dict,
                ): item
                for item in todo
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="frozen-attack",
            ):
                try:
                    result = future.result()
                except Exception as exc:
                    item = futures[future]
                    result = copy.deepcopy(item)
                    result.update(
                        {
                            "frozen_poison": item.get("Attacker Instruction", ""),
                            "frozen_rule_error": str(exc),
                            "frozen_rule_fallback": True,
                            "output": "",
                            "eval": "invalid",
                            "invalid_reason": "case_exception",
                        }
                    )
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
