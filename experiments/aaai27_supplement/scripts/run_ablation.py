#!/usr/bin/env python3
"""Bounded co-evolution ablation runner for AAAI-27 supplement experiments."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# The script is launched from InjecAgent/, but Python otherwise places this
# script's own directory on sys.path instead of the current working directory.
sys.path.insert(0, str(Path.cwd()))

from src.evaluate_attack_agent import build_tool_response, load_cached_results, make_case_id
from src.evaluate_prompted_agent import predict_one_case
from src.evaluate_protect_agent import (
    call_optimizer,
    call_protect_agent,
    empty_defense_strategy,
    evaluate_victim_round,
)
from src.models import MODELS
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
from src.utils import get_tool_dict


MODES = (
    "fixed",
    "attack_only",
    "defense_only",
    "full",
    "independent",
    "simultaneous",
    "no_feedback",
    "random_routing",
    "round_robin",
)


def round_is_hit(eval_result: str) -> bool:
    return eval_result in ("succ", "invalid")


def ablate_one_case(
    item: dict,
    base_system_prompt: str,
    base_user_prompt: str,
    params: dict,
    model_class,
    tool_dict: dict,
    rng: random.Random,
) -> dict:
    mode = params["ablation_mode"]
    history: list[dict] = []
    poison_text = item["Attacker Instruction"]
    defense_strategy = empty_defense_strategy()

    # Independent mode keeps an attack stream trained against the base harness
    # and a defense stream trained against the original seed poison.
    attack_poison = poison_text
    defense_poison_seed = item["Attacker Instruction"]

    first_hit_round = None
    protect_triggers = 0
    optimizer_triggers = 0
    final_output = ""
    final_eval = "unsucc"
    final_invalid_reason = ""

    for round_num in range(1, params["max_rounds"] + 1):
        current_poison = attack_poison if mode == "independent" else poison_text
        output, parsed, applied_instruction, defense_snapshot = evaluate_victim_round(
            item,
            current_poison,
            base_system_prompt,
            base_user_prompt,
            defense_strategy,
            params,
            model_class,
            tool_dict,
        )
        eval_result = parsed["eval"]
        record = {
            "round": round_num,
            "poison": current_poison,
            "defense_strategy": copy.deepcopy(defense_snapshot),
            "applied_instruction": applied_instruction,
            "eval": eval_result,
            "utility_eval": parsed.get("utility_eval", "invalid"),
            "victim_output": output,
        }
        if eval_result == "invalid":
            record["invalid_reason"] = parsed.get("invalid_reason", "")
        if round_is_hit(eval_result) and first_hit_round is None:
            first_hit_round = round_num

        final_output = output or final_output
        final_eval = eval_result
        final_invalid_reason = parsed.get("invalid_reason", "")
        post_round_action = None

        if round_num < params["max_rounds"]:
            if mode == "fixed":
                pass
            elif mode == "attack_only":
                if eval_result == "unsucc":
                    poison_text, record["optimizer"] = call_optimizer(
                        item,
                        history + [record],
                        round_num + 1,
                        params,
                        defense_strategy,
                    )
                    optimizer_triggers += 1
                    post_round_action = "optimize_poison"
            elif mode == "defense_only":
                if round_is_hit(eval_result):
                    defense_strategy, record["protect"] = call_protect_agent(
                        item,
                        history + [record],
                        round_num,
                        params,
                        poison_text,
                        output or "",
                        defense_strategy,
                    )
                    protect_triggers += 1
                    post_round_action = "optimize_defense"
            elif mode == "full":
                if round_is_hit(eval_result):
                    defense_strategy, record["protect"] = call_protect_agent(
                        item,
                        history + [record],
                        round_num,
                        params,
                        poison_text,
                        output or "",
                        defense_strategy,
                    )
                    protect_triggers += 1
                    post_round_action = "optimize_defense"
                elif eval_result == "unsucc":
                    poison_text, record["optimizer"] = call_optimizer(
                        item,
                        history + [record],
                        round_num + 1,
                        params,
                        defense_strategy,
                    )
                    optimizer_triggers += 1
                    post_round_action = "optimize_poison"
            elif mode == "independent":
                if eval_result == "unsucc":
                    attack_poison, record["optimizer"] = call_optimizer(
                        item,
                        history + [record],
                        round_num + 1,
                        params,
                        empty_defense_strategy(),
                    )
                    optimizer_triggers += 1
                    post_round_action = "optimize_poison_independent"
                if round_is_hit(eval_result):
                    defense_strategy, record["protect"] = call_protect_agent(
                        item,
                        history + [record],
                        round_num,
                        params,
                        defense_poison_seed,
                        output or "",
                        defense_strategy,
                    )
                    protect_triggers += 1
                    post_round_action = "optimize_defense_independent"
            elif mode == "simultaneous":
                poison_text, record["optimizer"] = call_optimizer(
                    item,
                    history + [record],
                    round_num + 1,
                    params,
                    defense_strategy,
                )
                defense_strategy, record["protect"] = call_protect_agent(
                    item,
                    history + [record],
                    round_num,
                    params,
                    poison_text,
                    output or "",
                    defense_strategy,
                )
                optimizer_triggers += 1
                protect_triggers += 1
                post_round_action = "optimize_both"
            elif mode == "no_feedback":
                if round_is_hit(eval_result):
                    defense_strategy, record["protect"] = call_protect_agent(
                        item,
                        [],
                        round_num,
                        params,
                        poison_text,
                        "",
                        defense_strategy,
                    )
                    protect_triggers += 1
                    post_round_action = "optimize_defense_no_feedback"
                elif eval_result == "unsucc":
                    poison_text, record["optimizer"] = call_optimizer(
                        item,
                        [],
                        round_num + 1,
                        params,
                        defense_strategy,
                    )
                    optimizer_triggers += 1
                    post_round_action = "optimize_poison_no_feedback"
            elif mode in ("random_routing", "round_robin"):
                if mode == "random_routing":
                    side = rng.choice(("attack", "defense"))
                else:
                    side = "attack" if round_num % 2 else "defense"
                if side == "attack":
                    poison_text, record["optimizer"] = call_optimizer(
                        item,
                        history + [record],
                        round_num + 1,
                        params,
                        defense_strategy,
                    )
                    optimizer_triggers += 1
                    post_round_action = "optimize_poison_unconditional"
                else:
                    defense_strategy, record["protect"] = call_protect_agent(
                        item,
                        history + [record],
                        round_num,
                        params,
                        poison_text,
                        output or "",
                        defense_strategy,
                    )
                    protect_triggers += 1
                    post_round_action = "optimize_defense_unconditional"
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        record["post_round_action"] = post_round_action
        history.append(record)

    result = copy.deepcopy(item)
    result["ablation_mode"] = mode
    result["ablation_seed"] = params["ablation_seed"]
    result["coevolution_history"] = history
    result["final_poison"] = attack_poison if mode == "independent" else poison_text
    result["final_defense_strategy"] = defense_strategy
    result["attacked_any_round"] = any(
        round_is_hit(record.get("eval", "")) for record in history
    )
    result["first_success_round"] = first_hit_round
    result["protect_triggers"] = protect_triggers
    result["optimizer_triggers"] = optimizer_triggers
    result["rounds_used"] = len(history)
    result["output"] = final_output
    result["eval"] = final_eval
    result["utility_final"] = history[-1].get("utility_eval") if history else "invalid"
    if final_eval == "invalid":
        result["invalid_reason"] = final_invalid_reason
    result["Tool Response"] = build_tool_response(
        item["Tool Response Template"], result["final_poison"], params["setting"]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model_type", default="GPT")
    parser.add_argument("--model_name", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--optimizer_model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--protect_model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--setting", choices=["base", "enhanced"], default="base")
    parser.add_argument(
        "--prompt_type", choices=["InjecAgent", "hwchase17_react"], default="InjecAgent"
    )
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--data",
        default="data/test_cases_ds_base.json",
        help="Case file to run, e.g. data/splits/train_cases_632.json",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "optimizer_model": args.optimizer_model,
        "protect_model": args.protect_model,
        "setting": args.setting,
        "prompt_type": args.prompt_type,
        "max_rounds": args.max_rounds,
        "limit": args.limit,
        "use_cache": args.use_cache,
        "only_get_score": False,
        "verbose": False,
        "ablation_mode": args.mode,
        "ablation_seed": args.seed,
    }
    print(json.dumps(params, indent=2))
    if args.dry_run:
        return 0

    data_path = Path(args.data)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        data = data[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cached_results(str(output_path)) if args.use_cache else {}
    model_class = MODELS[params["model_type"]](params)
    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]
    tool_dict = get_tool_dict()
    todo = []
    for index, item in enumerate(data):
        case_id = make_case_id(item)
        if case_id not in cached or cached[case_id].get("output") is None:
            todo.append((index, item))

    lock = threading.Lock()
    with output_path.open("w", encoding="utf-8") as handle:
        for item in cached.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()

        def run_item(index_and_item):
            index, item = index_and_item
            case_rng = random.Random(args.seed + index)
            return ablate_one_case(
                item,
                system_prompt,
                user_prompt,
                params,
                model_class,
                tool_dict,
                case_rng,
            )

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(run_item, item): item[1] for item in todo
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"ablation:{args.mode}",
            ):
                try:
                    result = future.result()
                except Exception as exc:
                    item = futures[future]
                    result = copy.deepcopy(item)
                    result.update(
                        {
                            "ablation_mode": args.mode,
                            "ablation_seed": args.seed,
                            "coevolution_history": [],
                            "final_poison": item.get("Attacker Instruction", ""),
                            "final_defense_strategy": empty_defense_strategy(),
                            "attacked_any_round": False,
                            "first_success_round": None,
                            "protect_triggers": 0,
                            "optimizer_triggers": 0,
                            "rounds_used": 0,
                            "output": "",
                            "eval": "invalid",
                            "invalid_reason": "case_exception",
                            "case_error": str(exc),
                        }
                    )
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
