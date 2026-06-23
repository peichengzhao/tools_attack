"""
Adaptive Attack on InjecAgent 544 S1 (no defense).

Multi-round feedback-driven poison optimization via OPTIMIZER_MODEL; victim via VICTIM_MODEL.
Primary table metric: ASR@{max_rounds} (final). Also reports ASR-any@{max_rounds} and Utility.
"""

from __future__ import annotations

import copy
import json
import os
import traceback

from tqdm import tqdm

from src.evaluate_attack_agent import (
    _fallback_poison,
    build_tool_response,
    evaluate_victim_round,
    load_cached_results,
    make_case_id,
)
from src.llm_client import chat_completion_with_retry
from src.models import MODELS
from src.params import parse_arguments
from src.prompts.adaptive_optimizer_prompt import (
    ADAPTIVE_ATTACK_OPTIMIZER_SYSTEM,
    adaptive_instruction_from_result,
    build_adaptive_optimizer_user_message,
)
from src.prompts.agent_prompts import PROMPT_DICT
from src.prompts.optimizer_prompt import parse_optimizer_response
from src.utility_parsing import attach_user_task_utility, get_s1_utility_score
from src.utils import get_adaptive_attack_score, get_tool_dict


def call_adaptive_optimizer(
    item: dict, history: list, round_num: int, params: dict
) -> tuple[str, dict]:
    fallback_poison = _fallback_poison(item, history)
    user_message = build_adaptive_optimizer_user_message(
        attacker_instruction=item["Attacker Instruction"],
        s1_target_tool=item["Attacker Tools"][0],
        user_instruction=item["User Instruction"],
        user_tool=item["User Tool"],
        tool_response_template=item["Tool Response Template"],
        victim_model=params["model_name"],
        round_num=round_num,
        max_rounds=params["max_rounds"],
        optimization_history=history,
        setting=params["setting"],
        case_id=make_case_id(item),
        attacker_tools=item["Attacker Tools"],
    )
    raw, error, _ = chat_completion_with_retry(
        messages=[
            {"role": "system", "content": ADAPTIVE_ATTACK_OPTIMIZER_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        model=params["optimizer_model"],
        label=f"adaptive:{params['optimizer_model']}",
    )
    if error:
        return fallback_poison, {
            "api_error": error,
            "skipped_optimizer": True,
            "fallback_poison": True,
        }

    parsed = parse_optimizer_response(raw or "")
    poison = adaptive_instruction_from_result(parsed)
    if not poison:
        poison = fallback_poison
        parsed["fallback_poison"] = True
    return poison, parsed


def optimize_one_case_adaptive(
    item: dict,
    system_prompt: str,
    user_prompt: str,
    params: dict,
    model_class,
    tool_dict: dict,
) -> dict:
    history: list[dict] = []
    poison_text = item["Attacker Instruction"]
    final_output = ""
    final_eval = "unsucc"
    final_invalid_reason = ""
    success_round = None

    for round_num in range(1, params["max_rounds"] + 1):
        optimizer_meta: dict = {}
        if round_num > 1:
            new_poison, optimizer_meta = call_adaptive_optimizer(
                item, history, round_num, params
            )
            poison_text = (
                _fallback_poison(item, history)
                if optimizer_meta.get("skipped_optimizer")
                else new_poison
            )

        output, parsed, applied_instruction = evaluate_victim_round(
            item, poison_text, system_prompt, user_prompt, params, model_class, tool_dict
        )

        round_record = {
            "round": round_num,
            "poison": poison_text,
            "applied_instruction": applied_instruction,
            "eval": parsed["eval"],
            "victim_output": output,
        }
        if parsed["eval"] == "invalid":
            round_record["invalid_reason"] = parsed.get("invalid_reason", "")
        if round_num > 1:
            round_record["optimizer"] = optimizer_meta
        if parsed.get("invalid_reason") == "api_error":
            round_record["skipped_round"] = True

        history.append(round_record)
        final_output = output or final_output
        final_eval = parsed["eval"]
        final_invalid_reason = parsed.get("invalid_reason", "")

        if parsed["eval"] == "succ":
            success_round = round_num
            break

    result = copy.deepcopy(item)
    result["optimization_history"] = history
    result["final_poison"] = poison_text
    result["success_round"] = success_round
    result["attacked_any_round"] = success_round is not None
    result["rounds_used"] = len(history)
    result["output"] = final_output
    result["eval"] = final_eval
    if final_eval == "invalid":
        result["invalid_reason"] = final_invalid_reason
    result["Tool Response"] = build_tool_response(
        item["Tool Response Template"], poison_text, params["setting"]
    )
    attach_user_task_utility(result)
    return result


def main(params: dict) -> None:
    print(params)

    model_class = MODELS[params["model_type"]](params)
    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]
    tool_dict = get_tool_dict()
    max_rounds = params["max_rounds"]

    output_dir = (
        f"./results/adaptive_agent_{params['model_type']}_{params['model_name']}_"
        f"{params['optimizer_model']}_{params['prompt_type']}_{params['setting']}"
    )
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"test_cases_ds_{params['setting']}.json"
    test_case_file = os.path.join("./data", file_name)
    output_file = os.path.join(output_dir, file_name)

    with open(test_case_file, "r") as f:
        data = json.load(f)

    if params["limit"] > 0:
        data = data[: params["limit"]]

    cached = load_cached_results(output_file) if params["use_cache"] else {}

    if not params["only_get_score"]:
        with open(output_file, "w") as f:
            for item in tqdm(data, desc="adaptive_agent"):
                case_id = make_case_id(item)
                try:
                    if case_id in cached and cached[case_id].get("output"):
                        result = cached[case_id]
                        if not result.get("utility_eval"):
                            attach_user_task_utility(result)
                    else:
                        result = optimize_one_case_adaptive(
                            item,
                            system_prompt,
                            user_prompt,
                            params,
                            model_class,
                            tool_dict,
                        )
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"Error on case {case_id}: {e}")
                    traceback.print_exc()
                    fallback = copy.deepcopy(item)
                    fallback.update(
                        {
                            "optimization_history": [],
                            "final_poison": item["Attacker Instruction"],
                            "success_round": None,
                            "attacked_any_round": False,
                            "rounds_used": 0,
                            "output": "",
                            "eval": "invalid",
                            "invalid_reason": "case_exception",
                            "case_error": str(e),
                        }
                    )
                    f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
                    f.flush()

    scores = get_adaptive_attack_score(output_file, max_rounds=max_rounds)
    scores.update(get_s1_utility_score(output_file))
    print(json.dumps(scores, indent=True, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_arguments(agent_type="adaptive_agent"))
