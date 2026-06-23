"""
Experiment 8: AgentDojo-style template attacks on InjecAgent 544 S1 (no defense).

Attacks: naive | combined | important_instructions | tool_knowledge
Metrics: ASR-all (S1) via get_s1_score; user-task Utility via get_s1_utility_score
Data:   runtime formatting from data/test_cases_ds_{setting}.json (strategy A)
"""

from __future__ import annotations

import json
import os

from tqdm import tqdm

from src.evaluate_prompted_agent import predict_one_case
from src.models import MODELS
from src.output_parsing import evaluate_output_prompted
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
from src.template_attacks import (
    COMBINED_TEMPLATE_ID,
    apply_template_attack,
    make_template_case_id,
)
from src.utility_parsing import attach_user_task_utility, get_s1_utility_score
from src.utils import get_s1_score, get_tool_dict


def load_cached_results(output_file: str) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if not os.path.exists(output_file):
        return cached
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                cached[make_template_case_id(item)] = item
            except json.JSONDecodeError:
                continue
    return cached


def main(params: dict) -> None:
    print(params)
    attack_type = params["attack_type"]
    model_class = MODELS[params["model_type"]](params)
    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]
    tool_dict = get_tool_dict()

    attack_tag = attack_type
    if attack_type == "combined":
        attack_tag = f"combined_t{COMBINED_TEMPLATE_ID}"
    output_dir = (
        f"./results/template_{attack_tag}_{params['model_type']}_"
        f"{params['model_name']}_{params['prompt_type']}_{params['setting']}"
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
            for item in tqdm(data, desc=f"template_{attack_type}"):
                case_id = make_template_case_id(item)
                try:
                    if case_id in cached and cached[case_id].get("output"):
                        result = cached[case_id]
                    else:
                        attacked = apply_template_attack(item, attack_type)
                        output = predict_one_case(
                            system_prompt,
                            user_prompt,
                            params,
                            attacked,
                            model_class,
                            tool_dict,
                        )
                        parsed = evaluate_output_prompted(
                            output,
                            attacked["Attacker Tools"][0],
                            attacked["Attacker Tools"],
                            attacked["User Tool"],
                            attacked["formatted_poison"],
                        )
                        result = attacked
                        result["output"] = output
                        result["eval"] = parsed["eval"]
                        if result["eval"] == "invalid":
                            result["invalid_reason"] = parsed.get("invalid_reason")
                        attach_user_task_utility(result)

                    if not result.get("utility_eval"):
                        attach_user_task_utility(result)

                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"Error on case {case_id}: {e} in {output_file}")

    scores = get_s1_score(output_file)
    scores.update(get_s1_utility_score(output_file))
    print(json.dumps(scores, indent=True))


if __name__ == "__main__":
    main(parse_arguments(agent_type="template_attacks"))
