"""
Experiment 6 (InjecAgent track): benign utility on clean user cases.

Dataset: InjecAgent/data/user_cases.jsonl → poison-free user_cases_clean (17 cases)
Defense: --defense none (base prompt) | max_defense (MAX_DEFENSE_STRATEGY)
Metric: utility rate / over-refusal rate (see utility_parsing.py)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

_TOOLS_ATTACK_ROOT = Path(__file__).resolve().parents[2]
if str(_TOOLS_ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ATTACK_ROOT))

from src.evaluate_prompted_agent import predict_one_case  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.params import parse_arguments  # noqa: E402
from src.prompts.agent_prompts import PROMPT_DICT  # noqa: E402
from src.prompts.max_defense_harness import MAX_DEFENSE_STRATEGY  # noqa: E402
from src.prompts.protect_prompt import apply_defense_strategy  # noqa: E402
from src.utility_integration import (  # noqa: E402
    USER_CASES_CLEAN_PATH,
    export_clean_user_cases_json,
    load_clean_user_cases,
)
from src.utility_parsing import evaluate_injecagent_utility, get_injecagent_utility_score  # noqa: E402
from src.utils import get_tool_dict  # noqa: E402


def make_case_id(item: dict) -> str:
    return "|".join([
        item.get("User Tool", ""),
        item.get("User Instruction", "")[:120],
    ])


def resolve_prompts(params: dict) -> tuple[str, str]:
    base_system, base_user = PROMPT_DICT[params["prompt_type"]]
    if params.get("defense") == "max_defense":
        harness_file = os.environ.get("TOOLS_ATTACK_HARNESS_STRATEGY_FILE", "").strip()
        strategy = (
            json.loads(Path(harness_file).read_text(encoding="utf-8"))
            if harness_file
            else MAX_DEFENSE_STRATEGY
        )
        return apply_defense_strategy(base_system, base_user, strategy)
    return base_system, base_user


def main(params: dict) -> None:
    print(params)

    if not USER_CASES_CLEAN_PATH.exists() or params.get("refresh_clean_data"):
        export_clean_user_cases_json()
        print(f"Exported clean cases to {USER_CASES_CLEAN_PATH}")

    data = load_clean_user_cases()
    if params["limit"] > 0:
        data = data[: params["limit"]]

    defense_tag = params.get("defense") or "none"
    system_prompt, user_prompt = resolve_prompts(params)
    model_class = MODELS[params["model_type"]](params)

    output_dir = (
        f"./results/injecagent_utility_{defense_tag}_"
        f"{params['model_type']}_{params['model_name']}_{params['prompt_type']}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "user_cases_clean.jsonl")

    tool_dict = get_tool_dict()
    cached: dict[str, str] = {}
    if params["use_cache"] and os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out = row.get("output")
                if isinstance(out, str) and out:
                    cached[make_case_id(row)] = out
        print(f"Loaded {len(cached)} cached outputs from {output_file}")

    if not params["only_get_score"]:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(data, desc=f"injecagent_utility_{defense_tag}"):
                case_id = make_case_id(item)
                try:
                    if params["use_cache"] and case_id in cached:
                        output = cached[case_id]
                    else:
                        output = predict_one_case(
                            system_prompt,
                            user_prompt,
                            params,
                            item,
                            model_class,
                            tool_dict,
                        )

                    result = dict(item)
                    result["output"] = output
                    result["defense"] = defense_tag
                    parsed = evaluate_injecagent_utility(
                        output, item["User Instruction"]
                    )
                    result["eval"] = parsed["eval"]
                    if parsed.get("invalid_reason"):
                        result["utility_reason"] = parsed["invalid_reason"]

                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as exc:
                    print(f"Error on case {case_id}: {exc}")

    scores = get_injecagent_utility_score(output_file)
    print(json.dumps({k: v for k, v in scores.items() if k != "counts"}, indent=True, ensure_ascii=False))
    print(json.dumps(scores.get("counts", {}), indent=True))


if __name__ == "__main__":
    main(parse_arguments(agent_type="injecagent_utility"))
