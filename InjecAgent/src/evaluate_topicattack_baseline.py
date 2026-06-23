"""
TopicAttack baseline (no defense): conv_attack on 510 DH grant cases.

Dataset: related_work/topicattack/injectagent/data/crafted_instruction_data_agent_grant.json
Attack:   TopicAttack conv_attack
Defense:  none — base InjecAgent SYS_PROMPT + USER_PROMPT only
Victim:   model_api.VICTIM_MODEL via InjecAgent GPTModel + llm_client
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

from paths import TOPICATTACK_GRANT_CASES  # noqa: E402

from src.evaluate_prompted_agent import predict_one_case  # noqa: E402
from src.evaluate_topicattack_max_defense import make_case_id  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.output_parsing import evaluate_output_prompted  # noqa: E402
from src.params import parse_arguments  # noqa: E402
from src.prompts.agent_prompts import PROMPT_DICT  # noqa: E402
from src.topicattack_integration import apply_conv_attack  # noqa: E402
from src.utils import get_dh_s1_score, get_tool_dict  # noqa: E402


def main(params: dict) -> None:
    print(params)

    test_case_file = params.get("test_case_file") or str(TOPICATTACK_GRANT_CASES)
    if not os.path.isfile(test_case_file):
        raise FileNotFoundError(f"TopicAttack dataset not found: {test_case_file}")

    model_class = MODELS[params["model_type"]](params)
    system_prompt, user_prompt = PROMPT_DICT[params["prompt_type"]]

    output_dir = (
        f"./results/topicattack_baseline_{params['model_type']}_"
        f"{params['model_name']}_{params['prompt_type']}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_cases_conv_attack_baseline.jsonl")

    with open(test_case_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if params["limit"] > 0:
        data = data[: params["limit"]]

    tool_dict = get_tool_dict()
    cached: dict[str, str] = {}
    if params["use_cache"] and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
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
            for item in tqdm(data, desc="topicattack_baseline"):
                case_id = make_case_id(item)
                try:
                    attacked = apply_conv_attack(item)
                    if params["use_cache"] and case_id in cached:
                        output = cached[case_id]
                    else:
                        output = predict_one_case(
                            system_prompt,
                            user_prompt,
                            params,
                            attacked,
                            model_class,
                            tool_dict,
                        )

                    result = dict(attacked)
                    result["output"] = output
                    parsed = evaluate_output_prompted(
                        output,
                        attacked["Attacker Tools"][0],
                        attacked["Attacker Tools"],
                        attacked["User Tool"],
                        attacked["Attacker Instruction"],
                    )
                    result["eval"] = parsed["eval"]
                    if parsed["eval"] == "invalid":
                        result["invalid_reason"] = parsed.get("invalid_reason", "")

                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as exc:
                    print(f"Error on case {case_id}: {exc}")

    scores = get_dh_s1_score(output_file)
    print(json.dumps(scores, indent=True, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_arguments(agent_type="topicattack_baseline"))
