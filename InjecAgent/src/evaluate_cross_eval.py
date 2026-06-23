"""
Cross-evaluation runner: attack × static defense × victim model on InjecAgent 544 S1.

Each cell reports ASR (Utility) from a single-round victim run.
invalid does NOT count toward ASR.
"""

from __future__ import annotations

import copy
import json
import os
import traceback
from pathlib import Path

from tqdm import tqdm

from src.cross_eval.attack_payloads import AttackPayloadProvider
from src.cross_eval.constants import (
    CROSS_EVAL_ATTACKS,
    CROSS_EVAL_DEFENSES,
    CROSS_EVAL_MODELS,
    EXTRA_CROSS_EVAL_ATTACKS,
    EXTRA_CROSS_EVAL_DEFENSES,
    expand_model_selection,
    OFFLINE_POISON_ATTACKS,
)
from src.cross_eval.scoring import (
    collect_cross_eval_results,
    score_cross_eval_file,
    write_pivot_markdown,
    write_summary_csv,
)
from src.evaluate_attack_agent import make_case_id
from src.evaluate_prompted_agent import predict_one_case
from src.models import MODELS
from src.output_parsing import evaluate_output_prompted
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
from src.prompts.static_defense_baselines import (
    apply_static_defense_to_prompts,
    prepare_tool_response,
)
from src.prompts.protect_prompt import apply_defense_strategy
from src.utility_parsing import attach_user_task_utility
from src.utils import get_tool_dict


def cross_eval_output_name(model: str, defense: str, attack: str) -> str:
    return f"{model}|{defense}|{attack}.jsonl"


def cross_eval_output_path(
    params: dict,
    model: str,
    defense: str,
    attack: str,
) -> str:
    subdir = "smoke" if params.get("smoke") else "full"
    out_dir = os.path.join("./results/cross_eval", subdir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, cross_eval_output_name(model, defense, attack))


def load_cached_jsonl(output_file: str) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if not os.path.exists(output_file):
        return cached
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                cached[make_case_id(item)] = item
            except json.JSONDecodeError:
                continue
    return cached


def load_ours_harness(path: str | None) -> dict:
    if not path:
        raise ValueError(
            "defense='ours' requires --harness_file with a selected harness strategy"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "selected" in payload:
        return payload["selected"]["candidate"]["strategy"]
    if "strategy" in payload:
        return payload["strategy"]
    return payload


def apply_cross_eval_defense_to_prompts(
    base_system: str,
    base_user: str,
    defense: str,
    user_instruction: str,
    harness_file: str | None,
) -> tuple[str, str]:
    if defense == "none":
        return base_system, base_user
    if defense == "ours":
        strategy = load_ours_harness(harness_file)
        return apply_defense_strategy(base_system, base_user, strategy)
    return apply_static_defense_to_prompts(
        base_system,
        base_user,
        defense,
        user_instruction,
    )


def run_one_cell(
    params: dict,
    model_name: str,
    defense: str,
    attack: str,
    data: list[dict],
    tool_dict: dict,
) -> dict:
    run_params = copy.deepcopy(params)
    run_params["model_name"] = model_name

    base_system, base_user = PROMPT_DICT[run_params["prompt_type"]]
    model_class = MODELS[run_params["model_type"]](run_params)
    payload_provider = AttackPayloadProvider(
        attack=attack,
        model_name=model_name,
        setting=run_params["setting"],
        cases=data,
        refresh_chatinject_map=run_params.get("refresh_chatinject_map", False),
        ours_attack_path=run_params.get("ours_attacks"),
        allow_offline_fallback=run_params.get("allow_offline_fallback", False),
    )

    output_file = cross_eval_output_path(run_params, model_name, defense, attack)
    cached = load_cached_jsonl(output_file) if run_params.get("use_cache") else {}

    if not run_params.get("only_get_score"):
        with open(output_file, "w", encoding="utf-8") as fout:
            desc = f"{model_name}|{defense}|{attack}"
            for item in tqdm(data, desc=desc):
                case_id = make_case_id(item)
                try:
                    if case_id in cached and cached[case_id].get("output") is not None:
                        result = cached[case_id]
                        if not result.get("utility_eval"):
                            attach_user_task_utility(result)
                    else:
                        poison, poison_meta = payload_provider.get_poison(item)
                        system_prompt, user_prompt = apply_cross_eval_defense_to_prompts(
                            base_system,
                            base_user,
                            defense,
                            item.get("User Instruction", ""),
                            run_params.get("harness_file"),
                        )
                        case_item = copy.deepcopy(item)
                        case_item["Tool Response"] = prepare_tool_response(
                            item["Tool Response Template"],
                            poison,
                            defense=defense,
                            setting=run_params["setting"],
                        )
                        case_item["cross_eval_poison"] = poison
                        case_item["cross_eval_meta"] = {
                            "model": model_name,
                            "defense": defense,
                            "attack": attack,
                            **poison_meta,
                        }

                        output = predict_one_case(
                            system_prompt,
                            user_prompt,
                            run_params,
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

                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                except Exception as exc:
                    print(f"Error {case_id}: {exc}")
                    traceback.print_exc()
                    fallback = copy.deepcopy(item)
                    fallback.update(
                        {
                            "cross_eval_meta": {
                                "model": model_name,
                                "defense": defense,
                                "attack": attack,
                                "error": str(exc),
                            },
                            "output": "",
                            "eval": "invalid",
                            "invalid_reason": "case_exception",
                        }
                    )
                    fout.write(json.dumps(fallback, ensure_ascii=False) + "\n")
                    fout.flush()

    scores = score_cross_eval_file(output_file)
    scores.update(
        {
            "model": model_name,
            "defense": defense,
            "attack": attack,
            "output_file": output_file,
        }
    )
    return scores


def expand_selection(values: tuple[str, ...], selected: str | None) -> list[str]:
    if not selected or selected == "all":
        return list(values)
    if selected not in values:
        raise ValueError(f"Invalid selection {selected!r}; expected one of {values} or 'all'")
    return [selected]


def main(params: dict) -> None:
    if params.get("setup_check"):
        from src.cross_eval.setup_check import run_setup_check

        report = run_setup_check(
            write_manifest=True,
            ours_attacks=params.get("ours_attacks", ""),
            test_data=params.get("test_data", ""),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["status"] == "error":
            raise SystemExit(1)
        return

    print(json.dumps(params, indent=2, ensure_ascii=False))

    data_file = params.get("test_data") or os.path.join(
        "./data",
        f"test_cases_ds_{params['setting']}.json",
    )
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)
    if params.get("limit", 0) > 0:
        data = data[: params["limit"]]

    tool_dict = get_tool_dict()

    models = expand_model_selection(params.get("model"))
    defenses = expand_selection(
        (*CROSS_EVAL_DEFENSES, *EXTRA_CROSS_EVAL_DEFENSES),
        params.get("defense"),
    )
    attacks = expand_selection(
        (*CROSS_EVAL_ATTACKS, *EXTRA_CROSS_EVAL_ATTACKS),
        params.get("attack"),
    )
    if params.get("skip_offline_attacks"):
        attacks = [a for a in attacks if a not in OFFLINE_POISON_ATTACKS]

    all_scores: list[dict] = []
    for model_name in models:
        for defense in defenses:
            for attack in attacks:
                scores = run_one_cell(
                    params, model_name, defense, attack, data, tool_dict
                )
                print(json.dumps(scores, indent=2, ensure_ascii=False))
                all_scores.append(scores)

    subdir = "smoke" if params.get("smoke") else "full"
    results_dir = Path("./results/cross_eval") / subdir
    rows = collect_cross_eval_results(results_dir)
    if rows:
        csv_path = write_summary_csv(
            rows, results_dir / "summary_cross_eval.csv"
        )
        print(f"Wrote {csv_path}")
        for model in models:
            md = write_pivot_markdown(
                rows,
                results_dir / f"pivot_{model.replace('.', '_')}.md",
                model,
            )
            print(f"Wrote {md}")


if __name__ == "__main__":
    main(parse_arguments(agent_type="cross_eval"))
