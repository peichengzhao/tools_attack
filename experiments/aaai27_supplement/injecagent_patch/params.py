import argparse

from src.llm_client import OPTIMIZER_MODEL, PROTECT_MODEL, VICTIM_MODEL

def define_arguments_prompted(parser):
    parser.add_argument("--model_type", help="Type of the model to evaluate", default="GPT")
    parser.add_argument("--model_name", help="Name of the model to evaluate", default=VICTIM_MODEL)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all)")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Use existing dataset")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only get score without inference")
    
    
def define_arguments_finetuned(parser):
    parser.add_argument("--model_name", help="Name of the model to evaluate", default=VICTIM_MODEL)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)


def define_arguments_protect_agent(parser):
    parser.add_argument("--model_type", help="Type of the victim model", default="GPT")
    parser.add_argument("--model_name", help="Name of the victim model", default=VICTIM_MODEL)
    parser.add_argument("--optimizer_model", help="Name of the attack optimizer model", default=OPTIMIZER_MODEL)
    parser.add_argument("--protect_model", help="Name of the protect agent model", default=PROTECT_MODEL)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--max_rounds", type=int, default=8, help="Co-evolution rounds per case (no early stop)")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all)")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Resume from existing output file")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only compute scores")
    parser.add_argument("--verbose", default=False, action='store_true', help="Print poison/defense co-evolution per round")


def define_arguments_attack_agent(parser):
    parser.add_argument("--model_type", help="Type of the victim model", default="GPT")
    parser.add_argument("--model_name", help="Name of the victim model", default=VICTIM_MODEL)
    parser.add_argument("--optimizer_model", help="Name of the attack optimizer model", default=OPTIMIZER_MODEL)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--max_rounds", type=int, default=6, help="Max optimization rounds per case")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all)")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Resume from existing output file")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only compute scores")


def define_arguments_topicattack_max_defense(parser):
    parser.add_argument("--model_type", help="Type of the victim model", default="GPT")
    parser.add_argument("--model_name", help="Name of the victim model", default=VICTIM_MODEL)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], default='InjecAgent')
    parser.add_argument("--setting", choices=['base', 'enhanced'], default='base', help="Recorded for parity; conv_attack uses grant JSON as-is")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all 510)")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Resume from existing output file")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only compute scores")
    parser.add_argument(
        "--test_case_file",
        type=str,
        default="",
        help="TopicAttack grant JSON (default: related_work/topicattack/.../crafted_instruction_data_agent_grant.json)",
    )


def define_arguments_adaptive_agent(parser):
    parser.add_argument("--model_type", help="Type of the victim model", default="GPT")
    parser.add_argument("--model_name", help="Name of the victim model", default=VICTIM_MODEL)
    parser.add_argument(
        "--optimizer_model",
        help="Adaptive optimizer model (feedback-driven poison mutation)",
        default=OPTIMIZER_MODEL,
    )
    parser.add_argument("--setting", choices=["base", "enhanced"], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=["InjecAgent", "hwchase17_react"], help="prompt type", required=True)
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=8,
        help="Max adaptive optimization rounds per case (early stop on succ)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all)")
    parser.add_argument("--use_cache", default=False, action="store_true", help="Resume from existing output file")
    parser.add_argument("--only_get_score", default=False, action="store_true", help="Only compute scores")


def define_arguments_cross_eval(parser):
    parser.add_argument("--model_type", help="Type of the victim model", default="GPT")
    parser.add_argument(
        "--model",
        default="all",
        help="Victim model or 'all' (deepseek-v3.2, deepseek-v4-flash, glm-5.1, glm-5.1-fp8, qwen3.5-27b; any API name allowed)",
    )
    parser.add_argument(
        "--defense",
        default="all",
        help="Defense or 'all' (sandwich, spotlight, struq, secalign, ours, none)",
    )
    parser.add_argument(
        "--attack",
        default="all",
        help="Attack row or 'all' (naive, injecagent_default, ..., ours_attack)",
    )
    parser.add_argument("--setting", choices=["base", "enhanced"], default="base")
    parser.add_argument("--prompt_type", choices=["InjecAgent", "hwchase17_react"], default="InjecAgent")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all)")
    parser.add_argument(
        "--test_data",
        default="",
        help="Optional custom case JSON, e.g. data/splits/test_cases_211.json",
    )
    parser.add_argument("--use_cache", default=False, action="store_true", help="Resume cell output jsonl")
    parser.add_argument("--only_get_score", default=False, action="store_true", help="Score existing outputs only")
    parser.add_argument("--smoke", default=False, action="store_true", help="Write under results/cross_eval/smoke/")
    parser.add_argument(
        "--harness_file",
        default="",
        help="Selected harness JSON required when --defense ours",
    )
    parser.add_argument(
        "--ours_attacks",
        default="",
        help="Frozen Ours-Attack JSONL required when --attack ours_attack",
    )
    parser.add_argument(
        "--allow_offline_fallback",
        default=False,
        action="store_true",
        help="Use the seed attack when Adaptive poison is missing (smoke only)",
    )
    parser.add_argument(
        "--refresh_chatinject_map",
        default=False,
        action="store_true",
        help="Rebuild data/cross_eval/chatinject_case_goal_map.json",
    )
    parser.add_argument(
        "--setup_check",
        default=False,
        action="store_true",
        help="Validate setup + write manifest; do not call victim API",
    )
    parser.add_argument(
        "--skip_offline_attacks",
        default=False,
        action="store_true",
        help="Exclude injecagent_enhanced and adaptive_attack rows",
    )


def define_arguments_template_attacks(parser):
    parser.add_argument("--model_type", help="Type of the victim model", default="GPT")
    parser.add_argument("--model_name", help="Name of the victim model", default=VICTIM_MODEL)
    parser.add_argument(
        "--attack_type",
        choices=["naive", "combined", "important_instructions", "tool_knowledge"],
        required=True,
        help="Template attack variant (Experiment 8)",
    )
    parser.add_argument("--setting", choices=["base", "enhanced"], default="base", help="Dataset setting (544 S1)")
    parser.add_argument("--prompt_type", choices=["InjecAgent", "hwchase17_react"], default="InjecAgent")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all)")
    parser.add_argument("--use_cache", default=False, action="store_true", help="Resume from existing output file")
    parser.add_argument("--only_get_score", default=False, action="store_true", help="Only compute scores")


def define_arguments_injecagent_utility(parser):
    parser.add_argument("--model_type", help="Type of the victim model", default="GPT")
    parser.add_argument("--model_name", help="Name of the victim model", default=VICTIM_MODEL)
    parser.add_argument("--prompt_type", choices=["InjecAgent", "hwchase17_react"], default="InjecAgent")
    parser.add_argument(
        "--defense",
        choices=["none", "max_defense"],
        default="none",
        help="none = base InjecAgent prompt; max_defense = static MAX_DEFENSE_STRATEGY",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all 17)")
    parser.add_argument("--use_cache", default=False, action="store_true", help="Resume from existing output file")
    parser.add_argument("--only_get_score", default=False, action="store_true", help="Only compute scores")
    parser.add_argument(
        "--refresh_clean_data",
        default=False,
        action="store_true",
        help="Regenerate InjecAgent/data/user_cases_clean.json from user_cases.jsonl",
    )


def parse_arguments(agent_type=None):
    parser = argparse.ArgumentParser()
    if agent_type == "prompted":
        define_arguments_prompted(parser)
    elif agent_type == "finetuned":
        define_arguments_finetuned(parser)
    elif agent_type == "attack_agent":
        define_arguments_attack_agent(parser)
    elif agent_type == "adaptive_agent":
        define_arguments_adaptive_agent(parser)
    elif agent_type == "protect_agent":
        define_arguments_protect_agent(parser)
    elif agent_type == "topicattack_max_defense":
        define_arguments_topicattack_max_defense(parser)
    elif agent_type == "topicattack_baseline":
        define_arguments_topicattack_max_defense(parser)
    elif agent_type == "injecagent_utility":
        define_arguments_injecagent_utility(parser)
    elif agent_type == "template_attacks":
        define_arguments_template_attacks(parser)
    elif agent_type == "cross_eval":
        define_arguments_cross_eval(parser)
    args = parser.parse_args()
    params = args.__dict__
    return params
