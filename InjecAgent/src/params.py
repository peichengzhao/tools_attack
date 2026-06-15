import argparse

from src.llm_client import ATTACK_MODEL, OPTIMIZER_MODEL

def define_arguments_prompted(parser):
    parser.add_argument("--model_type", help="Type of the model to evaluate", default="GPT")
    parser.add_argument("--model_name", help="Name of the model to evaluate", default=ATTACK_MODEL)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--use_cache", default=False, action='store_true', help="Use existing dataset")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only get score without inference")
    
    
def define_arguments_finetuned(parser):
    parser.add_argument("--model_name", help="Name of the model to evaluate", default=ATTACK_MODEL)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)


def define_arguments_attack_agent(parser):
    parser.add_argument("--model_type", help="Type of the victim model", default="GPT")
    parser.add_argument("--model_name", help="Name of the victim model", default=ATTACK_MODEL)
    parser.add_argument("--optimizer_model", help="Name of the attack optimizer model", default=OPTIMIZER_MODEL)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--max_rounds", type=int, default=6, help="Max optimization rounds per case")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N cases (0 = all)")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Resume from existing output file")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only compute scores")


def parse_arguments(agent_type=None):
    parser = argparse.ArgumentParser()
    if agent_type == "prompted":
        define_arguments_prompted(parser)
    elif agent_type == "finetuned":
        define_arguments_finetuned(parser)
    elif agent_type == "attack_agent":
        define_arguments_attack_agent(parser)
    args = parser.parse_args()
    params = args.__dict__
    return params