"""Example model configuration for supplement experiments."""

import os


OPENAI_BASE_URL = "https://api.example.com/v1"
OPENAI_API_KEY = os.environ.get("AAAI27_OPENAI_API_KEY", "").strip()

ATTACK_MODEL = "deepseek-v4.1-flash-1017"
DEFENSE_MODEL = "deepseek-v4.1-flash-1017"
JUDGE_MODEL = "deepseek-v4.1-flash-1017"
WORK_AGENT = "deepseek-v4.1-flash-1017"

# Legacy InjecAgent role aliases.
VICTIM_MODEL = WORK_AGENT
OPTIMIZER_MODEL = ATTACK_MODEL
PROTECT_MODEL = DEFENSE_MODEL
