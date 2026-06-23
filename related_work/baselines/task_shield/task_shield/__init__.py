"""
Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection

论文复现实现。

Usage:
    from task_shield import TaskShieldDefense
    from task_shield.task_shield_pipeline import build_task_shield_pipeline
"""

from task_shield.task_shield_defense import TaskShieldDefense
from task_shield.llm_client import DeepSeekClient
from task_shield.instruction_extractor import InstructionExtractor
from task_shield.alignment_checker import AlignmentChecker
from task_shield.feedback_generator import FeedbackGenerator

__version__ = "1.0.0"
__all__ = [
    "TaskShieldDefense",
    "DeepSeekClient",
    "InstructionExtractor",
    "AlignmentChecker",
    "FeedbackGenerator",
]
