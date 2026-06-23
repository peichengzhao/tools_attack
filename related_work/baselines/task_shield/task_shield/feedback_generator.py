"""
反馈生成模块。

对应论文 Section 3.3: Feedback Generation
当检测到不对齐的指令或工具调用时，生成结构化反馈消息，
引导 Agent 回到正确的任务轨道上。

论文中的设计:
- 反馈以 user message 的形式插入到对话历史中
- 明确提醒 Agent 当前的真正任务
- 指出被检测到的可疑指令
- 指导 Agent 忽略这些指令并继续原始任务
"""

import logging

from task_shield.prompts import (
    FEEDBACK_CONTENT_TEMPLATE,
    FEEDBACK_TOOL_CALL_TEMPLATE,
)

logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """
    反馈生成器。

    当 Task Shield 检测到不对齐的指令或工具调用时，
    生成结构化的反馈消息插入到对话中。
    """

    def generate_content_feedback(
        self,
        user_task_set: list[str],
        misaligned_instructions: list[str],
    ) -> str:
        """
        为不对齐的内容指令生成反馈。

        当工具输出中包含与用户任务不相关的指令（疑似注入）时调用。

        Args:
            user_task_set: 用户任务集
            misaligned_instructions: 被判定为不对齐的指令列表

        Returns:
            格式化的反馈消息字符串
        """
        user_tasks_formatted = "\n".join(
            [f"  - {task}" for task in user_task_set]
        )
        misaligned_formatted = "\n".join(
            [f"  - \"{instr}\"" for instr in misaligned_instructions]
        )

        feedback = FEEDBACK_CONTENT_TEMPLATE.format(
            user_tasks=user_tasks_formatted,
            misaligned_instructions=misaligned_formatted,
        )

        logger.info(
            f"生成内容对齐反馈: 拦截 {len(misaligned_instructions)} 条可疑指令"
        )
        return feedback

    def generate_tool_call_feedback(
        self,
        user_task_set: list[str],
        blocked_tool_calls: list[str],
    ) -> str:
        """
        为不对齐的工具调用生成反馈。

        当 Agent 试图执行与用户任务不相关的工具调用时调用。

        Args:
            user_task_set: 用户任务集
            blocked_tool_calls: 被阻止的工具调用描述列表

        Returns:
            格式化的反馈消息字符串
        """
        user_tasks_formatted = "\n".join(
            [f"  - {task}" for task in user_task_set]
        )
        blocked_formatted = "\n".join(
            [f"  - {call}" for call in blocked_tool_calls]
        )

        feedback = FEEDBACK_TOOL_CALL_TEMPLATE.format(
            user_tasks=user_tasks_formatted,
            blocked_tool_calls=blocked_formatted,
        )

        logger.info(
            f"生成工具调用反馈: 阻止 {len(blocked_tool_calls)} 个可疑工具调用"
        )
        return feedback
