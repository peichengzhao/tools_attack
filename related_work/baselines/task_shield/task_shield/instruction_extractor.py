"""
指令提取模块。

对应论文 Section 3.1: Instruction Extraction
从对话消息中提取可执行的指令/任务。

核心功能:
1. 从用户初始消息中提取 User Task Set（用户任务集）
2. 从工具输出或其他外部内容中提取新出现的指令
"""

import logging
from typing import Any

from task_shield.llm_client import DeepSeekClient
from task_shield.prompts import (
    TASK_EXTRACTION_SYSTEM,
    TASK_EXTRACTION_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class InstructionExtractor:
    """
    指令提取器。

    使用 LLM 从文本中提取可执行的指令/任务。

    论文中的设计:
    - User Task Set 在对话开始时从用户消息中提取一次，作为"基准"
    - 每次工具返回结果后，从结果中提取新指令，进行对齐检查

    Args:
        client: DeepSeek API 客户端实例
    """

    def __init__(self, client: DeepSeekClient):
        self.client = client

    def extract_user_tasks(self, user_message: str) -> list[str]:
        """
        从用户的初始消息中提取用户任务集 (User Task Set)。

        这是 Task Shield 的第一步。用户任务集定义了 Agent 应该完成的
        所有合法任务，后续所有指令/工具调用都必须与之对齐。

        Args:
            user_message: 用户的原始输入消息

        Returns:
            用户任务列表，例如: ["查找航班信息", "预订最便宜的航班"]
        """
        user_prompt = TASK_EXTRACTION_USER_TEMPLATE.format(message=user_message)

        try:
            tasks = self.client.chat_json(
                system_prompt=TASK_EXTRACTION_SYSTEM,
                user_prompt=user_prompt,
            )

            if not isinstance(tasks, list):
                logger.warning(
                    f"任务提取结果不是列表类型: {type(tasks)}, 内容: {tasks}"
                )
                if isinstance(tasks, dict) and "tasks" in tasks:
                    tasks = tasks["tasks"]
                else:
                    tasks = [str(tasks)]

            # 确保所有元素都是字符串
            tasks = [str(t) for t in tasks if t]

            logger.info(f"从用户消息中提取了 {len(tasks)} 个任务: {tasks}")
            return tasks

        except Exception as e:
            logger.error(f"用户任务提取失败: {e}")
            # 降级策略：直接将整个用户消息作为单个任务
            return [user_message]

    def extract_new_instructions(self, content: str) -> list[str]:
        """
        从外部内容（如工具输出）中提取新出现的指令。

        这些新指令可能是合法的子任务指引，也可能是注入的恶意指令。
        提取后需要经过 AlignmentChecker 检查对齐性。

        Args:
            content: 要检查的文本内容（通常是工具的返回结果）

        Returns:
            新指令列表。如果没有发现任何指令，返回空列表。
        """
        if not content or not content.strip():
            return []

        user_prompt = TASK_EXTRACTION_USER_TEMPLATE.format(message=content)

        try:
            instructions = self.client.chat_json(
                system_prompt=TASK_EXTRACTION_SYSTEM,
                user_prompt=user_prompt,
            )

            if not isinstance(instructions, list):
                logger.warning(
                    f"指令提取结果不是列表类型: {type(instructions)}, "
                    f"内容: {instructions}"
                )
                if isinstance(instructions, dict) and "instructions" in instructions:
                    instructions = instructions["instructions"]
                elif isinstance(instructions, dict) and "tasks" in instructions:
                    instructions = instructions["tasks"]
                else:
                    instructions = []

            instructions = [str(i) for i in instructions if i]

            if instructions:
                logger.info(
                    f"从外部内容中提取了 {len(instructions)} 条新指令: {instructions}"
                )
            else:
                logger.debug("外部内容中未发现新指令")

            return instructions

        except Exception as e:
            logger.error(f"指令提取失败: {e}")
            return []

    def extract_instructions_from_tool_results(
        self, messages: list[dict[str, Any]]
    ) -> list[str]:
        """
        从对话历史中最新的工具结果消息中提取指令。

        遍历最近的 tool result 消息，合并提取所有新指令。

        Args:
            messages: 完整的对话消息列表

        Returns:
            从所有最近工具结果中提取的新指令列表
        """
        all_instructions = []

        # 从后往前找到所有连续的 tool result 消息
        tool_results = []
        for msg in reversed(messages):
            role = msg.get("role", "")
            if role == "tool":
                tool_results.append(msg)
            elif role == "assistant":
                # 遇到 assistant 消息就停止（只检查最新一轮的工具结果）
                break
            else:
                break

        for tool_msg in tool_results:
            content = ""
            # 兼容不同的消息格式
            if isinstance(tool_msg.get("content"), str):
                content = tool_msg["content"]
            elif isinstance(tool_msg.get("content"), list):
                # AgentDojo 使用 content blocks 格式
                for block in tool_msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("content", "") + "\n"
                    elif isinstance(block, str):
                        content += block + "\n"

            if content.strip():
                instructions = self.extract_new_instructions(content)
                all_instructions.extend(instructions)

        return all_instructions
