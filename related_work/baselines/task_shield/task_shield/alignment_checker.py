"""
对齐检查模块。

对应论文 Section 3.2: Alignment Checking
检查从外部内容中提取的新指令 / Agent 生成的工具调用是否与用户任务集对齐。

核心思想:
- 对每条新指令，计算它与每个用户任务的 ContributesTo 分数
- 如果某条指令对所有用户任务的 ContributesTo 分数都为 0，则该指令不对齐（可能是注入）
- 对工具调用执行类似的检查
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from task_shield.llm_client import DeepSeekClient
from task_shield.prompts import (
    CONTENT_CHECKER_SYSTEM,
    CONTENT_CHECKER_USER_TEMPLATE,
    TOOL_CALL_CHECKER_SYSTEM,
    TOOL_CALL_CHECKER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """单条指令/工具调用的对齐检查结果。"""

    instruction: str
    """被检查的指令或工具调用描述"""

    is_aligned: bool
    """是否与用户任务对齐"""

    contributes_to_score: int
    """ContributesTo 分数（0 或 1）"""

    reasoning: str = ""
    """LLM 的推理过程"""

    matched_user_task: str | None = None
    """匹配到的用户任务（如果对齐的话）"""


@dataclass
class BatchAlignmentResult:
    """批量对齐检查结果。"""

    is_all_aligned: bool
    """是否所有指令/工具调用都对齐"""

    results: list[AlignmentResult] = field(default_factory=list)
    """每条指令/工具调用的详细结果"""

    misaligned_items: list[str] = field(default_factory=list)
    """不对齐的指令/工具调用列表"""


class AlignmentChecker:
    """
    对齐检查器。

    检查新指令或工具调用是否与用户的原始任务集对齐。

    论文中的核心逻辑:
    - ContributesTo(new_instruction, user_task) ∈ {0, 1}
    - 对每条 new_instruction，遍历所有 user_task 计算 ContributesTo
    - 如果 max(ContributesTo scores) == 0 → 该指令不对齐（疑似注入）
    - 只要有一条指令不对齐 → 整体判定为不对齐

    Args:
        client: DeepSeek API 客户端实例
    """

    def __init__(self, client: DeepSeekClient):
        self.client = client

    def check_content_alignment(
        self,
        new_instructions: list[str],
        user_task_set: list[str],
    ) -> BatchAlignmentResult:
        """
        检查从外部内容中提取的新指令是否与用户任务集对齐。

        对应论文 Algorithm 1 中的 Content Alignment Check 步骤。

        Args:
            new_instructions: 从工具输出中提取的新指令列表
            user_task_set: 用户任务集

        Returns:
            BatchAlignmentResult 对象，包含整体和逐条的对齐结果
        """
        if not new_instructions:
            return BatchAlignmentResult(is_all_aligned=True)

        results = []
        misaligned_items = []

        for instruction in new_instructions:
            result = self._check_single_instruction_alignment(
                instruction, user_task_set
            )
            results.append(result)
            if not result.is_aligned:
                misaligned_items.append(instruction)

        is_all_aligned = len(misaligned_items) == 0

        if not is_all_aligned:
            logger.warning(
                f"检测到 {len(misaligned_items)} 条不对齐的指令: {misaligned_items}"
            )
        else:
            logger.debug(f"所有 {len(new_instructions)} 条指令均与用户任务对齐")

        return BatchAlignmentResult(
            is_all_aligned=is_all_aligned,
            results=results,
            misaligned_items=misaligned_items,
        )

    def check_tool_call_alignment(
        self,
        tool_calls: list[dict[str, Any]],
        user_task_set: list[str],
    ) -> BatchAlignmentResult:
        """
        检查 Agent 生成的工具调用是否与用户任务集对齐。

        对应论文 Algorithm 1 中的 Tool Call Alignment Check 步骤。

        Args:
            tool_calls: 工具调用列表，每个元素包含 function name 和 arguments
            user_task_set: 用户任务集

        Returns:
            BatchAlignmentResult 对象
        """
        if not tool_calls:
            return BatchAlignmentResult(is_all_aligned=True)

        results = []
        misaligned_items = []

        for tool_call in tool_calls:
            result = self._check_single_tool_call_alignment(tool_call, user_task_set)
            results.append(result)
            if not result.is_aligned:
                call_desc = self._format_tool_call(tool_call)
                misaligned_items.append(call_desc)

        is_all_aligned = len(misaligned_items) == 0

        if not is_all_aligned:
            logger.warning(
                f"检测到 {len(misaligned_items)} 个不对齐的工具调用: "
                f"{misaligned_items}"
            )

        return BatchAlignmentResult(
            is_all_aligned=is_all_aligned,
            results=results,
            misaligned_items=misaligned_items,
        )

    def _check_single_instruction_alignment(
        self,
        instruction: str,
        user_task_set: list[str],
    ) -> AlignmentResult:
        """
        检查单条指令是否与用户任务集中的至少一个任务对齐。

        论文中的逻辑:
        - 对每个 user_task 计算 ContributesTo(instruction, user_task)
        - 如果 max score > 0，则对齐；否则不对齐

        为了减少 API 调用次数，这里将所有用户任务打包到一次调用中，
        让 LLM 一次性判断该指令是否贡献于任何用户任务。
        """
        # 格式化用户任务列表
        user_tasks_formatted = "\n".join(
            [f"  {i + 1}. {task}" for i, task in enumerate(user_task_set)]
        )

        user_prompt = CONTENT_CHECKER_USER_TEMPLATE.format(
            user_tasks=user_tasks_formatted,
            new_instruction=instruction,
        )

        try:
            response = self.client.chat_json(
                system_prompt=CONTENT_CHECKER_SYSTEM,
                user_prompt=user_prompt,
            )

            if isinstance(response, dict):
                contributes_to = int(response.get("contributes_to", 0))
                reasoning = response.get("reasoning", "")
            else:
                logger.warning(f"对齐检查返回意外格式: {response}")
                contributes_to = 0
                reasoning = str(response)

            return AlignmentResult(
                instruction=instruction,
                is_aligned=(contributes_to == 1),
                contributes_to_score=contributes_to,
                reasoning=reasoning,
                matched_user_task=(
                    "至少一个用户任务" if contributes_to == 1 else None
                ),
            )

        except Exception as e:
            logger.error(f"对齐检查失败 (instruction={instruction[:50]}...): {e}")
            # 失败时保守策略：视为对齐（避免误杀正常指令）
            return AlignmentResult(
                instruction=instruction,
                is_aligned=True,
                contributes_to_score=1,
                reasoning=f"对齐检查失败，使用保守策略: {e}",
            )

    def _check_single_tool_call_alignment(
        self,
        tool_call: dict[str, Any],
        user_task_set: list[str],
    ) -> AlignmentResult:
        """检查单个工具调用是否与用户任务集对齐。"""

        function_name = self._get_function_name(tool_call)
        arguments = self._get_arguments(tool_call)

        user_tasks_formatted = "\n".join(
            [f"  {i + 1}. {task}" for i, task in enumerate(user_task_set)]
        )

        user_prompt = TOOL_CALL_CHECKER_USER_TEMPLATE.format(
            user_tasks=user_tasks_formatted,
            function_name=function_name,
            arguments=json.dumps(arguments, ensure_ascii=False, indent=2),
        )

        try:
            response = self.client.chat_json(
                system_prompt=TOOL_CALL_CHECKER_SYSTEM,
                user_prompt=user_prompt,
            )

            if isinstance(response, dict):
                contributes_to = int(response.get("contributes_to", 0))
                reasoning = response.get("reasoning", "")
            else:
                contributes_to = 0
                reasoning = str(response)

            call_desc = self._format_tool_call(tool_call)

            return AlignmentResult(
                instruction=call_desc,
                is_aligned=(contributes_to == 1),
                contributes_to_score=contributes_to,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"工具调用对齐检查失败: {e}")
            call_desc = self._format_tool_call(tool_call)
            return AlignmentResult(
                instruction=call_desc,
                is_aligned=True,
                contributes_to_score=1,
                reasoning=f"检查失败，使用保守策略: {e}",
            )

    @staticmethod
    def _get_function_name(tool_call: dict[str, Any]) -> str:
        """从工具调用中提取函数名（兼容多种格式）。"""
        if "function" in tool_call:
            func = tool_call["function"]
            if isinstance(func, dict):
                return func.get("name", "unknown")
            return str(func)
        return tool_call.get("name", tool_call.get("function_name", "unknown"))

    @staticmethod
    def _get_arguments(tool_call: dict[str, Any]) -> dict:
        """从工具调用中提取参数（兼容多种格式）。"""
        if "function" in tool_call:
            func = tool_call["function"]
            if isinstance(func, dict):
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        return json.loads(args)
                    except json.JSONDecodeError:
                        return {"raw": args}
                return args
        return tool_call.get("args", tool_call.get("arguments", {}))

    @staticmethod
    def _format_tool_call(tool_call: dict[str, Any]) -> str:
        """格式化工具调用为可读字符串。"""
        name = AlignmentChecker._get_function_name(tool_call)
        args = AlignmentChecker._get_arguments(tool_call)
        return f"{name}({json.dumps(args, ensure_ascii=False)})"
