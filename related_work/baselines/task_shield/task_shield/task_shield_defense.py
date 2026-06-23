"""
Task Shield 主防御类。

对应论文 Algorithm 1 (Appendix B.2):
完整实现 Task Shield 的防御流程，作为 AgentDojo Pipeline 的一个元素。

Task Shield 在 Agent 执行循环中的两个关键位置介入:
1. 工具执行后 → LLM 调用前: 检查工具返回内容中是否包含注入指令 (Content Check)
2. LLM 回复后 → 工具执行前: 检查 LLM 生成的工具调用是否被劫持 (Tool Call Check)
"""

import json
import logging
from collections.abc import Sequence
from typing import Any

from task_shield.llm_client import DeepSeekClient
from task_shield.instruction_extractor import InstructionExtractor
from task_shield.alignment_checker import AlignmentChecker
from task_shield.feedback_generator import FeedbackGenerator

logger = logging.getLogger(__name__)


class TaskShieldDefense:
    """
    Task Shield 防御机制。

    实现论文 Algorithm 1 的核心逻辑:
    1. 初始化时从用户消息中提取 User Task Set
    2. 每轮循环中:
       a. 检查工具输出中的新指令是否对齐 (Content Alignment Check)
       b. 如果不对齐，生成反馈注入对话
       c. 检查 LLM 的工具调用是否对齐 (Tool Call Alignment Check)
       d. 如果不对齐，阻止工具调用并生成反馈

    Args:
        client: DeepSeek API 客户端实例（如果为 None，会自动创建）
        enable_content_check: 是否启用内容对齐检查
        enable_tool_call_check: 是否启用工具调用对齐检查
    """

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        enable_content_check: bool = True,
        enable_tool_call_check: bool = True,
    ):
        self.client = client or DeepSeekClient()
        self.extractor = InstructionExtractor(self.client)
        self.checker = AlignmentChecker(self.client)
        self.feedback_gen = FeedbackGenerator()

        self.enable_content_check = enable_content_check
        self.enable_tool_call_check = enable_tool_call_check

        # 状态：用户任务集（在第一次调用时提取并缓存）
        self._user_task_set: list[str] | None = None

        # 统计信息
        self.stats = {
            "content_checks": 0,
            "content_blocks": 0,
            "tool_call_checks": 0,
            "tool_call_blocks": 0,
            "instructions_extracted": 0,
        }

    @property
    def user_task_set(self) -> list[str]:
        """获取用户任务集。"""
        if self._user_task_set is None:
            raise RuntimeError("User task set 尚未初始化，请先调用 initialize()")
        return self._user_task_set

    def initialize(self, user_query: str) -> list[str]:
        """
        初始化 Task Shield: 从用户查询中提取 User Task Set。

        这应该在 Agent 开始处理用户请求之前调用一次。

        Args:
            user_query: 用户的原始查询

        Returns:
            提取的用户任务列表
        """
        self._user_task_set = self.extractor.extract_user_tasks(user_query)
        logger.info(f"Task Shield 初始化完成，用户任务集: {self._user_task_set}")
        return self._user_task_set

    def check_content(self, messages: list[dict[str, Any]]) -> tuple[
        bool, list[dict[str, Any]]
    ]:
        """
        内容对齐检查 (Content Alignment Check)。

        在工具执行返回结果后调用。检查工具输出中是否包含不对齐的指令。
        如果检测到不对齐指令，在消息末尾追加反馈消息。

        对应论文 Algorithm 1, Lines 8-15。

        Args:
            messages: 当前的对话消息列表

        Returns:
            (is_safe, updated_messages) 元组:
            - is_safe: 是否安全（未检测到注入）
            - updated_messages: 可能追加了反馈的消息列表
        """
        if not self.enable_content_check:
            return True, messages

        self.stats["content_checks"] += 1

        # 从最新的工具结果中提取新指令
        new_instructions = self.extractor.extract_instructions_from_tool_results(
            messages
        )
        self.stats["instructions_extracted"] += len(new_instructions)

        if not new_instructions:
            logger.debug("工具输出中未发现新指令，跳过对齐检查")
            return True, messages

        logger.info(f"从工具输出中提取到 {len(new_instructions)} 条新指令，开始对齐检查")

        # 检查对齐性
        result = self.checker.check_content_alignment(
            new_instructions, self.user_task_set
        )

        if result.is_all_aligned:
            logger.debug("所有新指令均与用户任务对齐")
            return True, messages

        # 检测到不对齐指令 → 生成反馈
        self.stats["content_blocks"] += 1
        logger.warning(
            f"检测到 {len(result.misaligned_items)} 条不对齐指令，生成防御反馈"
        )

        feedback = self.feedback_gen.generate_content_feedback(
            self.user_task_set, result.misaligned_items
        )

        # 将反馈作为 user message 追加到对话中
        updated_messages = list(messages) + [
            {
                "role": "user",
                "content": feedback,
            }
        ]

        return False, updated_messages

    def check_tool_calls(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        工具调用对齐检查 (Tool Call Alignment Check)。

        在 LLM 生成包含工具调用的回复后调用。检查这些工具调用是否与用户任务对齐。
        如果检测到不对齐的工具调用，移除这些调用并生成反馈。

        对应论文 Algorithm 1, Lines 16-24。

        Args:
            messages: 当前的对话消息列表（最后一条应为 assistant 消息含 tool_calls）

        Returns:
            (is_safe, updated_messages) 元组:
            - is_safe: 是否安全
            - updated_messages: 可能修改过的消息列表
        """
        if not self.enable_tool_call_check:
            return True, messages

        # 找到最后一条 assistant 消息
        last_msg = messages[-1] if messages else None
        if not last_msg or last_msg.get("role") != "assistant":
            return True, messages

        # 提取工具调用
        tool_calls = self._extract_tool_calls_from_message(last_msg)

        if not tool_calls:
            return True, messages

        self.stats["tool_call_checks"] += 1
        logger.info(f"检查 {len(tool_calls)} 个工具调用的对齐性")

        # 检查对齐性
        result = self.checker.check_tool_call_alignment(
            tool_calls, self.user_task_set
        )

        if result.is_all_aligned:
            logger.debug("所有工具调用均与用户任务对齐")
            return True, messages

        # 有不对齐的工具调用
        self.stats["tool_call_blocks"] += 1

        # 分离对齐和不对齐的工具调用
        aligned_calls = []
        misaligned_calls = []
        for i, (tc, ar) in enumerate(zip(tool_calls, result.results)):
            if ar.is_aligned:
                aligned_calls.append(tc)
            else:
                misaligned_calls.append(tc)

        logger.warning(
            f"阻止 {len(misaligned_calls)} 个不对齐的工具调用，"
            f"保留 {len(aligned_calls)} 个对齐的工具调用"
        )

        # 生成反馈
        blocked_descriptions = [
            self.checker._format_tool_call(tc) for tc in misaligned_calls
        ]
        feedback = self.feedback_gen.generate_tool_call_feedback(
            self.user_task_set, blocked_descriptions
        )

        # 构建更新后的消息列表
        updated_messages = list(messages)

        if aligned_calls:
            # 如果还有对齐的工具调用，保留它们，只移除不对齐的
            updated_messages[-1] = self._rebuild_message_with_calls(
                last_msg, aligned_calls
            )
            # 在后面追加反馈（作为对被阻止调用的 tool result）
            for tc in misaligned_calls:
                tool_call_id = tc.get("id", "blocked")
                updated_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": feedback,
                    }
                )
        else:
            # 所有工具调用都被阻止，移除 assistant 的工具调用消息
            # 替换为反馈消息
            updated_messages = updated_messages[:-1]  # 移除最后的 assistant 消息
            updated_messages.append(
                {
                    "role": "user",
                    "content": feedback,
                }
            )

        return False, updated_messages

    def get_stats(self) -> dict:
        """获取 Task Shield 的运行统计信息。"""
        return dict(self.stats)

    def reset(self):
        """重置 Task Shield 的状态（用于处理新的用户请求）。"""
        self._user_task_set = None
        self.stats = {
            "content_checks": 0,
            "content_blocks": 0,
            "tool_call_checks": 0,
            "tool_call_blocks": 0,
            "instructions_extracted": 0,
        }

    @staticmethod
    def _extract_tool_calls_from_message(
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """从 assistant 消息中提取工具调用列表。"""
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return []

        result = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                result.append(tc)
            else:
                # 处理 AgentDojo 的 FunctionCall 对象
                try:
                    result.append(
                        {
                            "id": getattr(tc, "id", "unknown"),
                            "function": {
                                "name": getattr(tc, "function", "unknown"),
                                "arguments": json.dumps(
                                    getattr(tc, "args", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
                except Exception:
                    logger.warning(f"无法解析工具调用: {tc}")
        return result

    @staticmethod
    def _rebuild_message_with_calls(
        original_message: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """用过滤后的工具调用重建 assistant 消息。"""
        new_msg = dict(original_message)
        new_msg["tool_calls"] = tool_calls
        return new_msg
