"""
DeepSeek API 客户端封装。

使用 OpenAI 兼容的 API 格式调用 DeepSeek-v4。
用于 Task Shield 内部的 LLM 调用（指令提取、对齐检查等）。
"""

import json
import logging
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """
    DeepSeek API 客户端。

    DeepSeek 使用 OpenAI 兼容的 API 格式，因此直接复用 openai 库。
    该客户端用于 Task Shield 内部的 LLM 调用（不是 Agent 本身的 LLM 调用）。

    Args:
        api_key: DeepSeek API Key，默认从环境变量 DEEPSEEK_API_KEY 读取
        base_url: DeepSeek API 地址，默认从环境变量读取
        model: 模型名称，默认从环境变量读取
        temperature: 生成温度，对于判定类任务建议使用较低的温度
        max_retries: 最大重试次数
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.temperature = temperature
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "DeepSeek API Key 未设置。请设置环境变量 DEEPSEEK_API_KEY 或在 .env 文件中配置。"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.max_retries,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        """
        发送聊天请求并获取文本回复。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 覆盖默认温度

        Returns:
            模型的文本回复
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature if temperature is not None else self.temperature,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"DeepSeek API 调用失败: {e}")
            raise

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> Any:
        """
        发送聊天请求并解析 JSON 回复。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 覆盖默认温度

        Returns:
            解析后的 JSON 对象（dict 或 list）
        """
        response_text = self.chat(system_prompt, user_prompt, temperature)

        # 尝试从响应中提取 JSON
        # 有时 LLM 会在 JSON 外面包裹 markdown 代码块
        json_text = response_text

        # 去除可能的 markdown 代码块标记
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        json_text = json_text.strip()

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning(
                f"无法解析 JSON 响应，原始文本:\n{response_text}\n"
                f"清理后文本:\n{json_text}"
            )
            # 尝试宽松解析：查找第一个 [ 或 { 到最后一个 ] 或 }
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start_idx = json_text.find(start_char)
                end_idx = json_text.rfind(end_char)
                if start_idx != -1 and end_idx != -1:
                    try:
                        return json.loads(json_text[start_idx : end_idx + 1])
                    except json.JSONDecodeError:
                        continue

            logger.error(f"JSON 解析彻底失败，返回空结果")
            # 根据期望格式返回默认值
            return []


def create_deepseek_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """
    创建一个指向 DeepSeek 的 OpenAI 客户端。

    该客户端用于 AgentDojo 的 Agent Pipeline（Agent 本身的 LLM 调用）。
    这样可以让 AgentDojo 的 OpenAILLM 类直接使用 DeepSeek 作为后端。

    Args:
        api_key: DeepSeek API Key
        base_url: DeepSeek API 地址

    Returns:
        配置为使用 DeepSeek 的 OpenAI 客户端
    """
    _api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    _base_url = base_url or os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )

    if not _api_key:
        raise ValueError(
            "DeepSeek API Key 未设置。请设置环境变量 DEEPSEEK_API_KEY 或在 .env 文件中配置。"
        )

    return OpenAI(api_key=_api_key, base_url=_base_url)
