"""GLM (智谱 BigModel) Teacher 适配器 — Anthropic 兼容接口

Docs: https://open.bigmodel.cn/api/anthropic
Base URL: https://open.bigmodel.cn/api/anthropic
格式: Anthropic Messages API 兼容
"""

import json
import httpx

from .base import BaseTeacher, TeacherResponse


class GLMTeacher(BaseTeacher):
    """GLM / 智谱 BigModel API (Anthropic 兼容格式)

    通过 Anthropic Messages API 调用 GLM-5.2
    Coding Plan 订阅，无需按量付费。
    """

    DEFAULT_MODEL = "glm-5.2"
    BASE_URL = "https://open.bigmodel.cn/api/anthropic"

    AVAILABLE_MODELS = [
        "glm-5.2",            # GLM-5.2 最新旗舰
        "glm-4-plus",         # GLM-4 旗舰
        "glm-4",              # GLM-4 标准
    ]

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **kwargs):
        super().__init__(api_key, model, self.BASE_URL, **kwargs)

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> TeacherResponse:
        """调用 GLM 对话 API (Anthropic 兼容格式)

        Args:
            messages: OpenAI 格式 [{"role": "system/user/assistant", "content": "..."}]
                     内部会自动转换为 Anthropic 格式

        Returns:
            TeacherResponse
        """
        # OpenAI 格式 → Anthropic 格式转换
        system = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system += msg["content"] + "\n"
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": temperature,
        }
        if system.strip():
            payload["system"] = system.strip()

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # 同步请求
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{self.BASE_URL}/v1/messages",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        # 解析 Anthropic 响应
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        usage = data.get("usage", {})

        return TeacherResponse(
            text=text,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
            raw=data,
        )
