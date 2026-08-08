"""Kimi (Moonshot AI) Teacher 适配器"""

from openai import OpenAI

from .base import BaseTeacher, TeacherResponse


class KimiTeacher(BaseTeacher):
    """Kimi / Moonshot API

    Docs: https://platform.moonshot.cn/docs
    Base URL: https://api.moonshot.cn/v1
    """

    DEFAULT_MODEL = "moonshot-v1-128k"
    BASE_URL = "https://api.moonshot.cn/v1"

    AVAILABLE_MODELS = [
        "moonshot-v1-8k",      # 8K 上下文
        "moonshot-v1-32k",     # 32K 上下文
        "moonshot-v1-128k",    # 128K 上下文
        "moonshot-v1-auto",    # 自动选择
    ]

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **kwargs):
        super().__init__(api_key, model, self.BASE_URL, **kwargs)
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> TeacherResponse:
        """调用 Kimi 对话 API"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return TeacherResponse(
            text=resp.choices[0].message.content,
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
            raw=resp.model_dump(),
        )
