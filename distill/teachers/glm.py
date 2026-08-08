"""GLM (智谱 BigModel) Teacher 适配器"""

from openai import OpenAI

from .base import BaseTeacher, TeacherResponse


class GLMTeacher(BaseTeacher):
    """GLM / 智谱 BigModel API

    Docs: https://open.bigmodel.cn/dev/api
    Base URL: https://open.bigmodel.cn/api/paas/v4
    """

    DEFAULT_MODEL = "glm-5.2"
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    AVAILABLE_MODELS = [
        "glm-5.2",           # GLM-5.2 最新旗舰
        "glm-4-plus",        # GLM-4 旗舰
        "glm-4",             # GLM-4 标准
        "glm-4-air",         # GLM-4 轻量
        "glm-4-airx",        # GLM-4 轻量极速
        "glm-4-flash",       # GLM-4 免费极速
        "glm-4-long",        # GLM-4 长文本
    ]

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **kwargs):
        super().__init__(api_key, model, self.BASE_URL, **kwargs)
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> TeacherResponse:
        """调用 GLM 对话 API"""
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
