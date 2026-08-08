"""Teacher 模型统一接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TeacherResponse:
    """Teacher 模型返回结构"""
    text: str
    model: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    raw: Optional[dict] = None


class BaseTeacher(ABC):
    """Teacher 模型统一基类"""

    def __init__(self, api_key: str, model: str, base_url: str = "", **kwargs):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> TeacherResponse:
        """对话接口

        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]

        Returns:
            TeacherResponse
        """
        ...

    def chat_simple(self, prompt: str, system: str = "") -> str:
        """简化调用，直接返回文本"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.chat(messages)
        return resp.text

    def __repr__(self):
        return f"<{self.__class__.__name__} model={self.model}>"


def create_teacher(teacher_type: str, api_key: str, model: str = "", **kwargs) -> BaseTeacher:
    """工厂函数 — 创建 Teacher 实例

    Args:
        teacher_type: "kimi" | "glm" | "deepseek"
        api_key: API Key
        model: 指定模型名（可选，用默认值）

    Returns:
        BaseTeacher 实例
    """
    from .kimi import KimiTeacher
    from .glm import GLMTeacher
    from .deepseek import DeepSeekTeacher

    teachers = {
        "kimi": KimiTeacher,
        "glm": GLMTeacher,
        "deepseek": DeepSeekTeacher,
    }

    if teacher_type not in teachers:
        raise ValueError(f"Unknown teacher type: {teacher_type}. Supported: {list(teachers.keys())}")

    cls = teachers[teacher_type]
    return cls(api_key=api_key, model=model or cls.DEFAULT_MODEL, **kwargs)
