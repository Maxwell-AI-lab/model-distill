"""Teacher 模型适配器测试"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill.teachers import create_teacher, KimiTeacher, GLMTeacher, DeepSeekTeacher


def test_teacher_creation():
    """测试创建 Teacher 实例"""
    # 无需 API Key 的测试
    teachers_config = [
        ("kimi", KimiTeacher, "moonshot-v1-32k"),
        ("glm", GLMTeacher, "glm-4-plus"),
        ("deepseek", DeepSeekTeacher, "deepseek-chat"),
    ]

    for name, cls, expected_default in teachers_config:
        # 测试默认模型
        assert cls.DEFAULT_MODEL, f"{name} 缺少默认模型"
        # 测试 base URL
        assert cls.BASE_URL, f"{name} 缺少 base URL"
        # 测试可用模型列表
        assert len(cls.AVAILABLE_MODELS) > 0, f"{name} 缺少可用模型列表"


def test_factory_function():
    """测试工厂函数"""
    # 需要 fake key，只测试类型创建
    teacher = create_teacher("kimi", api_key="fake-key")
    assert isinstance(teacher, KimiTeacher)

    teacher = create_teacher("glm", api_key="fake-key")
    assert isinstance(teacher, GLMTeacher)

    teacher = create_teacher("deepseek", api_key="fake-key")
    assert isinstance(teacher, DeepSeekTeacher)

    # 测试未知类型
    try:
        create_teacher("unknown", api_key="fake-key")
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass


def test_chat_simple_signature():
    """测试接口方法存在"""
    teacher = create_teacher("glm", api_key="fake-key")
    assert hasattr(teacher, "chat")
    assert hasattr(teacher, "chat_simple")
    assert callable(teacher.chat)
    assert callable(teacher.chat_simple)


if __name__ == "__main__":
    test_teacher_creation()
    test_factory_function()
    test_chat_simple_signature()
    print("✅ 所有测试通过!")
