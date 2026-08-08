"""快速使用脚本 — 直接调用 Teacher 生成数据"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill.teachers import create_teacher
from distill.data import DataGenerator, QualityFilter, DataFormatter


def main():
    # 1. 创建 Teacher（选择一个）
    # 需要设置环境变量: KIMI_API_KEY / GLM_API_KEY / DEEPSEEK_API_KEY

    examples = [
        ("kimi", "KIMI_API_KEY", "moonshot-v1-32k"),
        ("glm", "GLM_API_KEY", "glm-4-plus"),
        ("deepseek", "DEEPSEEK_API_KEY", "deepseek-chat"),
    ]

    teacher_type = os.environ.get("TEACHER_TYPE", "glm")
    teacher_info = next((e for e in examples if e[0] == teacher_type), examples[1])
    api_key = os.environ.get(teacher_info[1], "")

    if not api_key:
        print(f"❌ 请设置环境变量 {teacher_info[1]}")
        sys.exit(1)

    teacher = create_teacher(teacher_type, api_key=api_key, model=teacher_info[2])
    print(f"✅ Teacher: {teacher}")

    # 2. 测试调用
    response = teacher.chat_simple("你好，请简单介绍一下你自己")
    print(f"\n💬 Teacher 回复: {response[:200]}...")

    # 3. 生成数据
    gen = DataGenerator(teacher, {
        "scene": "通用问答",
        "system_prompt": "你是一个知识渊博的AI助手。",
        "topic_seeds": ["科技", "历史", "文化", "生活", "教育"],
        "num_samples": 10,
    })

    data = gen.generate_batch("data/quick_test.jsonl")

    # 4. 过滤
    filt = QualityFilter()
    clean = filt.filter_batch(data)

    # 5. 格式转换
    DataFormatter.to_chatml(clean, "data/quick_test_chatml.jsonl")

    print("\n🎉 快速测试完成!")


if __name__ == "__main__":
    main()
