#!/usr/bin/env python3
"""快速验证 GLM-5.2 API + 生成小批数据 (10题)"""
import os
import sys

os.environ["GLM_API_KEY"] = "4dbec255f842461ca9d26501f361ab2f.ayr6tCjeyOrkWngV"
sys.path.insert(0, "/data/z00666713/model-distill")

from distill.teachers import create_teacher

# 1. 验证 API
print("=== 验证 GLM-5.2 API ===")
key = os.environ["GLM_API_KEY"]
teacher = create_teacher("glm", api_key=key)
resp = teacher.chat_simple("1+1=?")
print(f"✅ GLM-5.2 OK: {resp[:80]}")

# 2. 加载数据集
print("\n=== 加载 HumanEval ===")
from distill.data.datasets import DatasetLoader

tasks = DatasetLoader.load_mixed(
    sources=["humaneval"],
    cache_dir="/data/z00666713/model-distill/data/raw",
    total_limit=10,
)
print(f"加载 {len(tasks)} 题")

# 3. 生成蒸馏数据
print("\n=== 开始生成 (10题) ===")
from distill.data.code_generator import CodeDistillGenerator

gen = CodeDistillGenerator({"glm": teacher})
results = gen.generate_batch(
    tasks,
    output_path="/data/z00666713/model-distill/data/distill_raw_10.jsonl",
)

# 4. 提取训练数据
print("\n=== 提取训练数据 ===")
from distill.data.code_generator import extract_training_data

train_data = extract_training_data(
    raw_path="/data/z00666713/model-distill/data/distill_raw_10.jsonl",
    output_path="/data/z00666713/model-distill/data/train_chatml_10.jsonl",
)

# 5. 检查结果
print("\n=== 结果 ===")
for item in train_data:
    meta = item.get("meta", {})
    print(f"  {meta.get('task_id', '?')} | teacher={meta.get('teacher', '?')} | "
          f"difficulty={meta.get('difficulty', '?')}")

print(f"\n✅ 生成完成: {len(train_data)} 条训练数据")
