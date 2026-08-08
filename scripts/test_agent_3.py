#!/usr/bin/env python3
"""测试 Agent 轨迹生成 (3题)"""
import os, sys, json

os.environ["GLM_API_KEY"] = "4dbec255f842461ca9d26501f361ab2f.ayr6tCjeyOrkWngV"
sys.path.insert(0, ".")
os.chdir("/data/z00666713/model-distill")

from distill.teachers import create_teacher
from distill.data.datasets import DatasetLoader
from distill.data.agent_generator import AgentTrajectoryGenerator

key = os.environ["GLM_API_KEY"]
teacher = create_teacher("glm", api_key=key, model="glm-5.2")
print(f"Teacher: {teacher}")

tasks = DatasetLoader.load_mixed(sources=["humaneval"], cache_dir="data/raw", total_limit=3)
print(f"Tasks: {len(tasks)}")

gen = AgentTrajectoryGenerator(teacher, max_rounds=5)
results = gen.generate_batch(tasks, output_path="data/agent_test_3.jsonl")

# 打印第一条完整轨迹
if results:
    r = results[0]
    print(f"\n{'='*60}")
    print(f"轨迹: {r['task_id']} | 通过: {r['passed']} | 轮数: {r['rounds']}")
    print(f"{'='*60}")
    for i, msg in enumerate(r["trajectory"]):
        role = msg["role"]
        content = msg["content"]
        # 每条消息最多显示 500 字符
        if len(content) > 500:
            content = content[:500] + "\n... (截断)"
        print(f"\n--- [{i}] {role} ---")
        print(content)
    print(f"\n{'='*60}")
