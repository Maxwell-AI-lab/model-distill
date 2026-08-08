#!/usr/bin/env python3
"""生成完整训练数据 (500题) + 训练 + 评估"""
import os
import sys

os.environ["GLM_API_KEY"] = "4dbec255f842461ca9d26501f361ab2f.ayr6tCjeyOrkWngV"
sys.path.insert(0, ".")
os.chdir("/data/z00666713/model-distill")

from rich.console import Console
console = Console()

# ═══ Step 1: 数据生成 ═══
console.print("\n[bold yellow]═══ Step 1: 数据生成 (500题) ═══[/bold yellow]\n")

from distill.teachers import create_teacher
from distill.data.datasets import DatasetLoader
from distill.data.code_generator import CodeDistillGenerator, extract_training_data, PLANNING_SYSTEM

key = os.environ["GLM_API_KEY"]
teacher = create_teacher("glm", api_key=key)
console.print(f"✅ Teacher: {teacher}")

# 加载数据 — HumanEval + MBPP
tasks = DatasetLoader.load_mixed(
    sources=["humaneval", "mbpp"],
    cache_dir="data/raw",
    total_limit=500,
)
console.print(f"✅ 加载 {len(tasks)} 题")

# 划分
train_tasks, eval_tasks = DatasetLoader.train_eval_split(tasks, eval_ratio=0.1)
DatasetLoader.save_jsonl(train_tasks, "data/tasks_train.jsonl")
DatasetLoader.save_jsonl(eval_tasks, "data/tasks_eval.jsonl")

# GLM-5.2 生成
gen = CodeDistillGenerator({"glm": teacher})
gen.generate_batch(train_tasks, output_path="data/distill_raw.jsonl")

# 提取训练数据
train_data = extract_training_data(
    raw_path="data/distill_raw.jsonl",
    output_path="data/train_chatml.jsonl",
)

console.print(f"\n[bold green]✅ 数据生成完成: {len(train_data)} 条训练数据[/bold green]")
console.print(f"   评估集: {len(eval_tasks)} 题")
console.print("\n下一步: 运行训练脚本\n")
