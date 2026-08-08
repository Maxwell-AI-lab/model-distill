#!/usr/bin/env python3
"""生成 50 题数据 + 验证报告"""
import json
import os
import sys
import re
import subprocess
from datetime import datetime
from pathlib import Path

os.environ["GLM_API_KEY"] = "4dbec255f842461ca9d26501f361ab2f.ayr6tCjeyOrkWngV"
sys.path.insert(0, ".")
os.chdir("/data/z00666713/model-distill")

from rich.console import Console
console = Console()

# ═══ Step 1: 数据生成 ═══
console.print("\n[bold yellow]═══ 数据生成 (50题) ═══[/bold yellow]\n")

from distill.teachers import create_teacher
from distill.data.datasets import DatasetLoader
from distill.data.code_generator import CodeDistillGenerator, extract_training_data

key = os.environ["GLM_API_KEY"]
teacher = create_teacher("glm", api_key=key)
console.print(f"✅ Teacher: {teacher}")

tasks = DatasetLoader.load_mixed(
    sources=["humaneval", "mbpp"],
    cache_dir="data/raw",
    total_limit=50,
)
console.print(f"✅ 加载 {len(tasks)} 题")

train_tasks, eval_tasks = DatasetLoader.train_eval_split(tasks, eval_ratio=0.2)
DatasetLoader.save_jsonl(train_tasks, "data/tasks_train.jsonl")
DatasetLoader.save_jsonl(eval_tasks, "data/tasks_eval.jsonl")

gen = CodeDistillGenerator({"glm": teacher})
gen.generate_batch(train_tasks, output_path="data/distill_raw.jsonl")

train_data = extract_training_data(
    raw_path="data/distill_raw.jsonl",
    output_path="data/train_chatml.jsonl",
)

console.print(f"\n[bold green]✅ 完成: {len(train_data)} 条训练数据, {len(eval_tasks)} 题评估集[/bold green]")

# ═══ Step 2: 验证报告 ═══
console.print("\n[bold yellow]═══ 生成报告 ═══[/bold yellow]\n")

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

raw = load_jsonl("data/distill_raw.jsonl")
train = load_jsonl("data/train_chatml.jsonl")

# 统计
valid_count = 0
has_plan = 0
has_code = 0
has_boundary = 0
plan_steps_list = []
code_lengths = []

for item in raw:
    best = item.get("best_teacher", "")
    if not best:
        continue
    resp = item.get("responses", {}).get(best, {})
    if "error" in resp:
        continue
    valid_count += 1
    if resp.get("plan"):
        has_plan += 1
        steps = re.findall(r"^\d+\.", resp["plan"], re.MULTILINE)
        plan_steps_list.append(len(steps))
    if resp.get("code"):
        has_code += 1
        code_lengths.append(len(resp["code"]))
    if resp.get("boundary"):
        has_boundary += 1

report = []
report.append("# 数据生成验证报告")
report.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"> Teacher: GLM-5.2 (智谱 BigModel, Anthropic 兼容 API)")
report.append(f"> Student: Qwen3-8B\n")

report.append("## 总览\n")
report.append(f"| 指标 | 值 |")
report.append(f"|------|------|")
report.append(f"| 题目总数 | {len(tasks)} |")
report.append(f"| 训练集 | {len(train_tasks)} 题 |")
report.append(f"| 评估集 | {len(eval_tasks)} 题 |")
report.append(f"| Teacher 生成 | {len(raw)} 条 |")
report.append(f"| 有效训练数据 | {len(train)} 条 |")
if valid_count:
    report.append(f"| 有效率 | {valid_count}/{len(raw)} ({valid_count/len(raw)*100:.1f}%) |")
    report.append(f"| 有解题计划 | {has_plan}/{valid_count} ({has_plan/valid_count*100:.1f}%) |")
    report.append(f"| 有代码实现 | {has_code}/{valid_count} ({has_code/valid_count*100:.1f}%) |")
    report.append(f"| 有边界分析 | {has_boundary}/{valid_count} ({has_boundary/valid_count*100:.1f}%) |")
    if plan_steps_list:
        report.append(f"| 平均计划步骤 | {sum(plan_steps_list)/len(plan_steps_list):.1f} 步 |")
        report.append(f"| 步骤范围 | {min(plan_steps_list)}~{max(plan_steps_list)} 步 |")
    if code_lengths:
        report.append(f"| 平均代码长度 | {sum(code_lengths)/len(code_lengths):.0f} 字符 |")

# 样例
report.append("\n## 数据样例 (2条)\n")
for i, item in enumerate(raw[:2]):
    best = item.get("best_teacher", "")
    resp = item.get("responses", {}).get(best, {})
    report.append(f"### 样例 {i+1}: {item.get('task_id', '')}\n")
    report.append(f"**题目**: {item['prompt'][:150]}...\n")
    if resp.get("plan"):
        report.append(f"**解题计划**:\n```\n{resp['plan'][:300]}...\n```\n")
    if resp.get("code"):
        report.append(f"**代码**:\n```python\n{resp['code'][:200]}...\n```\n")

# 文件清单
report.append("\n## 产出文件\n")
report.append("| 文件 | 行数 | 大小 |")
report.append("|------|------|------|")
for fname in ["data/tasks_train.jsonl", "data/tasks_eval.jsonl", "data/distill_raw.jsonl", "data/train_chatml.jsonl"]:
    p = Path(fname)
    if p.exists():
        size = p.stat().st_size
        lines = sum(1 for _ in open(p, encoding="utf-8"))
        size_str = f"{size/1024:.1f}KB"
        report.append(f"| {fname} | {lines} | {size_str} |")

report.append("\n## 下一步\n")
report.append("- [ ] NPU 训练 (Qwen3-8B + LoRA)")
report.append("- [ ] 评估 pass@1")
report.append("- [ ] 分析结果，决定是否扩大到 500 题\n")

report_text = "\n".join(report)
with open("docs/DATA_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
