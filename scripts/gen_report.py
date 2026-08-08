#!/usr/bin/env python3
"""数据生成验证报告 — 统计数据量、质量、样例展示"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
os.chdir("/data/z00666713/model-distill")

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# ═══ 加载数据 ═══
raw = load_jsonl("data/distill_raw.jsonl")
train = load_jsonl("data/train_chatml.jsonl")
tasks_train = load_jsonl("data/tasks_train.jsonl")
tasks_eval = load_jsonl("data/tasks_eval.jsonl")

# ═══ 统计 ═══
report = []
report.append("# 数据生成验证报告")
report.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"> Teacher: GLM-5.2 (智谱 BigModel, Anthropic 兼容 API)\n")

# 1. 总览
report.append("## 1. 总览\n")
report.append(f"| 指标 | 值 |")
report.append(f"|------|------|")
report.append(f"| 题目总数 | {len(tasks_train) + len(tasks_eval)} |")
report.append(f"| 训练集 | {len(tasks_train)} 题 |")
report.append(f"| 评估集 | {len(tasks_eval)} 题 |")
report.append(f"| Teacher 生成 | {len(raw)} 条 |")
report.append(f"| 有效训练数据 | {len(train)} 条 |")

# 2. 数据来源分布
report.append("\n## 2. 数据来源分布\n")
source_counts = {}
for t in tasks_train + tasks_eval:
    src = t.get("source", "unknown")
    source_counts[src] = source_counts.get(src, 0) + 1
report.append("| 来源 | 数量 |")
report.append("|------|------|")
for src, cnt in sorted(source_counts.items()):
    report.append(f"| {src} | {cnt} |")

# 3. 质量统计
report.append("\n## 3. 质量统计\n")
valid_count = 0
has_plan = 0
has_code = 0
has_boundary = 0
plan_steps = []
code_lengths = []

for item in raw:
    best_teacher = item.get("best_teacher", "")
    if not best_teacher:
        continue
    resp = item.get("responses", {}).get(best_teacher, {})
    if "error" in resp:
        continue
    valid_count += 1
    if resp.get("plan"):
        has_plan += 1
        # 计算步骤数
        import re
        steps = re.findall(r"^\d+\.", resp["plan"], re.MULTILINE)
        plan_steps.append(len(steps))
    if resp.get("code"):
        has_code += 1
        code_lengths.append(len(resp["code"]))
    if resp.get("boundary"):
        has_boundary += 1

report.append(f"| 质量指标 | 值 |")
report.append(f"|----------|------|")
report.append(f"| 有效数据 | {valid_count}/{len(raw)} ({valid_count/len(raw)*100:.1f}%) |" if raw else "| 有效数据 | 0 |")
report.append(f"| 有解题计划 | {has_plan}/{valid_count} ({has_plan/valid_count*100:.1f}%) |" if valid_count else "| 有计划 | 0 |")
report.append(f"| 有代码实现 | {has_code}/{valid_count} ({has_code/valid_count*100:.1f}%) |" if valid_count else "| 有代码 | 0 |")
report.append(f"| 有边界分析 | {has_boundary}/{valid_count} ({has_boundary/valid_count*100:.1f}%) |" if valid_count else "| 有边界 | 0 |")
if plan_steps:
    report.append(f"| 平均计划步骤数 | {sum(plan_steps)/len(plan_steps):.1f} 步 |")
    report.append(f"| 最少/最多步骤 | {min(plan_steps)}/{max(plan_steps)} 步 |")
if code_lengths:
    report.append(f"| 平均代码长度 | {sum(code_lengths)/len(code_lengths):.0f} 字符 |")

# 4. 训练数据格式
report.append("\n## 4. 训练数据格式\n")
report.append("```jsonl")
if train:
    sample = train[0]
    report.append(json.dumps({
        "messages": [
            {"role": "system", "content": sample["messages"][0]["content"][:80] + "..."},
            {"role": "user", "content": sample["messages"][1]["content"][:80] + "..."},
            {"role": "assistant", "content": sample["messages"][2]["content"][:120] + "..."},
        ]
    }, ensure_ascii=False, indent=2))
report.append("```")

# 5. 样例展示 (3条)
report.append("\n## 5. 数据样例 (3条)\n")
for i, item in enumerate(raw[:3]):
    best = item.get("best_teacher", "")
    resp = item.get("responses", {}).get(best, {})
    report.append(f"### 样例 {i+1}: {item.get('task_id', '')}\n")
    report.append(f"**来源**: {item.get('source', '')} | **Teacher**: {best}\n")
    report.append(f"**题目**:\n```\n{item['prompt'][:200]}...\n```\n")
    if resp.get("plan"):
        report.append(f"**解题计划**:\n{resp['plan'][:300]}...\n")
    if resp.get("code"):
        report.append(f"**代码实现**:\n```python\n{resp['code'][:200]}...\n```\n")

# 6. 文件清单
report.append("\n## 6. 产出文件\n")
report.append("| 文件 | 行数 | 大小 | 说明 |")
report.append("|------|------|------|------|")
for fname, desc in [
    ("data/tasks_train.jsonl", "训练题目"),
    ("data/tasks_eval.jsonl", "评估题目"),
    ("data/distill_raw.jsonl", "Teacher 原始生成"),
    ("data/train_chatml.jsonl", "ChatML 训练数据"),
]:
    p = Path(fname)
    if p.exists():
        size = p.stat().st_size
        lines = sum(1 for _ in open(p, encoding="utf-8"))
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
        report.append(f"| {fname} | {lines} | {size_str} | {desc} |")

# 7. 下一步
report.append("\n## 7. 下一步\n")
report.append("- [ ] NPU 训练: `python3 scripts/run_distill.py train`")
report.append("- [ ] 评估对比: `python3 scripts/run_distill.py eval`")
report.append("- [ ] 生成训练报告\n")

# ═══ 写入文件 ═══
report_text = "\n".join(report)
report_path = "docs/DATA_REPORT.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print(f"\n✅ 报告已保存到 {report_path}")
