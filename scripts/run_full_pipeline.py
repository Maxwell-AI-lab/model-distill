#!/usr/bin/env python3
"""完整蒸馏流水线: Agent 轨迹生成 -> NPU 训练 -> 评估 -> 报告"""
import os, sys, json, time
from datetime import datetime
from pathlib import Path

os.environ["GLM_API_KEY"] = "4dbec255f842461ca9d26501f361ab2f.ayr6tCjeyOrkWngV"
sys.path.insert(0, ".")
os.chdir("/data/z00666713/model-distill")

from rich.console import Console
console = Console()

# Phase 1: Agent 轨迹生成
console.print("\n[bold yellow]=== Phase 1: Agent 轨迹生成 ===[/bold yellow]\n")

from distill.teachers import create_teacher
from distill.data.datasets import DatasetLoader
from distill.data.agent_generator import AgentTrajectoryGenerator, trajectories_to_training_data

GLM_KEY = os.environ.get("GLM_API_KEY", "")
teacher = create_teacher("glm", api_key=GLM_KEY, model="glm-5.2")

train_tasks = []
with open("data/tasks_train.jsonl") as f:
    for line in f:
        train_tasks.append(json.loads(line))
console.print(f"训练题: {len(train_tasks)} 道")

gen = AgentTrajectoryGenerator(teacher, max_rounds=5)
results = gen.generate_batch(train_tasks, output_path="data/agent_trajectories.jsonl")

train_data = trajectories_to_training_data(results, output_path="data/agent_train.jsonl", only_passed=True)
console.print(f"训练数据: {len(train_data)} 条")

# Phase 2: NPU 训练
console.print("\n[bold yellow]=== Phase 2: NPU 训练 ===[/bold yellow]\n")

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

MODEL_PATH = "/data/model/Qwen3-8B"
OUTPUT_DIR = "outputs/agent-distill"

console.print(f"加载模型: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="npu:0"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="data/agent_train.jsonl", split="train")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=3,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=2e-4, warmup_ratio=0.1, lr_scheduler_type="cosine",
    logging_steps=5, save_steps=999999, bf16=True, no_cuda=True,
    report_to="none", seed=42,
    max_length=4096,
)

trainer = SFTTrainer(
    model=model, args=training_args, train_dataset=dataset,
    processing_class=tokenizer,
)

console.print("\n训练中...\n")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
console.print(f"\n模型保存到: {OUTPUT_DIR}")

# Phase 3: Agent 评估
console.print("\n[bold yellow]=== Phase 3: Agent 评估 ===[/bold yellow]\n")

eval_tasks = []
with open("data/tasks_eval.jsonl") as f:
    for line in f:
        eval_tasks.append(json.loads(line))

console.print(f"评估题: {len(eval_tasks)} 道")

from distill.eval.agent_eval import AgentEvaluator
model.eval()
evaluator = AgentEvaluator(model, tokenizer, device="npu:0", max_rounds=5)
summary = evaluator.evaluate_batch(eval_tasks)

# 报告
gen_passed = sum(1 for r in results if r["passed"])
gen_total = len(results)
gen_avg = sum(r["rounds"] for r in results) / gen_total if gen_total else 0

report_lines = [
    "# Agent 轨迹蒸馏完整报告",
    "",
    f"> 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "> Teacher: GLM-5.2 (Anthropic API, Coding Plan)",
    "> Student: Qwen3-8B + LoRA(r=64)",
    "> 硬件: 昇腾 910B3 x 8",
    "",
    "## Phase 1: Agent 轨迹生成",
    "",
    "| 指标 | 值 |",
    "|------|------|",
    f"| 训练题数 | {len(train_tasks)} |",
    f"| Agent 轨迹 | {gen_total} 条 |",
    f"| Teacher 通过率 | {gen_passed}/{gen_total} ({gen_passed/gen_total*100:.1f}%) |",
    f"| 平均交互轮数 | {gen_avg:.1f} |",
    f"| 有效训练数据 | {len(train_data)} 条 |",
    "",
    "## Phase 2: 训练",
    "",
    "| 参数 | 值 |",
    "|------|------|",
    "| 基座模型 | Qwen3-8B |",
    "| 微调方式 | LoRA |",
    "| rank / alpha | 64 / 128 |",
    "| 精度 | bf16 |",
    "| Epochs | 3 |",
    f"| 训练数据 | {len(train_data)} 条 |",
    "| 序列长度 | 4096 |",
    "",
    "## Phase 3: Agent 评估 (蒸馏后模型)",
    "",
    "| 指标 | 值 |",
    "|------|------|",
    f"| 评估题数 | {summary['total']} |",
    f"| pass@1 | {summary['passed']}/{summary['total']} ({summary['pass_rate']*100:.1f}%) |",
    f"| 首次通过率 | {summary['first_attempt_passed']} ({summary['first_pass_rate']*100:.1f}%) |",
    f"| 纠错成功率 | {summary['error_corrected']} ({summary['correction_rate']*100:.1f}%) |",
    f"| 平均轮数 | {summary['avg_rounds']:.1f} |",
    "",
    "## 结论",
    "",
    f"- GLM-5.2 Agent 轨迹质量: {gen_passed}/{gen_total} 通过",
    f"- 蒸馏后 Qwen3-8B pass@1: {summary['pass_rate']*100:.1f}%",
    f"- 纠错能力: {summary['correction_rate']*100:.1f}%",
    "",
]

report_text = "\n".join(report_lines)
with open("docs/AGENT_DISTILL_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n" + "="*60)
print(report_text)
print("="*60)
print("\n报告已保存到 docs/AGENT_DISTILL_REPORT.md")
