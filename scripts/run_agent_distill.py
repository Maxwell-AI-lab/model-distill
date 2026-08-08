#!/usr/bin/env python3
"""
Agent 轨迹蒸馏 — 完整流程

Step 1: 用 GLM-5.2 以 Agent 模式生成轨迹数据
Step 2: NPU 上训练 Qwen3-8B
Step 3: Agent 评估 (pass@1 + 纠错率)

用法:
  python3 scripts/run_agent_distill.py generate   # 生成轨迹
  python3 scripts/run_agent_distill.py train      # 训练
  python3 scripts/run_agent_distill.py eval       # 评估
  python3 scripts/run_agent_distill.py all        # 全流程
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")
os.chdir("/data/z00666713/model-distill")
os.environ["GLM_API_KEY"] = "4dbec255f842461ca9d26501f361ab2f.ayr6tCjeyOrkWngV"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def cmd_generate(args):
    """Step 1: GLM-5.2 Agent 模式生成轨迹"""
    console.print(Panel.fit("🤖 Step 1: Agent 轨迹生成", style="bold yellow"))

    from distill.teachers import create_teacher
    from distill.data.datasets import DatasetLoader
    from distill.data.agent_generator import AgentTrajectoryGenerator, trajectories_to_training_data

    key = os.environ["GLM_API_KEY"]
    teacher = create_teacher("glm", api_key=key, model="glm-5.2")

    from distill.eval.agent_eval import AgentEvaluator
    import torch
    import torch_npu

    # 加载评估题
    eval_tasks = []
    with open("data/tasks_eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            eval_tasks.append(json.loads(line))

    console.print(f"📋 评估题: {len(eval_tasks)} 道")

    # 加载原始模型 (未蒸馏 baseline) 或蒸馏后模型
    model_path = args.model_path or "/data/model/Qwen3-8B"
    console.print(f"🤖 加载模型: {model_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="npu:0",
    )
    model.eval()

    # 评估
    evaluator = AgentEvaluator(model, tokenizer, device="npu:0", max_rounds=5)
    summary = evaluator.evaluate_batch(eval_tasks)

    # 打印报告
    table = Table(title="Agent 评估结果", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="dim")
    table.add_column("值", justify="right")
    table.add_row("总题数", str(summary["total"]))
    table.add_row("✅ pass@1", f"{summary['passed']}/{summary['total']} ({summary['pass_rate']*100:.1f}%)")
    table.add_row("首次通过", f"{summary['first_attempt_passed']} ({summary['first_pass_rate']*100:.1f}%)")
    table.add_row("🔧 纠错成功", f"{summary['error_corrected']} ({summary['correction_rate']*100:.1f}%)")
    table.add_row("平均轮数", f"{summary['avg_rounds']:.1f}")
    console.print(table)

    # 保存
    model_name = os.path.basename(model_path)
    with open(f"data/eval_agent_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n💾 详细结果: data/eval_agent_{model_name}.json")


def cmd_all(args):
    cmd_generate(args)
    cmd_train(args)
    args.model_path = "outputs/agent-distill"
    cmd_eval(args)


def main():
    parser = argparse.ArgumentParser(description="Agent 轨迹蒸馏")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="生成 Agent 轨迹")
    gen.add_argument("--num", "-n", type=int, default=40)

    tr = sub.add_parser("train", help="训练")
    tr.add_argument("--model", "-m", default="/data/model/Qwen3-8B")
    tr.add_argument("--data", "-d", default="data/agent_train.jsonl")
    tr.add_argument("--output", "-o", default="outputs/agent-distill")
    tr.add_argument("--epochs", "-e", type=int, default=3)

    ev = sub.add_parser("eval", help="评估")
    ev.add_argument("--model-path", "-m", default=None)

    al = sub.add_parser("all", help="全流程")
    al.add_argument("--num", "-n", type=int, default=40)
    al.add_argument("--model", "-m", default="/data/model/Qwen3-8B")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "all":
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
