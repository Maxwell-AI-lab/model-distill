#!/usr/bin/env python3
"""
完整蒸馏流水线 — 代码任务规划能力蒸馏

使用方法:
  # Step 1: 生成数据 (任意有网络的机器)
  python scripts/run_distill.py generate --teachers deepseek,glm,kimi

  # Step 2: 训练 (在 NPU 机器上)
  python scripts/run_distill.py train --model Qwen/Qwen2.5-Coder-7B-Instruct

  # Step 3: 评估 (在 NPU 机器上)
  python scripts/run_distill.py eval

  # 一键全流程
  python scripts/run_distill.py all
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel

console = Console()


# ── Step 1: 数据生成 ─────────────────────────────────────────

def cmd_generate(args):
    """Step 1: 用 Teacher API 生成蒸馏数据"""
    console.print(Panel.fit("📝 Step 1: 数据生成", style="bold yellow"))

    # 加载数据集
    from distill.data.datasets import DatasetLoader

    tasks = DatasetLoader.load_mixed(
        sources=args.sources.split(","),
        cache_dir="data/raw",
        total_limit=args.num_tasks,
    )

    train_tasks, eval_tasks = DatasetLoader.train_eval_split(
        tasks, eval_ratio=args.eval_ratio
    )

    # 保存原始题目
    DatasetLoader.save_jsonl(train_tasks, "data/tasks_train.jsonl")
    DatasetLoader.save_jsonl(eval_tasks, "data/tasks_eval.jsonl")

    # 创建 Teachers
    from distill.teachers import create_teacher

    teachers = {}
    teacher_configs = {
        "deepseek": ("DEEPSEEK_API_KEY", "deepseek-chat", "deepseek"),
        "glm": ("GLM_API_KEY", "glm-4-plus", "glm"),
        "kimi": ("KIMI_API_KEY", "moonshot-v1-32k", "kimi"),
    }

    for name in args.teachers.split(","):
        name = name.strip()
        env_key, default_model, teacher_type = teacher_configs[name]
        api_key = os.environ.get(env_key, "")
        if not api_key:
            console.print(f"⚠️ 跳过 {name}: 未设置 {env_key}", style="yellow")
            continue
        model = args.teacher_models.get(name, default_model)
        teachers[name] = create_teacher(teacher_type, api_key=api_key, model=model)
        console.print(f"  ✅ {name}: {teachers[name]}")

    if not teachers:
        console.print("❌ 没有可用的 Teacher!", style="red")
        sys.exit(1)

    # 生成蒸馏数据
    from distill.data.code_generator import CodeDistillGenerator

    generator = CodeDistillGenerator(teachers)
    generator.generate_batch(train_tasks, output_path="data/distill_raw.jsonl")

    # 提取训练数据
    from distill.data.code_generator import extract_training_data, PLANNING_SYSTEM

    extract_training_data(
        raw_path="data/distill_raw.jsonl",
        output_path="data/train_chatml.jsonl",
        use_best_only=True,
        system_prompt=PLANNING_SYSTEM,
    )

    console.print("\n✅ 数据生成完成!", style="bold green")
    console.print("   下一步: python scripts/run_distill.py train")


# ── Step 2: 训练 ─────────────────────────────────────────────

def cmd_train(args):
    """Step 2: NPU 上 SFT 训练"""
    console.print(Panel.fit("🔥 Step 2: NPU 训练", style="bold yellow"))

    # 检测 NPU
    from distill.train.npu_adapter import check_npu_available, print_npu_status

    if not check_npu_available():
        console.print("❌ NPU 不可用! 请在昇腾节点上运行此命令", style="red")
        console.print("   如果是在 GPU 机器上训练，请使用 scripts/train.py", style="dim")
        sys.exit(1)

    print_npu_status()

    # 启动训练
    from distill.train.npu_sft import NPUSFTTrainer, NPUSFTConfig

    config = NPUSFTConfig(
        model_name_or_path=args.model,
        train_file=args.train_data,
        use_lora=not args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        output_dir=args.output,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc,
        max_seq_length=args.max_seq_length,
    )

    trainer = NPUSFTTrainer(config)
    trainer.train()

    console.print("\n✅ 训练完成!", style="bold green")
    console.print("   下一步: python scripts/run_distill.py eval")


# ── Step 3: 评估 ─────────────────────────────────────────────

def cmd_eval(args):
    """Step 3: 评估蒸馏效果"""
    console.print(Panel.fit("📊 Step 3: 评估", style="bold yellow"))

    from distill.eval.code_eval import CodeEvaluator

    evaluator = CodeEvaluator()

    # 加载评估题
    eval_tasks = []
    with open(args.eval_data, "r", encoding="utf-8") as f:
        for line in f:
            eval_tasks.append(json.loads(line))

    console.print(f"📋 评估题数: {len(eval_tasks)}")

    # 加载 Student 模型 (如果有)
    if args.model_path:
        import torch
        import torch_npu  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"🤖 加载 Student 模型: {args.model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="npu:0",
        )
        model.eval()

        # 生成答案
        from distill.data.code_generator import PLANNING_SYSTEM, PLANNING_USER

        model_outputs = []
        for i, task in enumerate(eval_tasks):
            user_msg = PLANNING_USER.format(prompt=task["prompt"])
            messages = [
                {"role": "system", "content": PLANNING_SYSTEM},
                {"role": "user", "content": user_msg},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to("npu:0")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            model_outputs.append(response)

            console.print(f"  [{i+1}/{len(eval_tasks)}] 完成")

        # 评估
        test_cases_list = []
        for task in eval_tasks:
            tests = task.get("test_list", [])
            if not tests and task.get("test"):
                # HumanEval 格式
                tests = [task["test"]]
            test_cases_list.append(tests)

        summary = evaluator.evaluate_batch(
            model_outputs, test_cases_list,
            task_ids=[t.get("task_id", "") for t in eval_tasks]
        )
        evaluator.print_report(summary, title="Student 模型评估")

        # 保存结果
        with open("data/eval_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    else:
        console.print("⚠️ 未指定模型路径，跳过推理评估", style="yellow")
        console.print("   用法: python scripts/run_distill.py eval --model-path outputs/sft_npu")


# ── 全流程 ───────────────────────────────────────────────────

def cmd_all(args):
    """一键全流程"""
    cmd_generate(args)
    cmd_train(args)
    cmd_eval(args)


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="run_distill",
        description="🔧 代码任务规划能力蒸馏 — 完整流水线",
    )
    subparsers = parser.add_subparsers(dest="command")

    # generate
    gen = subparsers.add_parser("generate", help="生成蒸馏数据")
    gen.add_argument("--teachers", "-t", default="glm", help="Teacher列表(逗号分隔)")
    gen.add_argument("--sources", "-s", default="humaneval,mbpp", help="数据源")
    gen.add_argument("--num-tasks", "-n", type=int, default=500, help="题目数")
    gen.add_argument("--eval-ratio", type=float, default=0.1, help="评估集比例")

    # train
    tr = subparsers.add_parser("train", help="NPU 训练")
    tr.add_argument("--model", "-m", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    tr.add_argument("--train-data", "-d", default="data/train_chatml.jsonl")
    tr.add_argument("--output", "-o", default="outputs/sft_npu")
    tr.add_argument("--epochs", "-e", type=int, default=3)
    tr.add_argument("--batch-size", "-b", type=int, default=4)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--lora-r", type=int, default=64)
    tr.add_argument("--lora-alpha", type=int, default=128)
    tr.add_argument("--full-finetune", action="store_true", help="全参微调(默认LoRA)")
    tr.add_argument("--max-seq-length", type=int, default=2048)
    tr.add_argument("--nnodes", type=int, default=1)
    tr.add_argument("--nproc", type=int, default=8)

    # eval
    ev = subparsers.add_parser("eval", help="评估效果")
    ev.add_argument("--model-path", "-m", default="outputs/sft_npu")
    ev.add_argument("--eval-data", "-d", default="data/tasks_eval.jsonl")

    # all
    al = subparsers.add_parser("all", help="一键全流程")
    al.add_argument("--teachers", "-t", default="glm")
    al.add_argument("--sources", "-s", default="humaneval,mbpp")
    al.add_argument("--num-tasks", "-n", type=int, default=500)
    al.add_argument("--model", "-m", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    al.add_argument("--epochs", "-e", type=int, default=3)

    args = parser.parse_args()

    if args.command == "generate":
        args.teacher_models = {}
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "all":
        args.teacher_models = {}
        args.train_data = "data/train_chatml.jsonl"
        args.eval_data = "data/tasks_eval.jsonl"
        args.output = "outputs/sft_npu"
        args.batch_size = 4
        args.grad_accum = 4
        args.lr = 2e-4
        args.lora_r = 64
        args.lora_alpha = 128
        args.full_finetune = False
        args.max_seq_length = 2048
        args.nnodes = 1
        args.nproc = 8
        args.model_path = "outputs/sft_npu"
        args.eval_ratio = 0.1
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
