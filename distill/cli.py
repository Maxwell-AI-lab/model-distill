"""CLI 入口"""

import argparse
import sys

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        prog="distill",
        description="🔧 Model Distill — 大模型能力蒸馏流水线",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run — 完整流水线
    run_parser = subparsers.add_parser("run", help="运行完整蒸馏流水线")
    run_parser.add_argument("--config", "-c", required=True, help="实验配置 YAML 文件")
    run_parser.add_argument("--skip-generate", action="store_true", help="跳过数据生成")
    run_parser.add_argument("--skip-train", action="store_true", help="跳过训练")
    run_parser.add_argument("--skip-eval", action="store_true", help="跳过评估")

    # generate — 仅生成数据
    gen_parser = subparsers.add_parser("generate", help="生成蒸馏数据")
    gen_parser.add_argument("--config", "-c", required=True, help="实验配置 YAML")
    gen_parser.add_argument("--output", "-o", default="data/generated.jsonl", help="输出路径")

    # train — 仅训练
    train_parser = subparsers.add_parser("train", help="训练 Student 模型")
    train_parser.add_argument("--config", "-c", required=True, help="实验配置 YAML")
    train_parser.add_argument("--method", "-m", default="sft", choices=["sft", "dpo"], help="训练方法")

    # eval — 仅评估
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--config", "-c", required=True, help="实验配置 YAML")
    eval_parser.add_argument("--model", help="Student 模型路径")

    # info — 查看配置
    info_parser = subparsers.add_parser("info", help="查看可用 Teacher 模型")
    info_parser.add_argument("--teacher", "-t", choices=["kimi", "glm", "deepseek"], help="指定 Teacher")

    args = parser.parse_args()

    if args.command == "run":
        from .pipeline import DistillPipeline
        from .utils import DistillConfig
        config = DistillConfig.from_yaml(args.config)
        pipeline = DistillPipeline(config)
        pipeline.run(
            skip_generate=args.skip_generate,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
        )

    elif args.command == "generate":
        from .utils import DistillConfig
        config = DistillConfig.from_yaml(args.config)
        import os
        from .teachers import create_teacher
        from .data import DataGenerator

        api_key = os.environ.get(config.teacher_api_key_env, "")
        teacher = create_teacher(config.teacher_type, api_key=api_key, model=config.teacher_model)
        gen = DataGenerator(teacher, {
            "scene": config.scene,
            "system_prompt": config.system_prompt,
            "topic_seeds": config.topic_seeds,
            "num_samples": config.num_samples,
        })
        gen.generate_batch(args.output)

    elif args.command == "train":
        from .utils import DistillConfig
        config = DistillConfig.from_yaml(args.config)
        if args.method == "sft":
            from .train import SFTTrainer, SFTConfig
            train_config = SFTConfig(
                model_name_or_path=config.student_model,
                output_dir=config.output_dir,
            )
            trainer = SFTTrainer(train_config)
            trainer.train()

    elif args.command == "eval":
        console.print("📊 评估功能开发中...", style="yellow")

    elif args.command == "info":
        console.print("\n🔧 支持的 Teacher 模型:\n", style="bold")
        from .teachers import KimiTeacher, GLMTeacher, DeepSeekTeacher

        teachers = {
            "Kimi (Moonshot)": KimiTeacher,
            "GLM (智谱)": GLMTeacher,
            "DeepSeek": DeepSeekTeacher,
        }
        for name, cls in teachers.items():
            console.print(f"  📡 {name}")
            console.print(f"     默认模型: {cls.DEFAULT_MODEL}")
            console.print(f"     可用模型: {', '.join(cls.AVAILABLE_MODELS)}")
            console.print(f"     Base URL: {cls.BASE_URL}\n")

        console.print("  🎯 Student 模型推荐:")
        students = [
            "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-14B",
        ]
        for s in students:
            console.print(f"     • {s}")
        console.print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
