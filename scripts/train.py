"""训练脚本"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill.train import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="SFT 训练")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-1.5B", help="Student 模型")
    parser.add_argument("--data", "-d", required=True, help="训练数据 (ChatML JSONL)")
    parser.add_argument("--output", "-o", default="outputs/sft", help="输出目录")
    parser.add_argument("--epochs", "-e", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--no-lora", action="store_true", help="不用 LoRA")
    parser.add_argument("--no-4bit", action="store_true", help="不用 4bit 量化")
    args = parser.parse_args()

    config = SFTConfig(
        model_name_or_path=args.model,
        train_file=args.data,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        use_lora=not args.no_lora,
        use_4bit=not args.no_4bit,
    )

    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
