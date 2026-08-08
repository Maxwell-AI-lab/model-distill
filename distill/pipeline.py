"""蒸馏流水线编排 — 串联数据生成、训练、评估"""

import os
from pathlib import Path

from rich.console import Console

from .teachers import create_teacher
from .data import DataGenerator, QualityFilter, DataFormatter
from .eval import LLMJudge
from .utils import DistillConfig

console = Console()


class DistillPipeline:
    """完整蒸馏流水线"""

    def __init__(self, config: DistillConfig):
        self.config = config
        self.work_dir = Path(f"experiments/{config.name}")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 Teacher
        api_key = os.environ.get(config.teacher_api_key_env, "")
        if not api_key:
            raise ValueError(f"环境变量 {config.teacher_api_key_env} 未设置")
        self.teacher = create_teacher(
            config.teacher_type, api_key=api_key, model=config.teacher_model
        )
        console.print(f"👨‍🏫 Teacher: {self.teacher}")

    def run(self, skip_generate: bool = False, skip_train: bool = False, skip_eval: bool = False):
        """运行完整流水线"""
        console.print(f"\n🚀 蒸馏实验: {self.config.name}", style="bold cyan")
        console.print(f"   场景: {self.config.scene}")
        console.print(f"   Teacher: {self.teacher.model}")
        console.print(f"   Student: {self.config.student_model}\n")

        data_path = self.work_dir / "raw_data.jsonl"
        clean_path = self.work_dir / "clean_data.jsonl"
        train_path = self.work_dir / "train_chatml.jsonl"

        # Step 1: 数据生成
        if not skip_generate:
            console.print("\n[1/4] 📝 数据生成", style="bold yellow")
            gen = DataGenerator(self.teacher, {
                "scene": self.config.scene,
                "system_prompt": self.config.system_prompt,
                "topic_seeds": self.config.topic_seeds,
                "difficulty_levels": self.config.difficulty_levels,
                "num_samples": self.config.num_samples,
            })
            gen.generate_batch(str(data_path))

            # 质量过滤
            import json
            raw_data = [json.loads(l) for l in open(data_path, encoding="utf-8")]
            filt = QualityFilter()
            clean_data = filt.filter_batch(raw_data)

            with open(clean_path, "w", encoding="utf-8") as f:
                for item in clean_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # 格式转换 → ChatML
            DataFormatter.to_chatml(clean_data, str(train_path), system_prompt=self.config.system_prompt)
        else:
            console.print("\n[1/4] ⏭️ 跳过数据生成", style="dim")

        # Step 2: 训练
        if not skip_train:
            console.print("\n[2/4] 🔧 模型训练", style="bold yellow")
            output_dir = str(self.work_dir / "model")
            if self.config.train_method == "sft":
                from .train import SFTTrainer, SFTConfig
                train_config = SFTConfig(
                    model_name_or_path=self.config.student_model,
                    train_file=str(train_path),
                    use_lora=self.config.use_lora,
                    use_4bit=self.config.use_4bit,
                    learning_rate=self.config.learning_rate,
                    num_train_epochs=self.config.num_train_epochs,
                    per_device_train_batch_size=self.config.per_device_train_batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    output_dir=output_dir,
                )
                trainer = SFTTrainer(train_config)
                trainer.train()
        else:
            console.print("\n[2/4] ⏭️ 跳过训练", style="dim")

        # Step 3: 评估
        if not skip_eval:
            console.print("\n[3/4] 📊 评估", style="bold yellow")
            # 评估逻辑 — 对比 Student 和 Teacher
            console.print("   (评估需要训练好的模型，后续完善)")
        else:
            console.print("\n[3/4] ⏭️ 跳过评估", style="dim")

        # Step 4: 保存配置
        console.print("\n[4/4] 💾 保存实验配置", style="bold yellow")
        self.config.to_yaml(str(self.work_dir / "config.yaml"))
        console.print(f"   ✅ 配置已保存: {self.work_dir / 'config.yaml'}")

        console.print(f"\n🎉 蒸馏实验完成!", style="bold green")
        return str(self.work_dir)
