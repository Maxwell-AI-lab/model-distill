"""DPO (Direct Preference Optimization) 训练器"""

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()


@dataclass
class DPOConfig:
    """DPO 训练配置"""
    # 模型
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    # 数据
    train_file: str = "data/dpo.jsonl"
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    # 训练
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    beta: float = 0.1
    # 输出
    output_dir: str = "outputs/dpo"
    save_steps: int = 500
    logging_steps: int = 10
    max_length: int = 2048
    max_prompt_length: int = 1024
    seed: int = 42


class DPOTrainer:
    """DPO 训练器 — 基于偏好数据的对齐训练"""

    def __init__(self, config: DPOConfig):
        self.config = config

    def prepare_data(self, data_path: str) -> str:
        """验证 DPO 数据格式"""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"DPO 数据不存在: {data_path}")

        valid = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                assert "prompt" in item and "chosen" in item and "rejected" in item
                assert item["rejected"], f"第 {valid+1} 条数据缺少 rejected"
                valid += 1

        console.print(f"✅ DPO 数据验证通过: {valid} 条")
        return str(data_path)

    def train(self):
        """启动 DPO 训练"""
        console.print("🚀 开始 DPO 训练", style="bold green")

        train_file = self.prepare_data(self.config.train_file)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType
        from datasets import load_dataset
        from trl import DPOTrainer as TRDPOTrainer, DPOConfig as TRDPOConfig
        from transformers import TrainingArguments

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("json", data_files=train_file, split="train")

        peft_config = None
        if self.config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )

        training_args = TRDPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            beta=self.config.beta,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            bf16=True,
            report_to="none",
            seed=self.config.seed,
        )

        trainer = TRDPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        trainer.train()
        trainer.save_model(self.config.output_dir)
        console.print(f"✅ DPO 训练完成! → {self.config.output_dir}", style="bold green")

        return self.config.output_dir
