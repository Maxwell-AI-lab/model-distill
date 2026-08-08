"""SFT (Supervised Fine-Tuning) 训练器"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class SFTConfig:
    """SFT 训练配置"""
    # 模型
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    # 数据
    train_file: str = "data/train_chatml.jsonl"
    eval_file: str = ""
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # 训练
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    # 量化
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    # 输出
    output_dir: str = "outputs/sft"
    save_steps: int = 500
    logging_steps: int = 10
    # 额外
    max_seq_length: int = 2048
    seed: int = 42


class SFTTrainer:
    """SFT 训练器 — 基于 HuggingFace TRL"""

    def __init__(self, config: SFTConfig):
        self.config = config

    def prepare_data(self, data_path: str) -> str:
        """准备数据 — 确保 ChatML 格式"""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"训练数据不存在: {data_path}")

        # 验证数据格式
        valid = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                assert "messages" in item, "数据必须是 ChatML 格式"
                valid += 1

        console.print(f"✅ 数据验证通过: {valid} 条训练样本")
        return str(data_path)

    def setup_model(self):
        """加载模型和 tokenizer"""
        console.print(f"📦 加载模型: {self.config.model_name_or_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}

        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            import torch
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype, torch.bfloat16),
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs,
        )

        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, tokenizer

    def train(self):
        """启动训练"""
        console.print("🚀 开始 SFT 训练", style="bold green")

        train_file = self.prepare_data(self.config.train_file)
        model, tokenizer = self.setup_model()

        from datasets import load_dataset
        from trl import SFTTrainer as TRSFTTrainer
        from transformers import TrainingArguments

        dataset = load_dataset("json", data_files=train_file, split="train")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            bf16=True,
            report_to="none",
            seed=self.config.seed,
        )

        trainer = TRSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        trainer.train()

        # 保存
        trainer.save_model(self.config.output_dir)
        console.print(f"✅ 训练完成! 模型保存到: {self.config.output_dir}", style="bold green")

        return self.config.output_dir
