"""NPU 版 SFT 训练器 — 适配昇腾 910B

关键适配:
1. 使用 torch_npu 替代 CUDA
2. bf16 精度 (910B 原生支持)
3. 不依赖 bitsandbytes (NPU 不支持)
4. 通过 torch.distributed + HCCL 实现多卡
5. LoRA 用 PEFT 原生实现
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class NPUSFTConfig:
    """NPU SFT 训练配置"""
    # 模型
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # 数据
    train_file: str = "data/train_chatml.jsonl"
    eval_file: str = ""
    max_seq_length: int = 2048

    # LoRA (NPU 支持)
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # 训练
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01

    # NPU 特定
    # 注意: 不用 4bit 量化 (NPU 不支持 bitsandbytes)
    # 910B 64GB 显存充足，直接 bf16 全精度
    dtype: str = "bfloat16"

    # 分布式
    use_distributed: bool = True  # 多卡数据并行
    nnodes: int = 1
    nproc_per_node: int = 8

    # 输出
    output_dir: str = "outputs/sft_npu"
    save_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    seed: int = 42


class NPUSFTTrainer:
    """NPU SFT 训练器 — 基于昇腾 910B"""

    def __init__(self, config: NPUSFTConfig):
        self.config = config
        self._setup_npu()

    def _setup_npu(self):
        """初始化 NPU 环境"""
        from .npu_adapter import setup_npu_env, get_npu_config, print_npu_status

        setup_npu_env()
        print_npu_status()

        self.npu_config = get_npu_config()
        self.device = self.npu_config["device"]
        self.device_count = self.npu_config["device_count"]

        console.print(f"🔧 NPU 环境就绪: {self.device_count} 张卡")

    def prepare_data(self) -> str:
        """验证训练数据"""
        data_path = Path(self.config.train_file)
        if not data_path.exists():
            raise FileNotFoundError(f"训练数据不存在: {data_path}")

        valid = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                assert "messages" in item, "数据必须是 ChatML 格式"
                valid += 1

        console.print(f"✅ 数据验证: {valid} 条训练样本")
        return str(data_path)

    def setup_model_and_tokenizer(self):
        """加载模型和 Tokenizer (NPU 适配)"""
        import torch
        import torch_npu  # noqa: F401 - 必须导入以注册 NPU 后端
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"📦 加载模型: {self.config.model_name_or_path}")
        console.print(f"   精度: {self.config.dtype}")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model — NPU 适配
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="npu:0",  # 先加载到 NPU 0
        )

        # LoRA
        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            # 全参微调 — 910B 64GB × 8卡 完全够
            console.print("   ⚠️ 全参微调模式")

        return model, tokenizer

    def train_single_node(self):
        """单节点多卡训练 (最常用)"""
        console.print("\n🚀 开始 NPU SFT 训练", style="bold green")
        console.print(f"   模型: {self.config.model_name_or_path}")
        console.print(f"   数据: {self.config.train_file}")
        console.print(f"   LoRA: {'r=' + str(self.config.lora_r) if self.config.use_lora else '全参微调'}")
        console.print(f"   精度: {self.config.dtype}")
        console.print(f"   卡数: {self.device_count}")
        console.print(f"   Epoch: {self.config.num_train_epochs}\n")

        train_file = self.prepare_data()
        model, tokenizer = self.setup_model_and_tokenizer()

        # 训练数据
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=train_file, split="train")

        # 训练参数
        from transformers import TrainingArguments
        import torch

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            # NPU 关键配置
            bf16=True,                    # 910B 原生支持 bf16
            no_cuda=True,                 # 禁用 CUDA
            use_cpu=False,
            dataloader_pin_memory=True,
            report_to="none",
            seed=self.config.seed,
            # 多卡
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
        )

        # TRL SFT Trainer
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        # 启动训练
        console.print("\n🔥 训练中...\n", style="bold yellow")
        trainer.train()

        # 保存
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        console.print(f"\n✅ 训练完成!", style="bold green")
        console.print(f"   模型保存到: {self.config.output_dir}")

        return self.config.output_dir

    def train_multi_node(self):
        """多节点分布式训练

        需要配合 torchrun 或 HCCL 启动脚本使用:
        torchrun --nproc_per_node=8 --nnodes=12 \\
            --master_addr=NODE0_IP --master_port=29500 \\
            scripts/train_npu.py --multi-node
        """
        console.print("\n🚀 开始多节点 NPU 训练", style="bold green")
        console.print(f"   节点数: {self.config.nnodes}")
        console.print(f"   每节点卡数: {self.config.nproc_per_node}")
        console.print(f"   总卡数: {self.config.nnodes * self.config.nproc_per_node}")

        # 初始化进程组
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa

        if not dist.is_initialized():
            dist.init_process_group(backend="hccl")

        self.train_single_node()

    def train(self):
        """启动训练 (自动选择单机/多机)"""
        if self.config.nnodes > 1:
            self.train_multi_node()
        else:
            self.train_single_node()
