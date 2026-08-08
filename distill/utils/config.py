"""配置管理"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DistillConfig:
    """完整蒸馏实验配置"""
    # 实验信息
    name: str = "default"
    description: str = ""

    # Teacher 配置
    teacher_type: str = "glm"  # kimi | glm | deepseek
    teacher_model: str = ""
    teacher_api_key_env: str = ""  # 从环境变量读取

    # Student 配置
    student_model: str = "Qwen/Qwen2.5-1.5B"

    # 场景
    scene: str = ""
    system_prompt: str = ""
    topic_seeds: list[str] = field(default_factory=list)
    difficulty_levels: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])

    # 数据生成
    num_samples: int = 100
    data_output_dir: str = "data"

    # 训练
    train_method: str = "sft"  # sft | dpo
    use_lora: bool = True
    use_4bit: bool = True
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    output_dir: str = "outputs"

    # 评估
    eval_sample_ratio: float = 0.1  # 抽样比例
    use_llm_judge: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "DistillConfig":
        """从 YAML 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 支持嵌套配置
        flattened = {}
        for key, val in data.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if sub_key in cls.__dataclass_fields__:
                        flattened[sub_key] = sub_val
            else:
                if key in cls.__dataclass_fields__:
                    flattened[key] = val

        return cls(**flattened)

    def to_yaml(self, path: str):
        """保存为 YAML"""
        data = {k: v for k, v in self.__dict__.items()}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
