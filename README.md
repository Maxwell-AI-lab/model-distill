# Model Distill

> 大模型能力蒸馏流水线 — 将 Kimi / GLM / DeepSeek 的能力蒸馏到 Qwen 小模型

## 🎯 项目目标

通过 API 黑盒蒸馏（Black-box Distillation），将大模型（Teacher）的特定场景能力，系统化地迁移到 Qwen 小模型（Student）中，实现：

- **低成本部署** — 1B~7B 小模型即可覆盖特定场景
- **能力对齐** — 尽可能逼近 Teacher 模型的效果
- **流水线化** — 从场景定义到模型部署的全流程自动化

## 🏗️ 架构

```
Teacher Models (API)          Student Models (Local)
┌──────────────┐              ┌──────────────┐
│  Kimi        │              │  Qwen 0.5B   │
│  GLM-4       │  ──蒸馏──→   │  Qwen 1.8B   │
│  DeepSeek    │              │  Qwen 4B     │
└──────────────┘              │  Qwen 7B     │
      │                       └──────────────┘
      ▼                              ▲
┌──────────────┐              ┌──────────────┐
│  数据生成     │  ──────────→  │  SFT / DPO   │
│  质量过滤     │              │  LoRA / QLoRA│
└──────────────┘              └──────────────┘
```

## 📦 核心模块

| 模块 | 说明 |
|------|------|
| `distill/data/` | 数据生成 — 调用 Teacher API 生成训练数据 |
| `distill/train/` | 训练流程 — SFT / DPO / LoRA 训练 |
| `distill/eval/` | 评估对齐 — Student vs Teacher 效果对比 |
| `distill/teachers/` | Teacher 模型适配器（Kimi / GLM / DeepSeek） |
| `distill/configs/` | 实验配置（YAML） |
| `distill/utils/` | 工具函数 |

## 🚀 快速开始

```bash
# 安装
pip install -e .

# 配置 API Key
export KIMI_API_KEY="your-kimi-key"
export GLM_API_KEY="your-glm-key"
export DEEPSEEK_API_KEY="your-deepseek-key"

# 一条命令跑通蒸馏流水线
distill run --config configs/example.yaml
```

## 📁 项目结构

```
model-distill/
├── README.md
├── setup.py / pyproject.toml
├── distill/
│   ├── __init__.py
│   ├── cli.py                 # CLI 入口
│   ├── pipeline.py            # 流水线编排
│   ├── teachers/              # Teacher 模型适配器
│   │   ├── __init__.py
│   │   ├── base.py            # 统一接口
│   │   ├── kimi.py            # Kimi (Moonshot)
│   │   ├── glm.py             # GLM (智谱)
│   │   └── deepseek.py        # DeepSeek
│   ├── data/                  # 数据生成与处理
│   │   ├── __init__.py
│   │   ├── generator.py       # 数据生成
│   │   ├── filter.py          # 质量过滤
│   │   └── formatter.py       # 格式转换
│   ├── train/                 # 训练模块
│   │   ├── __init__.py
│   │   ├── sft.py             # SFT 训练
│   │   ├── dpo.py             # DPO 训练
│   │   └── lora_utils.py      # LoRA/QLoRA 工具
│   ├── eval/                  # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py         # 评估指标
│   │   └── judge.py           # LLM-as-Judge
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # 配置管理
│       └── logger.py          # 日志
├── configs/                   # 实验配置
│   └── example.yaml
├── scripts/                   # 脚本
│   ├── generate_data.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    └── test_teachers.py
```

## 🔧 技术栈

- **Teacher API**: Kimi (Moonshot) / GLM (BigModel) / DeepSeek
- **Student**: Qwen 2.5 系列（0.5B / 1.8B / 4B / 7B）
- **训练**: PyTorch + HuggingFace Transformers + PEFT + DeepSpeed
- **配置**: YAML + dataclass
- **追踪**: WandB 集成

## 📄 License

MIT
