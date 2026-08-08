# Model Distill

> 大模型代码规划能力蒸馏流水线 — 将 GLM-5.2 的规划能力蒸馏到 Qwen3-8B

## 🎯 项目目标

通过 API 黑盒蒸馏，将 **GLM-5.2**（Teacher）的代码任务规划能力，系统化地迁移到 **Qwen3-8B**（Student）小模型中：

- **规划能力蒸馏** — 让小模型学会"先规划解题步骤，再写代码"
- **低成本部署** — 8B 模型在单节点昇腾 910B 上即可推理
- **流水线化** — 数据生成 → 训练 → 评估全自动化

## 🏗️ 架构

```
        Teacher                           Student
┌────────────────────┐           ┌────────────────────┐
│   GLM-5.2 API      │           │   Qwen3-8B         │
│   (智谱 Coding Plan)│  ──蒸馏──→│   (本地部署)        │
│                    │           │                    │
│   open.bigmodel.cn │           │   昇腾 910B × 8    │
│   /api/anthropic   │           │   bf16 + LoRA      │
└────────────────────┘           └────────────────────┘
        │                                ▲
        ▼                                │
┌────────────────────┐           ┌────────────────────┐
│   数据生成          │           │   SFT 训练          │
│   HumanEval + MBPP │  ────────→│   LoRA (r=64)      │
│   500题 × 计划+代码 │           │   3 epochs          │
│   质量过滤          │           │                    │
└────────────────────┘           └────────────────────┘
```

## 📊 完整流水线

```
Phase 1: 数据生成                 Phase 2: 训练              Phase 3: 评估
┌─────────────────────┐          ┌──────────────┐          ┌──────────────┐
│ HumanEval + MBPP    │          │              │          │              │
│      ↓              │          │  Qwen3-8B    │          │  50 道评估题  │
│ 采样 500 题         │          │  + LoRA      │          │      ↓       │
│      ↓              │          │              │          │  Student 做题 │
│ GLM-5.2 API 生成    │  train   │  910B × 8    │  model   │      ↓       │
│ (计划+代码)         │ ───────→ │  bf16        │ ───────→ │  代码执行测试 │
│      ↓              │  .jsonl  │              │          │      ↓       │
│ 质量过滤 (~400条)   │          │  3 epochs    │          │  pass@1 报告  │
│      ↓              │          │  ~2-3 小时    │          │              │
│ ChatML 格式化       │          │              │          │              │
└─────────────────────┘          └──────────────┘          └──────────────┘
```

详见架构图：[docs/pipeline.html](docs/pipeline.html)

## 🚀 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| 硬件 | 昇腾 910B（8卡，61GB HBM/卡） |
| CANN | 9.1.0-beta.3 |
| 镜像 | `deepseek-rl:910b-cann9.1-vllm0.23-v25` |
| Teacher API | GLM-5.2 (智谱 Coding Plan) |

### 一键运行

```bash
# 1. 克隆项目到集群节点
git clone https://github.com/Maxwell-AI-lab/model-distill.git
cd model-distill

# 2. 起训练容器
docker run -d \
  --name distill-train \
  --privileged --network host --ipc host \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /tmp:/tmp -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /data:/data \
  --entrypoint /bin/sh \
  deepseek-rl:910b-cann9.1-vllm0.23-v25 \
  -c 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && sleep infinity'

# 3. 容器内安装依赖
docker exec distill-train /bin/sh -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  pip install trl pyyaml rich httpx
'

# 4. 配置 API Key
export GLM_API_KEY="your-api-key"

# 5. 一键全流程
docker exec distill-train /bin/sh -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd /data/z00666713/model-distill
  python3 scripts/run_distill.py all
'

# 或者分步执行：
# python3 scripts/run_distill.py generate   # 生成数据
# python3 scripts/run_distill.py train      # NPU 训练
# python3 scripts/run_distill.py eval       # 评估
```

### 关键路径

```
集群跳板机: 119.8.234.170
项目目录:   /data/z00666713/model-distill/
GLM-5.2:   /data/model/GLM-5.2-bf16-4L-32E-dummy/
Qwen3-8B:  /data/model/Qwen3-8B/
```

## 📁 项目结构

```
model-distill/
├── README.md                         # 本文档
├── DESIGN.md                         # 整体技术设计方案
├── docs/
│   ├── PLANNING_DISTILLATION.md      # 长程规划蒸馏方案
│   └── pipeline.html                 # 流水线架构图
├── pyproject.toml
├── configs/
│   ├── code_planning.yaml            # 代码蒸馏实验配置 (当前)
│   └── example.yaml                  # 通用场景配置示例
├── distill/
│   ├── teachers/                     # Teacher 模型适配层
│   │   ├── base.py                   # 统一基类 + 工厂函数
│   │   ├── glm.py                    # GLM-5.2 (Anthropic 兼容接口)
│   │   ├── kimi.py                   # Kimi (Moonshot, OpenAI 接口)
│   │   └── deepseek.py               # DeepSeek (OpenAI 接口)
│   ├── data/
│   │   ├── datasets.py               # HumanEval / MBPP / CodeContests 加载
│   │   ├── code_generator.py         # 代码规划蒸馏数据生成 (核心)
│   │   ├── generator.py              # 通用数据生成
│   │   ├── filter.py                 # 质量过滤
│   │   └── formatter.py              # ChatML/Alpaca/ShareGPT 转换
│   ├── train/
│   │   ├── npu_sft.py                # NPU SFT 训练器 (910B 适配)
│   │   ├── npu_adapter.py            # NPU 环境检测与适配
│   │   ├── sft.py                    # GPU SFT 训练器
│   │   └── dpo.py                    # DPO 训练器
│   ├── eval/
│   │   ├── code_eval.py              # 代码执行评估 (pass@1)
│   │   ├── judge.py                  # LLM-as-Judge
│   │   └── metrics.py                # ROUGE/BLEU 文本指标
│   ├── pipeline.py                   # 流水线编排
│   ├── cli.py                        # CLI 入口
│   └── utils/
│       ├── config.py                 # 配置管理
│       └── logger.py                 # 日志
├── scripts/
│   ├── run_distill.py                # 一键全流程入口
│   ├── check_npu_env.py             # NPU 环境检测
│   ├── generate_data.py             # 独立数据生成
│   ├── train.py                     # 独立训练
│   └── evaluate.py                  # 独立评估
└── tests/
    └── test_teachers.py
```

## 🔧 技术栈

| 组件 | 选型 |
|------|------|
| **Teacher** | GLM-5.2 (智谱 BigModel, Anthropic 兼容 API, Coding Plan) |
| **Student** | Qwen3-8B |
| **数据源** | HumanEval (164题) + MBPP (974题) |
| **训练** | PyTorch 2.10 + torch_npu + PEFT (LoRA) + TRL (SFT) |
| **精度** | bf16 (910B 原生支持) |
| **硬件** | 昇腾 910B3 × 8 (61GB HBM/卡) |
| **容器** | deepseek-rl:910b-cann9.1-vllm0.23-v25 |
| **评估** | 代码执行 (pass@1) + LLM-as-Judge |

## 📈 训练参数

| 参数 | 值 |
|------|------|
| 微调方式 | LoRA |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| 精度 | bf16 |
| Epochs | 3 |
| Learning Rate | 2e-4 (cosine) |
| Batch Size | 4 × 4 (grad accum) |
| Max Seq Length | 2048 |

## 📦 产出

| 产出 | 位置 | 说明 |
|------|------|------|
| 训练数据 | `data/train_chatml.jsonl` | ~400 条 ChatML 格式 |
| LoRA 模型 | `outputs/code-planning-v1/` | ~200MB Adapter |
| 评估结果 | `data/eval_results.json` | pass@1 + 计划质量 |

## 📄 License

MIT
