# Model Distill — 技术设计方案

## 一、背景与问题

### 1.1 现状

大模型（GPT-4、Kimi、GLM-4、DeepSeek-V3 等）能力越来越强，但：

| 痛点 | 说明 |
|------|------|
| **部署成本高** | 需要大量 GPU 资源，推理延迟高 |
| **API 依赖** | 调用 API 持续付费，数据隐私不可控 |
| **场景过载** | 大模型是通才，但垂直场景只需要 10% 的能力 |
| **落地困难** | 企业需要的是能跑在有限算力上的模型 |

### 1.2 目标

**把大模型（Teacher）的特定场景能力，系统化地蒸馏到 Qwen 小模型（Student）中。**

- 蒸馏后的小模型可以在单卡甚至 CPU 上运行
- 在目标场景上逼近 Teacher 效果
- 整个过程流水线化、可复现、可扩展

---

## 二、蒸馏策略

### 2.1 为什么选 API 黑盒蒸馏？

我们有 Kimi、GLM、DeepSeek 的 API，但拿不到模型权重和 logits。所以只能做**黑盒蒸馏**：

```
黑盒蒸馏 = 用大模型生成数据 → 用数据训练小模型
```

这其实是一种特殊的"知识蒸馏"——不依赖 Teacher 的内部状态，只依赖 Teacher 的输出。

### 2.2 蒸馏的三个层次

```
┌──────────────────────────────────────────────────────────┐
│                    蒸馏的三个层次                          │
│                                                          │
│  Level 1: 答案蒸馏 (Response Distillation)               │
│  ── Teacher 回答问题 → Student 模仿回答                   │
│  ── 最简单，效果最直接                                    │
│                                                          │
│  Level 2: 思维蒸馏 (Reasoning Distillation)               │
│  ── Teacher 展示推理过程 → Student 学习思考方式           │
│  ── 让 Teacher 输出 Chain-of-Thought                     │
│                                                          │
│  Level 3: 反馈蒸馏 (Feedback Distillation / DPO)         │
│  ── Teacher 当裁判 → Student 偏好对齐                    │
│  ── 生成 chosen/rejected 对做 DPO 训练                   │
└──────────────────────────────────────────────────────────┘
```

**项目支持全部三层，由浅入深。**

### 2.3 多 Teacher 融合策略

一个 Teacher 的能力有限，三个 Teacher 可以：

```
策略 A: 单独蒸馏 → 各自训练 → 模型合并 (Model Merge)
策略 B: 混合数据 → 三个 Teacher 各生成 N 条 → 合并训练
策略 C: 投票机制 → 同一问题三个 Teacher 回答 → 取最优做训练数据
策略 D: 分工协作 → 不同场景用最擅长的 Teacher
        Kimi → 长文本场景
        GLM → 对话/通用
        DeepSeek → 推理/代码
```

**MVP 先做策略 B（混合数据），后续支持策略 C/D。**

---

## 三、系统架构

### 3.1 整体架构图

```
┌────────────────────────────────────────────────────────────────┐
│                      Model Distill Pipeline                     │
│                                                                │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────┐ │
│  │ 场景定义  │→  │  数据生成     │→  │  训练    │→  │  评估   │ │
│  │ config    │   │  Generation  │   │  Train   │   │  Eval   │ │
│  └──────────┘   └──────────────┘   └──────────┘   └─────────┘ │
│                       │                │               │       │
│                       ▼                ▼               ▼       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Teachers (API 适配层)                        │  │
│  │  ┌────────┐    ┌────────┐    ┌──────────┐               │  │
│  │  │  Kimi  │    │  GLM   │    │ DeepSeek │               │  │
│  │  │ Moonshot│   │BigModel│    │          │               │  │
│  │  └────────┘    └────────┘    └──────────┘               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Student (Qwen 小模型)                        │  │
│  │  0.5B │ 1.8B │ 3B │ 4B │ 7B │ 14B                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              实验管理 & 追踪                               │  │
│  │  Config YAML │ WandB │ 实验对比 │ 模型版本               │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
                    ┌─────────────┐
                    │ 场景 + 种子  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ Prompt 构建  │
                    └──────┬──────┘
                           ▼
              ┌─────────────────────────┐
              │   Teacher API 调用       │
              │   (Kimi/GLM/DeepSeek)   │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   原始数据 (JSONL)       │
              │   question + answer     │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   质量过滤               │
              │   去重 / 去噪 / 检查    │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   格式转换               │
              │   ChatML / Alpaca / DPO │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   训练                   │
              │   SFT → DPO (可选)      │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   评估                   │
              │   ROUGE + LLM-Judge     │
              └────────────┬────────────┘
                           ▼
                    ┌─────────────┐
                    │ 蒸馏后模型   │
                    └─────────────┘
```

---

## 四、核心模块详细设计

### 4.1 Teacher 适配层 (`distill/teachers/`)

**统一接口，屏蔽不同 API 差异：**

```python
class BaseTeacher(ABC):
    def chat(self, messages: list[dict], **kwargs) -> TeacherResponse
    def chat_simple(self, prompt: str, system: str = "") -> str
```

| Teacher | Base URL | 默认模型 | 特点 |
|---------|----------|----------|------|
| Kimi | api.moonshot.cn/v1 | moonshot-v1-128k | 长文本强，128K上下文 |
| GLM | open.bigmodel.cn/api/paas/v4 | glm-4-plus | 对话强，中文能力突出 |
| DeepSeek | api.deepseek.com/v1 | deepseek-chat | 推理强，性价比高 |

**设计要点：**
- 三个 API 都兼容 OpenAI 格式，所以底层用 `openai` SDK 统一调用
- `create_teacher()` 工厂函数，一行切换 Teacher
- 后续可轻松扩展其他 Teacher（Claude、GPT 等）

### 4.2 数据生成模块 (`distill/data/`)

**这是蒸馏的核心——数据质量决定最终效果。**

#### 4.2.1 数据生成策略

```
┌─────────────────────────────────────────────┐
│            数据生成策略                       │
│                                             │
│  1. 种子驱动生成                             │
│     人工提供主题种子 → Teacher 生成具体问答   │
│                                             │
│  2. 多样性采样                               │
│     不同难度、不同话题、不同风格              │
│                                             │
│  3. 思维链增强                               │
│     让 Teacher 输出推理过程（CoT）           │
│     Student 不仅学答案，还学思考方式         │
│                                             │
│  4. 自定义 Prompt                           │
│     高级用户直接提供 prompt 列表             │
└─────────────────────────────────────────────┘
```

#### 4.2.2 质量过滤

| 过滤规则 | 说明 |
|---------|------|
| 空值检查 | 问题或答案为空 |
| 长度检查 | 太短或太长 |
| 拒绝检测 | "我无法回答"、"作为AI" 等 |
| 去重 | 完全重复的问题 |
| 格式检查 | JSON 解析失败 |

#### 4.2.3 输出格式

支持三种格式自动转换：
- **ChatML** — Qwen 原生格式（推荐）
- **ShareGPT** — 通用 SFT 格式
- **Alpaca** — 经典指令微调格式
- **DPO** — 偏好对格式（后续阶段）

### 4.3 训练模块 (`distill/train/`)

#### 4.3.1 SFT 训练

```
训练配置:
├── 基础模型: Qwen2.5-1.5B (可调整)
├── 微调方式: LoRA / QLoRA (参数高效)
│   ├── rank: 16
│   ├── alpha: 32
│   └── target_modules: q,k,v,o,gate,up,down
├── 量化: 4-bit NF4 (降低显存)
├── 学习率: 2e-4 (cosine schedule)
└── Epochs: 3
```

**为什么用 LoRA？**
- 全参数微调需要 8x A100，LoRA 单卡就能跑
- 效果接近全参微调（在蒸馏场景下够用）
- 可以方便地合并多个 LoRA adapter

#### 4.3.2 DPO 训练（Level 2）

```
SFT 模型 → DPO 对齐 → 更贴近 Teacher 的偏好
```

需要构造 `chosen`（Teacher 答案）vs `rejected`（Student 旧答案）的偏好对。

### 4.4 评估模块 (`distill/eval/`)

**双重评估体系：**

| 评估方式 | 指标 | 特点 |
|---------|------|------|
| **文本匹配** | ROUGE-1/2/L, BLEU, Exact Match | 客观、快、免费 |
| **LLM-as-Judge** | 准确性/完整性/连贯性/简洁性 (1-10分) | 主观、更接近人类判断 |

LLM-as-Judge 流程：
```
同一问题 → Teacher 答案 (参考) + Student 答案 (预测)
         → Judge 模型 (用另一个 Teacher) 打分
```

### 4.5 流水线编排 (`distill/pipeline.py`)

**一个 YAML 配置驱动完整实验：**

```yaml
name: "customer-service-v1"
teacher_type: "glm"
teacher_model: "glm-4-plus"
student_model: "Qwen/Qwen2.5-1.5B"
scene: "电商客服对话"
num_samples: 200
train_method: "sft"
use_lora: true
```

```bash
distill run --config configs/experiment.yaml
```

---

## 五、项目结构

```
model-distill/
├── README.md                      # 项目文档
├── DESIGN.md                      # 本文档 — 技术设计
├── pyproject.toml                 # 项目配置
├── distill/                       # 核心代码
│   ├── __init__.py
│   ├── cli.py                     # CLI 入口 (distill run/generate/train/eval/info)
│   ├── pipeline.py                # 流水线编排
│   ├── teachers/                  # Teacher 适配层
│   │   ├── base.py                # 统一基类
│   │   ├── kimi.py                # Kimi (Moonshot)
│   │   ├── glm.py                 # GLM (智谱)
│   │   └── deepseek.py            # DeepSeek
│   ├── data/                      # 数据生成
│   │   ├── generator.py           # 批量数据生成
│   │   ├── filter.py              # 质量过滤
│   │   └── formatter.py           # 多格式转换
│   ├── train/                     # 训练
│   │   ├── sft.py                 # SFT 训练器
│   │   └── dpo.py                 # DPO 训练器
│   ├── eval/                      # 评估
│   │   ├── metrics.py             # 文本匹配指标
│   │   └── judge.py               # LLM-as-Judge
│   └── utils/                     # 工具
│       ├── config.py              # 配置管理
│       └── logger.py              # 日志
├── configs/                       # 实验配置
│   └── example.yaml
├── scripts/                       # 独立脚本
│   ├── generate_data.py
│   ├── train.py
│   └── evaluate.py
└── tests/                         # 测试
    └── test_teachers.py
```

---

## 六、使用流程

### 6.1 典型工作流

```bash
# 1. 配置 API Key
export GLM_API_KEY="your-key"
export KIMI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"

# 2. 编写实验配置
cp configs/example.yaml configs/my_scene.yaml
vim configs/my_scene.yaml

# 3. 一键蒸馏
distill run --config configs/my_scene.yaml

# 或者分步执行：
distill generate --config configs/my_scene.yaml -o data/raw.jsonl
distill train --config configs/my_scene.yaml -m sft
distill eval --config configs/my_scene.yaml
```

### 6.2 自定义场景

只需改 YAML 配置：

```yaml
name: "medical-qa-v1"
scene: "医疗健康问答"
system_prompt: "你是一个专业的医疗助手..."
topic_seeds: ["感冒发烧", "高血压", "糖尿病", "饮食营养", "运动健康"]
num_samples: 500
student_model: "Qwen/Qwen2.5-7B"
```

---

## 七、Roadmap

### Phase 1 — MVP（当前 ✅）
- [x] Teacher 适配层（Kimi / GLM / DeepSeek）
- [x] 基础数据生成 + 过滤
- [x] SFT 训练（LoRA/QLoRA）
- [x] 基础评估（ROUGE + LLM-Judge）
- [x] CLI + 流水线编排

### Phase 2 — 增强
- [ ] CoT 思维链数据生成
- [ ] DPO 偏好对齐训练
- [ ] 多 Teacher 数据融合
- [ ] 数据增强（改写、扰动）
- [ ] WandB 实验追踪集成

### Phase 3 — 高级
- [ ] 自动化场景发现（用 Teacher 分析目标场景需要哪些能力）
- [ ] 多阶段蒸馏（SFT → DPO → RLHF）
- [ ] 模型合并（mergekit）
- [ ] 导出部署（量化、ONNX、vLLM）
- [ ] Web UI 可视化

### Phase 4 — 平台化
- [ ] 分布式训练支持
- [ ] 模型注册表 + 版本管理
- [ ] A/B 测试框架
- [ ] 在线学习闭环

---

## 八、技术选型说明

| 选择 | 理由 |
|------|------|
| OpenAI SDK 调 Teacher | 三个 API 都兼容 OpenAI 格式，一套代码通吃 |
| Qwen 2.5 做 Student | 开源、中文好、尺寸丰富（0.5B~72B）、社区活跃 |
| LoRA/QLoRA | 参数高效，单卡可训，效果够用 |
| YAML 配置 | 实验可复现，方便版本管理 |
| Rich 终端输出 | 美观、进度条、彩色日志 |
| TRL 库做训练 | HuggingFace 生态，SFT/DPO 都支持 |
