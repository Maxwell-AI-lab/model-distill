# 代码任务规划能力蒸馏 — 技术设计方案

## 一、项目概述

### 1.1 目标

将 **GLM-5.2**（Teacher）的代码任务规划能力，通过 API 黑盒蒸馏，迁移到 **Qwen3-8B**（Student）小模型中。

### 1.2 核心价值

- 蒸馏后的小模型学会"先规划解题步骤，再写代码"的范式
- 在代码任务上显著提升通过率和代码质量
- 可在单节点昇腾 910B 上本地部署，无需 API 调用

## 二、Teacher 与 Student

### Teacher: GLM-5.2

| 项目 | 值 |
|------|------|
| 提供方 | 智谱 BigModel |
| API | Anthropic 兼容接口 |
| 地址 | `https://open.bigmodel.cn/api/anthropic` |
| 认证 | x-api-key + anthropic-version |
| 订阅 | Coding Plan |
| 优势 | 代码规划、推理、中文理解均顶级 |

### Student: Qwen3-8B

| 项目 | 值 |
|------|------|
| 提供方 | 阿里通义 |
| 参数量 | 8.2B |
| 本地路径 | `/data/model/Qwen3-8B` |
| 优势 | 代码基础好、中文强、开源活跃 |

### 备选 Student

| 模型 | 参数量 | 场景 |
|------|--------|------|
| Qwen3-0.6B | 0.6B | 极轻量部署 |
| Qwen3-14B | 14B | 更高精度 |
| Qwen3.5-35B-A3B | 35B (MoE) | 最强效果 |

## 三、蒸馏策略

### 3.1 黑盒蒸馏

通过 Teacher API 生成"题目 → 解题计划 + 代码实现"的训练数据，SFT 训练 Student。

### 3.2 四阶段蒸馏法

| 阶段 | 方法 | 目标 | 状态 |
|------|------|------|------|
| **Phase 1** | SFT | 学会"先规划再编码" | ✅ 当前 |
| **Phase 2** | DPO | 区分好/差规划 | 待定 |
| **Phase 3** | 自我反思 | 发现自己计划的漏洞 | 待定 |
| **Phase 4** | 压缩 | 7B → 1.5B 再蒸馏 | 待定 |

### 3.3 数据生成策略

```
输入: HumanEval (164题) + MBPP (974题)
      ↓ 随机采样 500 题
      ↓ 90% 训练 / 10% 评估
      ↓
GLM-5.2 API 生成:
      每道题 → ## 解题计划 (步骤分解 + 依赖关系)
             → ## 边界分析 (特殊情况处理)
             → ## 代码实现 (完整可运行代码)
             → ## 复杂度分析 (时间/空间)
      ↓
质量过滤:
      ✅ 代码必须跑通测试用例
      ✅ 格式完整 (计划+代码)
      ✅ 非拒绝回答
      ↓
格式化: ChatML 格式
```

## 四、训练方案

### 4.1 硬件环境

| 项目 | 值 |
|------|------|
| 硬件 | 昇腾 910B3 × 8 (61GB HBM/卡) |
| 节点 | 119.8.234.170 (aura-1.novalocal) |
| 系统 | Huawei Cloud EulerOS 2.0 (aarch64) |
| CANN | 9.1.0-beta.3 |
| 容器 | deepseek-rl:910b-cann9.1-vllm0.23-v25 |

### 4.2 训练配置

| 参数 | 值 | 说明 |
|------|------|------|
| 微调方式 | LoRA | 参数高效，单节点足够 |
| rank | 64 | LoRA 秩 |
| alpha | 128 | 缩放因子 |
| dropout | 0.05 | 正则化 |
| 精度 | bf16 | 910B 原生支持 |
| Epochs | 3 | |
| Learning Rate | 2e-4 | cosine schedule |
| Batch Size | 4 | per_device |
| Grad Accum | 4 | 有效 batch=16 |
| Max Seq | 2048 | |

### 4.3 NPU 关键配置

```python
# 必须设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# torch_npu 适配
import torch_npu  # 注册 NPU 后端
torch.npu.set_device(0)

# bf16 精度 (不用 bitsandbytes 量化，NPU 不支持)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="npu:0",
)

# LoRA target modules (Qwen3)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

## 五、评估方案

### 5.1 评估集

50 道编程题（HumanEval + MBPP 各抽，训练时未见过）。

### 5.2 评估维度

| 维度 | 方法 | 说明 |
|------|------|------|
| **pass@1** | 代码执行 | 运行测试用例，通过率 (核心指标) |
| **计划质量** | LLM-as-Judge | GLM-5.2 按准确性/完整性/连贯性/简洁性打分 |
| **规划深度** | 统计 | 平均步骤数、有无边界分析 |
| **对比基线** | 对照 | 原始 Qwen3-8B (未蒸馏) 的 pass@1 |

### 5.3 预期效果

| 指标 | 原始 Qwen3-8B | 蒸馏后 | GLM-5.2 |
|------|--------------|--------|---------|
| pass@1 | ~60% | ~75%+ | ~90%+ |
| 有计划输出 | 偶尔 | 几乎总是 | 总是 |
| 平均步骤 | 2~3步 | 6~8步 | 7~9步 |
| 部署成本 | 本地 | 本地 | API |

## 六、执行计划

### Week 1: MVP

| 天 | 任务 |
|----|------|
| Day 1 | 确认方案，适配 NPU 代码 |
| Day 2-3 | 调 GLM-5.2 API 生成数据 (500题) |
| Day 4 | 数据清洗 + 格式化 |
| Day 5-6 | 单节点 8 卡 SFT 训练 |
| Day 7 | 评估 + 分析 |

### Week 2: 迭代优化

- 根据评估结果调整 (补数据 / 调参数)
- 可选: DPO 偏好对齐
- 可选: 压缩到更小模型

## 七、关键路径

```
跳板机:     119.8.234.170
项目目录:   /data/z00666713/model-distill/
代码仓库:   github.com/Maxwell-AI-lab/model-distill
Teacher:    GLM-5.2 API (Coding Plan)
Student:    /data/model/Qwen3-8B
容器镜像:   deepseek-rl:910b-cann9.1-vllm0.23-v25
```

## 八、Roadmap

### Phase 1 — MVP (当前)
- [x] Teacher 适配 (GLM-5.2 Anthropic 接口)
- [x] 数据集加载 (HumanEval / MBPP)
- [x] 代码规划数据生成
- [x] NPU SFT 训练器 (910B 适配)
- [x] 代码执行评估 (pass@1)
- [x] 流水线编排 + 一键脚本

### Phase 2 — 增强
- [ ] CoT 思维链数据生成
- [ ] DPO 偏好对齐训练
- [ ] 数据增强 (改写、扰动)
- [ ] 更大训练规模 (多节点)

### Phase 3 — 高级
- [ ] 通用任务规划蒸馏
- [ ] 模型压缩 (7B → 1.5B)
- [ ] 导出部署 (量化、vLLM)
- [ ] Web UI 可视化
