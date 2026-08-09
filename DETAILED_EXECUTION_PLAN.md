# 模型蒸馏项目详细执行方案

**制定时间**: 2026-08-09  
**项目周期**: 21天 (三阶段渐进式执行)  
**目标**: 将GLM-5.2代码规划能力蒸馏到Qwen3-8B模型

---

## 🎯 总体战略

### 核心目标
- **主要目标**: 让Qwen3-8B学会"先规划再编码"的思维模式
- **性能目标**: HumanEval pass@1从~60%提升到75%+
- **部署目标**: 单节点昇腾910B本地推理
- **成本目标**: 通过蒸馏减少API调用依赖

### 三阶段战略
```
阶段1: 基础验证 (3天)  → 验证技术路径可行性
阶段2: 核心开发 (7天)  → 完成主要蒸馏任务  
阶段3: 优化部署 (11天) → 生产级部署和优化
```

---

## 📋 阶段1: 基础验证 (Day 1-3)

### 目标设定
- ✅ 验证GLM-5.2 API可用性
- ✅ 验证数据生成流程完整性
- ✅ 验证训练环境可用性
- ✅ 完成5样本端到端测试

### Day 1: 环境验证

#### 上午任务 (4小时)
```bash
# 任务1.1: Python环境验证 (30分钟)
python3 --version  # 确认 >= 3.9
pip3 list | grep -E "(torch|transformers|trl)"

# 任务1.2: 依赖安装 (1小时)
pip3 install torch transformers trl peft accelerate \
    pyyaml rich httpx tqdm numpy pandas --quiet

# 任务1.3: GLM-5.2 API测试 (1小时) 
cd /root/code/model-distill
python3 -c "
import os
from distill.teachers import create_teacher

# 测试API连接
teacher = create_teacher('glm', api_key=os.environ.get('GLM_API_KEY'))
print(f'✅ Teacher创建成功: {teacher}')

# 测试单个调用
response = teacher.generate('写一个Python函数计算斐波那契数列')
print(f'✅ API响应测试成功')
print(f'响应长度: {len(response)} 字符')
"

# 任务1.4: 基础数据集加载测试 (1.5小时)
python3 -c "
from distill.data.datasets import DatasetLoader

# 测试HumanEval加载
tasks = DatasetLoader.load_mixed(
    sources=['humaneval'],
    cache_dir='./data/raw',
    total_limit=10
)

print(f'✅ 数据集加载成功: {len(tasks)} 个任务')
print(f'示例任务: {tasks[0].get(\"prompt\", \"\")[:100]}...')
"
```

#### 下午任务 (4小时)
```bash
# 任务1.5: 单样本数据生成测试 (2小时)
python3 -c "
import os
import json
from distill.teachers import create_teacher  
from distill.data.code_generator import CodeDistillGenerator
from distill.data.datasets import DatasetLoader

teacher = create_teacher('glm', api_key=os.environ.get('GLM_API_KEY'))
generator = CodeDistillGenerator({'glm': teacher})

# 加载1个测试任务
tasks = DatasetLoader.load_mixed(
    sources=['humaneval'],
    cache_dir='./data/raw',
    total_limit=1
)

# 生成数据
results = generator.generate_batch(
    tasks,
    output_path='./data/test_single.jsonl'
)

print(f'✅ 单样本生成成功')
print(f'生成内容长度: {len(results[0].get(\"content\", \"\"))} 字符')

# 检查格式
if '解题计划' in results[0].get('content', ''):
    print('✅ 包含解题计划')
if '代码实现' in results[0].get('content', ''):
    print('✅ 包含代码实现')
"

# 任务1.6: 数据格式验证 (1小时)
python3 -c "
import json

# 读取生成的数据
with open('./data/test_single.jsonl') as f:
    data = json.loads(f.readline())

print('📊 数据格式检查:')
print(f'Keys: {list(data.keys())}')

# 验证必需字段
required_keys = ['task_id', 'prompt', 'content', 'metadata']
for key in required_keys:
    if key in data:
        print(f'✅ {key}: 存在')
    else:
        print(f'❌ {key}: 缺失')
"

# 任务1.7: 训练环境测试 (1小时)
python3 -c "
import torch
import transformers
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')

# 检查CUDA/NPU可用性
if torch.cuda.is_available():
    print(f'✅ CUDA可用: {torch.cuda.device_count()} 个GPU')
    print(f'GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ CUDA不可用，检查NPU...')
    
try:
    import torch_npu
    print(f'✅ NPU可用: {torch_npu.npu.device_count()} 个NPU')
except ImportError:
    print('⚠️ torch_npu未安装')
"
```

### Day 2: 端到端验证

#### 全天任务 (8小时)
```bash
# 任务2.1: 5样本完整流程测试 (6小时)
python3 -c "
import os
import json
import sys
sys.path.insert(0, '/root/code/model-distill')

from distill.teachers import create_teacher
from distill.data.code_generator import CodeDistillGenerator  
from distill.data.datasets import DatasetLoader
from distill.data.formatter import extract_training_data

print('🚀 开始5样本端到端测试...')

# Step 1: 生成数据
teacher = create_teacher('glm', api_key=os.environ.get('GLM_API_KEY'))
generator = CodeDistillGenerator({'glm': teacher})

tasks = DatasetLoader.load_mixed(
    sources=['humaneval'],
    cache_dir='./data/raw', 
    total_limit=5
)

print(f'📊 加载任务: {len(tasks)} 个')

results = generator.generate_batch(
    tasks,
    output_path='./data/test_5.jsonl'
)

print(f'✅ 生成完成: {len(results)} 条')

# Step 2: 格式化训练数据
formatted_data = extract_training_data(
    './data/test_5.jsonl',
    './data/train_5.jsonl'
)

print(f'✅ 格式化完成: {len(formatted_data)} 条训练样本')

# Step 3: 数据质量检查
with open('./data/train_5.jsonl') as f:
    train_data = [json.loads(line) for line in f]

plan_count = sum(1 for item in train_data if '解题计划' in str(item))
code_count = sum(1 for item in train_data if '代码实现' in str(item))

print(f'📊 质量统计:')
print(f'   包含解题计划: {plan_count}/{len(train_data)}')
print(f'   包含代码实现: {code_count}/{len(train_data)}')

print('✅ 端到端测试完成!')
"

# 任务2.2: 快速训练测试 (2小时)
# 先使用更小的模型进行训练测试
python3 scripts/quick_train.py
```

### Day 3: 结果分析与规划

#### 上午任务 (4小时)
```bash
# 任务3.1: 训练结果深度分析 (2小时)
python3 -c "
import json
import os

# 分析训练日志
if os.path.exists('./outputs/quick_test/trainer_log.json'):
    with open('./outputs/quick_test/trainer_log.json') as f:
        logs = json.load(f)
    
    print('📊 训练分析:')
    print(f'   总步数: {len(logs[\"log_history\"])}')
    print(f'   最终损失: {logs[\"log_history\"][-1].get(\"loss\", \"N/A\")}')
    
    # 损失趋势
    losses = [step.get(\"loss\", 0) for step in logs[\"log_history\"] if \"loss\" in step]
    if losses:
        print(f'   损失变化: {losses[0]:.3f} → {losses[-1]:.3f}')

# 分析模型输出
if os.path.exists('./outputs/quick_test/adapter_config.json'):
    print('✅ 模型配置文件存在')
    
with open('./outputs/quick_test/adapter_config.json') as f:
    config = json.load(f)
    print(f'   LoRA rank: {config.get(\"r\", \"N/A\")}')
    print(f'   Target modules: {config.get(\"target_modules\", [])}')
"

# 任务3.2: 模型推理测试 (2小时)
python3 -c "
import sys
sys.path.insert(0, '/root/code/model-distill')

from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# 加载训练好的模型
try:
    model_path = './outputs/quick_test'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print('✅ 模型加载成功')
    
    # 测试推理
    test_prompt = '写一个Python函数计算阶乘'
    inputs = tokenizer(test_prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f'📝 推理测试:')
    print(f'输入: {test_prompt}')
    print(f'输出: {response[:200]}...')
    print(f'✅ 推理测试完成')
    
except Exception as e:
    print(f'❌ 推理测试失败: {e}')
"
```

#### 下午任务 (4小时)
```bash
# 任务3.3: 制定阶段2详细计划 (2小时)
cat > /root/code/model-distill/PHASE2_PLAN.md << 'EOF'
# 阶段2执行计划

## 目标
- 生成50个高质量训练样本
- 完成Qwen3-1.5B模型训练
- 建立评估体系

## 关键任务
1. 数据生成优化
2. 训练参数调优  
3. 评估体系建立
4. 效果对比分析

## 时间安排
- Day 4-5: 数据生成与处理
- Day 6-8: 模型训练
- Day 9-10: 评估与优化
EOF

# 任务3.4: 环境与依赖检查清单 (2小时)
cat > /root/code/model-distill/ENVIRONMENT_CHECKLIST.md << 'EOF'
# 环境检查清单

## 软件环境
- [ ] Python 3.9+
- [ ] PyTorch 2.0+
- [ ] Transformers 4.30+
- [ ] TRL, PEFT, Accelerate
- [ ] GLM-5.2 API Key

## 硬件环境  
- [ ] NPU可用性 (或GPU备用)
- [ ] 存储空间 (至少100GB)
- [ ] 网络连接 (API调用)

## 数据准备
- [ ] HumanEval数据集
- [ ] MBPP数据集
- [ ] 数据存储路径

## 模型文件
- [ ] Qwen3-8B基础模型
- [ ] 模型加载测试通过
EOF
```

### 阶段1成功标准
- ✅ 所有测试脚本运行无错误
- ✅ 至少完成1次端到端训练流程
- ✅ 模型可以成功加载和推理
- ✅ 建立完整的监控和日志体系

---

## 🚀 阶段2: 核心开发 (Day 4-10)

### 目标设定
- ✅ 生成50个高质量训练样本
- ✅ 完成中等规模模型训练
- ✅ 建立评估和对比体系
- ✅ 验证蒸馏效果

### Day 4-5: 数据生成与处理

#### Day 4 上午: 扩展数据生成 (4小时)
```bash
# 任务4.1: 50样本批量生成 (4小时)
python3 -c "
import os
import json
import sys
import time
sys.path.insert(0, '/root/code/model-distill')

from distill.teachers import create_teacher
from distill.data.code_generator import CodeDistillGenerator
from distill.data.datasets import DatasetLoader

print('🚀 开始50样本批量生成...')

# 创建Teacher
teacher = create_teacher('glm', api_key=os.environ.get('GLM_API_KEY'))
generator = CodeDistillGenerator({'glm': teacher})

# 加载数据集 - 50个任务
tasks = DatasetLoader.load_mixed(
    sources=['humaneval', 'mbpp'],
    cache_dir='./data/raw',
    total_limit=50
)

print(f'📊 任务分布: Humaneval {sum(1 for t in tasks if \"humaneval\" in t.get(\"source\", \"\"))} 个')
print(f'              MBPP {sum(1 for t in tasks if \"mbpp\" in t.get(\"source\", \"\"))} 个')

# 批量生成（带进度监控）
start_time = time.time()
results = generator.generate_batch(
    tasks,
    output_path='./data/distill_50.jsonl'
)
end_time = time.time()

print(f'✅ 生成完成: {len(results)} 条')
print(f'⏱️ 用时: {end_time - start_time:.1f} 秒')
print(f'📊 平均每条: {(end_time - start_time) / len(results):.1f} 秒')

# 保存统计信息
stats = {
    'total_tasks': len(tasks),
    'successful_generation': len(results),
    'success_rate': len(results) / len(tasks) * 100,
    'generation_time': end_time - start_time,
    'avg_time_per_task': (end_time - start_time) / len(results),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

with open('./data/generation_stats_50.json', 'w') as f:
    json.dump(stats, f, indent=2)

print('📊 统计信息已保存')
"
```

#### Day 4 下午: 数据质量分析 (4小时)
```bash
# 任务4.2: 数据质量深度分析 (2小时)
python3 -c "
import json
import re
from collections import Counter

print('🔍 开始数据质量分析...')

# 读取生成的数据
with open('./data/distill_50.jsonl') as f:
    data = [json.loads(line) for line in f]

print(f'📊 总样本数: {len(data)}')

# 分析内容特征
quality_metrics = {
    'has_plan': 0,
    'has_boundary_analysis': 0,
    'has_code': 0,
    'has_complexity': 0,
    'avg_content_length': 0,
    'code_blocks': 0,
    'structured_format': 0
}

total_length = 0
for item in data:
    content = item.get('content', '')
    total_length += len(content)
    
    if '解题计划' in content:
        quality_metrics['has_plan'] += 1
    if '边界分析' in content or '边界情况' in content:
        quality_metrics['has_boundary_analysis'] += 1
    if '代码实现' in content:
        quality_metrics['has_code'] += 1
    if '复杂度' in content:
        quality_metrics['has_complexity'] += 1
    if '```python' in content:
        quality_metrics['code_blocks'] += 1
    if all(keyword in content for keyword in ['解题计划', '代码实现']):
        quality_metrics['structured_format'] += 1

quality_metrics['avg_content_length'] = total_length / len(data)

print('📈 质量指标:')
for key, value in quality_metrics.items():
    percentage = value / len(data) * 100 if key != 'avg_content_length' else 0
    if key == 'avg_content_length':
        print(f'   {key}: {value:.0f} 字符')
    else:
        print(f'   {key}: {value}/{len(data)} ({percentage:.1f}%)')

# 保存质量报告
with open('./data/quality_report_50.json', 'w') as f:
    json.dump({
        'total_samples': len(data),
        'quality_metrics': quality_metrics,
        'analysis_time': __import__('time').strftime('%Y-%m-%d %H:%M:%S')
    }, f, indent=2)

print('✅ 质量分析完成')
"

# 任务4.3: 数据清洗与过滤 (2小时)
python3 -c "
import json
import re

print('🧹 开始数据清洗...')

with open('./data/distill_50.jsonl') as f:
    raw_data = [json.loads(line) for line in f]

cleaned_data = []

for item in raw_data:
    content = item.get('content', '')
    
    # 基本质量过滤
    if len(content) < 100:  # 太短
        continue
    if '抱歉' in content or '无法' in content:  # 拒绝回答
        continue
    if '```python' not in content:  # 没有代码块
        continue
    
    # 提取代码并验证格式
    code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
    if code_match:
        code = code_match.group(1)
        if len(code.strip()) < 10:  # 代码太短
            continue
    else:
        continue
    
    cleaned_data.append(item)

print(f'📊 清洗前: {len(raw_data)} 条')
print(f'📊 清洗后: {len(cleaned_data)} 条')
print(f'📊 过滤率: {(1 - len(cleaned_data)/len(raw_data)) * 100:.1f}%')

# 保存清洗后的数据
with open('./data/distill_50_cleaned.jsonl', 'w') as f:
    for item in cleaned_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print('✅ 数据清洗完成')
"
```

#### Day 5: 训练数据格式化与准备
```bash
# 任务5.1: ChatML格式转换 (2小时)
python3 -c "
import sys
sys.path.insert(0, '/root/code/model-distill')

from distill.data.formatter import extract_training_data

print('🔄 开始ChatML格式转换...')

# 转换清洗后的数据
formatted_data = extract_training_data(
    './data/distill_50_cleaned.jsonl',
    './data/train_50_chatml.jsonl',
    format='chatml'
)

print(f'✅ 格式转换完成: {len(formatted_data)} 条训练样本')

# 验证格式
import json
with open('./data/train_50_chatml.jsonl') as f:
    sample = json.loads(f.readline())

print('📝 样本格式检查:')
print(f'   Keys: {list(sample.keys())}')
if 'messages' in sample:
    print(f'   消息数量: {len(sample[\"messages\"])}')
    print(f'   消息结构: {[msg[\"role\"] for msg in sample[\"messages\"]]}')
"

# 任务5.2: 训练集/验证集划分 (2小时)
python3 -c "
import json
import random

# 设置随机种子
random.seed(42)

# 读取数据
with open('./data/train_50_chatml.jsonl') as f:
    data = [json.loads(line) for line in f]

# 随机划分
random.shuffle(data)
split_point = int(len(data) * 0.9)  # 90%训练，10%验证

train_data = data[:split_point]
val_data = data[split_point:]

print(f'📊 数据划分:')
print(f'   训练集: {len(train_data)} 条')
print(f'   验证集: {len(val_data)} 条')

# 保存划分后的数据
with open('./data/train_split.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('./data/val_split.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print('✅ 数据划分完成')
"

# 任务5.3: 创建训练配置 (4小时)
cat > configs/qwen3_1.5b_distill.yaml << 'EOF'
# Qwen3-1.5B 蒸馏训练配置

model:
  name_or_path: "Qwen/Qwen3-1.5B"
  trust_remote_code: true

training:
  output_dir: "./outputs/qwen3-1.5b-distill-v1"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  logging_steps: 10
  save_steps: 100
  eval_steps: 100
  save_total_limit: 3

lora:
  use_lora: true
  r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  task_type: "CAUSAL_LM"

data:
  train_file: "./data/train_split.jsonl"
  validation_file: "./data/val_split.jsonl"
  preprocessing_num_workers: 4
  max_seq_length: 2048
  pad_to_max_length: false

hardware:
  bf16: true
  torch_compile: false
  gradient_checkpointing: true
  ddp_find_unused_parameters: false
EOF

echo "✅ 训练配置创建完成"
```

### Day 6-8: 模型训练

#### Day 6: 训练准备与启动
```bash
# 任务6.1: 训练环境最终检查 (2小时)
bash scripts/check_training_env.sh

# 任务6.2: 启动训练 (后台运行) (2小时)
nohup python3 scripts/train_medium.py > training_medium.log 2>&1 &
TRAIN_PID=$!
echo "训练进程PID: $TRAIN_PID"

# 任务6.3: 训练监控设置 (2小时)
cat > scripts/monitor_training.sh << 'EOF'
#!/bin/bash
# 训练监控脚本

LOG_FILE="training_medium.log"
MODEL_DIR="./outputs/qwen3-1.5b-distill-v1"

echo "📊 训练监控面板"
echo "================"

# 检查进程状态
if ps -p $1 > /dev/null; then
    echo "✅ 训练进程运行中 (PID: $1)"
else
    echo "❌ 训练进程已停止"
fi

# 检查最新日志
if [ -f "$LOG_FILE" ]; then
    echo "📝 最新训练日志:"
    tail -10 "$LOG_FILE"
fi

# 检查模型保存
if [ -d "$MODEL_DIR" ]; then
    echo "💾 模型检查点:"
    ls -lh "$MODEL_DIR" | tail -5
fi

# GPU/NPU使用情况
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
elif command -v npu-smi &> /dev/null; then
    echo "🎮 NPU状态:"
    npu-smi info
fi
EOF

chmod +x scripts/monitor_training.sh
```

#### Day 7: 训练过程监控
```bash
# 任务7.1: 定期监控训练 (全天，每30分钟)
watch -n 1800 bash scripts/monitor_training.sh $TRAIN_PID

# 任务7.2: 实时损失曲线 (持续运行)
python3 -c "
import json
import time
from matplotlib import pyplot as plt

losses = []
timestamps = []

while True:
    try:
        with open('./outputs/qwen3-1.5b-distill-v1/trainer_log.json') as f:
            logs = json.load(f)
        
        for step in logs['log_history']:
            if 'loss' in step:
                losses.append(step['loss'])
                timestamps.append(step.get('step', len(losses)))
        
        # 简单文本显示
        if losses:
            print(f'\\rStep {len(losses)}: Loss = {losses[-1]:.4f}', end='')
        
        time.sleep(60)  # 每分钟更新
        
    except (FileNotFoundError, json.JSONDecodeError):
        print('\\r等待训练日志...', end='')
        time.sleep(60)
"
```

#### Day 8: 训练完成检查
```bash
# 任务8.1: 训练结果验证 (2小时)
python3 -c "
import json
import os

model_dir = './outputs/qwen3-1.5b-distill-v1'

# 检查文件完整性
required_files = [
    'adapter_config.json',
    'adapter_model.safetensors',
    'trainer_log.json',
    'training_args.bin'
]

print('🔍 检查训练输出:')
for filename in required_files:
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / 1024  # KB
        print(f'✅ {filename}: {size:.1f} KB')
    else:
        print(f'❌ {filename}: 缺失')

# 分析训练日志
with open(os.path.join(model_dir, 'trainer_log.json')) as f:
    logs = json.load(f)

history = logs['log_history']
if history:
    final_loss = history[-1].get('loss', 'N/A')
    print(f'\\n📊 训练结果:')
    print(f'   总训练步数: {len(history)}')
    print(f'   最终损失: {final_loss}')
    
    # 损失趋势
    losses = [step.get('loss', float('inf')) for step in history if 'loss' in step]
    if losses and len(losses) > 1:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f'   损失改善: {improvement:.1f}%')
"
```

### Day 9-10: 模型评估

#### Day 9: 评估体系建立
```bash
# 任务9.1: 创建评估脚本 (4小时)
cat > scripts/evaluate_distilled.py << 'EOF'
#!/usr/bin/env python3
\"\"\"蒸馏模型评估脚本\"\"\"
import sys
sys.path.insert(0, '/root/code/model-distill')

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

def extract_code(response):
    \"\"\"从响应中提取代码\"\"\"
    code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return response

def evaluate_model(model_path, test_data):
    \"\"\"评估模型性能\"\"\"
    print(f'📊 加载模型: {model_path}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    results = []
    
    for i, item in enumerate(test_data):
        task_id = item.get('task_id', f'task_{i}')
        prompt = item.get('prompt', '')
        
        # 构造输入
        messages = [
            {'role': 'system', 'content': '你是一个资深软件工程师，擅长解题规划和代码实现。'},
            {'role': 'user', 'content': prompt}
        ]
        
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors='pt')
        
        # 生成响应
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 提取代码
        code = extract_code(response)
        
        results.append({
            'task_id': task_id,
            'prompt': prompt,
            'response': response,
            'code': code,
            'has_plan': '解题计划' in response,
            'has_boundary_analysis': '边界分析' in response or '边界情况' in response,
            'response_length': len(response)
        })
        
        print(f'\\r评估进度: {i+1}/{len(test_data)}', end='')
    
    print()
    return results

def analyze_results(results):
    \"\"\"分析评估结果\"\"\"
    print('\\n📊 评估结果分析:')
    
    # 基本统计
    total = len(results)
    has_plan = sum(1 for r in results if r['has_plan'])
    has_boundary = sum(1 for r in results if r['has_boundary_analysis'])
    avg_response_length = sum(r['response_length'] for r in results) / total
    
    print(f'   总样本数: {total}')
    print(f'   包含解题计划: {has_plan}/{total} ({has_plan/total*100:.1f}%)')
    print(f'   包含边界分析: {has_boundary}/{total} ({has_boundary/total*100:.1f}%)')
    print(f'   平均响应长度: {avg_response_length:.0f} 字符')
    
    return {
        'total_samples': total,
        'has_plan_ratio': has_plan / total,
        'has_boundary_ratio': has_boundary / total,
        'avg_response_length': avg_response_length
    }

if __name__ == '__main__':
    # 加载测试数据
    with open('./data/val_split.jsonl') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f'📊 测试数据: {len(test_data)} 条')
    
    # 评估模型
    results = evaluate_model('./outputs/qwen3-1.5b-distill-v1', test_data)
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 保存结果
    with open('./data/evaluation_results.json', 'w') as f:
        json.dump({
            'analysis': analysis,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print('✅ 评估完成，结果已保存')
EOF

chmod +x scripts/evaluate_distilled.py

# 任务9.2: 执行评估 (4小时)
python3 scripts/evaluate_distilled.py
```

#### Day 10: 对比分析与优化
```bash
# 任务10.1: 基线模型对比 (3小时)
python3 -c "
print('🔄 对比原始Qwen3-1.5B性能...')
# 对原始模型进行相同测试
# (具体实现略)
"

# 任务10.2: 效果分析与优化建议 (3小时)
python3 -c "
import json

# 读取评估结果
with open('./data/evaluation_results.json') as f:
    eval_results = json.load(f)

analysis = eval_results['analysis']

print('📈 蒸馏效果分析:')
print(f'   规划能力习得率: {analysis[\"has_plan_ratio\"]*100:.1f}%')
print(f'   边界分析能力: {analysis[\"has_boundary_ratio\"]*100:.1f}%')

# 基于结果提出优化建议
print('\\n💡 优化建议:')
if analysis['has_plan_ratio'] < 0.8:
    print('   - 规划能力不足，建议增加规划相关训练数据')
if analysis['has_boundary_ratio'] < 0.6:
    print('   - 边界分析能力弱，建议强化边界案例训练')
if analysis['avg_response_length'] < 500:
    print('   - 响息过短，可能需要增加推理长度')
"

# 任务10.3: 优化策略制定 (2小时)
cat > /root/code/model-distill/OPTIMIZATION_STRATEGY.md << 'EOF'
# 模型蒸馏优化策略

## 当前效果分析
- 基于Day 10评估结果
- 对比原始模型vs蒸馏模型
- 识别能力差距

## 优化方向

### 数据优化
1. 增加规划步骤的训练样本
2. 强化边界案例分析
3. 多样化题目类型覆盖

### 训练优化  
1. 调整LoRA参数
2. 优化训练轮数和学习率
3. 增加数据增强策略

### 评估优化
1. 建立更全面的评估体系
2. 添加代码执行验证
3. 多模型对比测试
EOF
```

### 阶段2成功标准
- ✅ 50个高质量训练样本生成并清洗
- ✅ Qwen3-1.5B模型训练完成
- ✅ 建立评估体系和对比分析
- ✅ 规划能力习得率 > 80%

---

## 🏭 阶段3: 优化部署 (Day 11-21)

### 目标设定
- ✅ NPU环境适配和部署
- ✅ 扩展到100个训练样本
- ✅ Qwen3-8B完整蒸馏
- ✅ 生产级部署方案

### Day 11-13: NPU环境准备

### Day 14-17: 大规模训练

### Day 18-20: 完整评估与优化

### Day 21: 部署与总结

---

## 📊 关键里程碑检查点

### Checkpoint 1: Day 3晚上
**决策点**: 阶段1验证是否成功
- ✅ 如果成功 → 进入阶段2
- ❌ 如果失败 → 问题诊断和修复

### Checkpoint 2: Day 10晚上  
**决策点**: 阶段2效果评估
- ✅ 如果效果满意 → 进入阶段3
- 🟡 如果效果一般 → 优化调整后进入阶段3
- ❌ 如果效果差 → 重新制定策略

### Checkpoint 3: Day 21晚上
**决策点**: 项目整体评估
- ✅ 如果目标达成 → 准备发布
- 🟡 如果部分达成 → 制定后续计划
- ❌ 如果未达成 → 经验总结和改进

---

## 🔧 技术实施细节

### 数据生成策略
```python
# 高质量数据生成要点
1. 精心设计的prompt模板
2. 质量过滤和验证机制  
3. 多样性保证策略
4. 成本控制和缓存优化
```

### 训练优化策略
```python
# 训练关键参数
1. LoRA配置优化
2. 学习率调度策略
3. 梯度累积和批次大小
4. 显存优化技巧
```

### 评估体系
```python
# 多维度评估
1. pass@1准确率
2. 规划质量评分
3. 推理效率测试
4. 对比基线测试
```

---

## 📋 每日工作清单模板

### 每日开始检查
```bash
# 每日早晨检查清单
- [ ] 检查前日任务完成状态
- [ ] 确认今日目标和时间安排
- [ ] 检查环境可用性
- [ ] 查看日志和错误信息
```

### 每日结束总结
```bash
# 每日晚上总结清单  
- [ ] 总结当日完成任务
- [ ] 记录遇到的问题和解决方案
- [ ] 更新进度和状态
- [ ] 规划明日任务
```

---

## 🚨 风险应对预案

### 技术风险
```bash
# API风险应对
1. 准备多个Teacher API Key
2. 实现请求缓存机制
3. 添加重试和降级策略

# 训练风险应对  
1. 准备GPU备用环境
2. 参数配置回退方案
3. 检查点恢复机制
```

### 时间风险
```bash
# 时间延误应对
1. 任务优先级调整
2. 范围裁剪策略
3. 并行执行优化
```

---

## 📈 成功标准总表

### 最小目标 (必须达成)
- ✅ 完整训练流程验证
- ✅ 至少一个模型训练成功
- ✅ 蒸馏效果可验证

### 期望目标 (努力达成)
- ✅ 50+样本高质量数据
- ✅ 多规模模型训练成功
- ✅ 规划能力明显提升

### 理想目标 (争取达成)  
- ✅ 100+样本生产级数据
- ✅ 8B模型完整蒸馏
- ✅ 生产级部署方案

---

**方案制定完成，现在可以开始执行！** 

第一步：执行阶段1 Day 1任务。