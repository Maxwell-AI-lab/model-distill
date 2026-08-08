#!/usr/bin/env python3
"""NPU 环境检测脚本 — 在昇腾节点上运行，检测硬件和软件环境"""

import subprocess
import sys
import os

def run(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.stdout else result.stderr.strip()
    except Exception as e:
        return f"Error: {e}"

print("=" * 60)
print("           昇腾 NPU 环境检测")
print("=" * 60)

# 1. 基础信息
print("\n📦 1. 系统信息")
print(f"   Python: {sys.version}")
print(f"   OS: {run('cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2')}")
print(f"   Kernel: {run('uname -r')}")
print(f"   CPU: {run('nproc')} cores")

# 2. NPU 硬件
print("\n🖥️  2. NPU 硬件")
print(f"   npu-smi: {run('npu-smi info 2>/dev/null | head -20')}")
print(f"   卡数量: {run('npu-smi info -l 2>/dev/null | grep -c \"NPU ID\"') or 'unknown'}")

# 3. CANN 版本
print("\n🔧 3. CANN 版本信息")
cann_version = run("cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || cat /usr/local/Ascend/nnae/latest/version.cfg 2>/dev/null")
print(f"   CANN: {cann_version}")

# 4. Python 框架
print("\n📚 4. Python 框架检测")

# PyTorch
try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
except ImportError:
    print(f"   ❌ PyTorch: 未安装")

# torch_npu
try:
    import torch_npu
    print(f"   ✅ torch_npu: {torch_npu.__version__}")
    # 测试 NPU 是否可用
    if torch.npu.is_available():
        print(f"   ✅ NPU 可用: {torch.npu.device_count()} 张")
        for i in range(torch.npu.device_count()):
            props = torch.npu.get_device_properties(i)
            print(f"      NPU {i}: {props.name if hasattr(props, 'name') else 'Ascend'} (total memory: {props.total_memory / 1024**3:.1f} GB)")
    else:
        print(f"   ⚠️ torch_npu 已安装但 NPU 不可用")
except ImportError:
    print(f"   ❌ torch_npu: 未安装")

# MindSpore
try:
    import mindspore
    print(f"   ✅ MindSpore: {mindspore.__version__}")
except ImportError:
    print(f"   ❌ MindSpore: 未安装")

# transformers
try:
    import transformers
    print(f"   ✅ transformers: {transformers.__version__}")
except ImportError:
    print(f"   ❌ transformers: 未安装")

# peft
try:
    import peft
    print(f"   ✅ peft: {peft.__version__}")
except ImportError:
    print(f"   ❌ peft: 未安装")

# deepspeed
try:
    import deepspeed
    print(f"   ✅ deepspeed: {deepspeed.__version__}")
except ImportError:
    print(f"   ❌ deepspeed: 未安装")

# accelerate
try:
    import accelerate
    print(f"   ✅ accelerate: {accelerate.__version__}")
except ImportError:
    print(f"   ❌ accelerate: 未安装")

# bitsandbytes
try:
    import bitsandbytes
    print(f"   ✅ bitsandbytes: {bitsandbytes.__version__}")
except ImportError:
    print(f"   ❌ bitsandbytes: 未安装")

# 5. 环境变量
print("\n🌍 5. 关键环境变量")
env_vars = [
    "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME",
    "LD_LIBRARY_PATH", "PYTHONPATH",
    "HCCL_CONNECT_TIMEOUT",
]
for var in env_vars:
    val = os.environ.get(var, "")
    if val:
        print(f"   {var} = {val[:100]}{'...' if len(val) > 100 else ''}")
    else:
        print(f"   {var} = (未设置)")

# 6. 网络互通检测
print("\n🌐 6. 多机网络检测")
hostname = run("hostname")
ip = run("hostname -I | awk '{print $1}'")
print(f"   主机名: {hostname}")
print(f"   IP: {ip}")
print(f"   HCCL 端口: 通常使用 10000-20000 范围")

# 7. 存储检测
print("\n💾 7. 存储信息")
print(f"   磁盘: {run('df -h / | tail -1')}")

print("\n" + "=" * 60)
print("  检测完成！请把以上输出发给我，我来适配训练代码。")
print("=" * 60)
