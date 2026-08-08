"""NPU (昇腾 910B) 环境检测与训练适配

自动检测 torch_npu 环境，配置 NPU 训练参数。
"""

import os
import torch

# ── NPU 环境检测 ─────────────────────────────────────────────

def check_npu_available() -> bool:
    """检测 NPU 是否可用"""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


def get_npu_info() -> dict:
    """获取 NPU 硬件信息"""
    if not check_npu_available():
        return {"available": False}

    import torch_npu
    info = {
        "available": True,
        "device_count": torch.npu.device_count(),
        "devices": [],
    }
    for i in range(torch.npu.device_count()):
        props = torch.npu.get_device_properties(i)
        info["devices"].append({
            "id": i,
            "name": getattr(props, "name", "Ascend 910B"),
            "total_memory_gb": props.total_memory / 1024**3,
        })
    return info


def setup_npu_env():
    """配置 NPU 环境变量（需要在 import torch_npu 之前调用部分项）"""
    # HCCL 通信超时 (多卡训练需要)
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "7200")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "7200")

    # ACL 日志级别
    os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "1")  # ERROR

    # 内存优化
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")


# ── NPU 训练适配 ─────────────────────────────────────────────

def get_npu_config(device_ids: list[int] = None):
    """获取 NPU 训练配置

    Returns:
        dict: 包含 device, dtype, config 等
    """
    setup_npu_env()

    if not check_npu_available():
        raise RuntimeError(
            "NPU 不可用。请确认:\n"
            "1. 已安装 CANN 工具链\n"
            "2. 已安装 torch_npu (pip install torch_npu)\n"
            "3. npu-smi info 能显示设备"
        )

    import torch_npu

    if device_ids is None:
        device_ids = list(range(torch.npu.device_count()))

    config = {
        "device_ids": device_ids,
        "device_count": len(device_ids),
        "device": f"npu:{device_ids[0]}",
        "dtype": torch.float16,
        "bf16": True,  # 910B 原生支持 bf16
    }

    # 设置默认设备
    torch.npu.set_device(device_ids[0])

    return config


def patch_transformers_for_npu():
    """为 NPU 打补丁 transformers 库

    主要处理:
    1. 设备映射: cuda → npu
    2. bf16 精度: 910B 原生支持
    3. 注意: bitsandbytes 4bit 量化不支持 NPU，需要用 bf16 全精度
    """
    import transformers

    # Monkey-patch: 让 transformers 识别 NPU
    original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def npu_from_pretrained(*args, **kwargs):
        # 强制不用 bitsandbytes
        kwargs.pop("quantization_config", None)
        kwargs.pop("load_in_4bit", None)
        kwargs.pop("load_in_8bit", None)

        # 使用 bf16 (910B 原生支持)
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.bfloat16

        return original_from_pretrained(*args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = npu_from_pretrained


def create_npu_distributed_config(
    num_nodes: int = 1,
    num_devices_per_node: int = 8,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    node_rank: int = 0,
):
    """创建 NPU 分布式训练环境变量配置

    用于多机多卡 HCCL 通信。
    """
    world_size = num_nodes * num_devices_per_node

    env_config = {
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "WORLD_SIZE": str(world_size),
        "NODE_RANK": str(node_rank),
        "NNODES": str(num_nodes),
        "NPROC_PER_NODE": str(num_devices_per_node),
        # HCCL 相关
        "HCCL_CONNECT_TIMEOUT": "7200",
        "HCCL_EXEC_TIMEOUT": "7200",
        # NPU 相关
        "ASCEND_LAUNCH_BLOCKING": "0",
    }

    for key, val in env_config.items():
        os.environ[key] = val

    return env_config


# ── 便捷函数 ─────────────────────────────────────────────────

def print_npu_status():
    """打印 NPU 状态信息"""
    info = get_npu_info()

    print("\n" + "=" * 50)
    print("         昇腾 NPU 状态")
    print("=" * 50)

    if not info["available"]:
        print("  ❌ NPU 不可用")
        print("=" * 50)
        return

    print(f"  ✅ 可用设备: {info['device_count']} 张")
    for dev in info["devices"]:
        print(f"     NPU {dev['id']}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")
    print("=" * 50 + "\n")
