"""数据格式化 — 转换为训练格式"""

import json
from pathlib import Path


class DataFormatter:
    """数据格式转换

    将蒸馏数据转换为不同训练框架所需的格式。
    """

    @staticmethod
    def to_sharegpt(data: list[dict], output_path: str) -> str:
        """转换为 ShareGPT 格式（通用 SFT 格式）

        格式: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                formatted = {
                    "conversations": [
                        {"from": "human", "value": item["question"]},
                        {"from": "gpt", "value": item["answer"]},
                    ]
                }
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

        print(f"✅ ShareGPT 格式: {len(data)} 条 → {output_path}")
        return str(output_path)

    @staticmethod
    def to_alpaca(data: list[dict], output_path: str) -> str:
        """转换为 Alpaca 格式

        格式: {"instruction": "...", "input": "", "output": "..."}
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                formatted = {
                    "instruction": item["question"],
                    "input": "",
                    "output": item["answer"],
                }
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

        print(f"✅ Alpaca 格式: {len(data)} 条 → {output_path}")
        return str(output_path)

    @staticmethod
    def to_chatml(data: list[dict], output_path: str, system_prompt: str = "") -> str:
        """转换为 ChatML 格式（Qwen 原生格式）

        格式: {"messages": [{"role": "system", "content": "..."}, {"role": "user", ...}, {"role": "assistant", ...}]}
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": item["question"]})
                messages.append({"role": "assistant", "content": item["answer"]})
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

        print(f"✅ ChatML 格式: {len(data)} 条 → {output_path}")
        return str(output_path)

    @staticmethod
    def to_dpo(data: list[dict], output_path: str) -> str:
        """生成 DPO 格式（需要 chosen / rejected 对）

        注意: rejected 数据需要额外生成或用规则构造
        这里先输出 chosen，rejected 后续补充
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                formatted = {
                    "prompt": item["question"],
                    "chosen": item["answer"],
                    "rejected": "",  # 待填充
                }
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

        print(f"✅ DPO 格式: {len(data)} 条 → {output_path} (rejected 待补充)")
        return str(output_path)
