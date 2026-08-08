"""开源代码数据集加载器

支持的数据集:
- HumanEval (164题) — HumanEval-X 中文版可选
- MBPP (974题)
- CodeSearchNet (代码搜索)
- 自定义 JSONL 数据集
"""

import json
import random
from pathlib import Path
from typing import Optional


class DatasetLoader:
    """数据集统一加载器"""

    @staticmethod
    def load_humaneval(
        split: str = "test",
        cache_dir: str = "data/raw",
        language: str = "python",
    ) -> list[dict]:
        """加载 HumanEval 数据集

        Args:
            split: "test" | "train" (HumanEval 只有 test)
            cache_dir: 缓存目录
            language: "python" | "cpp" | "java" | "go" (HumanEval-X)

        Returns:
            标准化的题目列表
        """
        try:
            from datasets import load_dataset

            if language == "python":
                ds = load_dataset("openai_humaneval", split="test", cache_dir=cache_dir)
            else:
                # HumanEval-X 多语言版
                ds = load_dataset(
                    "THUDM/humaneval-x",
                    f"humaneval-x-{language}",
                    split="test",
                    cache_dir=cache_dir,
                )
        except Exception as e:
            print(f"⚠️ 在线加载 HumanEval 失败: {e}")
            print("   尝试加载本地缓存...")
            return DatasetLoader._load_local_jsonl(f"{cache_dir}/humaneval.jsonl")

        results = []
        for item in ds:
            results.append({
                "task_id": item.get("task_id", ""),
                "prompt": item.get("prompt", ""),
                "canonical_solution": item.get("canonical_solution", ""),
                "test": item.get("test", ""),
                "entry_point": item.get("entry_point", ""),
                "language": language,
                "source": "humaneval",
                "difficulty": "medium",  # HumanEval 没有难度标注，统一标记
            })

        print(f"✅ HumanEval 加载完成: {len(results)} 题 ({language})")
        return results

    @staticmethod
    def load_mbpp(
        split: str = "train",
        cache_dir: str = "data/raw",
    ) -> list[dict]:
        """加载 MBPP (Mostly Basic Python Problems) 数据集

        Args:
            split: "train" | "test" | "validation" | "prompt"
            cache_dir: 缓存目录
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("mbpp", split=split, cache_dir=cache_dir)
        except Exception as e:
            print(f"⚠️ 在线加载 MBPP 失败: {e}")
            print("   尝试加载本地缓存...")
            return DatasetLoader._load_local_jsonl(f"{cache_dir}/mbpp_{split}.jsonl")

        results = []
        for item in ds:
            # MBPP 格式: text, code, test_list, task_id, difficulty
            results.append({
                "task_id": f"mbpp_{item.get('task_id', '')}",
                "prompt": item.get("text", ""),
                "canonical_solution": item.get("code", ""),
                "test_list": item.get("test_list", []),
                "difficulty": item.get("difficulty", "medium"),
                "language": "python",
                "source": "mbpp",
            })

        print(f"✅ MBPP 加载完成: {split}: {len(results)} 题")
        return results

    @staticmethod
    def load_codecontests(
        split: str = "train",
        cache_dir: str = "data/raw",
        limit: int = 500,
    ) -> list[dict]:
        """加载 CodeContests 竞赛题（较难）

        Args:
            split: "train" | "valid" | "test"
            cache_dir: 缓存目录
            limit: 最多加载题数
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("code_contests", split=split, cache_dir=cache_dir)
        except Exception as e:
            print(f"⚠️ CodeContests 加载失败: {e}")
            return []

        results = []
        for i, item in enumerate(ds):
            if i >= limit:
                break
            results.append({
                "task_id": f"cc_{item.get('name', i)}",
                "prompt": item.get("description", ""),
                "canonical_solution": "",  # 竞赛题标准答案较长
                "difficulty": {1: "easy", 2: "medium", 3: "hard"}.get(
                    item.get("difficulty", 2), "medium"
                ),
                "language": "python",
                "source": "codecontests",
            })

        print(f"✅ CodeContests 加载完成: {len(results)} 题")
        return results

    @staticmethod
    def load_custom(path: str) -> list[dict]:
        """加载自定义数据集 (JSONL 格式)

        格式要求: 每行一个 JSON，包含 prompt 字段，可选 test/difficulty 等
        """
        data = DatasetLoader._load_local_jsonl(path)

        # 标准化
        results = []
        for item in data:
            results.append({
                "task_id": item.get("task_id", f"custom_{len(results)}"),
                "prompt": item.get("prompt", item.get("description", "")),
                "test": item.get("test", ""),
                "test_list": item.get("test_list", []),
                "difficulty": item.get("difficulty", "medium"),
                "language": item.get("language", "python"),
                "source": "custom",
            })

        print(f"✅ 自定义数据集加载: {len(results)} 题 ← {path}")
        return results

    @staticmethod
    def load_mixed(
        sources: list[str] = None,
        cache_dir: str = "data/raw",
        total_limit: int = 1000,
        seed: int = 42,
    ) -> list[dict]:
        """混合加载多个数据集

        Args:
            sources: 数据源列表 ["humaneval", "mbpp", "codecontests"]
            cache_dir: 缓存目录
            total_limit: 总题数限制
            seed: 随机种子
        """
        if sources is None:
            sources = ["humaneval", "mbpp"]

        all_data = []

        for source in sources:
            if source == "humaneval":
                data = DatasetLoader.load_humaneval(cache_dir=cache_dir)
            elif source == "mbpp":
                data = DatasetLoader.load_mbpp(split="train", cache_dir=cache_dir)
            elif source == "codecontests":
                data = DatasetLoader.load_codecontests(cache_dir=cache_dir, limit=300)
            elif source.startswith("custom:"):
                data = DatasetLoader.load_custom(source.split("custom:", 1)[1])
            else:
                print(f"⚠️ 未知数据源: {source}")
                continue
            all_data.extend(data)

        # 打乱并截断
        random.seed(seed)
        random.shuffle(all_data)
        if len(all_data) > total_limit:
            all_data = all_data[:total_limit]

        # 统计
        source_counts = {}
        diff_counts = {}
        for item in all_data:
            source_counts[item.get("source", "unknown")] = source_counts.get(
                item.get("source", "unknown"), 0
            ) + 1
            diff_counts[item.get("difficulty", "unknown")] = diff_counts.get(
                item.get("difficulty", "unknown"), 0
            ) + 1

        print(f"\n📊 混合数据集统计:")
        print(f"   总计: {len(all_data)} 题")
        print(f"   来源: {source_counts}")
        print(f"   难度: {diff_counts}")

        return all_data

    @staticmethod
    def _load_local_jsonl(path: str) -> list[dict]:
        """加载本地 JSONL 文件"""
        path = Path(path)
        if not path.exists():
            return []
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    @staticmethod
    def save_jsonl(data: list[dict], path: str):
        """保存为 JSONL"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"💾 保存: {len(data)} 条 → {path}")

    @staticmethod
    def train_eval_split(
        data: list[dict],
        eval_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[dict], list[dict]]:
        """划分训练集和评估集"""
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)

        eval_size = max(1, int(len(data) * eval_ratio))
        eval_data = data[:eval_size]
        train_data = data[eval_size:]

        print(f"📂 数据划分: 训练 {len(train_data)} 条 / 评估 {len(eval_data)} 条")
        return train_data, eval_data
