"""数据生成器 — 用 Teacher 模型生成训练数据"""

import json
import random
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ..teachers.base import BaseTeacher


class DataGenerator:
    """蒸馏数据生成器

    通过 Teacher 模型 API，针对特定场景批量生成高质量问答数据。
    """

    def __init__(self, teacher: BaseTeacher, task_config: dict):
        """
        Args:
            teacher: Teacher 模型实例
            task_config: 任务配置，包含场景定义、prompt 模板等
        """
        self.teacher = teacher
        self.task_config = task_config
        self.scene = task_config.get("scene", "")
        self.num_samples = task_config.get("num_samples", 100)
        self.system_prompt = task_config.get("system_prompt", "")
        self.topic_seeds = task_config.get("topic_seeds", [])
        self.difficulty_levels = task_config.get("difficulty_levels", ["easy", "medium", "hard"])

    def _build_generation_prompt(self, topic: str, difficulty: str, index: int) -> str:
        """构建数据生成 prompt"""
        return f"""你是一个专业的数据生成专家。请根据以下要求生成一条高质量的问答数据：

【场景】{self.scene}
【主题】{topic}
【难度】{difficulty}
【编号】第 {index} 条

请生成一条该场景下的典型问题，并给出高质量的回答。

要求：
1. 问题要真实、自然，符合实际使用场景
2. 回答要准确、详细、有逻辑
3. 难度为 {difficulty}：{"基础水平，问题直接，答案简洁" if difficulty == "easy" else "中等水平，有一定复杂度" if difficulty == "medium" else "高难度，问题复杂，需要深入分析"}"
4. 避免重复和模板化

请严格按以下 JSON 格式输出（不要有其他内容）：
```json
{{"question": "问题内容", "answer": "回答内容", "topic": "{topic}", "difficulty": "{difficulty}"}}
```"""

    def generate_single(self, topic: str, difficulty: str, index: int) -> Optional[dict]:
        """生成单条数据"""
        prompt = self._build_generation_prompt(topic, difficulty, index)
        try:
            text = self.teacher.chat_simple(prompt, system=self.system_prompt)
            # 提取 JSON
            text = text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
            data["source_model"] = self.teacher.model
            return data
        except (json.JSONDecodeError, Exception) as e:
            return {"question": "", "answer": "", "error": str(e), "topic": topic, "difficulty": difficulty}

    def generate_batch(self, output_path: str = "data/generated.jsonl") -> list[dict]:
        """批量生成数据

        Args:
            output_path: 输出文件路径 (JSONL)

        Returns:
            生成的数据列表
        """
        results = []
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成任务列表
        tasks = []
        for i in range(self.num_samples):
            topic = random.choice(self.topic_seeds) if self.topic_seeds else self.scene
            difficulty = random.choice(self.difficulty_levels)
            tasks.append((topic, difficulty, i + 1))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task(f"Generating via {self.teacher.model}...", total=len(tasks))

            with open(output_path, "w", encoding="utf-8") as f:
                for topic, difficulty, idx in tasks:
                    item = self.generate_single(topic, difficulty, idx)
                    if item and "error" not in item:
                        results.append(item)
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    progress.advance(task)

        print(f"\n✅ 生成完成: {len(results)}/{self.num_samples} 条有效数据 → {output_path}")
        return results

    def generate_from_prompts(self, prompts: list[str], output_path: str = "data/generated.jsonl") -> list[dict]:
        """从自定义 prompt 列表生成数据（更灵活的控制）

        Args:
            prompts: 自定义 prompt 列表
            output_path: 输出路径

        Returns:
            生成的数据列表
        """
        results = []
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts):
                try:
                    answer = self.teacher.chat_simple(prompt, system=self.system_prompt)
                    item = {
                        "question": prompt,
                        "answer": answer,
                        "source_model": self.teacher.model,
                        "index": i,
                    }
                    results.append(item)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"⚠️ Error at {i}: {e}")

        print(f"✅ 生成完成: {len(results)}/{len(prompts)} 条 → {output_path}")
        return results
