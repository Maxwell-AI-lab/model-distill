"""代码任务蒸馏数据生成器

专门用于生成"先规划、再编码"的训练数据。
用 Teacher 模型对编程题生成分步解题计划 + 代码实现。
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ..teachers.base import BaseTeacher

console = Console()


# ── Prompt 模板 ──────────────────────────────────────────────

PLANNING_SYSTEM = """你是一个资深软件工程师。面对编程任务，你总是先制定清晰的解题计划，再写代码。
你的计划包括：步骤分解、依赖关系、边界情况分析。你写的代码简洁、正确、有注释。"""

PLANNING_USER = """请解决以下编程任务。

【题目】
{prompt}

要求：
1. 先制定分步解题计划（标注每一步要做什么）
2. 分析关键边界情况和风险
3. 写出完整的 Python 实现
4. 确保代码可以直接运行

请按以下格式回答：

## 解题计划
1. [步骤1] ...
2. [步骤2] ...
...

## 边界分析
- [需要处理的边界情况]

## 代码实现
```python
[完整代码]
```

## 复杂度
- 时间：O(?)
- 空间：O(?)"""

PLANNING_USER_WITH_TESTS = """请解决以下编程任务。

【题目】
{prompt}

【测试用例】
{test_cases}

要求：
1. 先制定分步解题计划
2. 分析边界情况
3. 写出完整实现，确保通过所有测试用例

请按以下格式回答：

## 解题计划
1. [步骤1] ...
2. [步骤2] ...
...

## 边界分析
- [边界情况]

## 代码实现
```python
[完整代码]
```

## 复杂度
- 时间：O(?)
- 空间：O(?)"""


class CodeDistillGenerator:
    """代码任务蒸馏数据生成器

    对每道编程题，调用 Teacher 生成"计划+代码"的完整解答。
    支持多 Teacher 并行生成，自动质量过滤。
    """

    def __init__(self, teachers: dict[str, BaseTeacher]):
        """
        Args:
            teachers: {"deepseek": DeepSeekTeacher, "glm": GLMTeacher, ...}
        """
        self.teachers = teachers

    def generate_for_task(self, task: dict) -> dict:
        """对单道题目生成蒸馏数据

        用所有 Teacher 各生成一次，返回完整结果。

        Returns:
            {
                "task_id": "...",
                "prompt": "...",
                "responses": {
                    "deepseek": {"plan": ..., "code": ..., "raw": ...},
                    "glm": {...},
                    ...
                },
                "best_teacher": "deepseek",
            }
        """
        prompt = task["prompt"]
        test_cases = task.get("test_list", [])
        has_tests = bool(test_cases)

        if has_tests:
            test_str = "\n".join(f"assert {t}" for t in test_cases)
            user_msg = PLANNING_USER_WITH_TESTS.format(
                prompt=prompt, test_cases=test_str
            )
        else:
            user_msg = PLANNING_USER.format(prompt=prompt)

        responses = {}
        for name, teacher in self.teachers.items():
            try:
                text = teacher.chat_simple(user_msg, system=PLANNING_SYSTEM)
                parsed = self._parse_response(text)
                parsed["raw"] = text
                parsed["teacher_model"] = teacher.model
                responses[name] = parsed
            except Exception as e:
                responses[name] = {"error": str(e), "plan": "", "code": ""}

        # 选最优
        best = self._select_best(task, responses)

        return {
            "task_id": task.get("task_id", ""),
            "prompt": prompt,
            "test_cases": test_cases,
            "difficulty": task.get("difficulty", ""),
            "source": task.get("source", ""),
            "responses": responses,
            "best_teacher": best,
        }

    def generate_batch(
        self,
        tasks: list[dict],
        output_path: str = "data/distill_raw.jsonl",
    ) -> list[dict]:
        """批量生成蒸馏数据

        Args:
            tasks: 编程题目列表
            output_path: 输出路径

        Returns:
            全部生成结果
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        teacher_names = list(self.teachers.keys())

        console.print(
            f"\n🚀 开始生成蒸馏数据\n"
            f"   题目数: {len(tasks)}\n"
            f"   Teachers: {teacher_names}\n"
            f"   预计 API 调用: {len(tasks) * len(teacher_names)} 次\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task_bar = progress.add_task("Generating...", total=len(tasks))

            with open(output_path, "w", encoding="utf-8") as f:
                for i, task in enumerate(tasks):
                    result = self.generate_for_task(task)
                    results.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()  # 实时写入，防止中断丢数据

                    progress.advance(task_bar)

        # 统计
        stats = self._compute_stats(results)
        console.print(f"\n📊 生成统计:")
        console.print(f"   总题数: {stats['total']}")
        for name in teacher_names:
            valid = stats["valid_per_teacher"].get(name, 0)
            console.print(f"   {name}: {valid} 条有效 ({valid/stats['total']*100:.0f}%)")
        console.print(f"   输出: {output_path}")

        return results

    def _parse_response(self, text: str) -> dict:
        """解析 Teacher 的 Markdown 回复

        提取: 计划、边界分析、代码、复杂度
        """
        result = {"plan": "", "boundary": "", "code": "", "complexity": ""}

        # 提取解题计划
        plan_match = re.search(r"## 解题计划\s*\n(.*?)(?=## |$)", text, re.DOTALL)
        if plan_match:
            result["plan"] = plan_match.group(1).strip()

        # 提取边界分析
        boundary_match = re.search(r"## 边界分析\s*\n(.*?)(?=## |$)", text, re.DOTALL)
        if boundary_match:
            result["boundary"] = boundary_match.group(1).strip()

        # 提取代码
        code_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if code_match:
            result["code"] = code_match.group(1).strip()

        # 提取复杂度
        complexity_match = re.search(r"## 复杂度\s*\n(.*?)(?=## |$)", text, re.DOTALL)
        if complexity_match:
            result["complexity"] = complexity_match.group(1).strip()

        return result

    def _select_best(self, task: dict, responses: dict) -> str:
        """选择最优 Teacher 的结果作为训练标签

        策略:
        1. 优先选有代码且能通过测试的
        2. 其次选有代码且计划详细的
        3. 最后选有代码的
        """
        scored = []
        for name, resp in responses.items():
            if "error" in resp or not resp.get("code"):
                continue

            score = 0
            # 有计划加分
            if resp.get("plan"):
                score += len(resp["plan"]) // 50  # 计划越详细分越高
            # 有代码加分
            if resp.get("code"):
                score += 10
            # 有边界分析加分
            if resp.get("boundary"):
                score += 5

            # 代码验证
            test_cases = task.get("test_list", [])
            if test_cases and resp.get("code"):
                passed = self._run_code_tests(resp["code"], test_cases)
                if passed:
                    score += 100  # 通过测试大幅加分

            scored.append((name, score))

        if not scored:
            return ""

        scored.sort(key=lambda x: -x[1])
        return scored[0][0]

    def _run_code_tests(self, code: str, test_cases: list[str]) -> bool:
        """运行测试用例验证代码（安全沙盒）

        Args:
            code: 待测代码
            test_cases: 测试断言列表 ["assert x==1", ...]

        Returns:
            是否全部通过
        """
        try:
            # ⚠️ 注意：实际使用时需要更严格的安全沙盒
            test_code = code + "\n\n" + "\n".join(test_cases)
            exec(test_code, {"__name__": "__test__"})
            return True
        except Exception:
            return False

    def _compute_stats(self, results: list[dict]) -> dict:
        """计算生成统计"""
        stats = {"total": len(results)}
        valid_per_teacher = {}
        for result in results:
            for name, resp in result.get("responses", {}).items():
                if "error" not in resp and resp.get("code"):
                    valid_per_teacher[name] = valid_per_teacher.get(name, 0) + 1
        stats["valid_per_teacher"] = valid_per_teacher
        return stats


# ── 后处理: 从生成结果提取训练数据 ───────────────────────────


def extract_training_data(
    raw_path: str,
    output_path: str = "data/train_chatml.jsonl",
    use_best_only: bool = True,
    system_prompt: str = PLANNING_SYSTEM,
) -> list[dict]:
    """从生成结果中提取 SFT 训练数据

    Args:
        raw_path: generate_batch 的输出文件
        output_path: 训练数据输出路径
        use_best_only: True=每题只取最优Teacher; False=所有有效结果都保留

    Returns:
        ChatML 格式的训练数据
    """
    raw_data = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    train_samples = []
    skipped = 0

    for item in raw_data:
        prompt = item["prompt"]
        test_cases = item.get("test_cases", [])

        if use_best_only:
            best = item.get("best_teacher", "")
            if not best:
                skipped += 1
                continue
            candidates = [(best, item["responses"].get(best, {}))]
        else:
            candidates = [
                (name, resp)
                for name, resp in item.get("responses", {}).items()
                if "error" not in resp and resp.get("code")
            ]

        for teacher_name, resp in candidates:
            if "error" in resp or not resp.get("code"):
                skipped += 1
                continue

            # 代码验证
            if test_cases and resp.get("code"):
                try:
                    test_code = resp["code"] + "\n\n" + "\n".join(test_cases)
                    exec(test_code, {"__name__": "__test__"})
                except Exception:
                    skipped += 1
                    continue

            # 构造 ChatML
            user_content = PLANNING_USER.format(prompt=prompt) if not test_cases else \
                PLANNING_USER_WITH_TESTS.format(
                    prompt=prompt,
                    test_cases="\n".join(f"assert {t}" for t in test_cases),
                )

            # 重构 assistant 回复 (从解析结果重新组装 Markdown)
            assistant_parts = []
            if resp.get("plan"):
                assistant_parts.append(f"## 解题计划\n{resp['plan']}")
            if resp.get("boundary"):
                assistant_parts.append(f"## 边界分析\n{resp['boundary']}")
            if resp.get("code"):
                assistant_parts.append(f"## 代码实现\n```python\n{resp['code']}\n```")
            if resp.get("complexity"):
                assistant_parts.append(f"## 复杂度\n{resp['complexity']}")

            assistant_content = "\n\n".join(assistant_parts)

            train_samples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "meta": {
                    "task_id": item.get("task_id", ""),
                    "teacher": teacher_name,
                    "teacher_model": resp.get("teacher_model", ""),
                    "difficulty": item.get("difficulty", ""),
                    "source": item.get("source", ""),
                },
            })

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    console.print(
        f"\n✅ 训练数据提取完成:\n"
        f"   有效: {len(train_samples)} 条\n"
        f"   跳过: {skipped} 条\n"
        f"   输出: {output_path}"
    )

    return train_samples
