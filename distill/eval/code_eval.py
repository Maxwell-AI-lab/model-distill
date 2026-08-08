"""代码执行评估 — 在沙盒中运行 Student 代码，检查测试用例

评估蒸馏后小模型的代码能力：
1. 代码通过率 (pass@1)
2. 计划质量 (是否有计划、步骤数)
3. 执行时间 / 代码长度等统计
"""

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


class CodeExecutor:
    """安全代码执行器 — 在子进程中运行，限制超时"""

    TIMEOUT = 10  # 秒

    @staticmethod
    def run_code(code: str, test_cases: list[str]) -> dict:
        """运行代码并执行测试

        Args:
            code: Python 代码字符串
            test_cases: 测试断言列表 ["assert x==1", ...]

        Returns:
            {"passed": bool, "error": str, "execution_time": float}
        """
        full_code = code + "\n\n" + "\n".join(test_cases)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            start = time.time()
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=CodeExecutor.TIMEOUT,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                return {"passed": True, "error": "", "execution_time": elapsed}
            else:
                return {
                    "passed": False,
                    "error": result.stderr.strip()[-500:],  # 截取最后 500 字符
                    "execution_time": elapsed,
                }
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Timeout", "execution_time": CodeExecutor.TIMEOUT}
        except Exception as e:
            return {"passed": False, "error": str(e), "execution_time": 0}
        finally:
            Path(temp_path).unlink(missing_ok=True)


class CodeEvaluator:
    """代码蒸馏效果评估器"""

    def __init__(self, executor: CodeExecutor = None):
        self.executor = executor or CodeExecutor()

    def extract_code(self, text: str) -> str:
        """从模型输出中提取代码"""
        # 尝试 ```python ... ```
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 尝试 ``` ... ```
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 无代码块，返回原文
        return text.strip()

    def extract_plan(self, text: str) -> str:
        """从模型输出中提取解题计划"""
        match = re.search(r"## 解题计划\s*\n(.*?)(?=## |$)", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def count_plan_steps(self, plan_text: str) -> int:
        """计算计划步骤数"""
        steps = re.findall(r"^\d+\.", plan_text, re.MULTILINE)
        return len(steps)

    def evaluate_single(
        self,
        model_output: str,
        test_cases: list[str],
        task_id: str = "",
    ) -> dict:
        """评估单条结果

        Args:
            model_output: 模型的完整输出 (计划+代码)
            test_cases: 测试用例
            task_id: 任务 ID

        Returns:
            评估结果
        """
        code = self.extract_code(model_output)
        plan = self.extract_plan(model_output)
        plan_steps = self.count_plan_steps(plan)

        result = {
            "task_id": task_id,
            "has_plan": bool(plan),
            "plan_steps": plan_steps,
            "has_code": bool(code),
            "code_length": len(code),
        }

        # 运行测试
        if code and test_cases:
            exec_result = self.executor.run_code(code, test_cases)
            result["passed"] = exec_result["passed"]
            result["error"] = exec_result["error"]
            result["execution_time"] = exec_result["execution_time"]
        else:
            result["passed"] = False
            result["error"] = "No code or no test cases"
            result["execution_time"] = 0

        return result

    def evaluate_batch(
        self,
        model_outputs: list[str],
        test_cases_list: list[list[str]],
        task_ids: list[str] = None,
    ) -> dict:
        """批量评估

        Returns:
            {"pass_rate": float, "details": [...], "stats": {...}}
        """
        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(len(model_outputs))]

        details = []
        passed_count = 0
        has_plan_count = 0
        total_plan_steps = 0

        for output, tests, tid in zip(model_outputs, test_cases_list, task_ids):
            result = self.evaluate_single(output, tests, tid)
            details.append(result)

            if result["passed"]:
                passed_count += 1
            if result["has_plan"]:
                has_plan_count += 1
                total_plan_steps += result["plan_steps"]

        total = len(details)
        summary = {
            "total": total,
            "passed": passed_count,
            "pass_rate": passed_count / total if total else 0,
            "has_plan_rate": has_plan_count / total if total else 0,
            "avg_plan_steps": total_plan_steps / has_plan_count if has_plan_count else 0,
            "avg_code_length": sum(d["code_length"] for d in details) / total if total else 0,
            "avg_execution_time": sum(d["execution_time"] for d in details) / total if total else 0,
            "details": details,
        }

        return summary

    @staticmethod
    def print_report(summary: dict, title: str = "评估结果"):
        """打印评估报告"""
        console.print(f"\n📊 {title}", style="bold cyan")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("指标", style="dim")
        table.add_column("值", justify="right")

        table.add_row("总题数", str(summary["total"]))
        table.add_row(
            "✅ 通过",
            f"[green]{summary['passed']}[/green] ({summary['pass_rate']*100:.1f}%)"
        )
        table.add_row(
            "📝 有计划",
            f"{summary['has_plan_rate']*100:.1f}%"
        )
        table.add_row(
            "📋 平均步骤数",
            f"{summary['avg_plan_steps']:.1f}"
        )
        table.add_row(
            "📏 平均代码长度",
            f"{summary['avg_code_length']:.0f} 字符"
        )
        table.add_row(
            "⏱️ 平均执行时间",
            f"{summary['avg_execution_time']:.2f}s"
        )

        console.print(table)

        # 失败案例
        failed = [d for d in summary["details"] if not d["passed"]]
        if failed:
            console.print(f"\n❌ 失败案例 ({len(failed)}):", style="red")
            for d in failed[:5]:  # 只显示前5个
                console.print(f"   {d['task_id']}: {d['error'][:80]}")
