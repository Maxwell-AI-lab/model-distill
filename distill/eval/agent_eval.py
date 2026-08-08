"""Agent 评估器 — 让 Student 模型以 Agent 模式做题

评估流程:
1. 给 Student 一道编程题
2. Student 可以调用 run_code 工具 (最多 N 轮)
3. 记录完整交互轨迹
4. 统计 pass@1 + 纠错率 + 平均轮数
"""

import json
import os
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


class AgentEvaluator:
    """Agent 模式评估器"""

    def __init__(self, model, tokenizer, device="npu:0", max_rounds=5):
        """
        Args:
            model: HuggingFace 模型
            tokenizer: 对应的 tokenizer
            device: 推理设备
            max_rounds: 每题最多交互轮数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_rounds = max_rounds

    def evaluate_single(self, task: dict) -> dict:
        """评估单道题

        Returns:
            {
                "task_id": "...",
                "passed": bool,
                "rounds": int,
                "first_attempt_passed": bool,
                "error_corrected": bool,
                "trajectory": [...],
            }
        """
        import torch

        prompt = task.get("prompt", "")
        test_cases = task.get("test_list", [])
        if not test_cases and task.get("test"):
            test_cases = [task["test"]]

        # 初始消息
        system_msg = AGENT_SYSTEM
        user_msg = f"请解决以下编程任务：\n\n{prompt}\n"
        if test_cases:
            user_msg += f"\n测试用例:\n"
            for tc in test_cases[:5]:
                user_msg += f"{tc}\n"
        user_msg += "\n请先用 run_code 工具实现代码，然后运行测试用例验证。"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        trajectory = [m.copy() for m in messages]
        final_code = ""
        first_attempt_passed = None  # 第一次代码执行的结果

        for round_num in range(1, self.max_rounds + 1):
            # 模型生成
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # 检查完成
            if "<done>" in response:
                response = re.sub(r"<done>", "", response).strip()
                final_code = self._extract_code(response)
                trajectory.append({"role": "assistant", "content": response})
                break

            # 检查工具调用
            tool_calls = re.findall(TOOL_CALL_PATTERN, response, re.DOTALL)
            trajectory.append({"role": "assistant", "content": response})
            messages.append({"role": "assistant", "content": response})

            if tool_calls:
                for tc_str in tool_calls:
                    try:
                        tc = json.loads(tc_str)
                        code = tc.get("code", "")
                        tool_result = run_python_code(code)
                    except json.JSONDecodeError:
                        tool_result = "❌ 工具调用格式错误"

                    tool_msg = f"🔧 run_code 结果:\n{tool_result}"
                    trajectory.append({"role": "tool", "content": tool_msg})
                    messages.append({"role": "user", "content": tool_msg})

                    # 记录第一次尝试结果
                    if "assert" in code.lower() or "test" in code.lower():
                        current_pass = "❌" not in tool_result and "Error" not in tool_result
                        if first_attempt_passed is None:
                            first_attempt_passed = current_pass

                if first_attempt_passed is True:
                    messages.append({
                        "role": "user",
                        "content": "测试全部通过！请用 <done> 标记结束，并简要总结。"
                    })
            else:
                if round_num < self.max_rounds:
                    messages.append({
                        "role": "user",
                        "content": "请继续。用 run_code 工具实现代码或运行测试。完成后用 <done> 标记。"
                    })

        # 最终验证
        if not final_code:
            final_code = self._extract_last_code(trajectory)

        passed = False
        if final_code and test_cases:
            passed = self._verify(final_code, test_cases)

        error_corrected = (first_attempt_passed is False) and passed

        return {
            "task_id": task.get("task_id", ""),
            "passed": passed,
            "rounds": round_num,
            "first_attempt_passed": first_attempt_passed or False,
            "error_corrected": error_corrected,
            "trajectory": trajectory,
            "final_code": final_code[:500],
        }

    def evaluate_batch(self, tasks: list[dict]) -> dict:
        """批量评估"""
        results = []

        console.print(f"\n🤖 Agent 评估")
        console.print(f"   题数: {len(tasks)}")
        console.print(f"   最大轮数: {self.max_rounds}\n")

        for i, task in enumerate(tasks):
            console.print(f"  [{i+1}/{len(tasks)}] {task.get('task_id', '')}...", end=" ")
            result = self.evaluate_single(task)
            results.append(result)
            status = "✅" if result["passed"] else "❌"
            correction = " (纠错成功!)" if result["error_corrected"] else ""
            console.print(f"{status}{correction} ({result['rounds']} 轮)")

        # 统计
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        first_pass = sum(1 for r in results if r["first_attempt_passed"])
        corrected = sum(1 for r in results if r["error_corrected"])
        avg_rounds = sum(r["rounds"] for r in results) / total if total else 0

        summary = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0,
            "first_attempt_passed": first_pass,
            "first_pass_rate": first_pass / total if total else 0,
            "error_corrected": corrected,
            "correction_rate": corrected / total if total else 0,
            "avg_rounds": avg_rounds,
            "details": results,
        }

        return summary

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if not match:
            match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_last_code(self, trajectory: list[dict]) -> str:
        for msg in reversed(trajectory):
            if msg["role"] == "assistant":
                code = self._extract_code(msg["content"])
                if code:
                    return code
                tool_calls = re.findall(TOOL_CALL_PATTERN, msg["content"], re.DOTALL)
                for tc_str in reversed(tool_calls):
                    try:
                        tc = json.loads(tc_str)
                        if tc.get("code"):
                            return tc["code"]
                    except json.JSONDecodeError:
                        continue
        return ""

    def _verify(self, code: str, test_cases: list[str]) -> bool:
        full_code = code + "\n\n" + "\n".join(test_cases)
        result = run_python_code(full_code)
        return "❌" not in result and "Error" not in result


def run_python_code(code: str, timeout: int = 10) -> str:
    """执行 Python 代码"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        if result.returncode == 0:
            return output if output else "✅ 执行成功"
        else:
            error_lines = error.split("\n")
            return f"❌ 执行失败:\n" + "\n".join(error_lines[-5:])
    except subprocess.TimeoutExpired:
        return "❌ 超时"
    except Exception as e:
        return f"❌ 异常: {e}"
    finally:
        Path(temp_path).unlink(missing_ok=True)


# ── 导入 Agent prompt (跟 agent_generator 共享) ──────────────

from .agent_generator import AGENT_SYSTEM, TOOL_CALL_PATTERN
