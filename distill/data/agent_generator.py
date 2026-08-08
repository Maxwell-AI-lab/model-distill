"""Agent 轨迹数据生成器

让 GLM-5.2 以 Agent 模式解题:
1. 分析题目 → 制定计划
2. 写代码 → 调用 run_code 执行
3. 看结果 → 如果失败则分析修正
4. 再验证 → 直到通过或达到轮数上限

记录完整的思考→行动→纠错轨迹。
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

console = Console()


# ── 工具: 代码执行器 ─────────────────────────────────────────

def run_python_code(code: str, timeout: int = 10) -> str:
    """执行 Python 代码，返回输出或错误信息。
    使用 exec() 直接执行，避免子进程启动开销 (NPU 容器中子进程启动极慢)。
    """
    import io
    import contextlib

    stdout_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, {"__name__": "__main__", "__builtins__": __builtins__})
        output = stdout_buf.getvalue().strip()
        return output if output else "✅ 执行成功"
    except AssertionError as e:
        return f"❌ 断言失败: {e}"
    except Exception as e:
        return f"❌ 执行失败: {type(e).__name__}: {e}"


# ── Agent 系统 Prompt ────────────────────────────────────────

AGENT_SYSTEM = """你是一个专业的编程 Agent。你有以下工具可以使用：

## 工具

### run_code
执行 Python 代码并返回结果。你可以用它来运行你的实现代码或测试用例。
使用方式: 在回复中写入以下标记来调用:

<tool_call>
{"name": "run_code", "code": "你的Python代码"}
</tool_call>

## 工作流程

每次收到编程任务后，请按以下步骤工作：

1. **分析**: 理解题目需求，制定解题计划
2. **实现**: 编写代码，用 run_code 工具执行
3. **验证**: 运行测试用例检查结果
4. **修正**: 如果测试失败，分析原因并修正
5. **总结**: 确认通过后，简要总结方案

每一步只做一件事，不要一次性输出所有内容。等待工具返回结果后再继续。

当所有测试通过后，用 <done> 标记结束。"""

TOOL_CALL_PATTERN = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
DONE_PATTERN = r"<done>"


# ── Agent 轨迹生成器 ─────────────────────────────────────────

class AgentTrajectoryGenerator:
    """Agent 轨迹生成器

    让 Teacher (GLM-5.2) 以 Agent 模式解题，
    通过多轮交互 + 代码执行工具，记录完整思考轨迹。
    """

    def __init__(self, teacher, max_rounds: int = 6):
        """
        Args:
            teacher: Teacher 模型实例
            max_rounds: 最大交互轮数
        """
        self.teacher = teacher
        self.max_rounds = max_rounds

    def solve(self, task: dict) -> dict:
        """用 Agent 模式解决一道题，返回完整轨迹

        Args:
            task: {"prompt": "...", "test_list": [...], "task_id": "..."}

        Returns:
            {
                "task_id": "...",
                "prompt": "...",
                "test_cases": [...],
                "trajectory": [messages],
                "final_code": "...",
                "passed": bool,
                "rounds": int,
            }
        """
        task_id = task.get("task_id", "")
        prompt = task.get("prompt", "")
        test_cases = task.get("test_list", [])
        if not test_cases and task.get("test"):
            test_cases = [task["test"]]

        # 第一轮：给题目
        user_msg = self._build_initial_prompt(prompt, test_cases)

        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        trajectory = [m.copy() for m in messages]
        final_code = ""
        passed = False

        for round_num in range(1, self.max_rounds + 1):
            # 调用 Teacher
            try:
                resp = self.teacher.chat(messages, temperature=0.3, max_tokens=2048)
                assistant_text = resp.text
            except Exception as e:
                assistant_text = f"(API 调用失败: {e})"

            # 检查是否完成
            if "<done>" in assistant_text:
                # 去掉 <done> 标记
                assistant_text = re.sub(r"<done>", "", assistant_text).strip()

                # 提取最终代码
                final_code = self._extract_code(assistant_text)
                trajectory.append({"role": "assistant", "content": assistant_text})
                break

            # 检查是否有工具调用
            tool_calls = re.findall(TOOL_CALL_PATTERN, assistant_text, re.DOTALL)

            if tool_calls:
                trajectory.append({"role": "assistant", "content": assistant_text})
                messages.append({"role": "assistant", "content": assistant_text})

                # 执行每个工具调用
                for tc_str in tool_calls:
                    try:
                        tc = json.loads(tc_str)
                        code = tc.get("code", "")
                        if not code:
                            continue
                        tool_result = run_python_code(code)
                    except json.JSONDecodeError:
                        tool_result = "❌ 工具调用格式错误"
                        code = ""

                    # 记录工具结果
                    tool_msg = f"🔧 run_code 结果:\n{tool_result}"
                    trajectory.append({"role": "tool", "content": tool_msg})
                    messages.append({"role": "user", "content": tool_msg})

                    # 如果是测试用例执行，检查是否通过
                    if test_cases and "assert" in code.lower():
                        passed = "❌" not in tool_result and "Error" not in tool_result

                # 如果测试通过了，提示 Agent 可以结束
                if passed:
                    messages.append({
                        "role": "user",
                        "content": "测试全部通过！请用 <done> 标记结束，并简要总结你的解决方案。"
                    })
            else:
                # 没有工具调用，普通回复
                trajectory.append({"role": "assistant", "content": assistant_text})
                messages.append({"role": "assistant", "content": assistant_text})

                # 如果不是最后一轮，提示继续
                if round_num < self.max_rounds:
                    messages.append({
                        "role": "user",
                        "content": "请继续。如果完成了实现和验证，请用 <done> 标记结束。"
                    })

        # 最终验证
        if not final_code:
            final_code = self._extract_last_code(trajectory)

        if final_code and test_cases:
            passed = self._verify_code(final_code, test_cases)

        return {
            "task_id": task_id,
            "prompt": prompt,
            "test_cases": test_cases,
            "trajectory": trajectory,
            "final_code": final_code,
            "passed": passed,
            "rounds": round_num,
        }

    def _build_initial_prompt(self, problem: str, test_cases: list[str]) -> str:
        """构建初始用户消息"""
        msg = f"请解决以下编程任务：\n\n{problem}\n"
        if test_cases:
            msg += f"\n测试用例:\n"
            for tc in test_cases[:5]:
                msg += f"{tc}\n"
        msg += "\n请先用 run_code 工具实现代码，然后运行测试用例验证。"
        return msg

    def _extract_code(self, text: str) -> str:
        """从文本中提取 Python 代码"""
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if not match:
            match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_last_code(self, trajectory: list[dict]) -> str:
        """从轨迹中提取最后一段代码"""
        for msg in reversed(trajectory):
            if msg["role"] == "assistant":
                code = self._extract_code(msg["content"])
                if code:
                    return code
                # 也检查 tool_call 里的代码
                tool_calls = re.findall(TOOL_CALL_PATTERN, msg["content"], re.DOTALL)
                for tc_str in reversed(tool_calls):
                    try:
                        tc = json.loads(tc_str)
                        if tc.get("code"):
                            return tc["code"]
                    except json.JSONDecodeError:
                        continue
        return ""

    def _verify_code(self, code: str, test_cases: list[str]) -> bool:
        """验证代码是否通过所有测试用例"""
        full_code = code + "\n\n" + "\n".join(test_cases)
        return "✅" not in run_python_code(full_code) and run_python_code(full_code) and "❌" not in run_python_code(full_code)

    def generate_batch(self, tasks: list[dict], output_path: str = "data/agent_trajectories.jsonl") -> list[dict]:
        """批量生成 Agent 轨迹"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        console.print(f"\n🤖 Agent 轨迹生成")
        console.print(f"   题目: {len(tasks)}")
        console.print(f"   Teacher: {self.teacher.model}")
        console.print(f"   最大轮数: {self.max_rounds}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            for i, task in enumerate(tasks):
                console.print(f"  [{i+1}/{len(tasks)}] {task.get('task_id', '')}...", end=" ")

                result = self.solve(task)
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

                status = "✅" if result["passed"] else "❌"
                console.print(f"{status} ({result['rounds']} 轮)")

        # 统计
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        avg_rounds = sum(r["rounds"] for r in results) / total if total else 0

        console.print(f"\n📊 生成统计:")
        console.print(f"   总题数: {total}")
        console.print(f"   通过: {passed}/{total} ({passed/total*100:.1f}%)")
        console.print(f"   平均轮数: {avg_rounds:.1f}")
        console.print(f"   输出: {output_path}")

        return results


# ── 训练数据格式化 ───────────────────────────────────────────

def trajectories_to_training_data(
    trajectories: list[dict],
    output_path: str = "data/agent_train.jsonl",
    only_passed: bool = True,
) -> list[dict]:
    """将 Agent 轨迹转换为训练数据

    Args:
        trajectories: AgentTrajectoryGenerator.generate_batch() 的输出
        output_path: 输出路径
        only_passed: 是否只用最终通过的轨迹

    Returns:
        ChatML 格式的多轮对话训练数据
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_data = []

    for traj in trajectories:
        if only_passed and not traj["passed"]:
            continue

        # 轨迹已经是 messages 格式，直接用
        messages = traj["trajectory"]

        # 过滤掉空的或太长的
        if len(messages) < 4:  # 至少 system+user+assistant+tool
            continue

        # 检查内容长度
        total_len = sum(len(m.get("content", "")) for m in messages)
        if total_len > 8000:  # 太长跳过
            continue

        train_data.append({
            "messages": messages,
            "meta": {
                "task_id": traj.get("task_id", ""),
                "rounds": traj.get("rounds", 0),
                "passed": traj.get("passed", False),
                "total_length": total_len,
            },
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    console.print(f"\n✅ 训练数据转换完成:")
    console.print(f"   有效轨迹: {len(train_data)} 条")
    console.print(f"   输出: {output_path}")

    return train_data
