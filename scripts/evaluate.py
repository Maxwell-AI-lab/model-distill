"""评估脚本 — 对比 Student 和 Teacher"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill.eval import compute_metrics, print_metrics
from distill.eval.judge import LLMJudge


def main():
    parser = argparse.ArgumentParser(description="评估蒸馏效果")
    parser.add_argument("--student-file", "-s", required=True, help="Student 结果文件 (JSONL)")
    parser.add_argument("--teacher-file", "-t", required=True, help="Teacher 结果文件 (JSONL)")
    parser.add_argument("--use-llm-judge", "-j", action="store_true", help="使用 LLM-as-Judge")
    parser.add_argument("--judge-type", default="glm", choices=["kimi", "glm", "deepseek"])
    args = parser.parse_args()

    # 加载数据
    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    student_data = load_jsonl(args.student_file)
    teacher_data = load_jsonl(args.teacher_file)

    # 提取答案
    predictions = [d.get("answer", d.get("response", "")) for d in student_data]
    references = [d.get("answer", d.get("response", "")) for d in teacher_data]

    # 基础指标
    print("\n📐 基础文本匹配指标:")
    metrics = compute_metrics(predictions, references)
    print_metrics(metrics)

    # LLM-as-Judge
    if args.use_llm_judge:
        import os
        from distill.teachers import create_teacher

        env_map = {"kimi": "KIMI_API_KEY", "glm": "GLM_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
        api_key = os.environ.get(env_map[args.judge_type], "")
        if api_key:
            judge_teacher = create_teacher(args.judge_type, api_key=api_key)
            judge = LLMJudge(judge_teacher)

            questions = [d.get("question", "") for d in student_data]
            summary = judge.evaluate_batch(questions, references, predictions)
            LLMJudge.print_summary(summary)
        else:
            print(f"⚠️ 未设置 {env_map[args.judge_type]}，跳过 LLM-as-Judge")


if __name__ == "__main__":
    main()
