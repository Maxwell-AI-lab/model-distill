"""LLM-as-Judge — 用 Teacher 模型评估 Student 模型"""

from ..teachers.base import BaseTeacher


class LLMJudge:
    """用大模型作为裁判，评估蒸馏后小模型的质量"""

    def __init__(self, judge_model: BaseTeacher):
        self.judge = judge_model

    JUDGE_PROMPT = """你是一个专业的评估专家。请评估学生模型的回答质量。

【问题】{question}
【参考答案（Teacher）】{reference}
【学生答案（Student）】{prediction}

请从以下维度评分（1-10分）：

1. 准确性 (accuracy): 回答是否正确，有无事实错误
2. 完整性 (completeness): 是否覆盖了关键信息点
3. 连贯性 (coherence): 逻辑是否清晰，结构是否合理
4. 简洁性 (conciseness): 是否简洁有效，无冗余

请严格按以下 JSON 格式输出：
```json
{{"accuracy": 8, "completeness": 7, "coherence": 9, "conciseness": 8, "overall": 8, "comment": "简要评价"}}
```"""

    def evaluate_single(self, question: str, reference: str, prediction: str) -> dict:
        """评估单条"""
        prompt = self.JUDGE_PROMPT.format(
            question=question, reference=reference, prediction=prediction
        )

        try:
            text = self.judge.chat_simple(prompt, system="你是一个严格公正的评估专家。")
            # 提取 JSON
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            import json
            result = json.loads(text)
            return result
        except Exception as e:
            return {"error": str(e), "overall": 0}

    def evaluate_batch(self, questions: list[str], references: list[str], predictions: list[str]) -> dict:
        """批量评估

        Returns:
            {"avg_accuracy": x, "avg_completeness": x, ..., "details": [...]}
        """
        all_results = []
        for q, ref, pred in zip(questions, references, predictions):
            result = self.evaluate_single(q, ref, pred)
            result["question"] = q[:50] + "..." if len(q) > 50 else q
            all_results.append(result)

        # 统计平均分
        valid = [r for r in all_results if "error" not in r]
        summary = {}
        for dim in ["accuracy", "completeness", "coherence", "conciseness", "overall"]:
            scores = [r.get(dim, 0) for r in valid]
            summary[f"avg_{dim}"] = sum(scores) / len(scores) if scores else 0

        summary["total"] = len(all_results)
        summary["valid"] = len(valid)
        summary["details"] = all_results

        return summary

    @staticmethod
    def print_summary(summary: dict):
        """打印评估摘要"""
        print("\n🏆 LLM-as-Judge 评估结果:")
        print("=" * 50)
        for key in ["avg_accuracy", "avg_completeness", "avg_coherence", "avg_conciseness", "avg_overall"]:
            print(f"  {key:25s}: {summary.get(key, 0):.2f} / 10")
        print(f"  {'有效评估':25s}: {summary.get('valid', 0)}/{summary.get('total', 0)}")
        print("=" * 50)
