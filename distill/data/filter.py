"""数据质量过滤器"""

import re
from typing import Optional


class QualityFilter:
    """蒸馏数据质量过滤

    多维度过滤低质量数据，保留高质量训练样本。
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.min_length = self.config.get("min_length", 10)
        self.max_length = self.config.get("max_length", 8192)
        self.min_answer_length = self.config.get("min_answer_length", 20)
        self.duplicate_threshold = self.config.get("duplicate_threshold", 0.9)

    def filter_single(self, item: dict) -> tuple[bool, str]:
        """过滤单条数据

        Returns:
            (是否通过, 原因)
        """
        question = item.get("question", "")
        answer = item.get("answer", "")

        # 空值检查
        if not question.strip() or not answer.strip():
            return False, "empty_content"

        # 长度检查
        if len(question) < self.min_length:
            return False, "question_too_short"
        if len(answer) < self.min_answer_length:
            return False, "answer_too_short"
        if len(answer) > self.max_length:
            return False, "answer_too_long"

        # 质量检查 — 拒绝模型拒绝回答
        reject_patterns = [
            r"我无法", r"我不能", r"I cannot", r"I can't",
            r"作为AI", r"作为一个人工智能",
            r"很抱歉", r"对不起，我",
        ]
        for pattern in reject_patterns:
            if re.search(pattern, answer[:100]):
                return False, "refusal_response"

        # 重复检查 — 问题或答案完全重复
        if question.strip() == answer.strip():
            return False, "identical_qa"

        return True, "pass"

    def filter_batch(self, data: list[dict]) -> list[dict]:
        """批量过滤"""
        results = []
        seen_questions = set()
        stats = {"pass": 0}
        rejected = 0

        for item in data:
            ok, reason = self.filter_single(item)
            if not ok:
                stats[reason] = stats.get(reason, 0) + 1
                rejected += 1
                continue

            # 去重
            q = item["question"].strip()
            if q in seen_questions:
                stats["duplicate"] = stats.get("duplicate", 0) + 1
                rejected += 1
                continue

            seen_questions.add(q)
            results.append(item)
            stats["pass"] += 1

        print(f"📊 过滤结果: {len(results)}/{len(data)} 通过 ({rejected} 条被过滤)")
        for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
            if reason != "pass":
                print(f"   ❌ {reason}: {count}")
        print(f"   ✅ pass: {stats['pass']}")

        return results
