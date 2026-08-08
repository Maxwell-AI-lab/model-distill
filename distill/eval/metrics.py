"""评估指标"""

from typing import Optional


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """计算基础文本匹配指标

    Args:
        predictions: 模型预测列表
        references: 参考答案列表

    Returns:
        指标字典
    """
    if len(predictions) != len(references):
        raise ValueError(f"长度不匹配: pred={len(predictions)}, ref={len(references)}")

    results = {}

    # Exact Match
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    results["exact_match"] = exact_matches / len(references) if references else 0

    # 平均长度
    results["avg_pred_length"] = sum(len(p) for p in predictions) / len(predictions) if predictions else 0
    results["avg_ref_length"] = sum(len(r) for r in references) / len(references) if references else 0

    # ROUGE (中文用 rouge-chinese)
    try:
        from rouge_chinese import Rouge
        import jieba

        rouge = Rouge()
        scores = []
        for pred, ref in zip(predictions, references):
            pred_seg = " ".join(jieba.cut(pred.strip()))
            ref_seg = " ".join(jieba.cut(ref.strip()))
            if pred_seg.strip() and ref_seg.strip():
                score = rouge.get_scores(pred_seg, ref_seg)[0]
                scores.append(score)

        if scores:
            results["rouge_1"] = sum(s["rouge-1"]["f"] for s in scores) / len(scores)
            results["rouge_2"] = sum(s["rouge-2"]["f"] for s in scores) / len(scores)
            results["rouge_l"] = sum(s["rouge-l"]["f"] for s in scores) / len(scores)
    except ImportError:
        results["rouge_note"] = "Install rouge-chinese and jieba for ROUGE scores"

    # BLEU
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        smoothie = SmoothingFunction().method_1
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
            bleu_scores.append(bleu)
        results["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    except ImportError:
        results["bleu_note"] = "Install nltk for BLEU scores"

    return results


def print_metrics(metrics: dict):
    """格式化打印指标"""
    print("\n📊 评估结果:")
    print("=" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print("=" * 40)
