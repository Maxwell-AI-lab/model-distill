#!/usr/bin/env python3
"""查看生成数据样例"""
import json
import sys

lines = open("data/distill_raw.jsonl").readlines()
print(f"当前已生成: {len(lines)} 条\n{'='*60}\n")

for idx in [0, 2]:
    if idx >= len(lines):
        break
    item = json.loads(lines[idx])
    print(f"{'='*60}")
    print(f"题目 ID: {item.get('task_id', '')}")
    print(f"来源: {item.get('source', '')} | 难度: {item.get('difficulty', '')}")
    print(f"{'='*60}\n")

    print("【题目】")
    print(item['prompt'][:400])
    print()

    best = item.get('best_teacher', '')
    resp = item.get('responses', {}).get(best, {})
    print(f"【Teacher: {best} | 模型: {resp.get('teacher_model', '')}】\n")

    if resp.get('plan'):
        print("─── 解题计划 ───")
        print(resp['plan'])
        print()

    if resp.get('boundary'):
        print("─── 边界分析 ───")
        print(resp['boundary'])
        print()

    if resp.get('code'):
        print("─── 代码实现 ───")
        print(resp['code'])
        print()

    if resp.get('complexity'):
        print("─── 复杂度 ───")
        print(resp['complexity'])
        print()

    print(f"{'='*60}\n")
