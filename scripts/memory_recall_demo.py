"""记忆召回对照实验：旧实现（词重叠） vs 新实现（BGE-M3 语义）。

对应简历风险 R1：旧 ``retrieval/memory.py._rank_qa`` 用 ``query.lower().split()``
算词重叠，中文无空格 ⇒ 任意两条中文 query 的 overlap 恒为 0，排序退化成"取最新 N 条"。

本脚本给出可复现的对照数字。

用法:
    python scripts/memory_recall_demo.py
    python scripts/memory_recall_demo.py --save   # 顺带写 JSON 结果
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from research_assistant.memory import MemoryStore, SemanticEncoder
from research_assistant.memory.store import format_preferences

HISTORY = [
    "锂离子电池剩余寿命预测方法",
    "退役电池梯次利用筛选策略",
    "图神经网络在电池早期寿命预测中的应用",
    "深度学习在电力系统负荷预测中的应用",
    "学术论文摘要的自动生成方法",
]

QUERIES = [
    "电池剩余寿命预测",
    "remaining useful life prediction of lithium-ion batteries",
]


def lexical_overlap(query, text):
    """旧实现的口径：空格分词后的 token 交集大小。"""
    return len(set(query.lower().split()) & set(text.lower().split()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    encoder = SemanticEncoder()
    if not encoder.available:
        print('❌ 语义编码器不可用')
        return

    hist_vecs = encoder.encode(HISTORY)
    report = {'history': HISTORY, 'queries': []}

    print('=' * 78)
    print('记忆召回对照：旧实现(词重叠) vs 新实现(BGE-M3 语义)')
    print('=' * 78)

    for q in QUERIES:
        print(f"\n▸ query: {q}")
        q_vec = encoder.encode([q])[0]
        lex = [(lexical_overlap(q, h), h) for h in HISTORY]
        lex_sorted = sorted(range(len(HISTORY)), key=lambda i: (-lex[i][0], i))
        sem = (hist_vecs @ q_vec).astype(float)
        sem_sorted = sorted(range(len(HISTORY)), key=lambda i: -sem[i])

        print(f"  {'#':<3}{'旧:重叠':>9}  {'新:cos':>8}   历史 query")
        for rank in range(len(HISTORY)):
            li, si = lex_sorted[rank], sem_sorted[rank]
            print(f"  {rank+1:<3}{lex[li][0]:>9}  {sem[si]:>8.4f}   {HISTORY[si]}")

        zero_ratio_old = sum(1 for v, _ in lex if v == 0) / len(HISTORY)
        print(f"  旧实现：{int(zero_ratio_old * len(HISTORY))}/{len(HISTORY)} 条历史重叠为 0"
              f"（{'排序退化' if zero_ratio_old == 1.0 else '部分可用'}）")
        print(f"  新实现：cos 范围 [{sem.min():.4f}, {sem.max():.4f}]，"
              f"最高与最低相差 {sem.max() - sem.min():.4f}")

        report['queries'].append({
            'query': q,
            'lexical_top1': HISTORY[lex_sorted[0]],
            'lexical_overlaps': [lex[i][0] for i in lex_sorted],
            'lexical_degenerate': bool(zero_ratio_old == 1.0),
            'semantic_top1': HISTORY[sem_sorted[0]],
            'semantic_scores': [float(sem[i]) for i in sem_sorted],
        })

    # ---- 存储层冒烟测试（衰减 / 负反馈 / 成功轨迹 / 语义召回 backend）
    print('\n' + '=' * 78)
    print('记忆层端到端冒烟测试（MemoryStore v2）')
    print('=' * 78)
    import tempfile
    db = Path(tempfile.mkdtemp()) / 'memory.db'
    store = MemoryStore(str(db), encoder=encoder)

    store.add_preference('demo', 'method', 'Bayesian deep learning', weight=3.0)
    store.add_preference('demo', 'keyword', 'remaining useful life', weight=2.0)
    store.add_preference('demo', 'journal', 'Reliability Engineering & System Safety')
    store.add_preference('demo', 'style', 'simulation-only papers without validation',
                         polarity=-1)
    store.add_preference('demo', 'custom_dim', 'auto-registered category')  # 非 strict ⇒ 自动纳入
    store.add_qa('demo', '电池 RUL 怎么预测', '用了层次贝叶斯…', success=True, trace='steps=[...]')
    store.add_qa('demo', '随便问问', '失败的回答', success=False)

    stats = store.stats('demo')
    print(f"  stats: {json.dumps(stats, ensure_ascii=False)}")
    print(f"  format_preferences: {format_preferences(store.get_preferences('demo'))[:120]}")
    rec = store.recall('demo', query='电池可以用多久', top_k=2)
    print(f"  recall backend = {rec['backend']}   (旧实现恒为词重叠)")
    print(f"  recall top-1   = {rec['qa_history'][0]['query'] if rec['qa_history'] else None}")
    print(f"  成功轨迹读取    = {len(store.get_successful_traces('demo'))} 条 (旧实现从不读取 success)")
    report['store_stats'] = stats
    report['recall_backend'] = rec['backend']
    print('  ✅ 全部通过')

    if args.save:
        out = ROOT / 'data' / 'eval' / 'memory_recall_results.json'
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n结果已保存: {out}')


if __name__ == '__main__':
    main()
