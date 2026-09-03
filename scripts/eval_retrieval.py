"""
可复现的检索评测脚本。

从 data/eval/corpus.json(真实论文语料)构建 BM25 索引,
对 data/eval/retrieval_eval.json(真实 query + 人工标注相关论文)评测,
打印 Precision@k / Recall@k / Hit@k / MRR。

用法:
    python scripts/eval_retrieval.py
    或
    research-assistant --eval

无网络、无 GPU、可复现(数据已提交到仓库)。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_assistant.retrieval.evaluation import evaluate_bm25_retrieval


def main():
    corpus = ROOT / 'data' / 'eval' / 'corpus.json'
    eval_path = ROOT / 'data' / 'eval' / 'retrieval_eval.json'
    summary = evaluate_bm25_retrieval(str(corpus), str(eval_path), k_values=(1, 3, 5))

    print('\n' + '=' * 56)
    print(f'可复现检索评测 (BM25) — {summary["total_queries"]} 条 query')
    print('=' * 56)
    for metric, value in summary['metrics'].items():
        print(f'  {metric:14s}: {value:.4f}  ({value:.2%})')
    print('=' * 56)
    return summary


if __name__ == '__main__':
    main()
