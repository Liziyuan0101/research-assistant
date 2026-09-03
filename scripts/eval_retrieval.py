"""
可复现的检索评测脚本。

用法:
    python scripts/eval_retrieval.py --mode bm25     # 仅 BM25(快)
    python scripts/eval_retrieval.py --mode hybrid   # BM25 + BGE 稠密(需 FlagEmbedding)
    python scripts/eval_retrieval.py --mode both     # 两者对比

无网络、可复现(数据已提交到仓库)。hybrid 模式需安装 FlagEmbedding/GPU。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_assistant.retrieval.evaluation import (
    evaluate_bm25_retrieval,
    evaluate_hybrid_retrieval,
)


def print_summary(label, summary):
    print(f"\n[{label}] {summary['total_queries']} 条 query")
    for metric, value in summary['metrics'].items():
        print(f"  {metric:14s}: {value:.4f}  ({value:.2%})")


def main():
    parser = argparse.ArgumentParser(description='可复现检索评测')
    parser.add_argument('--mode', choices=['bm25', 'hybrid', 'both'], default='bm25',
                        help='评测模式(默认 bm25)')
    args = parser.parse_args()

    corpus = str(ROOT / 'data' / 'eval' / 'corpus.json')
    eval_path = str(ROOT / 'data' / 'eval' / 'retrieval_eval.json')

    print('=' * 56)
    if args.mode in ('bm25', 'both'):
        print_summary('BM25', evaluate_bm25_retrieval(corpus, eval_path))
    if args.mode in ('hybrid', 'both'):
        print_summary('BM25+BGE(混合)', evaluate_hybrid_retrieval(corpus, eval_path))
    print('=' * 56)


if __name__ == '__main__':
    main()
