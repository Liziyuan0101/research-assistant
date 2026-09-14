"""可复现的检索评测脚本。

用法:
    python scripts/eval_retrieval.py --mode bm25        # 仅 BM25(快, 纯离线)
    python scripts/eval_retrieval.py --mode hybrid      # BM25 + BGE 稠密(需 FlagEmbedding)
    python scripts/eval_retrieval.py --mode both        # 两者对比
    python scripts/eval_retrieval.py --mode personalized --lam 0 0.1 0.2 0.3
                                                        # 偏好回流消融(需 sentence-transformers)
    python scripts/eval_retrieval.py --mode personalized --prior-mode expand

无网络、可复现(数据已提交到仓库)。hybrid 模式需安装 FlagEmbedding/GPU。
personalized 模式在 --lam 0 时不加载编码器，因此基线仍可纯离线复现。

prior-mode 说明:
    boost   偏好编码为向量, 以 (1-λ)·base + λ·prior 混合进文档分数
    expand  偏好并入查询(查询侧个性化), 用 BM25 重算
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_assistant.retrieval.evaluation import (
    evaluate_bm25_retrieval,
    evaluate_hybrid_retrieval,
    evaluate_personalized_retrieval,
)

METRIC_KEYS = ['Precision@1', 'Precision@3', 'Precision@5', 'Recall@5', 'Hit@5', 'MRR']


def print_summary(label, summary):
    print(f"\n[{label}] {summary['total_queries']} 条 query")
    for metric, value in summary['metrics'].items():
        print(f"  {metric:14s}: {value:.4f}  ({value:.2%})")


def load_profiles(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)['profiles']


def run_personalized(corpus, eval_path, profiles_path, profile_names, lams, prior_modes):
    """偏好回流消融：同一 corpus / query，只变 λ / 画像 / 先验机制。"""
    profiles = load_profiles(profiles_path)

    from research_assistant.memory.encoder import SemanticEncoder
    encoder = SemanticEncoder()
    if not encoder.available:
        print('❌ 语义编码器不可用（需 sentence-transformers + 本地 BAAI/bge-m3）')
        return
    print(f'\n编码器就绪: {encoder.model_name}')

    results = {}
    for mode in prior_modes:
        for name in profile_names:
            if name not in profiles:
                print(f'  跳过未知画像: {name}')
                continue
            prof = profiles[name]
            by_lam = {}
            for lam in lams:
                s = evaluate_personalized_retrieval(
                    corpus, eval_path,
                    preferences=prof.get('positive', []),
                    negative_preferences=prof.get('negative', []),
                    lam=lam, encoder=encoder, prior_mode=mode,
                )
                by_lam[lam] = s['metrics']
            results[f'{mode}:{name}'] = {
                'prior_mode': mode,
                'label': prof.get('label', ''),
                'expectation': prof.get('expectation', ''),
                'by_lam': {str(l): by_lam[l] for l in lams},
            }

    # ---- P@5 汇总表
    print('\n' + '=' * 86)
    print('P@5 消融（同一 corpus / query，仅变先验强度与机制）')
    print('=' * 86)
    header = f"{'机制:画像':<34}" + ''.join(f"{'λ=' + format(l, '<4'):>10}" for l in lams) + f"{'best-λ0':>10}"
    print(header)
    print('-' * len(header))
    for key, data in results.items():
        cells = ''
        for l in lams:
            cells += f"{data['by_lam'][str(l)].get('Precision@5', 0):>10.2%}"
        base = data['by_lam'][str(lams[0])].get('Precision@5', 0.0)
        best = max(data['by_lam'][str(l)].get('Precision@5', 0.0) for l in lams)
        print(f"{key:<34}{cells}{best - base:>+10.2%}")
    print('=' * len(header))

    # ---- 完整指标
    print('\n完整指标:')
    for key, data in results.items():
        print(f"\n--- {key} :: {data['label']}")
        print(f"    预期: {data['expectation']}")
        print('    ' + f"{'λ':<6}" + ''.join(f"{k:>14}" for k in METRIC_KEYS))
        for lam, m in data['by_lam'].items():
            print('    ' + f"{lam:<6}" + ''.join(f"{m.get(k, 0):>14.4f}" for k in METRIC_KEYS))

    out = ROOT / 'data' / 'eval' / 'personalized_results.json'
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n结果已保存: {out}')


def main():
    parser = argparse.ArgumentParser(description='可复现检索评测')
    parser.add_argument('--mode', choices=['bm25', 'hybrid', 'both', 'personalized'],
                        default='bm25', help='评测模式(默认 bm25)')
    parser.add_argument('--lam', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3],
                        help='personalized 模式的 λ 列表(默认 0 0.1 0.2 0.3)')
    parser.add_argument('--profiles', nargs='+', default=['battery_research', 'off_domain_control'],
                        help='personalized 模式使用的画像名')
    parser.add_argument('--prior-mode', nargs='+', choices=['boost', 'expand'],
                        default=['boost', 'expand'], help='先验机制')
    args = parser.parse_args()

    corpus = str(ROOT / 'data' / 'eval' / 'corpus.json')
    eval_path = str(ROOT / 'data' / 'eval' / 'retrieval_eval.json')
    profiles_path = ROOT / 'data' / 'eval' / 'preference_profiles.json'

    print('=' * 56)
    if args.mode in ('bm25', 'both'):
        print_summary('BM25', evaluate_bm25_retrieval(corpus, eval_path))
    if args.mode in ('hybrid', 'both'):
        print_summary('BM25+BGE(混合)', evaluate_hybrid_retrieval(corpus, eval_path))
    if args.mode == 'personalized':
        run_personalized(corpus, eval_path, profiles_path, args.profiles, args.lam, args.prior_mode)
    print('=' * 56)


if __name__ == '__main__':
    main()
