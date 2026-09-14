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
import tempfile
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


def run_crossdomain(corpus, eval_path, profiles_path, lams, target_domain='电池/能源'):
    """跨域消融：把查询按「目标域 vs 其他域」分组，看偏好先验**何时帮忙、何时帮倒忙**。

    这是原 4 条同域 query 做不到的实验：同域语料下画像对所有文档近似等权 → 没有区分度。
    """
    profiles = load_profiles(profiles_path)
    samples = json.loads(Path(eval_path).read_text(encoding='utf-8'))['samples']

    def domain_of(s):
        return (s.get('metadata') or {}).get('domain') or s.get('domain')

    groups = {
        f'target({target_domain})': [s for s in samples if domain_of(s) == target_domain],
        'other_domains': [s for s in samples if domain_of(s) != target_domain],
    }

    # 为每个子集写一个临时 eval 文件（不改库 API）
    sub_paths = {}
    for name, subs in groups.items():
        if not subs:
            continue
        p = Path(tempfile.gettempdir()) / f'eval_subset_{abs(hash(name))}.json'
        p.write_text(json.dumps({'samples': subs}, ensure_ascii=False), encoding='utf-8')
        sub_paths[name] = (str(p), len(subs))

    from research_assistant.memory.encoder import SemanticEncoder
    encoder = SemanticEncoder()
    if not encoder.available:
        print('❌ 语义编码器不可用（需 sentence-transformers + 本地 BAAI/bge-m3）')
        return
    n_docs = len(json.loads(Path(corpus).read_text(encoding='utf-8')))
    print(f'\n语料 {n_docs} 篇 | 查询分组: '
          + ', '.join(f'{k}={v[1]}' for k, v in sub_paths.items()))
    print(f'编码器: {encoder.model_name}')

    results = {}
    for mode_ in ('boost', 'tiebreak', 'expand'):
        for pname in profiles:
            prof = profiles.get(pname)
            if prof is None:
                continue
            for gname, (spath, _n) in sub_paths.items():
                row = {}
                for lam in lams:
                    s = evaluate_personalized_retrieval(
                        corpus, spath,
                        preferences=prof.get('positive', []),
                        negative_preferences=prof.get('negative', []),
                        lam=lam, encoder=encoder, prior_mode=mode_,
                    )
                    row[lam] = s['metrics']
                results[f'{mode_}:{pname}:{gname}'] = {
                    'prior_mode': mode_, 'profile': pname, 'group': gname,
                    'label': prof.get('label', ''),
                    'by_lam': {str(l): row[l] for l in lams},
                }

    # ---- P@5 汇总
    print('\n' + '=' * 92)
    print('P@5 消融 —— 跨域语料（同一语料/查询，仅变先验强度、机制与画像）')
    print('=' * 92)
    header = f"{'机制:画像:查询组':<42}" + ''.join(f"{'λ=' + format(l, '<4'):>9}" for l in lams) + f"{'Δ最佳':>9}"
    print(header)
    print('-' * len(header))
    for key, data in results.items():
        cells = ''.join(f"{data['by_lam'][str(l)].get('Precision@5', 0):>9.2%}" for l in lams)
        base = data['by_lam'][str(lams[0])].get('Precision@5', 0.0)
        best = max(data['by_lam'][str(l)].get('Precision@5', 0.0) for l in lams)
        print(f'{key:<42}{cells}{best - base:>+9.2%}')
    print('=' * len(header))

    print('\n完整指标:')
    keys = ['Precision@1', 'Precision@3', 'Precision@5', 'Recall@5', 'Hit@5', 'MRR']
    for key, data in results.items():
        print(f"\n--- {key}  ({data['label'][:40]})")
        print('    ' + f"{'λ':<6}" + ''.join(f'{k:>13}' for k in keys))
        for lam, m in data['by_lam'].items():
            print('    ' + f'{lam:<6}' + ''.join(f"{m.get(k, 0):>13.4f}" for k in keys))

    out = ROOT / 'data' / 'eval' / 'crossdomain_results.json'
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n结果已保存: {out}')


def main():
    parser = argparse.ArgumentParser(description='可复现检索评测')
    parser.add_argument('--mode', choices=['bm25', 'hybrid', 'both', 'personalized', 'crossdomain'],
                        default='bm25', help='评测模式(默认 bm25)')
    parser.add_argument('--lam', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3],
                        help='personalized/crossdomain 模式的 λ 列表')
    parser.add_argument('--profiles', nargs='+', default=['battery_research', 'off_domain_control'],
                        help='使用的画像名')
    parser.add_argument('--prior-mode', nargs='+', choices=['boost', 'expand', 'tiebreak'],
                        default=['boost', 'expand'], help='先验机制')
    parser.add_argument('--corpus', type=str, default=None,
                        help='语料 JSON（默认 data/eval/corpus.json）')
    parser.add_argument('--eval', dest='eval_path', type=str, default=None,
                        help='评测集 JSON（默认 data/eval/retrieval_eval.json）')
    args = parser.parse_args()

    corpus = args.corpus or str(ROOT / 'data' / 'eval' / 'corpus.json')
    eval_path = args.eval_path or str(ROOT / 'data' / 'eval' / 'retrieval_eval.json')
    profiles_path = ROOT / 'data' / 'eval' / 'preference_profiles.json'

    print('=' * 56)
    if args.mode == 'crossdomain':
        run_crossdomain(corpus, eval_path, profiles_path, args.lam)
    else:
        if args.mode in ('bm25', 'both'):
            print_summary('BM25', evaluate_bm25_retrieval(corpus, eval_path))
        if args.mode in ('hybrid', 'both'):
            print_summary('BM25+BGE(混合)', evaluate_hybrid_retrieval(corpus, eval_path))
        if args.mode == 'personalized':
            run_personalized(corpus, eval_path, profiles_path, args.profiles, args.lam,
                             args.prior_mode)
    print('=' * 56)


if __name__ == '__main__':
    main()
