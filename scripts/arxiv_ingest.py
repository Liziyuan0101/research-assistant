"""
arXiv 论文抓取脚本：初次抓取 + 增量更新。

用法:
    # 初次抓取:按分类拉最近 N 篇(相关度排序)
    python scripts/arxiv_ingest.py --categories cs.AI cs.LG eess.SY --max-results 100

    # 增量更新:只拉最近 1 天新提交的论文
    python scripts/arxiv_ingest.py --categories cs.AI --days 1

    # 增量更新:只拉某日期(含)之后新提交的论文
    python scripts/arxiv_ingest.py --categories cs.AI --since 2026-09-01

输出:data/papers/arxiv_corpus.json(论文元数据,含 id/title/authors/abstract/categories/pdf_url),
可直接喂给 `ResearchAssistant.index_papers()` 或 `hybrid_retriever.add_papers()` 建索引。
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import arxiv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / 'data' / 'papers' / 'arxiv_corpus.json'


def _to_dict(r: arxiv.Result) -> dict:
    return {
        'id': r.entry_id.split('/abs/')[-1],
        'title': r.title,
        'authors': [a.name for a in r.authors],
        'abstract': r.summary,
        'published': r.published.date().isoformat(),
        'categories': list(r.categories),
        'source': 'arxiv',
        'pdf_url': r.pdf_url,
    }


def fetch(categories, max_results, since_date=None):
    """抓取论文。since_date 非空时,按提交时间新→旧,只取该日期(含)之后的。"""
    client = arxiv.Client()
    papers = []
    for cat in categories:
        if since_date:
            search = arxiv.Search(query=f"cat:{cat}", max_results=max_results,
                                  sort_by=arxiv.SortCriterion.SubmittedDate)
            for r in client.results(search):
                if r.published.date() >= since_date:
                    papers.append(_to_dict(r))
                else:
                    break  # 已按时间新→旧排序,遇到更早的就停
        else:
            search = arxiv.Search(query=f"cat:{cat}", max_results=max_results,
                                  sort_by=arxiv.SortCriterion.Relevance)
            for r in client.results(search):
                papers.append(_to_dict(r))
    return papers


def main():
    parser = argparse.ArgumentParser(description='Fetch arXiv papers (initial or incremental)')
    parser.add_argument('--categories', nargs='+', default=['cs.AI', 'cs.LG', 'eess.SY'],
                        help='arXiv 分类(默认 cs.AI cs.LG eess.SY)')
    parser.add_argument('--max-results', type=int, default=100, help='每分类最多抓取篇数')
    parser.add_argument('--since', type=str, default=None,
                        help='增量:只取该日期(YYYY-MM-DD,含)之后新提交的')
    parser.add_argument('--days', type=int, default=None,
                        help='增量:只取最近 N 天新提交的(与 --since 二选一)')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    since_date = None
    if args.since:
        since_date = date.fromisoformat(args.since)
    elif args.days:
        since_date = date.today() - timedelta(days=args.days)

    papers = fetch(args.categories, args.max_results, since_date)

    seen = set()
    uniq = []
    for p in papers:
        if p['id'] not in seen:
            seen.add(p['id'])
            uniq.append(p)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(uniq, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"抓取 {len(uniq)} 篇论文 -> {out}")


if __name__ == '__main__':
    main()
