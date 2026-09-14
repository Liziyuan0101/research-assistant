"""构造跨域评测集：语料 + 带标签的查询（规则构造，可复现，需人工抽查）。

为什么需要这个
--------------
原有评测集只有 **4 条 query / 23 篇同域语料**，导致两个问题：
1. `Precision@5` 的理论上限只有 60%（q3/q4 各只标 1 篇相关），任何更高的数字都不可能；
2. 语料**全为锂电池**，用户画像在"领域"这一维对全部文档近似等权 → 个性化先验**没有区分度**，
   所以测不出增益（§3.5）。

本脚本构造一个**跨域**语料（8 个领域 / 30 个主题 / 约 360 篇），使偏好先验第一次具备
可测的区分度，并让 P@k 的 5 篇上限问题消失。

相关性标签怎么来的（关键：避免循环论证）
----------------------------------------
**标签 = OpenAlex 的 `primary_topic.id` 是否等于该 query 的目标主题**，即"论文所属主题"
这一**分类学元数据**，而不是"论文文本里是否出现查询词"。检索系统看不到 topic id，
所以这不构成自我实现。
需要说明的残余偏差：查询文本由主题名人工改写而来，与论文文本仍存在词汇相关性，
因此绝对指标可能偏乐观；**结论应看对照与差分（λ 消融、同域 vs 离域画像），而非绝对值**。

用法：
    python scripts/build_crossdomain_eval.py            # 抓取并写文件
    python scripts/build_crossdomain_eval.py --dry-run  # 只打印计划
"""

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_CORPUS = ROOT / 'data' / 'eval' / 'corpus_crossdomain.json'
OUT_EVAL = ROOT / 'data' / 'eval' / 'retrieval_eval_crossdomain.json'
OUT_SPOTCHECK = ROOT / 'data' / 'eval' / 'crossdomain_spotcheck.md'

MAILTO = 'lzy17839229740@163.com'
UA = {'User-Agent': f'research-assistant-eval/0.1 (mailto:{MAILTO})'}
PER_TOPIC = 12
SEED = 20260914

# (OpenAlex topic id, 该主题的中文领域标签, 人工改写的自然语言查询)
TOPICS = [
    # ---- 电池 / 能源（用户本域）----
    ('T10663', '电池/能源', 'how to estimate the remaining useful life of lithium-ion batteries'),
    ('T10223', '电池/能源', 'control and optimization of microgrid energy storage systems'),
    ('T12737', '电池/能源', 'fault detection and diagnosis in photovoltaic power systems'),
    ('T10305', '电池/能源', 'stability analysis and optimization of smart power grids'),
    ('T10030', '电池/能源', 'electrocatalysts for hydrogen fuel cell energy conversion'),
    ('T10175', '电池/能源', 'design of DC-DC converters for electric vehicle charging'),
    # ---- 时序 / 预测 ----
    ('T11052', '时序预测', 'probabilistic forecasting of electricity load demand'),
    ('T11326', '时序预测', 'deep learning methods for stock market price forecasting'),
    ('T11344', '时序预测', 'traffic flow prediction with spatio-temporal deep learning'),
    ('T11512', '时序预测', 'anomaly detection in multivariate time series data'),
    # ---- NLP / LLM ----
    ('T10181', 'NLP/LLM', 'large language model fine-tuning for downstream NLP tasks'),
    ('T10664', 'NLP/LLM', 'sentiment analysis of user reviews and social media text'),
    ('T13910', 'NLP/LLM', 'computational text analysis with transformer models'),
    ('T13523', 'NLP/LLM', 'retrieval augmented generation for open domain question answering'),
    # ---- 检索 / 推荐 ----
    ('T10203', '检索推荐', 'collaborative filtering for personalized recommender systems'),
    # ---- 图模型 / 通用机器学习 ----
    ('T11273', '图模型/ML', 'graph neural networks for node classification and link prediction'),
    ('T10036', '图模型/ML', 'novel neural network architectures for representation learning'),
    ('T10462', '图模型/ML', 'reinforcement learning for robot control and manipulation'),
    # ---- 视觉 ----
    ('T12549', '视觉', 'real-time object detection in natural images'),
    ('T10052', '视觉', 'deep learning based medical image segmentation'),
    ('T10191', '视觉', 'point cloud processing for 3D scene understanding'),
    # ---- 材料 / 生命健康 ----
    ('T11948', '材料/健康', 'machine learning for predicting material properties'),
    ('T10534', '材料/健康', 'structural health monitoring of civil infrastructure'),
    ('T10044', '材料/健康', 'protein structure prediction and molecular dynamics'),
    # ---- 工业 / 供应链 ----
    ('T10876', '工业/供应链', 'industrial process fault detection and diagnosis'),
    ('T10328', '工业/供应链', 'demand forecasting and inventory optimization in supply chains'),
    # ---- 安全 / 隐私 ----
    ('T10764', '安全/隐私', 'federated learning for privacy-preserving machine learning'),
    ('T10400', '安全/隐私', 'intrusion detection in computer networks'),
    # ---- 语音 / 机器人 ----
    ('T10201', '语音/机器人', 'end-to-end automatic speech recognition'),
    ('T10653', '语音/机器人', 'learning-based robot manipulation and grasping'),
]


def get(url: str, retries: int = 4):
    last = None
    for i in range(retries):
        try:
            return json.loads(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=40).read())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 + 4 * i)
    raise RuntimeError(f'GET failed: {url}\n{last}')


def reconstruct_abstract(inv):
    """OpenAlex 用 inverted index 存摘要，需要还原成文本。"""
    if not inv:
        return ''
    pos = {}
    for word, positions in inv.items():
        for p in positions:
            pos[p] = word
    return ' '.join(pos[k] for k in sorted(pos))


def fetch_topic(topic_id: str, n: int, rng: random.Random):
    """抓取某主题下的论文。

    质量过滤（第一批抽查发现 OpenAlex 的 topic 分配有噪声，例如把
    "Air Conditioning Repairs Mosman" 归入光伏故障检测）：
      * ``type:article`` + ``language:en``  —— 排除非论文与非英文
      * 按被引排序取前 50 篇作为池子 —— 排除刚入库的未审校条目
      * 丢弃 ``primary_topic.score < 0.55`` —— 只留主题归属较有把握的
      * 摘要至少 300 字符
    """
    select = ('id,doi,title,publication_year,primary_topic,topics,cited_by_count,'
              'abstract_inverted_index,authorships')
    url = (f'https://api.openalex.org/works?per-page=50&mailto={MAILTO}'
           f'&select={select}'
           f'&filter=primary_topic.id:{topic_id},has_abstract:true,type:article,language:en'
           f'&sort=cited_by_count:desc')
    d = get(url)
    pool = []
    for w in d.get('results', []):
        pt = w.get('primary_topic') or {}
        if float(pt.get('score') or 0) < 0.55:
            continue
        abstract = reconstruct_abstract(w.get('abstract_inverted_index'))
        if not w.get('title') or len(abstract) < 300:
            continue
        pool.append({
            'id': w['id'].split('/')[-1],
            'doi': w.get('doi'),
            'title': w['title'],
            'authors': [a['author']['display_name'] for a in (w.get('authorships') or [])
                        if a.get('author')][:8],
            'abstract': abstract,
            'published': str(w.get('publication_year') or ''),
            'source': 'openalex',
            'cited_by_count': w.get('cited_by_count', 0),
            'primary_topic_id': topic_id,
            'primary_topic': pt.get('display_name'),
            'primary_topic_score': round(float(pt.get('score') or 0), 3),
        })
    rng.shuffle(pool)                     # 固定种子 → 可复现的抽样
    return pool[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--per-topic', type=int, default=PER_TOPIC)
    args = ap.parse_args()

    print(f'计划：{len(TOPICS)} 个主题 × {args.per_topic} 篇 ≈ {len(TOPICS) * args.per_topic} 篇语料')
    if args.dry_run:
        for tid, domain, q in TOPICS:
            print(f'  {tid:<8}{domain:<12}{q}')
        return

    rng = random.Random(SEED)
    corpus, queries = [], []
    for i, (tid, domain, query) in enumerate(TOPICS, 1):
        try:
            papers = fetch_topic(tid, args.per_topic, rng)
        except Exception as exc:  # noqa: BLE001
            print(f'  [{i:>2}/{len(TOPICS)}] {tid} FAILED: {exc}')
            continue
        corpus.extend(papers)
        queries.append({
            'query_id': f'q{i:02d}_{tid}',
            'query': query,
            'relevant_paper_ids': [p['id'] for p in papers],
            'source': 'openalex primary_topic (自动标签)',
            'difficulty': 'auto',
            'metadata': {'domain': domain, 'topic_id': tid},
        })
        print(f'  [{i:>2}/{len(TOPICS)}] {tid:<8}{domain:<12}{len(papers):>3} 篇  {query[:52]}')
        time.sleep(0.4)

    # 去重（同一篇可能被多个主题抓到？primary_topic 唯一，理论上不会）
    seen, dedup = set(), []
    for p in corpus:
        if p['id'] in seen:
            continue
        seen.add(p['id'])
        dedup.append(p)

    n_rel = [len(q['relevant_paper_ids']) for q in queries]
    print(f'\n语料 {len(dedup)} 篇（去重前 {len(corpus)}）| 查询 {len(queries)} 条')
    print(f'每条查询的相关文献数: min={min(n_rel)} max={max(n_rel)} '
          f'→ P@5 上限 = {min(1.0, min(n_rel) / 5):.0%}')

    OUT_CORPUS.write_text(json.dumps(dedup, ensure_ascii=False, indent=1), encoding='utf-8')
    OUT_EVAL.write_text(json.dumps({'samples': queries}, ensure_ascii=False, indent=1),
                        encoding='utf-8')
    print(f'已写: {OUT_CORPUS}\n      {OUT_EVAL}')

    # 抽查文件：人工核对标签是否正确
    lines = [
        '# 跨域评测集抽查',
        '',
        '> 这是**自动构造**的评测集：语料来自 OpenAlex，相关性标签 = 论文的 `primary_topic`',
        '> 与查询目标主题一致。**请抽查下面的样本**：如果某条查询下列出的论文明显不属于该主题，',
        '> 说明 OpenAlex 的主题分配有噪声，需要人工剔除后再用。',
        '',
        f'- 语料：{len(dedup)} 篇 ｜ 查询：{len(queries)} 条 ｜ 生成种子：`{SEED}`',
        f'- 每条查询的相关文献数：{min(n_rel)}–{max(n_rel)}（P@5 上限 {min(1.0, min(n_rel)/5):.0%}）',
        '',
    ]
    for q in queries[:8]:
        md = q.get('metadata') or {}
        lines.append(f"## {q['query_id']} ｜ {md.get('domain')}")
        lines.append(f"**查询**: {q['query']}")
        lines.append(f"**主题**: {md.get('topic_id')} ｜ 相关文献 {len(q['relevant_paper_ids'])} 篇")
        lines.append('')
        for pid in q['relevant_paper_ids'][:8]:
            p = next((x for x in dedup if x['id'] == pid), None)
            if p:
                lines.append(f"- `{p['id']}` ({p['published']}, cites={p['cited_by_count']}, "
                             f"topic_score={p.get('primary_topic_score')}) {p['title'][:104]}")
        lines.append('')
    lines.append('---')
    lines.append('抽查完成后，把判定结果告诉我：需要剔除哪些 `id` 或哪些查询。')
    OUT_SPOTCHECK.write_text('\n'.join(lines), encoding='utf-8')
    print(f'抽查文件: {OUT_SPOTCHECK}')


if __name__ == '__main__':
    sys.exit(main())
