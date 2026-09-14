"""端到端演示：单 Agent + Skills + 记忆个性化，真实运行（无需 LLM API key）。

跑：
    PYTHONPATH= HF_HUB_OFFLINE=1 <env>/python.exe scripts/demo_e2e.py

演示内容：
  0. 模块健康状态 + skills 列表（渐进披露的常驻/按需占用）
  1. 建索引（data/eval/corpus.json 的 23 篇）
  2. 无个性化混合检索
  3. 写入偏好 → 个性化混合检索（对比排序变化）
  4. Agent 任务路由：实验设计（无 LLM 时走内置模板，如实标注）
  5. Agent 任务路由：检索类任务（自动选中 paper-retrieval 且带个性化）
  6. 链式编排：一次任务串起 检索 → 实验设计
  7. 记忆画像 + 成功轨迹
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_assistant.assistant import ResearchAssistant

USER = 'demo_lzy'
CORPUS = ROOT / 'data' / 'eval' / 'corpus.json'
#: 独立数据目录 —— 让 demo 可重复运行（不污染 data/ 主索引，也不会重复追加 chunk）
DEMO_DIR = ROOT / 'data' / 'demo'


def hr(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def short(t, n=66):
    return (t or '?')[:n]


def main():
    # 幂等：每次跑都从干净的数据目录开始（否则 add_papers 会重复追加 chunk）
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    hr('0. 初始化（默认后端 = 单 Agent + Skills）')
    ra = ResearchAssistant(verbose=False, user_id=USER, prior_lambda=0.25,
                           data_dir=str(DEMO_DIR))
    for k, v in ra.health_report().items():
        print(f'  {k}: {v}')

    reg = ra.agent.registry
    loader = ra.agent.loader
    print(f'\n  skills ({len(reg.skills)} 个) — 渐进披露三级加载:')
    for s in reg.skills:
        print(f'    ▸ {s.name:<18} 正文 {s.body_chars:>5} 字符, references {len(s.references)} 个')
    print(f'  L0 常驻索引  : {reg.resident_chars()} 字符  ← 每请求都付')
    print(f'  L1/L2 已加载 : {loader.loaded_chars} 字符  ← 命中才付')

    hr('1. 建索引（真实混合检索：BM25 + BGE-M3 稠密）')
    corpus = json.loads(CORPUS.read_text(encoding='utf-8'))
    papers = [{'id': p['id'], 'title': p['title'], 'authors': p.get('authors', []),
               'abstract': p.get('abstract', ''), 'source': p.get('source', 'arxiv')}
              for p in corpus]
    n = ra.index_papers(papers)
    print(f'  语料 {len(papers)} 篇 → 建立 {n} 个 chunk')
    print(f'  稠密编码可用: {ra.hybrid_retriever.dense_available}')
    print(f'  索引统计: {ra.hybrid_retriever.get_stats()}')

    q = 'remaining useful life prediction of lithium-ion batteries'

    hr('2. 混合检索（无个性化，λ=0）')
    base = ra.hybrid_search(q, top_k=6, use_rerank=False, personalize=False)
    meta_a = dict(ra.hybrid_retriever.last_search_meta)
    print(f'  meta: {meta_a}')
    for i, r in enumerate(base, 1):
        print(f"    {i}. [{r.get('score', 0):.4f}] {short(r.get('paper', {}).get('title'))}")

    hr('3. 写入用户偏好 → 个性化检索（偏好回流打分先验）')
    prefs = [
        ('keyword', 'remaining useful life', 3.0, 1),
        ('method', 'Bayesian deep learning for uncertainty quantification', 3.0, 1),
        ('method', 'hierarchical Bayesian model for battery lifetime early prediction', 2.0, 1),
        ('journal', 'Reliability Engineering & System Safety', 2.0, 1),
        ('style', 'purely empirical curve fitting without physical interpretation', 1.0, -1),
    ]
    for cat, val, w, pol in prefs:
        ra.add_preference(cat, val, weight=w, polarity=pol)
        print(f"  {'✅' if pol > 0 else '🚫'} {cat} = {val[:58]}")

    pers = ra.hybrid_search(q, top_k=6, use_rerank=False)
    meta_b = dict(ra.hybrid_retriever.last_search_meta)
    print(f'\n  meta: {meta_b}')
    for i, r in enumerate(pers, 1):
        mark = '  ← 个性化' if r.get('personalized') else ''
        print(f"    {i}. [{r.get('score', 0):.4f}] {short(r.get('paper', {}).get('title'))}{mark}")

    ids_a = [r.get('paper', {}).get('paper_id') for r in base]
    ids_b = [r.get('paper', {}).get('paper_id') for r in pers]
    print(f"\n  排序是否变化: {ids_a != ids_b}")
    if ids_a != ids_b:
        for i, (a, b) in enumerate(zip(ids_a, ids_b), 1):
            if a != b:
                print(f'    位置 {i}: {a} → {b}')

    hr('4. Agent 路由：实验设计类任务')
    r4 = ra.run_agent('帮我设计一个消融实验，对比 PINN 和纯数据驱动模型的电池寿命预测效果')
    print(f"  命中/执行 skills: {r4['routing']['skills_matched']} / {r4['routing']['skills_executed']}")
    print(f"  路由跳数: {r4['routing']['hops']}  (LLM 调用 {r4['routing']['llm_calls_for_routing']})")
    print(f"  上下文: 常驻 {r4['routing']['resident_index_chars']} / 加载 {r4['routing']['loaded_chars']} 字符 {r4['routing']['load_by_level']}")
    for s in r4['steps']:
        print(f"  ─ {s['skill']}: {s['status']}  llm_used={s.get('llm_used')}")
        plan = (s.get('output') or {}).get('plan')
        if plan:
            print(f"      实验方案字段: {list(plan)[:8]}")

    hr('5. Agent 路由：检索类任务（自动带个性化）')
    r5 = ra.run_agent('帮我检索退役电池寿命预测的相关论文')
    print(f"  命中/执行 skills: {r5['routing']['skills_matched']} / {r5['routing']['skills_executed']}")
    for s in r5['steps']:
        print(f"  ─ {s['skill']}: {s['status']}")
        if s.get('backend'):
            print(f"      检索后端: {s['backend']}")
        out = s.get('output') or {}
        for i, p in enumerate((out.get('papers') or [])[:5], 1):
            print(f"      {i}. {short(p.get('title'), 58)}")

    hr('6. 链式编排（chain=True：按依赖顺序，检索结果喂给下游 skill）')
    r6 = ra.run_agent('检索电池早期寿命预测的论文，然后设计一个对比实验', chain=True)
    print(f"  命中 skills（按相关度）: {r6['routing']['skills_matched']}")
    print(f"  执行 skills（按依赖序）: {r6['routing']['skills_executed']}")
    for s in r6['steps']:
        out = s.get('output') or {}
        extra = ''
        if out.get('papers'):
            extra = f"  ({len(out['papers'])} 篇论文，来自上游检索)"
        if out.get('plan'):
            extra = f"  (实验方案已生成；上游检索到的论文 = {out.get('n_papers_context')} 篇)"
        print(f"  ─ {s['skill']}: {s['status']}{extra}")

    hr('7. 记忆画像与成功轨迹')
    print(json.dumps(ra.memory_report(), ensure_ascii=False, indent=2))
    traces = ra.memory.get_successful_traces(USER, limit=3)
    print(f'\n  成功轨迹 {len(traces)} 条（旧实现写入 success 却从不读取）:')
    for t in traces:
        print(f"    - {t['query'][:52]}  trace={t['trace']}")

    hr('完成')
    print('  全部流程真实执行完毕：无需 LLM API key（实验/写作走项目内置模板回退，已如实标注 llm_used）。')
    print(f'  数据目录: {DEMO_DIR}（每次运行前清空，保证可重复）')
    print(f'  记忆库  : {DEMO_DIR / "memory.db"}（user_id={USER}）')


if __name__ == '__main__':
    main()
