"""个性化接线的集成测试：偏好真的作用到了**产品检索器**的打分上。

这是"偏好回流检索打分先验"这句声明的回归测试 —— 之前该逻辑只存在于评测脚本里，
产品路径 HybridRetriever 完全没有入口，本文件锁住新接线。

依赖 sentence-transformers + 本地 BAAI/bge-m3；不可用时整体 skip。
"""

import json
from pathlib import Path

import pytest

from research_assistant.memory import MemoryStore
from research_assistant.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'data' / 'eval' / 'corpus.json'


@pytest.fixture(scope='module')
def retriever(tmp_path_factory):
    from research_assistant.memory.encoder import SemanticEncoder
    if not SemanticEncoder().available:
        pytest.skip('semantic encoder unavailable')
    corpus = json.loads(CORPUS.read_text(encoding='utf-8'))
    papers = [{'id': p['id'], 'title': p['title'], 'authors': p.get('authors', []),
               'abstract': p.get('abstract', ''), 'source': p.get('source', 'arxiv')}
              for p in corpus]
    r = HybridRetriever(config={}, data_dir=str(tmp_path_factory.mktemp('ra')), verbose=False)
    r.add_papers(papers, verbose=False)
    return r


@pytest.fixture()
def memory(tmp_path):
    return MemoryStore(str(tmp_path / 'm.db'))


QUERY = 'remaining useful life prediction of lithium-ion batteries'


def test_dense_backend_is_available(retriever):
    """没有 FlagEmbedding 也必须能稠密检索（sentence-transformers 回退）。"""
    assert retriever.dense_available is True


def test_search_returns_real_hybrid_results(retriever):
    results = retriever.search(QUERY, top_k=5, use_rerank=False, verbose=False)
    assert len(results) == 5
    assert all(r.get('paper', {}).get('paper_id') for r in results)
    meta = retriever.last_search_meta
    assert meta['backend'] in ('bm25+dense', 'bm25', 'dense')
    assert meta['n_dense'] > 0


def test_lambda_zero_is_not_personalized(retriever, memory):
    memory.add_preference('u', 'method', 'Bayesian deep learning', weight=3.0)
    retriever.search(QUERY, top_k=5, use_rerank=False, verbose=False,
                     memory=memory, user_id='u', prior_lambda=0.0)
    meta = retriever.last_search_meta
    assert meta['personalized'] is False
    assert meta['prior_lambda'] == 0.0


def test_no_memory_is_not_personalized(retriever):
    retriever.search(QUERY, top_k=5, use_rerank=False, verbose=False, prior_lambda=0.3)
    assert retriever.last_search_meta['personalized'] is False


def test_empty_profile_is_not_personalized(retriever, memory):
    """没有任何偏好时不应假装个性化生效。"""
    retriever.search(QUERY, top_k=5, use_rerank=False, verbose=False,
                     memory=memory, user_id='nobody', prior_lambda=0.3)
    assert retriever.last_search_meta['personalized'] is False


def test_personalization_reranks_results(retriever, memory):
    """有偏好且 λ>0 时：personalized=True，且打分被重算到 [0,1]。"""
    for cat, val, w in [
        ('keyword', 'remaining useful life', 3.0),
        ('method', 'Bayesian deep learning for uncertainty quantification', 3.0),
        ('method', 'hierarchical Bayesian model for battery lifetime early prediction', 2.0),
        ('journal', 'Reliability Engineering & System Safety', 2.0),
    ]:
        memory.add_preference('u', cat, val, weight=w)

    base = retriever.search(QUERY, top_k=6, use_rerank=False, verbose=False,
                            prior_lambda=0.0)
    ids_base = [r['paper']['paper_id'] for r in base]

    pers = retriever.search(QUERY, top_k=6, use_rerank=False, verbose=False,
                            memory=memory, user_id='u', prior_lambda=0.25)
    ids_pers = [r['paper']['paper_id'] for r in pers]

    assert retriever.last_search_meta['personalized'] is True
    assert all(0.0 <= r['score'] <= 1.0 for r in pers)
    assert set(ids_pers) == set(ids_base)          # 同一批候选，只是顺序变了
    assert ids_pers != ids_base, '个性化必须先验改变排序，否则等于没接线'


def test_negative_preference_is_accepted(retriever, memory):
    memory.add_preference('u', 'style', 'purely empirical curve fitting', polarity=-1)
    memory.add_preference('u', 'method', 'Bayesian deep learning', weight=2.0)
    retriever.search(QUERY, top_k=5, use_rerank=False, verbose=False,
                     memory=memory, user_id='u', prior_lambda=0.2)
    assert retriever.last_search_meta['personalized'] is True


def test_no_index_returns_empty_with_reason(tmp_path):
    """未建索引时必须返回空 + 明确原因，而不是抛错或静默返回假结果。"""
    r = HybridRetriever(config={}, data_dir=str(tmp_path), verbose=False)
    assert r.search('anything', verbose=False) == []
    assert r.last_search_meta.get('reason') == 'no-index'


def test_rerank_unavailable_is_graceful(retriever):
    """装了 FlagEmbedding 但本地**没有** reranker 权重时：必须优雅跳过而不是抛
    OSError（离线环境无网可下），并如实上报 ``reranker_skipped``。

    这是真实踩过的 bug：守卫条件写成"FlagEmbedding 能不能 import"是错的，
    必须是"reranker **权重**能不能加载"。
    """
    results = retriever.search(QUERY, top_k=5, use_rerank=True, verbose=False)
    assert len(results) == 5
    meta = retriever.last_search_meta
    assert 'reranker_skipped' in meta
    if meta['reranker_skipped']:
        assert all(r['reranked'] is False for r in results), '跳过精排却标了 reranked'
    else:
        assert all(r['reranked'] is True for r in results), '精排过却没标 reranked'


def test_enriched_results_expose_observability_flags(retriever, memory):
    """结果里要能看出"这条是否被个性化影响"，否则线上无法诊断。"""
    memory.add_preference('u', 'method', 'Bayesian deep learning', weight=3.0)
    plain = retriever.search(QUERY, top_k=3, use_rerank=False, verbose=False,
                             prior_lambda=0.0)
    assert all(r['personalized'] is False for r in plain)

    pers = retriever.search(QUERY, top_k=3, use_rerank=False, verbose=False,
                            memory=memory, user_id='u', prior_lambda=0.25)
    assert all(r['personalized'] is True for r in pers)


def test_assistant_hybrid_search_wires_personalization(tmp_path):
    """ResearchAssistant.hybrid_search 必须把 memory/user_id/λ 透传给检索器。"""
    from research_assistant.assistant import ResearchAssistant
    from research_assistant.memory.encoder import SemanticEncoder
    if not SemanticEncoder().available:
        pytest.skip('semantic encoder unavailable')

    ra = ResearchAssistant(verbose=False, user_id='wiretest', prior_lambda=0.25)
    if ra.hybrid_retriever is None:
        pytest.skip('hybrid retriever unavailable')

    # 自包含：先把评测语料灌进索引（避免依赖磁盘上已有索引）
    corpus = json.loads(CORPUS.read_text(encoding='utf-8'))
    ra.index_papers([{'id': p['id'], 'title': p['title'], 'authors': p.get('authors', []),
                      'abstract': p.get('abstract', ''), 'source': p.get('source', 'arxiv')}
                     for p in corpus])
    ra.add_preference('method', 'Bayesian deep learning', weight=3.0)

    ra.hybrid_search(QUERY, top_k=5, use_rerank=False, personalize=False)
    assert ra.hybrid_retriever.last_search_meta['personalized'] is False

    ra.hybrid_search(QUERY, top_k=5, use_rerank=False)
    assert ra.hybrid_retriever.last_search_meta['personalized'] is True
    assert ra.hybrid_retriever.last_search_meta['prior_lambda'] == 0.25
