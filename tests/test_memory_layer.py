"""记忆层测试：语义召回、可配置类别、衰减、负反馈、成功轨迹。

这些测试针对简历风险 R1（"向量化知识库"与实现不符）的修复。
不依赖网络：编码器不可用时语义相关用例自动跳过。
"""

import math
from datetime import datetime, timedelta

import pytest

from research_assistant.memory import MemoryStore, PreferenceSchema, SemanticEncoder
from research_assistant.memory.store import format_preferences


@pytest.fixture(scope='module')
def encoder():
    enc = SemanticEncoder()
    if not enc.available:
        pytest.skip('semantic encoder unavailable (sentence-transformers / bge-m3)')
    return enc


@pytest.fixture()
def store(tmp_path, encoder):
    return MemoryStore(str(tmp_path / 'memory.db'), encoder=encoder)


# ------------------------------------------------------------------ R1 核心
def test_lexical_overlap_is_zero_for_chinese():
    """固化旧实现的缺陷：空格外词下中文 query 的重叠恒为 0。

    这是 R1 的证据 —— 旧 `_rank_qa` 在中文场景下退化成"按时间取最新 N 条"。
    """
    a = '电池剩余寿命预测'
    b = '锂离子电池剩余寿命预测方法'
    assert len(set(a.lower().split()) & set(b.lower().split())) == 0


def test_semantic_recall_ranks_related_first(store, encoder):
    """新实现：语义召回能把真正相关的历史排到前面。"""
    store.add_qa('u1', '锂离子电池剩余寿命预测方法', 'ans-1')
    store.add_qa('u1', '深度学习在电力系统负荷预测中的应用', 'ans-2')
    store.add_qa('u1', '学术论文摘要的自动生成方法', 'ans-3')

    rec = store.recall('u1', query='电池可以用多久', top_k=3)
    assert rec['backend'] == 'semantic'
    assert rec['qa_history'][0]['query'] == '锂离子电池剩余寿命预测方法'


def test_semantic_recall_is_cross_lingual(store, encoder):
    """跨语言召回：中文 query 能命中英文历史（词重叠方案必失败）。"""
    store.add_qa('u2', 'remaining useful life prediction of lithium-ion batteries', 'en')
    store.add_qa('u2', 'sentiment analysis of movie reviews', 'other')

    rec = store.recall('u2', query='电池寿命预测', top_k=2)
    assert rec['qa_history'][0]['query'].startswith('remaining useful life')


def test_recall_falls_back_to_lexical_when_encoder_missing(tmp_path):
    """编码器不可用时必须**如实标注**降级，而不是假装语义可用。"""
    class _DeadEncoder(SemanticEncoder):
        @property
        def available(self):
            return False

    s = MemoryStore(str(tmp_path / 'm.db'), encoder=_DeadEncoder())
    s.add_qa('u3', 'battery life prediction', 'a')
    rec = s.recall('u3', query='battery life', top_k=1)
    assert rec['backend'] == 'lexical'


# ------------------------------------------------------------------ schema
def test_unknown_category_autoregisters_by_default(tmp_path):
    s = MemoryStore(str(tmp_path / 'm.db'), encoder=None)
    assert s.add_preference('u', 'brand_new_dim', 'v') is True
    assert 'brand_new_dim' in s.schema.categories


def test_strict_schema_rejects_unknown_category(tmp_path):
    schema = PreferenceSchema(strict=True)
    s = MemoryStore(str(tmp_path / 'm.db'), schema=schema, encoder=None)
    with pytest.raises(ValueError):
        s.add_preference('u', 'nope', 'v')


def test_schema_from_config():
    schema = PreferenceSchema.from_config({
        'preference_categories': ['journal', 'method'],
        'half_life_days': 30,
        'max_weight': 5,
    })
    assert schema.categories == ('journal', 'method')
    assert schema.half_life_days == 30
    assert schema.max_weight == 5


# ------------------------------------------------------------------ 衰减
def test_time_decay_halves_weight_at_half_life(tmp_path, monkeypatch):
    schema = PreferenceSchema(half_life_days=30.0)
    s = MemoryStore(str(tmp_path / 'm.db'), schema=schema, encoder=None)
    s.add_preference('u', 'method', 'X', weight=4.0)

    now = datetime.now()
    as_of = now + timedelta(days=30)
    eff = s.get_weighted_preferences('u', as_of=as_of)['method']['X']
    assert eff == pytest.approx(2.0, rel=0.02)   # 半衰期处应减半


def test_expired_preference_is_dropped_from_plain_view(tmp_path):
    schema = PreferenceSchema(half_life_days=1.0, min_effective_weight=0.05)
    s = MemoryStore(str(tmp_path / 'm.db'), schema=schema, encoder=None)
    s.add_preference('u', 'method', 'OLD', weight=1.0)

    as_of = datetime.now() + timedelta(days=30)   # 30 个半衰期
    assert s.get_preferences('u', as_of=as_of)['method'] == []
    # 但仍在库中，可用 include_expired 取回
    assert 'OLD' in s.get_weighted_preferences('u', as_of=as_of, include_expired=True)['method']


def test_weight_is_capped(tmp_path):
    schema = PreferenceSchema(max_weight=3.0)
    s = MemoryStore(str(tmp_path / 'm.db'), schema=schema, encoder=None)
    for _ in range(10):
        s.add_preference('u', 'method', 'X', weight=1.0)
    eff = s.get_weighted_preferences('u')['method']['X']
    assert eff <= 3.0 + 1e-9


# ------------------------------------------------------------------ 负反馈
def test_negative_preference_is_excluded_from_positive_view(tmp_path):
    s = MemoryStore(str(tmp_path / 'm.db'), encoder=None)
    s.add_preference('u', 'journal', 'RESS', polarity=1)
    s.add_preference('u', 'style', 'simulation-only', polarity=-1)

    pos = s.get_preferences('u')
    assert 'RESS' in pos['journal']
    assert 'simulation-only' not in pos.get('style', [])

    neg = s.get_weighted_preferences('u', polarity=-1)
    assert 'simulation-only' in neg['style']


def test_preference_texts_marks_negative(tmp_path):
    s = MemoryStore(str(tmp_path / 'm.db'), encoder=None)
    s.add_preference('u', 'method', 'Bayesian', polarity=1)
    s.add_preference('u', 'method', 'curve fitting only', polarity=-1)
    texts = s.preference_texts('u')
    assert 'Bayesian' in texts
    assert any(t.startswith('不偏好') and 'curve fitting only' in t for t in texts)


# ------------------------------------------------------------------ 成功轨迹
def test_successful_traces_only_returns_successes(tmp_path):
    s = MemoryStore(str(tmp_path / 'm.db'), encoder=None)
    s.add_qa('u', 'q-ok', 'a', success=True, trace='steps=[1,2]')
    s.add_qa('u', 'q-bad', 'a', success=False)

    traces = s.get_successful_traces('u')
    assert [t['query'] for t in traces] == ['q-ok']
    assert traces[0]['trace'] == 'steps=[1,2]'


# ------------------------------------------------------------------ 迁移 / 统计
def test_migration_adds_columns_to_legacy_db(tmp_path):
    """老库（无 polarity / trace 列）应能自动升级。"""
    import sqlite3
    db = tmp_path / 'legacy.db'
    conn = sqlite3.connect(db)
    conn.execute('''CREATE TABLE preferences (
        user_id TEXT NOT NULL, category TEXT NOT NULL, value TEXT NOT NULL,
        weight REAL DEFAULT 1.0, updated_at TIMESTAMP, PRIMARY KEY (user_id, category, value))''')
    conn.execute('''CREATE TABLE qa_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, query TEXT NOT NULL,
        answer TEXT, success INTEGER DEFAULT 1, created_at TIMESTAMP)''')
    conn.execute("INSERT INTO preferences VALUES ('u','method','legacy',2.0,'2026-01-01T00:00:00')")
    conn.commit()
    conn.close()

    s = MemoryStore(str(db), encoder=None)
    # 老数据仍可读
    assert 'legacy' in s.get_weighted_preferences('u')['method']
    # 新列可写
    s.add_preference('u', 'method', 'new', polarity=-1)
    assert 'new' in s.get_weighted_preferences('u', polarity=-1)['method']


def test_stats_reflects_storage(tmp_path):
    s = MemoryStore(str(tmp_path / 'm.db'), encoder=None)
    s.add_preference('u', 'method', 'A')
    s.add_preference('u', 'style', 'B', polarity=-1)
    s.add_qa('u', 'q', 'a', success=True)
    s.add_qa('u', 'q2', 'a2', success=False)

    st = s.stats('u')
    assert st['preferences_positive'] == 1
    assert st['preferences_negative'] == 1
    assert st['qa_total'] == 2
    assert st['qa_success'] == 1


def test_format_preferences_includes_new_categories(tmp_path):
    s = MemoryStore(str(tmp_path / 'm.db'), encoder=None)
    s.add_preference('u', 'dataset', 'NASA B0005')
    text = format_preferences(s.get_preferences('u'))
    assert '数据集' in text and 'NASA B0005' in text


# ------------------------------------------------------------------ 向后兼容
def test_legacy_import_path_still_works():
    """`retrieval.memory` 现在是 shim，必须不破坏老导入。"""
    from research_assistant.retrieval.memory import (
        MemoryStore as LegacyStore, format_preferences as legacy_fmt, PREFERENCE_CATEGORIES,
    )
    assert LegacyStore is MemoryStore
    assert PREFERENCE_CATEGORIES == ('journal', 'keyword', 'method', 'dataset', 'style')
    assert legacy_fmt({'method': ['X']}) == '用户科研偏好 — 方法: X'
