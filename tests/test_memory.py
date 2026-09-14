"""MemoryStore 记忆池测试：偏好与问答的 add/recall 往返。"""

import pytest

from research_assistant.retrieval.memory import MemoryStore, format_preferences


@pytest.fixture
def store(tmp_path):
    return MemoryStore(str(tmp_path / "memory.db"))


def test_add_and_get_preferences_roundtrip(store):
    store.add_preference("u1", "journal", "Nature")
    store.add_preference("u1", "keyword", "lithium battery")
    store.add_preference("u1", "method", "deep learning")
    prefs = store.get_preferences("u1")
    assert prefs["journal"] == ["Nature"]
    assert prefs["keyword"] == ["lithium battery"]
    assert prefs["method"] == ["deep learning"]


def test_preferences_isolated_by_user(store):
    store.add_preference("u1", "journal", "Nature")
    store.add_preference("u2", "journal", "JACS")
    assert store.get_preferences("u1")["journal"] == ["Nature"]
    assert store.get_preferences("u2")["journal"] == ["JACS"]


def test_unknown_preference_category_autoregisters_by_default(store):
    """行为变更（v2）：未知类别默认**告警并自动纳入**，而非 `raise ValueError`。

    动机：原实现硬编码 ('journal','keyword','method')，导致"对话中自动归纳新偏好维度"
    无法实现；严格模式仍可通过 PreferenceSchema(strict=True) 恢复旧行为。
    """
    assert store.add_preference("u1", "bogus", "x") is True
    assert "bogus" in store.schema.categories


def test_invalid_preference_category_raises_in_strict_mode(tmp_path):
    from research_assistant.memory import MemoryStore, PreferenceSchema
    strict = MemoryStore(str(tmp_path / "strict.db"), schema=PreferenceSchema(strict=True))
    with pytest.raises(ValueError):
        strict.add_preference("u1", "bogus", "x")


def test_add_qa_and_recall_ranks_relevant(store):
    store.add_qa("u1", "lithium battery remaining useful life", "answer A", success=True)
    store.add_qa("u1", "graph neural network", "answer B", success=False)
    result = store.recall("u1", query="lithium battery prediction")
    assert result["qa_history"][0]["query"].startswith("lithium battery")


def test_recall_no_query_returns_most_recent(store):
    store.add_qa("u1", "first query", "A")
    store.add_qa("u1", "second query", "B")
    result = store.recall("u1")
    assert result["qa_history"][0]["query"] == "second query"


def test_format_preferences(store):
    prefs = {"journal": ["Nature"], "keyword": ["battery"], "method": ["deep learning"]}
    text = format_preferences(prefs)
    assert "Nature" in text
    assert "battery" in text
    assert "deep learning" in text


def test_format_preferences_empty(store):
    assert format_preferences({}) == ""
