"""HybridRetriever BM25 检索测试(不加载 BGE/FAISS,GPU 无关)。"""

import pytest

from research_assistant.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def retriever(tmp_path):
    # data_dir 用临时目录,避免污染项目 data/
    return HybridRetriever(config={}, data_dir=str(tmp_path))


def test_tokenize_lowercases_and_splits(retriever):
    assert retriever._tokenize("Hello, World!") == ["hello", "world"]


def test_bm25_ranks_relevant_chunk_first(retriever):
    texts = [
        "lithium ion battery remaining useful life prediction",
        "graph neural network for molecular property prediction",
        "deep learning for computer vision image classification",
    ]
    retriever._build_bm25_index(texts, [1, 2, 3])

    results = retriever._bm25_search("battery remaining useful life", top_k=3)
    assert results, "BM25 应返回非空结果"
    assert results[0]["chunk_id"] == 1  # 电池相关文本应排第一
    assert results[0]["source"] == "bm25"
    assert results[0]["score"] > 0


def test_bm25_no_match_returns_empty(retriever):
    retriever._build_bm25_index(["battery capacity fade prediction"], [1])
    results = retriever._bm25_search("zzz qqq unrelated", top_k=3)
    assert results == []


def test_bm25_supports_multiple_results(retriever):
    retriever._build_bm25_index(
        [
            "battery state of health estimation",
            "battery capacity prediction",
            "graph neural network",
        ],
        [10, 20, 30],
    )
    results = retriever._bm25_search("battery", top_k=2)
    assert 1 <= len(results) <= 2
    assert all(r["source"] == "bm25" for r in results)
