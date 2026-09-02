"""MetadataStore SQLite CRUD 测试(网络无关、GPU 无关)。"""

import pytest

from research_assistant.retrieval.hybrid_retriever import (
    MetadataStore,
    PaperMetadata,
    TextChunk,
)


@pytest.fixture
def store(tmp_path):
    # 注意:MetadataStore 每个方法各自建立连接,故必须用文件而非 ":memory:"
    return MetadataStore(str(tmp_path / "papers.db"))


def _paper(paper_id="p1", title="A Paper", authors=None, source="arxiv"):
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=authors or ["Alice", "Bob"],
        abstract="abstract text",
        published="2020",
        source=source,
    )


def test_add_and_get_paper_roundtrip(store):
    assert store.add_paper(_paper()) is True
    got = store.get_paper("p1")
    assert got is not None
    assert got.title == "A Paper"
    assert got.authors == ["Alice", "Bob"]
    assert got.source == "arxiv"


def test_get_missing_paper_returns_none(store):
    assert store.get_paper("does-not-exist") is None


def test_add_paper_upsert_replaces(store):
    store.add_paper(_paper(title="First"))
    store.add_paper(_paper(title="Second"))
    got = store.get_paper("p1")
    assert got.title == "Second"  # INSERT OR REPLACE 语义


def test_add_chunk_and_retrieve(store):
    cid = store.add_chunk(
        TextChunk(chunk_id=0, paper_id="p1", content="hello world", chunk_type="abstract")
    )
    assert cid > 0
    got = store.get_chunk_by_id(cid)
    assert got.content == "hello world"
    assert got.chunk_type == "abstract"


def test_add_chunk_duplicate_returns_zero(store):
    c = TextChunk(chunk_id=0, paper_id="p1", content="dup", chunk_type="abstract")
    first = store.add_chunk(c)
    second = store.add_chunk(c)
    assert first > 0
    assert second == 0  # 相同 paper_id + chunk_type + content 被跳过


def test_get_chunks_by_paper(store):
    store.add_chunk(TextChunk(chunk_id=0, paper_id="p1", content="a", chunk_type="abstract"))
    store.add_chunk(TextChunk(chunk_id=0, paper_id="p1", content="b", chunk_type="section"))
    store.add_chunk(TextChunk(chunk_id=0, paper_id="p2", content="c", chunk_type="abstract"))
    chunks = store.get_chunks_by_paper("p1")
    assert len(chunks) == 2


def test_search_papers_by_title(store):
    store.add_paper(_paper("p1", title="Battery RUL prediction"))
    store.add_paper(_paper("p2", title="Graph neural networks"))
    results = store.search_papers_by_title("battery")
    assert len(results) == 1
    assert results[0].paper_id == "p1"


def test_get_stats_counts(store):
    store.add_paper(_paper("p1"))
    store.add_paper(_paper("p2"))
    store.add_chunk(TextChunk(chunk_id=0, paper_id="p1", content="x", chunk_type="abstract"))
    stats = store.get_stats()
    assert stats["total_papers"] == 2
    assert stats["total_chunks"] == 1
