"""PaperRetriever 纯函数测试(不触网、不加载 KeyBERT)。"""

import pytest

import research_assistant.retrieval.paper_retriever as pr_module
from research_assistant.retrieval.paper_retriever import PaperRetriever

# 跳过 KeyBERT(需下载模型),只测内置术语匹配与纯逻辑
pr_module.HAS_QUERY_ENHANCER = False


@pytest.fixture
def retriever(tmp_path):
    return PaperRetriever(
        {
            "cache_directory": str(tmp_path / "papers"),
            "max_results": 5,
            "download_pdf": False,
        }
    )


def test_reconstruct_abstract_reorders_words(retriever):
    inverted = {"world": [1], "hello": [0]}
    assert retriever._reconstruct_abstract(inverted) == "hello world"


def test_reconstruct_abstract_empty(retriever):
    assert retriever._reconstruct_abstract({}) == ""


def test_deduplicate_papers_by_title(retriever):
    papers = [
        {"title": "Same Title", "source": "arxiv"},
        {"title": "  same title  ", "source": "openalex"},  # 仅大小写/空白差异
        {"title": "Different Title", "source": "arxiv"},
    ]
    deduped = retriever._deduplicate_papers(papers)
    assert len(deduped) == 2
    titles = [p["title"] for p in deduped]
    assert "Different Title" in titles


def test_extract_keyphrases_matches_domain_terms(retriever):
    kp = retriever._extract_keyphrases_for_search(
        "remaining useful life of lithium-ion battery"
    )
    assert "remaining useful life" in kp
    assert "lithium-ion battery" in kp


def test_extract_keyphrases_falls_back_to_words(retriever):
    # 无领域术语命中时,退回按长度提取实词
    kp = retriever._extract_keyphrases_for_search("graph neural network analysis")
    assert len(kp) >= 1
    assert all(isinstance(k, str) for k in kp)
