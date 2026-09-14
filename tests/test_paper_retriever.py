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
            # 必须一并指向 tmp，否则 __init__ 会在仓库里建 data/cache/search（测试污染）
            "search_cache_dir": str(tmp_path / "search_cache"),
            "max_results": 5,
            "download_pdf": False,
        }
    )


def test_cache_key_is_deterministic(retriever):
    """缓存键必须跨进程稳定。

    回归测试：原实现用内置 hash()，而 Python 对 str 默认按进程随机加盐 ——
    _save_to_cache 写的文件名，下一次进程里 load_from_cache 永远找不到，
    缓存从未生效（且在 data/papers 里堆了 18 个垃圾文件）。
    """
    import hashlib

    assert retriever._cache_key("battery rul") == hashlib.sha1(b"battery rul").hexdigest()[:16]
    # 归一化：大小写 / 多余空白不影响键
    assert retriever._cache_key("  Battery   RUL  ") == retriever._cache_key("battery rul")


def test_cache_roundtrip_within_dir(retriever):
    """写入后能读回（同进程），且文件落在 search_cache_dir 而不是 PDF 目录。"""
    papers = [{"id": "x", "title": "T"}]
    retriever._save_to_cache("some query", papers)
    assert retriever.load_from_cache("some query") == papers
    assert retriever.search_cache_dir != retriever.cache_dir
    assert not list(retriever.cache_dir.glob("search_*.json"))


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
