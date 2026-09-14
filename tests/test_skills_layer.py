"""Skill 层测试：注册、渐进披露、确定性路由。

这些测试固化"路由不再消耗 LLM 调用"这一重构主张。
"""

import pytest

from research_assistant.skills import SkillLoader, SkillRegistry


@pytest.fixture(scope='module')
def registry():
    return SkillRegistry()


def test_registry_finds_all_skills(registry):
    names = {s.name for s in registry.skills}
    assert names == {'paper-retrieval', 'experiment-design', 'academic-writing'}


def test_frontmatter_is_parsed(registry):
    meta = registry.get('paper-retrieval')
    assert meta is not None
    assert meta.description, 'description 必须非空 —— 它是路由的唯一依据'
    assert 'PaperRetrievalTool' in meta.tools
    assert meta.references, 'references 应被索引以便按需加载'


def test_resident_index_is_smaller_than_total_bodies(registry):
    """渐进披露的核心断言：常驻 < 总量。"""
    total_body = sum(s.body_chars for s in registry.skills)
    assert registry.resident_chars() < total_body
    # 常驻索引里不应出现正文内容
    assert '## 何时用我' not in registry.index_text()


@pytest.mark.parametrize('task,expected', [
    ('帮我检索一下退役电池寿命预测的相关论文', 'paper-retrieval'),
    ('search for recent papers on retrieval augmented generation', 'paper-retrieval'),
    ('帮我设计一个消融实验，对比 PINN 和纯数据驱动模型', 'experiment-design'),
    ('帮我写一段论文摘要，关于梯次利用电池分选', 'academic-writing'),
    ('polish this abstract into academic English', 'academic-writing'),
])
def test_select_picks_expected_skill(registry, task, expected):
    metas = registry.select(task, top_k=1)
    assert metas, f'no skill matched: {task}'
    assert metas[0].name == expected


def test_select_returns_multiple_for_mixed_task(registry):
    """复合任务应命中多个 skill（对齐原 task_type == 'complex' 的链式能力）。"""
    metas = registry.select('先检索相关论文，然后帮我写摘要', top_k=3)
    assert len(metas) >= 2
    assert 'paper-retrieval' in {m.name for m in metas}


def test_select_is_deterministic(registry):
    task = '协助检索电池相关文献'
    assert [m.name for m in registry.select(task)] == [m.name for m in registry.select(task)]


def test_loader_is_lazy(registry):
    """未加载前 loaded_chars 为 0。"""
    loader = SkillLoader(registry)
    assert loader.loaded_chars == 0
    assert loader.load_events == []


def test_loader_counts_body_then_reference(registry):
    loader = SkillLoader(registry)
    body = loader.load('paper-retrieval')
    assert body is not None and body.chars > 0
    after_body = loader.loaded_chars
    assert after_body == body.chars

    ref = loader.load_reference('paper-retrieval', 'evaluation.md')
    assert ref and 'P@5' in ref
    assert loader.loaded_chars > after_body

    # 重复加载应命中缓存，不重复计量
    loader.load('paper-retrieval')
    assert loader.loaded_chars == after_body + len(ref)


def test_loader_body_excludes_frontmatter(registry):
    loader = SkillLoader(registry)
    body = loader.load('paper-retrieval')
    assert not body.body.startswith('---')
    assert body.frontmatter.get('name') == 'paper-retrieval'


def test_loader_mentions_references(registry):
    loader = SkillLoader(registry)
    body = loader.load('paper-retrieval')
    assert 'evaluation.md' in body.references_mentioned


def test_reset_meter(registry):
    loader = SkillLoader(registry)
    loader.load('paper-retrieval')
    loader.reset_meter()
    assert loader.loaded_chars == 0


def test_registry_stats(registry):
    st = registry.stats()
    assert st['n_skills'] == 3
    assert st['resident_chars'] > 0
    assert st['total_body_chars'] > st['resident_chars']


def test_missing_skill_returns_none(registry):
    loader = SkillLoader(registry)
    assert loader.load('does-not-exist') is None
    assert loader.load_reference('paper-retrieval', 'nope.md') is None
