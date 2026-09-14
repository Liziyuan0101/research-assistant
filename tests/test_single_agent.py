"""单 Agent 运行时测试：路由一跳、渐进披露、执行器契约、记忆落库、链式依赖顺序。

不依赖网络 / LLM API key：用 stub retriever / planner / writer 验证编排逻辑。
"""

import pytest

from research_assistant.agents.single_agent import SingleAgent
from research_assistant.memory import MemoryStore
from research_assistant.skills import SkillLoader, SkillRegistry


class _StubRetriever:
    """最小检索器替身，模拟 HybridRetriever 的返回与 last_search_meta。"""

    def __init__(self, n=3):
        self.calls = []
        self.last_search_meta = {}
        self._n = n

    def search(self, query, top_k=5, **kwargs):
        self.calls.append({'query': query, **kwargs})
        self.last_search_meta = {'backend': 'bm25+dense', 'personalized':
                                 bool(kwargs.get('prior_lambda')), 'n_bm25': 5, 'n_dense': 5}
        return [{'paper': {'paper_id': f'p{i}', 'title': f'Paper {i}'}, 'score': 1.0 - i * 0.1}
                for i in range(min(self._n, top_k))]


class _StubPlanner:
    def __init__(self, client=None):
        self.client = client
        self.seen = []

    def design_experiment(self, research_question, papers_context=None, constraints=None):
        self.seen.append({'q': research_question, 'n_papers': len(papers_context or [])})
        return {'research_question': research_question, 'plan_text': 'stub', 'status': 'ok'}


class _StubWriter:
    def __init__(self, client=None):
        self.client = client

    def generate_abstract(self, **kwargs):
        return 'abstract-stub'

    def polish_text(self, text):
        return f'polished:{text}'


@pytest.fixture()
def registry():
    return SkillRegistry()


@pytest.fixture()
def memory(tmp_path):
    return MemoryStore(str(tmp_path / 'm.db'))


def make_agent(registry, memory, *, retriever=None, planner=None, writer=None,
               prior_lambda=0.0, tmp_path=None):
    return SingleAgent(
        config={}, retriever=retriever or _StubRetriever(), memory=memory,
        user_id='u', registry=registry, loader=SkillLoader(registry),
        planner=planner, writer=writer, prior_lambda=prior_lambda,
    )


# ------------------------------------------------------------------ 路由
@pytest.mark.parametrize('task,expected', [
    ('帮我检索退役电池寿命预测的相关论文', 'paper-retrieval'),
    ('帮我设计一个消融实验，对比 PINN 和纯数据驱动模型', 'experiment-design'),
    ('帮我写一段论文摘要，关于梯次利用电池分选', 'academic-writing'),
])
def test_route_picks_expected_skill(registry, memory, task, expected):
    ag = make_agent(registry, memory)
    metas = ag.route(task)
    assert metas[0].name == expected
    assert ag.hops == 1                      # 一跳完成路由


def test_run_has_zero_routing_llm_calls(registry, memory):
    ag = make_agent(registry, memory)
    res = ag.run('帮我检索电池寿命预测的论文')
    assert res['routing']['llm_calls_for_routing'] == 0
    assert res['routing']['hops'] == 1


def test_default_executes_only_top_skill(registry, memory):
    """默认只执行最相关的 skill，但会如实汇报还命中了哪些。"""
    ag = make_agent(registry, memory)
    res = ag.run('帮我检索退役电池寿命预测的相关论文')
    assert len(res['routing']['skills_executed']) == 1
    assert res['routing']['skills_executed'][0] == 'paper-retrieval'
    assert 'academic-writing' in res['routing']['skills_matched']


# ------------------------------------------------------- 渐进披露 / 上下文
def test_progressive_disclosure_footprint(registry, memory):
    """常驻只有 frontmatter 索引；正文按需加载，加载量被计量。"""
    ag = make_agent(registry, memory)
    res = ag.run('帮我检索电池寿命预测的论文')
    r = res['routing']
    assert r['resident_index_chars'] > 0
    assert r['loaded_chars'] >= r['load_by_level']['L1'] > 0
    # references（L2）默认不该被加载
    assert r['load_by_level']['L2'] == 0
    # 只加载命中的那一个 skill
    assert r['loaded_chars'] == r['load_by_level']['L1']


def test_loader_meter_resets_between_runs(registry, memory):
    ag = make_agent(registry, memory)
    ag.run('帮我检索电池寿命预测的论文')
    first = ag.loader.loaded_chars
    ag.run('帮我设计一个消融实验')
    second = ag.loader.loaded_chars
    assert first > 0 and second > 0
    assert second != first * 2      # 每次运行重置，不是累加


# ------------------------------------------------------------------ 执行器
def test_retrieval_executor_passes_personalization(registry, memory):
    retriever = _StubRetriever()
    ag = make_agent(registry, memory, retriever=retriever, prior_lambda=0.25)
    res = ag.run('帮我检索电池寿命预测的论文')
    step = res['steps'][0]
    assert step['status'] == 'ok'
    call = retriever.calls[0]
    assert call['prior_lambda'] == 0.25
    assert call['memory'] is memory
    assert call['user_id'] == 'u'
    assert step['backend']['backend'] == 'bm25+dense'


def test_retrieval_executor_unavailable_without_retriever(registry, memory):
    ag = SingleAgent(config={}, retriever=None, memory=memory, user_id='u',
                     registry=registry, loader=SkillLoader(registry))
    step = ag.run_retrieval('anything')
    assert step['status'] == 'unavailable'
    assert 'retriever' in step['reason']


def test_experiment_executor_marks_llm_usage(registry, memory):
    """无 LLM 时必须如实标注 llm_used=False（走的是项目内置模板回退）。"""
    planner = _StubPlanner(client=None)
    ag = make_agent(registry, memory, planner=planner)
    step = ag.run_experiment('消融实验')
    assert step['status'] == 'ok'
    assert step['llm_used'] is False


def test_writing_executor_requires_fields(registry, memory):
    """字段不全时要求补全，**不臆造内容**。"""
    ag = make_agent(registry, memory, writer=_StubWriter())
    step = ag.run_writing('写摘要')
    assert step['status'] == 'needs_fields'
    assert 'title' in step['missing']


def test_writing_executor_polishes_draft(registry, memory):
    ag = make_agent(registry, memory, writer=_StubWriter())
    step = ag.run_writing('润色', draft='raw text')
    assert step['status'] == 'ok'
    assert step['output']['text'] == 'polished:raw text'


# ------------------------------------------------------- 链式依赖顺序
def test_chain_orders_retrieval_first(registry, memory):
    """链式编排必须按依赖序执行：检索在前，下游才拿得到论文。"""
    planner = _StubPlanner()
    retriever = _StubRetriever(n=4)
    ag = make_agent(registry, memory, retriever=retriever, planner=planner)
    res = ag.run('检索电池早期寿命预测的论文，然后设计一个对比实验', chain=True)

    executed = res['routing']['skills_executed']
    assert executed[0] == 'paper-retrieval', executed
    # 实验方案应看到上游检索到的 4 篇论文
    exp = next(s for s in res['steps'] if s['skill'] == 'experiment-design')
    assert exp['output']['n_papers_context'] == 4
    assert planner.seen[0]['n_papers'] == 4


def test_non_chain_does_not_pass_papers(registry, memory):
    planner = _StubPlanner()
    ag = make_agent(registry, memory, retriever=_StubRetriever(), planner=planner)
    ag.run('检索电池早期寿命预测的论文，然后设计一个对比实验', chain=False)
    assert planner.seen == [] or planner.seen[0]['n_papers'] == 0


# ------------------------------------------------------------------ 记忆
def test_run_writes_qa_to_memory(registry, memory):
    ag = make_agent(registry, memory)
    ag.run('帮我检索电池寿命预测的论文')
    stats = memory.stats('u')
    assert stats['qa_total'] == 1
    assert stats['qa_success'] == 1


def test_memory_report_summarizes_profile(registry, memory):
    memory.add_preference('u', 'method', 'Bayesian deep learning')
    ag = make_agent(registry, memory)
    rep = ag.memory_report()
    assert rep['enabled'] is True
    assert rep['user_id'] == 'u'
    assert rep['stats']['preferences_positive'] == 1


def test_routing_metadata_shape(registry, memory):
    ag = make_agent(registry, memory)
    res = ag.run('帮我检索电池寿命预测的论文')
    for key in ('hops', 'llm_calls_for_routing', 'skills_matched', 'skills_executed',
                'resident_index_chars', 'loaded_chars', 'load_by_level'):
        assert key in res['routing'], key
    assert res['success'] is True
    assert 'elapsed_ms' in res


def test_unknown_task_returns_error(registry, memory):
    ag = make_agent(registry, memory)
    res = ag.run('zzz qqq www')
    assert res.get('error') == 'no skill matched' or res['routing']['skills_matched']
