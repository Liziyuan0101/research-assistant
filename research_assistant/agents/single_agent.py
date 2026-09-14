"""单 Agent + Skills 渐进披露运行时（取代 supervisor 编排，作为默认入口）。

设计
----
* **路由只有一跳**：``SkillRegistry.select()`` 按 skill frontmatter 的词元覆盖率打分，
  **0 次 LLM 调用**。
* **渐进披露**：常驻只有 frontmatter 索引；被选中的 skill 才加载正文（L1），
  正文里提到的 ``references/*.md`` 再按需加载（L2）。加载量由 ``SkillLoader`` 计量。
* **记忆是共享层**：检索 skill 会把长期记忆里的偏好回流为检索打分先验，
  执行完把本次问答写回记忆（含 trace 与 success）。
* **诚实降级**：任何一步不可用都返回 ``status='unavailable'/'needs_fields'`` 并给出原因，
  不返回编造内容。无 LLM 时实验/写作走项目原有的 ``_fallback_*`` 模板，并在
  ``llm_used=False`` 中如实标注。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from ..memory import MemoryStore, build_profile
from ..skills import SkillLoader, SkillMeta, SkillRegistry

logger = logging.getLogger(__name__)

#: skill 名 → 执行器方法名
_EXECUTORS = {
    'paper-retrieval': '_exec_retrieval',
    'experiment-design': '_exec_experiment',
    'academic-writing': '_exec_writing',
}

#: 写作 skill 需要的字段（缺失时要求用户补，不臆造）
_WRITING_FIELDS = ('title', 'keywords', 'background', 'methods', 'results', 'conclusion')

#: 链式编排的**依赖顺序**：检索必须最先（experiment / writing 依赖它的输出）。
#: 注意这与 ``registry.select`` 的相关度排序是两个不同的东西 ——
#: 相关度决定"要不要执行"，依赖顺序决定"按什么次序执行"。
_SKILL_DEPENDENCY_ORDER = {
    'paper-retrieval': 0,
    'experiment-design': 1,
    'academic-writing': 2,
}


class SingleAgent:
    """单 Agent 运行时。

    Args:
        config: 配置字典。
        retriever: ``HybridRetriever``（检索 skill 用）。
        memory: ``MemoryStore``（个性化 + 问答落库）。
        user_id: 用户标识。
        skills_dir: skill 目录，默认 ``<repo>/skills``。
        planner / writer / citation_manager: 注入的执行器依赖（可选）。
        prior_lambda: 偏好回流强度 λ。``0`` 表示关闭个性化。
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        retriever=None,
        memory: Optional[MemoryStore] = None,
        user_id: str = 'default',
        skills_dir: Optional[str] = None,
        planner=None,
        writer=None,
        citation_manager=None,
        verbose: bool = False,
        prior_lambda: float = 0.0,
        registry: Optional[SkillRegistry] = None,
        loader: Optional[SkillLoader] = None,
    ) -> None:
        self.config = config or {}
        self.retriever = retriever
        self.memory = memory
        self.user_id = user_id
        self.planner = planner
        self.writer = writer
        self.citation_manager = citation_manager
        self.verbose = verbose
        self.prior_lambda = float(prior_lambda)

        self.registry = registry or SkillRegistry(skills_dir)
        self.loader = loader or SkillLoader(self.registry)
        self.hops = 0
        self.history: List[Dict] = []

    # -------------------------------------------------------------- 路由
    def route(self, task: str, top_k: int = 3) -> List[SkillMeta]:
        """一次调用即完成路由（旧实现要消耗 1 次 LLM 调用做 supervisor 决策）。"""
        metas = self.registry.select(task, top_k=top_k)
        self.hops += 1
        return metas

    # -------------------------------------------------------------- 主入口
    def run(self, task: str, top_k: int = 5, chain: bool = False) -> Dict:
        """执行一条任务。

        Args:
            chain: ``True`` 时把命中的多个 skill 依次串起来（检索结果喂给下游），
                对齐原 ``task_type == 'complex'`` 的链式能力；默认只执行最相关的那个。
        """
        t0 = time.time()
        self.loader.reset_meter()
        self.hops = 0

        profile = None
        if self.memory is not None:
            try:
                profile = build_profile(self.memory, self.user_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning('build_profile failed: %s', exc)

        metas = self.route(task, top_k=3)
        if not metas:
            return self._result(task, [], [], [], profile, t0,
                                error='no skill matched')

        if chain:
            # 按**依赖顺序**执行（检索在前），而不是按相关度顺序 ——
            # 否则下游的 experiment/writing 会拿不到检索结果。
            to_exec = sorted(metas, key=lambda m: _SKILL_DEPENDENCY_ORDER.get(m.name, 99))
        else:
            to_exec = metas[:1]
        context: Dict[str, Any] = {'papers': [], 'plan': None, 'draft': None}
        steps: List[Dict] = []

        for meta in to_exec:
            body = self.loader.load(meta.name)          # L1：按需加载正文
            if body is not None and self.verbose:
                logger.info('loaded skill %s (%d chars)', meta.name, body.chars)
            step = self._dispatch(meta, task, context, top_k)
            steps.append(step)
            context.update(step.get('context') or {})

        ok = any(s.get('status') == 'ok' for s in steps)
        self._remember(task, steps, ok)
        return self._result(task, [m.name for m in metas], [m.name for m in to_exec],
                            steps, profile, t0)

    # ------------------------------------------------------------ 执行器
    def _dispatch(self, meta: SkillMeta, task: str, context: Dict, top_k: int) -> Dict:
        name = _EXECUTORS.get(meta.name)
        if name is None:
            return {'skill': meta.name, 'status': 'unavailable', 'reason': 'no executor bound'}
        return getattr(self, name)(task, context, top_k)

    def _exec_retrieval(self, task: str, context: Dict, top_k: int) -> Dict:
        """检索 skill：混合检索 + 偏好回流（个性化真正落地的地方）。"""
        if self.retriever is None:
            return {'skill': 'paper-retrieval', 'status': 'unavailable',
                    'reason': 'retriever not configured'}
        try:
            results = self.retriever.search(
                task, top_k=top_k, use_rerank=False, verbose=self.verbose,
                memory=self.memory, user_id=self.user_id,
                prior_lambda=self.prior_lambda or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning('retrieval failed: %s', exc)
            return {'skill': 'paper-retrieval', 'status': 'error', 'reason': str(exc)}

        backend = dict(getattr(self.retriever, 'last_search_meta', {}) or {})
        papers = [r.get('paper') or {} for r in results]
        return {
            'skill': 'paper-retrieval', 'status': 'ok',
            'backend': backend,
            'output': {'papers': papers, 'scores': [r.get('score') for r in results]},
            'context': {'papers': papers},
        }

    def _exec_experiment(self, task: str, context: Dict, top_k: int) -> Dict:
        """实验设计 skill。无 LLM 时走项目内置的 ``_fallback_plan`` 模板（如实标注）。"""
        if self.planner is None:
            return {'skill': 'experiment-design', 'status': 'unavailable',
                    'reason': 'planner not configured'}
        try:
            plan = self.planner.design_experiment(
                task, papers_context=context.get('papers') or None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning('experiment design failed: %s', exc)
            return {'skill': 'experiment-design', 'status': 'error', 'reason': str(exc)}
        llm_used = bool(getattr(self.planner, 'client', None))
        n_ctx = len(context.get('papers') or [])
        return {'skill': 'experiment-design', 'status': 'ok', 'llm_used': llm_used,
                'output': {'plan': plan, 'n_papers_context': n_ctx},
                'context': {'plan': plan}}

    def _exec_writing(self, task: str, context: Dict, top_k: int) -> Dict:
        """写作 skill。字段不全时**要求补全**，不臆造（对齐 SKILL.md 的契约）。"""
        if self.writer is None:
            return {'skill': 'academic-writing', 'status': 'unavailable',
                    'reason': 'writer not configured'}

        draft = context.get('draft')
        if draft:
            try:
                text = self.writer.polish_text(draft)
                return {'skill': 'academic-writing', 'status': 'ok',
                        'llm_used': bool(getattr(self.writer, 'client', None)),
                        'output': {'section': 'polish', 'text': text,
                                   'word_count': len(text)},
                        'context': {'draft': text}}
            except Exception as exc:  # noqa: BLE001
                return {'skill': 'academic-writing', 'status': 'error', 'reason': str(exc)}

        fields = context.get('writing_fields') or {}
        missing = [f for f in _WRITING_FIELDS if not fields.get(f)]
        if missing:
            return {'skill': 'academic-writing', 'status': 'needs_fields',
                    'reason': '缺少必要字段，需用户提供（不臆造内容）',
                    'required': list(_WRITING_FIELDS), 'missing': missing}
        try:
            text = self.writer.generate_abstract(
                title=fields['title'], keywords=fields['keywords'],
                background=fields['background'], methods=fields['methods'],
                results=fields['results'], conclusion=fields['conclusion'],
            )
            return {'skill': 'academic-writing', 'status': 'ok',
                    'llm_used': bool(getattr(self.writer, 'client', None)),
                    'output': {'section': 'abstract', 'text': text,
                               'word_count': len(text)},
                    'context': {'draft': text}}
        except Exception as exc:  # noqa: BLE001
            return {'skill': 'academic-writing', 'status': 'error', 'reason': str(exc)}

    # ------------------------------------------- 与旧接口对齐的直接调用
    def run_retrieval(self, query: str, top_k: int = 5) -> Dict:
        return self._exec_retrieval(query, {}, top_k)

    def run_experiment(self, task: str, papers: Optional[List[Dict]] = None) -> Dict:
        return self._exec_experiment(task, {'papers': papers or []}, 5)

    def run_writing(self, task: str, draft: str = '', fields: Optional[Dict] = None) -> Dict:
        return self._exec_writing(task, {'draft': draft, 'writing_fields': fields or {}}, 5)

    # -------------------------------------------------------------- 记忆
    def _remember(self, task: str, steps: List[Dict], ok: bool) -> None:
        if self.memory is None:
            return
        try:
            summary = json.dumps(
                [{k: s.get(k) for k in ('skill', 'status', 'reason')} for s in steps],
                ensure_ascii=False,
            )
            self.memory.add_qa(self.user_id, task, summary, success=ok,
                               trace=json.dumps([s.get('skill') for s in steps],
                                                ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.warning('add_qa failed: %s', exc)

    def memory_report(self) -> Dict:
        """当前用户的记忆画像摘要（供 CLI/API 展示）。"""
        if self.memory is None:
            return {'enabled': False}
        out = {'enabled': True, 'user_id': self.user_id,
               'stats': self.memory.stats(self.user_id)}
        try:
            out['profile'] = build_profile(self.memory, self.user_id).summary()
        except Exception as exc:  # noqa: BLE001
            out['profile_error'] = str(exc)
        return out

    # -------------------------------------------------------------- 汇总
    def _result(self, task, matched, executed, steps, profile, t0, error=None) -> Dict:
        routing = {
            'hops': self.hops,
            'llm_calls_for_routing': 0,
            'skills_matched': matched,
            'skills_executed': executed,
            'resident_index_chars': self.registry.resident_chars(),
            'loaded_chars': self.loader.loaded_chars,
            'load_by_level': self.loader.stats()['by_level'],
        }
        res = {
            'task': task,
            'routing': routing,
            'profile': profile.summary() if profile is not None else None,
            'steps': steps,
            'success': any(s.get('status') == 'ok' for s in steps),
            'elapsed_ms': round((time.time() - t0) * 1000, 1),
        }
        if error:
            res['error'] = error
        self.history.append(res)
        return res
