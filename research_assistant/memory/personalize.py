"""偏好回流：把记忆变成**检索打分先验**。

这是本次重构的核心区分点。旧实现里偏好只被拼进 prompt（
``multi_agent.py:688`` 的"软个性化"）::

    pref_text = format_preferences(self.memory.get_preferences(self.user_id))
    task = f"{pref_text}\\n用户任务: {task}"

也就是说：标题写着"个性化检索"，**但检索器根本看不到偏好**。

本模块把用户偏好编码成向量，与候选文档算相似度，再以 ``λ`` 混合进检索分数::

    final = (1 - λ) * minmax(base_scores) + λ * minmax(pref_sim)

* ``λ = 0``  ⇒ 退化为原始检索（可作消融基线，保证可比）
* ``λ > 0``  ⇒ 偏好把"该用户更可能关心"的文档往前推

设计约束
--------
* **不引入新依赖**：复用 :class:`~research_assistant.memory.encoder.SemanticEncoder`，
  与 :class:`~research_assistant.retrieval.hybrid_retriever.HybridRetriever`
  共享同一份 BGE-M3 权重。
* **不做静默降级**：编码器不可用时返回 ``base_scores`` 原值并写日志，调用方可据此
  判断"零增益"到底来自 λ 太小还是先验根本没生效。
* **正负偏好分离**：负向偏好（``不偏好 X``）单独成一个向量，从分数里减掉，
  而不是混进正向均值里互相抵消。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .encoder import SemanticEncoder

logger = logging.getLogger(__name__)


def _minmax(x: np.ndarray) -> np.ndarray:
    """缩放到 ``[0, 1]``；全等值时返回全 0（避免除零）。"""
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@dataclass
class PreferenceProfile:
    """一个用户的偏好画像（向量形式）。"""

    user_id: str
    positive_texts: List[str] = field(default_factory=list)
    negative_texts: List[str] = field(default_factory=list)
    pos_vector: Optional[np.ndarray] = None
    neg_vector: Optional[np.ndarray] = None
    backend: str = 'none'

    @property
    def n_positive(self) -> int:
        return len(self.positive_texts)

    @property
    def is_empty(self) -> bool:
        return self.pos_vector is None and self.neg_vector is None

    def summary(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'n_positive': self.n_positive,
            'n_negative': len(self.negative_texts),
            'backend': self.backend,
            'is_empty': self.is_empty,
        }


def build_profile(
    store,
    user_id: str,
    encoder: Optional[SemanticEncoder] = None,
    include_negative: bool = True,
    as_of=None,
) -> PreferenceProfile:
    """从 :class:`MemoryStore` 构造偏好画像。

    Args:
        store: ``MemoryStore`` 实例（鸭子类型，只需 ``preference_texts`` / ``encoder``）。
        encoder: 显式编码器；``None`` 时用 ``store.encoder``。
    """
    encoder = encoder or getattr(store, 'encoder', None)
    texts = store.preference_texts(user_id, as_of=as_of, include_negative=include_negative)
    profile = PreferenceProfile(user_id=user_id)

    if not texts:
        profile.backend = 'empty'
        return profile
    if encoder is None or not encoder.available:
        profile.backend = 'unavailable'
        logger.warning('build_profile: encoder unavailable; prior will be a no-op')
        return profile

    pos = [t for t in texts if not t.startswith('不偏好')]
    neg = [t for t in texts if t.startswith('不偏好')]

    if pos:
        vecs = encoder.encode(pos)
        if vecs is not None and len(vecs):
            v = vecs.mean(axis=0)
            profile.pos_vector = v / (np.linalg.norm(v) + 1e-12)
            profile.positive_texts = pos
    if neg:
        vecs = encoder.encode(neg)
        if vecs is not None and len(vecs):
            v = vecs.mean(axis=0)
            profile.neg_vector = v / (np.linalg.norm(v) + 1e-12)
            profile.negative_texts = neg

    profile.backend = 'semantic' if not profile.is_empty else 'empty'
    return profile


def preference_prior(
    profile: PreferenceProfile,
    doc_texts: Sequence[str],
    encoder: SemanticEncoder,
    lam: float = 0.2,
    base_scores: Optional[Sequence[float]] = None,
    neg_lambda: Optional[float] = None,
) -> np.ndarray:
    """把偏好相似度混合进打分。

    Args:
        profile: :func:`build_profile` 的结果。
        doc_texts: 候选文档文本（与 ``base_scores`` 同序）。
        encoder: 编码器（须与建画像时同一份权重）。
        lam: 正向偏好权重 ``λ ∈ [0, 1]``。``0`` 表示不做个性化。
        base_scores: 原始检索分数；``None`` 时只返回偏好分数。
        neg_lambda: 负向偏好权重，默认 ``lam / 2``。

    Returns:
        与 ``doc_texts`` 等长的打分数组。
    """
    if base_scores is not None:
        base = np.asarray(base_scores, dtype='float32')
        base_norm = _minmax(base)
    else:
        base_norm = None

    if lam <= 0 or profile.is_empty:
        return base_norm if base_norm is not None else np.zeros(len(doc_texts), dtype='float32')

    vecs = encoder.encode(list(doc_texts))
    if vecs is None:
        logger.warning('preference_prior: encoder unavailable; returning base scores unchanged')
        return base_norm if base_norm is not None else np.zeros(len(doc_texts), dtype='float32')

    prior = np.zeros(len(doc_texts), dtype='float32')
    if profile.pos_vector is not None:
        prior += _minmax(vecs @ profile.pos_vector)

    nl = lam / 2 if neg_lambda is None else neg_lambda
    if nl > 0 and profile.neg_vector is not None:
        prior -= nl * _minmax(vecs @ profile.neg_vector)

    prior = _minmax(prior)
    if base_norm is None:
        return prior
    return (1.0 - lam) * base_norm + lam * prior


def boost_scores(*args, **kwargs) -> np.ndarray:
    """:func:`preference_prior` 的别名（语义化命名，便于检索器侧调用）。"""
    return preference_prior(*args, **kwargs)
