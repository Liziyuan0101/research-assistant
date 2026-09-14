"""偏好类别的可配置 schema。

旧实现（``retrieval/memory.py``）::

    PREFERENCE_CATEGORIES = ('journal', 'keyword', 'method')
    if category not in PREFERENCE_CATEGORIES:
        raise ValueError(f"Unsupported preference category: {category}")

问题：无法表达实验设计习惯、写作风格、常用数据集、负面偏好等，且扩展需要改源码。

新实现把类别、半衰期、权重上限收进一个 dataclass，可来自 ``config.yaml``。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Mapping, Tuple

#: 默认类别（向后兼容旧的 3 类，并补充 dataset / style / venue 等常用维度）
DEFAULT_CATEGORIES: Tuple[str, ...] = (
    'journal',    # 偏好期刊 / 会议
    'keyword',    # 研究关键词
    'method',     # 偏好方法
    'dataset',    # 常用数据集
    'style',      # 写作风格偏好
)


@dataclass
class PreferenceSchema:
    """偏好类别与权重衰减策略。

    Args:
        categories: 允许的偏好类别。
        strict: ``True`` 时未知类别抛 :class:`ValueError`（旧行为）；
            ``False``（默认）时记录告警并自动纳入，便于从对话中归纳新类别。
        half_life_days: 权重半衰期。``weight * 0.5 ** (age_days / half_life_days)``。
        max_weight: 单个偏好的权重上限，防止高频偏好锁死画像。
        min_effective_weight: 低于该有效权重的偏好视为已失效，不再注入。
    """

    categories: Tuple[str, ...] = DEFAULT_CATEGORIES
    strict: bool = False
    half_life_days: float = 90.0
    max_weight: float = 10.0
    min_effective_weight: float = 0.05
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.categories = tuple(dict.fromkeys(self.categories))  # 去重保序
        if self.half_life_days <= 0:
            raise ValueError('half_life_days must be > 0')
        if self.max_weight <= 0:
            raise ValueError('max_weight must be > 0')

    def validate(self, category: str) -> bool:
        """返回类别是否合法（不抛异常，交由调用方决定 strict 行为）。"""
        return category in self.categories

    def add_category(self, category: str) -> None:
        if not self.validate(category):
            self.categories = self.categories + (category,)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['categories'] = list(self.categories)
        return d

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> 'PreferenceSchema':
        """从 ``config.yaml`` 的 ``memory:`` 段构造。

        例::

            memory:
              preference_categories: [journal, keyword, method, dataset]
              half_life_days: 60
              max_weight: 8
              strict_categories: false
        """
        cfg = dict(cfg or {})
        kwargs: Dict[str, Any] = {}
        if 'preference_categories' in cfg:
            kwargs['categories'] = tuple(cfg['preference_categories'])
        for src, dst in (
            ('half_life_days', 'half_life_days'),
            ('max_weight', 'max_weight'),
            ('min_effective_weight', 'min_effective_weight'),
            ('strict_categories', 'strict'),
        ):
            if src in cfg:
                kwargs[dst] = cfg[src]
        return cls(**kwargs)
