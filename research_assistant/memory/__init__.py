"""
记忆层（memory）：独立于检索的个性化层。

与旧实现（``retrieval/memory.py``）的区别：
1. **可配置偏好类别** —— 旧实现硬编码 ``('journal','keyword','method')``，
   传别的类别直接 ``raise ValueError``；现在通过 :class:`PreferenceSchema` 配置。
2. **语义召回** —— 旧实现用 ``query.lower().split()`` 做词重叠排序，中文无空格，
   重叠恒为 0，实际退化成"取最新 N 条"；现在复用项目已有的 BGE-M3 向量召回。
3. **时间衰减 + 权重上限 + 负反馈** —— 旧实现 ``weight`` 只增不减。
4. **偏好回流检索** —— 旧实现只把偏好拼进 prompt；现在生成打分先验喂给检索器。
5. **成功轨迹复用** —— 旧实现存了 ``success`` 字段却从未读取。

架构位置：本包为顶层模块（不在 ``retrieval/`` 下），记忆是被检索/实验/写作
三个 skill 共享的个性化层，而非检索的子模块。
"""

from .schema import DEFAULT_CATEGORIES, PreferenceSchema
from .encoder import SemanticEncoder
from .store import MemoryStore, format_preferences
from .personalize import PreferenceProfile, boost_scores, build_profile

__all__ = [
    'DEFAULT_CATEGORIES',
    'PreferenceSchema',
    'SemanticEncoder',
    'MemoryStore',
    'format_preferences',
    'PreferenceProfile',
    'boost_scores',
    'build_profile',
]
