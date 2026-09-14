"""向后兼容 shim —— 记忆层已提升为顶层 :mod:`research_assistant.memory`。

本文件保留是为了不破坏既有导入（``multi_agent.py`` / ``assistant.py`` /
``tests/test_memory.py`` 都从 ``retrieval.memory`` 导入）。新代码请直接使用
``research_assistant.memory``。

迁移对照
--------
============================  ==========================================
旧（本文件所在地）             新
============================  ==========================================
``retrieval/memory.py``        ``memory/store.py``（存储 + 语义召回 + 衰减）
（无）                          ``memory/schema.py``（可配置偏好类别）
（无）                          ``memory/encoder.py``（BGE-M3 语义编码）
（无）                          ``memory/personalize.py``（偏好回流检索）
============================  ==========================================
"""

from ..memory.schema import DEFAULT_CATEGORIES, PreferenceSchema
from ..memory.store import MemoryStore, format_preferences

#: 旧模块级常量，保留以免外部引用断裂
PREFERENCE_CATEGORIES = DEFAULT_CATEGORIES

__all__ = [
    'MemoryStore',
    'format_preferences',
    'PREFERENCE_CATEGORIES',
    'PreferenceSchema',
    'DEFAULT_CATEGORIES',
]
