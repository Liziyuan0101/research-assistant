"""Skill 层：渐进披露（progressive disclosure）的 skill 注册与按需加载。

取代原先的 supervisor 路由：
* 旧设计 —— 1 个 supervisor LLM 调用决定把任务派给哪个 ReAct Agent，
  随后 agent 之间通过 ``AgentState`` 反复搬运上下文（≥2N 跳）。
* 新设计 —— skill 的 frontmatter 常驻上下文（约数十 token/个），
  单 Agent 用 frontmatter 的 description 做**一次**选择（N→1 跳），
  被选中后才加载正文，正文内引用的 references 再次按需加载。

本模块只做"选"与"取"，不含 LLM 调用，因此完全可离线测试与度量。
"""

from .registry import SkillMeta, SkillRegistry
from .loader import SkillBody, SkillLoader

__all__ = ['SkillMeta', 'SkillRegistry', 'SkillBody', 'SkillLoader']
