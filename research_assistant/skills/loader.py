"""Skill 加载器：渐进披露的"按需加载"那一半。

三级加载，每级只在需要时发生：

```
L0 常驻   frontmatter（SkillRegistry.index_text）      ← 始终在上下文
L1 正文   SKILL.md 的 markdown 体                       ← skill 被选中时
L2 参考   references/*.md                               ← 正文里被引用且确实需要时
```

``loaded_chars`` 会累计 L1+L2 实际进过上下文的字符数，用于度量 M1。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .registry import SkillRegistry

logger = logging.getLogger(__name__)

_FRONTMATTER_RE_START = '---'


@dataclass
class SkillBody:
    """一个已加载的 skill 正文。"""

    name: str
    body: str
    path: Optional[Path] = None
    frontmatter: Dict = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.body)

    @property
    def references_mentioned(self) -> List[str]:
        """正文里以 ``references/xxx.md`` 形式提到的资源。"""
        import re
        return sorted(set(re.findall(r'references/([\w\-.]+)', self.body)))


class SkillLoader:
    """按需加载 skill 正文与 references，并记录累计加载量。"""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry
        self._body_cache: Dict[str, SkillBody] = {}
        self._ref_cache: Dict[str, str] = {}
        self.loaded_chars = 0          # 累计进入上下文的字符数（L1+L2）
        self.load_events: List[Dict] = []

    # ------------------------------------------------------------ L1
    def load(self, name: str) -> Optional[SkillBody]:
        if name in self._body_cache:
            return self._body_cache[name]

        meta = self.registry.get(name)
        if meta is None or meta.path is None:
            logger.warning('skill not found: %s', name)
            return None
        text = meta.path.read_text(encoding='utf-8')
        body = text
        fm: Dict = {}
        if text.startswith(_FRONTMATTER_RE_START):
            parts = text.split(_FRONTMATTER_RE_START, 2)
            if len(parts) >= 3:
                body = parts[2].lstrip('\n')
                try:
                    import yaml
                    fm = yaml.safe_load(parts[1]) or {}
                except Exception:  # noqa: BLE001
                    fm = {}

        sb = SkillBody(name=name, body=body, path=meta.path, frontmatter=fm)
        self._body_cache[name] = sb
        self.loaded_chars += sb.chars
        self.load_events.append({'level': 'L1', 'skill': name, 'chars': sb.chars})
        return sb

    # ------------------------------------------------------------ L2
    def load_reference(self, name: str, rel_name: str) -> Optional[str]:
        key = f'{name}/{rel_name}'
        if key in self._ref_cache:
            return self._ref_cache[key]

        meta = self.registry.get(name)
        if meta is None or meta.path is None:
            return None
        path = meta.path.parent / 'references' / rel_name
        if not path.exists():
            logger.warning('reference not found: %s', path)
            return None
        text = path.read_text(encoding='utf-8')
        self._ref_cache[key] = text
        self.loaded_chars += len(text)
        self.load_events.append({'level': 'L2', 'skill': name, 'ref': rel_name,
                                 'chars': len(text)})
        return text

    def load_all_references(self, name: str) -> Dict[str, str]:
        """加载某 skill 的全部 references（仅在你确实需要全部时才调用）。"""
        meta = self.registry.get(name)
        if meta is None or meta.path is None:
            return {}
        out = {}
        for ref in meta.references:
            text = self.load_reference(name, ref)
            if text is not None:
                out[ref] = text
        return out

    # ------------------------------------------------------------ 度量
    def reset_meter(self) -> None:
        self.loaded_chars = 0
        self.load_events = []

    def stats(self) -> Dict:
        return {
            'resident_chars': self.registry.resident_chars(),
            'loaded_chars': self.loaded_chars,
            'n_load_events': len(self.load_events),
            'by_level': {
                lv: sum(e['chars'] for e in self.load_events if e['level'] == lv)
                for lv in ('L1', 'L2')
            },
        }
