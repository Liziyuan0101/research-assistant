"""Skill 注册表：扫描 ``skills/*/SKILL.md``，构建**常驻索引**并提供选择。

常驻索引 = 每个 skill 的 frontmatter（name/description/tools），
不含正文。这是渐进披露省下上下文的地方：正文与 references 只在命中后加载。

不依赖 LLM：选择用**词元重叠打分**完成，因此可在无 API key 的环境下完整测试与度量。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

# 触发词：中文短语以 '/' 分隔；英文按空格分词
_CN_STOP = set('的了和与及或在是为对把被和等一二三四五六七八九十这那你我他它')


def _tokenize(text: str) -> List[str]:
    """优先复用项目分词器（jieba），失败则退化为字符 bigram + 英文词。"""
    try:
        from ..utils.helpers import tokenize as _t
        toks = [t for t in _t(text) if t and t not in _CN_STOP]
        if toks:
            return toks
    except Exception:  # noqa: BLE001
        pass
    toks = re.findall(r'[A-Za-z0-9_]+', text.lower())
    cn = re.sub(r'[^\u4e00-\u9fff]', '', text)
    toks += [cn[i:i + 2] for i in range(max(len(cn) - 1, 0))]
    return [t for t in toks if t not in _CN_STOP]


@dataclass
class SkillMeta:
    """一个 skill 的常驻元数据（不含正文）。"""

    name: str
    description: str = ''
    tools: List[str] = field(default_factory=list)
    version: str = ''
    path: Optional[Path] = None
    body_chars: int = 0
    references: List[str] = field(default_factory=list)

    @property
    def head_text(self) -> str:
        """参与选择的文本 = name + description + tools（**不含正文**）。"""
        return ' '.join([self.name, self.description, ' '.join(self.tools)])

    def to_index_line(self) -> str:
        return f'- **{self.name}**: {self.description}'

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'tools': self.tools,
            'version': self.version,
            'body_chars': self.body_chars,
            'references': self.references,
        }


class SkillRegistry:
    """扫描并索引 ``skills/`` 目录。"""

    def __init__(self, skills_dir: Optional[str | Path] = None) -> None:
        if skills_dir is None:
            skills_dir = Path(__file__).resolve().parent.parent.parent / 'skills'
        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, SkillMeta] = {}
        self.reload()

    # ------------------------------------------------------------ 扫描
    def reload(self) -> 'SkillRegistry':
        self._skills = {}
        if not self.skills_dir.exists():
            logger.warning('skills dir not found: %s', self.skills_dir)
            return self
        for sk_md in sorted(self.skills_dir.glob('*/SKILL.md')):
            try:
                meta = self._parse(sk_md)
            except Exception as exc:  # noqa: BLE001
                logger.warning('skip %s: %s', sk_md, exc)
                continue
            self._skills[meta.name] = meta
        return self

    @staticmethod
    def _parse(path: Path) -> SkillMeta:
        text = path.read_text(encoding='utf-8')
        m = _FRONTMATTER_RE.match(text)
        fm: Dict = {}
        body = text
        if m:
            raw, body = m.group(1), text[m.end():]
            try:
                import yaml
                fm = yaml.safe_load(raw) or {}
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f'bad YAML frontmatter: {exc}') from exc
        name = str(fm.get('name') or path.parent.name)
        tools = fm.get('tools') or []
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.strip('[]').split(',') if t.strip()]
        refs_dir = path.parent / 'references'
        refs = sorted(p.name for p in refs_dir.glob('*')) if refs_dir.exists() else []
        return SkillMeta(
            name=name,
            description=str(fm.get('description') or '').strip(),
            tools=[str(t) for t in tools],
            version=str(fm.get('version') or ''),
            path=path,
            body_chars=len(body),
            references=refs,
        )

    # ------------------------------------------------------------ 索引
    @property
    def skills(self) -> List[SkillMeta]:
        return list(self._skills.values())

    def get(self, name: str) -> Optional[SkillMeta]:
        return self._skills.get(name)

    def index_text(self) -> str:
        """常驻索引文本（供单 Agent 的 system prompt 使用）。"""
        return '\n'.join(s.to_index_line() for s in self.skills)

    def resident_chars(self) -> int:
        """常驻上下文字符数 = 各 skill frontmatter 摘要之和。"""
        return sum(len(s.to_index_line()) for s in self.skills)

    # ------------------------------------------------------------ 选择
    def score(self, query: str, meta: SkillMeta) -> float:
        q = set(_tokenize(query))
        if not q:
            return 0.0
        head = set(_tokenize(meta.head_text))
        inter = q & head
        if not inter:
            return 0.0
        # 覆盖率（避免长 description 靠词数取胜）
        return len(inter) / max(len(q), 1)

    def select(self, query: str, top_k: int = 1) -> List[SkillMeta]:
        """按词元覆盖率选 skill。**一次调用即完成路由（N→1 跳）**。"""
        scored = [(self.score(query, s), s) for s in self.skills]
        scored.sort(key=lambda x: -x[0])
        hits = [s for sc, s in scored if sc > 0]
        if not hits:  # 兜底：无命中时给按名称字典序的第一个，并在 trace 中标注
            logger.warning('no skill matched %r; falling back', query)
            return self.skills[:1] if self.skills else []
        return hits[:top_k]

    def stats(self) -> Dict:
        return {
            'n_skills': len(self._skills),
            'skills': [s.to_dict() for s in self.skills],
            'resident_chars': self.resident_chars(),
            'total_body_chars': sum(s.body_chars for s in self.skills),
        }
