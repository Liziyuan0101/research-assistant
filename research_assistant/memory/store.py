"""用户记忆存储：科研偏好 + 历史问答 + 成功轨迹。

相对旧实现（``retrieval/memory.py``）的 5 项升级
------------------------------------------------
1. 类别可配置（:class:`~research_assistant.memory.schema.PreferenceSchema`），
   未知类别默认告警并纳入而非 ``raise ValueError``。
2. 召回由"空格分词词重叠"改为 **BGE-M3 语义召回**（中文场景下词重叠恒为 0，
   旧实现实际退化成按时间取最新 N 条）。
3. ``weight`` 增加 **时间衰减 + 上限 + 负反馈（polarity）**。
4. 新增成功轨迹读取（旧实现存了 ``success`` 字段却从未读）。
5. 位置从 ``retrieval/`` 提升为顶层 ``memory/``，作为 skill 共享层。

数据库向后兼容：老库缺列时自动 ``ALTER TABLE`` 补齐，不会破坏已有数据。
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .encoder import SemanticEncoder
from .schema import PreferenceSchema

logger = logging.getLogger(__name__)

_CREATE_PREFERENCES = '''
    CREATE TABLE IF NOT EXISTS preferences (
        user_id    TEXT NOT NULL,
        category   TEXT NOT NULL,
        value      TEXT NOT NULL,
        weight     REAL DEFAULT 1.0,
        polarity   INTEGER DEFAULT 1,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, category, value)
    )
'''

_CREATE_QA = '''
    CREATE TABLE IF NOT EXISTS qa_history (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    TEXT NOT NULL,
        query      TEXT NOT NULL,
        answer     TEXT,
        success    INTEGER DEFAULT 1,
        trace      TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''


class MemoryStore:
    """按 ``user_id`` 存储的长期记忆。

    Args:
        db_path: SQLite 路径。
        schema: 偏好类别与衰减策略，见 :class:`PreferenceSchema`。
        encoder: 语义编码器；``None`` 时自动创建一个（可延迟加载）。
        semantic: 是否启用语义召回；编码器不可用时会自动回退到词重叠。
    """

    def __init__(
        self,
        db_path: str,
        schema: Optional[PreferenceSchema] = None,
        encoder: Optional[SemanticEncoder] = None,
        semantic: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema or PreferenceSchema()
        self.encoder = encoder if encoder is not None else SemanticEncoder()
        self.semantic = semantic
        self._init_db()

    # ------------------------------------------------------------- 初始化
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_CREATE_PREFERENCES)
            cur.execute(_CREATE_QA)
            self._migrate(cur)
            cur.execute('CREATE INDEX IF NOT EXISTS idx_prefs_user ON preferences(user_id)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_qa_user ON qa_history(user_id)')
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _migrate(cur: sqlite3.Cursor) -> None:
        """为老库补齐新列（``polarity`` / ``trace``）。"""
        def columns(table: str) -> set:
            cur.execute(f'PRAGMA table_info({table})')
            return {row[1] for row in cur.fetchall()}

        pref_cols = columns('preferences')
        if 'polarity' not in pref_cols:
            cur.execute('ALTER TABLE preferences ADD COLUMN polarity INTEGER DEFAULT 1')
            logger.info('memory: migrated preferences.polarity')

        qa_cols = columns('qa_history')
        if 'trace' not in qa_cols:
            cur.execute('ALTER TABLE qa_history ADD COLUMN trace TEXT')
            logger.info('memory: migrated qa_history.trace')

    # --------------------------------------------------------------- 偏好
    def add_preference(
        self,
        user_id: str,
        category: str,
        value: str,
        weight: float = 1.0,
        polarity: int = 1,
    ) -> bool:
        """添加/更新偏好。同 ``(user_id, category, value)`` 累加权重并刷新时间戳。

        Args:
            polarity: ``+1`` 正向偏好，``-1`` 负向偏好（如"不喜欢纯仿真论文"）。
        """
        if not self.schema.validate(category):
            if self.schema.strict:
                raise ValueError(f'Unsupported preference category: {category}')
            logger.warning('preference category %r not in schema; auto-registering', category)
            self.schema.add_category(category)

        if polarity not in (1, -1):
            raise ValueError('polarity must be +1 or -1')

        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                '''
                INSERT INTO preferences (user_id, category, value, weight, polarity, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, category, value)
                DO UPDATE SET weight     = MIN(preferences.weight + excluded.weight, ?),
                              polarity   = excluded.polarity,
                              updated_at = excluded.updated_at
                ''',
                (user_id, category, value, weight, polarity, now, self.schema.max_weight),
            )
            conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning('add_preference failed: %s', exc)
            return False
        finally:
            conn.close()

    def _decayed(self, weight: float, updated_at: Optional[str], as_of: Optional[datetime]) -> float:
        """``weight * 0.5 ** (age_days / half_life_days)``，并夹在 ``[0, max_weight]``。"""
        if not updated_at:
            return min(weight, self.schema.max_weight)
        try:
            ts = datetime.fromisoformat(str(updated_at))
        except ValueError:
            return min(weight, self.schema.max_weight)
        now = as_of or datetime.now()
        age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
        decay = math.pow(0.5, age_days / self.schema.half_life_days)
        return min(weight * decay, self.schema.max_weight)

    def get_weighted_preferences(
        self,
        user_id: str,
        as_of: Optional[datetime] = None,
        polarity: Optional[int] = 1,
        include_expired: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """返回 ``{category: {value: 有效权重}}``（已衰减），按权重降序。"""
        sql = 'SELECT category, value, weight, polarity, updated_at FROM preferences WHERE user_id = ?'
        params: List[Any] = [user_id]
        if polarity is not None:
            sql += ' AND polarity = ?'
            params.append(polarity)

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        out: Dict[str, Dict[str, float]] = {}
        for category, value, weight, _pol, updated_at in rows:
            eff = self._decayed(float(weight), updated_at, as_of)
            if not include_expired and eff < self.schema.min_effective_weight:
                continue
            out.setdefault(category, {})[value] = eff

        return {c: dict(sorted(v.items(), key=lambda kv: -kv[1])) for c, v in out.items()}

    def get_preferences(
        self,
        user_id: str,
        as_of: Optional[datetime] = None,
        include_negative: bool = False,
    ) -> Dict[str, List[str]]:
        """向后兼容旧接口：``{category: [value, ...]}``，按有效权重降序。

        Args:
            include_negative: ``True`` 时负向偏好以 ``'!value'`` 形式并入。
        """
        result: Dict[str, List[str]] = {c: [] for c in self.schema.categories}
        weighted = self.get_weighted_preferences(user_id, as_of=as_of, polarity=1)
        for category, values in weighted.items():
            result.setdefault(category, [])
            result[category].extend(values.keys())

        if include_negative:
            neg = self.get_weighted_preferences(user_id, as_of=as_of, polarity=-1)
            for category, values in neg.items():
                result.setdefault(category, [])
                result[category].extend(f'!{v}' for v in values)

        return result

    def preference_texts(
        self,
        user_id: str,
        as_of: Optional[datetime] = None,
        include_negative: bool = True,
    ) -> List[str]:
        """把偏好渲染成"可编码文本"列表（供语义召回/先验使用）。

        负向偏好前缀 ``'不偏好'``，与 :mod:`personalize` 的符号约定一致。
        """
        texts: List[str] = []
        for _, values in self.get_weighted_preferences(user_id, as_of=as_of, polarity=1).items():
            texts.extend(values.keys())
        if include_negative:
            for _, values in self.get_weighted_preferences(user_id, as_of=as_of, polarity=-1).items():
                texts.extend(f'不偏好 {v}' for v in values)
        return texts

    # --------------------------------------------------------------- 问答
    def add_qa(
        self,
        user_id: str,
        query: str,
        answer: str,
        success: bool = True,
        trace: Optional[str] = None,
    ) -> int:
        """记录一次问答，返回自增 id。``trace`` 可存工具调用轨迹/中间产物。"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                '''
                INSERT INTO qa_history (user_id, query, answer, success, trace, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (user_id, query, answer, 1 if success else 0, trace,
                 datetime.now().isoformat()),
            )
            conn.commit()
            return int(cur.lastrowid)
        except Exception as exc:  # noqa: BLE001
            logger.warning('add_qa failed: %s', exc)
            return -1
        finally:
            conn.close()

    def get_successful_traces(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """读取**成功**的问答轨迹。

        旧实现写入了 ``success`` 却从未读取 —— 成功轨迹是最廉价的 few-shot 样本，
        这里把它开放出来供 skill 复用（升级项 5）。
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                '''
                SELECT query, answer, trace, created_at FROM qa_history
                WHERE user_id = ? AND success = 1 AND answer IS NOT NULL
                ORDER BY created_at DESC LIMIT ?
                ''',
                (user_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            {'query': q, 'answer': a, 'trace': t, 'created_at': c}
            for q, a, t, c in rows
        ]

    # --------------------------------------------------------------- 召回
    def recall(
        self,
        user_id: str,
        query: Optional[str] = None,
        top_k: int = 5,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """返回 ``{'preferences': {...}, 'qa_history': [...], 'backend': 'semantic'|'lexical'}``。

        有 ``query`` 时按语义相似度（BGE-M3）排序历史；编码器不可用则回退到
        **词重叠**并在 ``backend`` 中如实标注，便于观测降级。
        """
        preferences = self.get_preferences(user_id, as_of=as_of)

        conn = self._connect()
        try:
            rows = conn.execute(
                '''
                SELECT query, answer, success, created_at FROM qa_history
                WHERE user_id = ? ORDER BY created_at DESC
                ''',
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

        qa = [
            {'query': r[0], 'answer': r[1], 'success': bool(r[2]), 'created_at': r[3]}
            for r in rows
        ]

        backend = 'lexical'
        if query is not None and qa:
            ranked = None
            if self.semantic:
                ranked = self._rank_qa_semantic(query, qa)
                if ranked is not None:
                    backend = 'semantic'
            if ranked is None:
                ranked = self._rank_qa_lexical(query, qa)
            qa = ranked[:top_k]
        else:
            qa = qa[:top_k]

        return {'preferences': preferences, 'qa_history': qa, 'backend': backend}

    def _rank_qa_semantic(self, query: str, qa: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """语义排序；编码器不可用时返回 ``None`` 触发回退。"""
        if not self.encoder.available:
            return None
        vecs = self.encoder.encode([query] + [item['query'] or '' for item in qa])
        if vecs is None or len(vecs) != len(qa) + 1:
            return None
        q_vec, hist = vecs[0], vecs[1:]
        sims = hist @ q_vec
        order = sorted(range(len(qa)), key=lambda i: (-float(sims[i]), qa[i]['created_at']))
        return [dict(qa[i], similarity=float(sims[i])) for i in order]

    @staticmethod
    def _rank_qa_lexical(query: str, qa: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """旧行为（词重叠）。保留是为了**可对照**，不是推荐路径。"""
        query_tokens = set(query.lower().split())
        scored = []
        for item in qa:
            hist_tokens = set((item['query'] or '').lower().split())
            scored.append((len(query_tokens & hist_tokens), item))
        scored.sort(key=lambda x: (-x[0], x[1]['created_at']))
        return [item for _, item in scored]

    # --------------------------------------------------------------- 统计
    def stats(self, user_id: str) -> Dict[str, Any]:
        """记忆画像统计，供评测指标 M7（记忆命中率）与调试使用。"""
        conn = self._connect()
        try:
            n_pos = conn.execute(
                'SELECT COUNT(*) FROM preferences WHERE user_id = ? AND polarity = 1', (user_id,)
            ).fetchone()[0]
            n_neg = conn.execute(
                'SELECT COUNT(*) FROM preferences WHERE user_id = ? AND polarity = -1', (user_id,)
            ).fetchone()[0]
            n_qa = conn.execute(
                'SELECT COUNT(*) FROM qa_history WHERE user_id = ?', (user_id,)
            ).fetchone()[0]
            n_ok = conn.execute(
                'SELECT COUNT(*) FROM qa_history WHERE user_id = ? AND success = 1', (user_id,)
            ).fetchone()[0]
        finally:
            conn.close()

        alive = sum(len(v) for v in self.get_weighted_preferences(user_id).values())
        return {
            'preferences_stored': n_pos + n_neg,
            'preferences_positive': n_pos,
            'preferences_negative': n_neg,
            'preferences_alive': alive,          # 衰减后仍在有效期内
            'qa_total': n_qa,
            'qa_success': n_ok,
            'categories': list(self.schema.categories),
        }


def format_preferences(preferences: Dict[str, Sequence[str]]) -> str:
    """把偏好渲染成可注入生成 prompt 的短文本（保持旧接口签名）。"""
    labels = {'journal': '期刊', 'keyword': '关键词', 'method': '方法',
              'dataset': '数据集', 'style': '写作风格'}
    parts = []
    for category, values in preferences.items():
        if not values:
            continue
        label = labels.get(category, category)
        parts.append(f"{label}: {', '.join(str(v) for v in values)}")
    if not parts:
        return ''
    return '用户科研偏好 — ' + '; '.join(parts)
