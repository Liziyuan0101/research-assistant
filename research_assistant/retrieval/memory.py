"""
User Memory Pool
用户记忆池：短期（会话）+ 长期（科研偏好、历史问答）

按 user_id 存储，用于个性化检索与生成。
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 偏好类别
PREFERENCE_CATEGORIES = ('journal', 'keyword', 'method')


class MemoryStore:
    """用户记忆存储：科研偏好 + 历史问答（SQLite，按 user_id）。"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                user_id    TEXT NOT NULL,
                category   TEXT NOT NULL,
                value      TEXT NOT NULL,
                weight     REAL DEFAULT 1.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, category, value)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                query      TEXT NOT NULL,
                answer     TEXT,
                success    INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prefs_user ON preferences(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_qa_user ON qa_history(user_id)')

        conn.commit()
        conn.close()

    def add_preference(self, user_id: str, category: str, value: str, weight: float = 1.0) -> bool:
        """添加/更新一条偏好（同 user_id+category+value 则累加 weight）。"""
        if category not in PREFERENCE_CATEGORIES:
            raise ValueError(f"Unsupported preference category: {category}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO preferences (user_id, category, value, weight, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, category, value)
                DO UPDATE SET weight = weight + excluded.weight,
                              updated_at = excluded.updated_at
            ''', (user_id, category, value, weight, datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            logger.warning("add_preference failed: %s", e)
            return False
        finally:
            conn.close()

    def get_preferences(self, user_id: str) -> Dict[str, List[str]]:
        """返回 {journal: [...], keyword: [...], method: [...]}。"""
        result = {c: [] for c in PREFERENCE_CATEGORIES}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT category, value FROM preferences
            WHERE user_id = ?
            ORDER BY weight DESC, updated_at DESC
        ''', (user_id,))
        for category, value in cursor.fetchall():
            if category in result and value not in result[category]:
                result[category].append(value)
        conn.close()
        return result

    def add_qa(self, user_id: str, query: str, answer: str, success: bool = True) -> int:
        """记录一次问答，返回 id。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO qa_history (user_id, query, answer, success, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, query, answer, 1 if success else 0, datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.warning("add_qa failed: %s", e)
            return -1
        finally:
            conn.close()

    def recall(self, user_id: str, query: Optional[str] = None, top_k: int = 5) -> Dict:
        """返回 {'preferences': {...}, 'qa_history': [...]}。

        若提供 query，则按与历史 query 的 token 重叠度排序后取 top_k。
        """
        preferences = self.get_preferences(user_id)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT query, answer, success, created_at FROM qa_history
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        rows = cursor.fetchall()
        conn.close()

        qa = [
            {'query': r[0], 'answer': r[1], 'success': bool(r[2]), 'created_at': r[3]}
            for r in rows
        ]
        if query is not None and qa:
            qa = self._rank_qa(query, qa)[:top_k]
        else:
            qa = qa[:top_k]
        return {'preferences': preferences, 'qa_history': qa}

    @staticmethod
    def _rank_qa(query: str, qa: List[Dict]) -> List[Dict]:
        """按 query 与历史 query 的 token 重叠度排序（简单关键词匹配，GPU 无关）。"""
        query_tokens = set(query.lower().split())
        scored = []
        for item in qa:
            hist_tokens = set(item['query'].lower().split())
            overlap = len(query_tokens & hist_tokens)
            scored.append((overlap, item))
        scored.sort(key=lambda x: (-x[0], x[1]['created_at']))
        return [item for _, item in scored]


def format_preferences(preferences: Dict[str, List[str]]) -> str:
    """把偏好渲染成可注入生成 prompt 的短文本。"""
    labels = {'journal': '期刊', 'keyword': '关键词', 'method': '方法'}
    parts = []
    for category, label in labels.items():
        values = preferences.get(category, [])
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    if not parts:
        return ""
    return "用户科研偏好 — " + "; ".join(parts)
