"""懒加载语义编码器（BGE-M3 via sentence-transformers）。

为什么用 sentence-transformers 而不是 FlagEmbedding
--------------------------------------------------
``sentence-transformers`` 已在 ``pyproject.toml`` 的 ``finetune`` extra 中声明，
不引入新依赖；且本地 HF 缓存中的 ``BAAI/bge-m3`` 就是 sentence-transformers 格式。
更关键的是：它与 ``HybridRetriever`` 使用**同一份权重**，因此记忆层与检索层落在
同一向量空间 —— 偏好先验可以直接作用到检索打分上，而不需要额外的对齐层。

降级策略
--------
模型不可用（未安装 / 权重缺失 / 加载失败）时 :attr:`available` 为 ``False``，
:meth:`encode` 返回 ``None``；调用方必须显式回退（如词重叠排序），
**绝不静默返回全零向量**，否则会把"没生效"伪装成"效果为 0"。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'BAAI/bge-m3'


class SemanticEncoder:
    """BGE-M3 稠密编码器（带内存缓存与优雅降级）。

    Args:
        model_name: HF 模型名，默认与检索器一致。
        device: ``'cpu'`` / ``'cuda'`` / ``None``（自动）。
        offline: 是否强制离线（只用本地缓存），默认 ``True`` 以避免联网卡住。
        cache_size: 文本 → 向量 的内存缓存条数上限。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        offline: bool = True,
        cache_size: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.offline = offline
        self.cache_size = cache_size
        self._model = None
        self._failed = False
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ 状态
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def available(self) -> bool:
        """尝试加载一次；失败后记住失败状态，不再重复尝试。"""
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            self._load()
            return True
        except Exception as exc:  # noqa: BLE001 - 降级路径，需吞掉所有异常
            self._failed = True
            logger.warning(
                'SemanticEncoder unavailable (%s: %s); callers must fall back.',
                type(exc).__name__, exc,
            )
            return False

    def _load(self) -> None:
        if self.offline:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        from sentence_transformers import SentenceTransformer  # 延迟导入

        kwargs = {}
        if self.device:
            kwargs['device'] = self.device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        logger.info('SemanticEncoder loaded: %s', self.model_name)

    # ------------------------------------------------------------------ 编码
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Optional[np.ndarray]:
        """编码为 L2 归一化向量；不可用时返回 ``None``。"""
        if not texts:
            return np.zeros((0, 0), dtype='float32')
        if not self.available:
            return None

        missing: List[str] = []
        for t in texts:
            if t not in self._cache and t not in missing:
                missing.append(t)

        if missing:
            if len(self._cache) + len(missing) > self.cache_size:
                self._cache.clear()  # 简单策略：超限整体清空
            vecs = self._model.encode(
                missing, batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            vecs = np.asarray(vecs, dtype='float32')
            for t, v in zip(missing, vecs):
                self._cache[t] = v

        return np.stack([self._cache[t] for t in texts]).astype('float32')

    def encode_one(self, text: str) -> Optional[np.ndarray]:
        out = self.encode([text])
        return None if out is None else out[0]

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """归一化向量的余弦相似度（等价于点积）。"""
        return float(np.dot(a, b))
