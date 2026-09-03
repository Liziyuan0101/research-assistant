"""
Centralized LLM client construction
集中式 LLM 客户端构造
"""

import hashlib
import json
import logging
import os
import types
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False


def resolve_api_key(config: Optional[dict] = None) -> Optional[str]:
    """解析 API key:优先环境变量(DEEPSEEK/OPENAI),再 config['api_keys']['openai_api_key']。"""
    key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
    if key:
        return key
    if config:
        return config.get('api_keys', {}).get('openai_api_key')
    return None


def create_openai_client(config: Optional[dict] = None):
    """构建 OpenAI 兼容客户端(DeepSeek 等)。无 openai 库或无 api_key 时返回 None。"""
    if not HAS_OPENAI:
        return None
    api_key = resolve_api_key(config)
    if not api_key:
        return None
    cfg = config or {}
    llm_config = cfg.get('llm', {})
    api_keys_config = cfg.get('api_keys', {})
    base_url = llm_config.get('base_url') or api_keys_config.get('openai_base_url')
    return OpenAI(api_key=api_key, base_url=base_url)


logger = logging.getLogger(__name__)

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False


_cache_file: Optional[Path] = None


def set_llm_cache(path: Optional[str]):
    """设置 LLM 结果缓存文件路径(传 None 关闭缓存)。"""
    global _cache_file
    _cache_file = Path(path) if path else None


def _cache_key(kwargs) -> str:
    return hashlib.md5(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()


def _read_cache() -> dict:
    if not _cache_file or not _cache_file.exists():
        return {}
    try:
        return json.loads(_cache_file.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _write_cache(cache: dict):
    if not _cache_file:
        return
    _cache_file.parent.mkdir(parents=True, exist_ok=True)
    _cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')


def _make_response(content: str):
    """构造最小可用的响应对象(缓存命中时用,兼容 .choices[0].message.content)。"""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def chat_completion(client, retries: int = 3, **kwargs):
    """带指数退避重试的 LLM 调用(应对限流/网络抖动),并支持磁盘缓存。"""
    if client is None:
        return None

    key = _cache_key(kwargs) if _cache_file else None
    if key:
        cache = _read_cache()
        if key in cache:
            return _make_response(cache[key])

    if not HAS_TENACITY or retries <= 1:
        response = client.chat.completions.create(**kwargs)
    else:
        @retry(
            stop=stop_after_attempt(retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=lambda s: logger.warning("LLM call retry %s: %s", s.attempt_number, s.outcome.exception()),
        )
        def _call():
            return client.chat.completions.create(**kwargs)
        response = _call()

    if key:
        cache = _read_cache()
        cache[key] = response.choices[0].message.content
        _write_cache(cache)

    return response
