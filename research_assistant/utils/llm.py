"""
Centralized LLM client construction
集中式 LLM 客户端构造
"""

import logging
import os
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


def chat_completion(client, retries: int = 3, **kwargs):
    """带指数退避重试的 LLM 调用(应对限流/网络抖动)。"""
    if client is None:
        return None
    if not HAS_TENACITY or retries <= 1:
        return client.chat.completions.create(**kwargs)

    @retry(
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda s: logger.warning("LLM call retry %s: %s", s.attempt_number, s.outcome.exception()),
    )
    def _call():
        return client.chat.completions.create(**kwargs)

    return _call()
