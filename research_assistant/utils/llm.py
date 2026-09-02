"""
Centralized LLM client construction
集中式 LLM 客户端构造
"""

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
