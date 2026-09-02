"""utils 模块测试:helpers(配置/JSON)与 llm(api-key 解析)。"""

import pytest

from research_assistant.utils.helpers import load_config, save_json, load_json, format_paper_info
from research_assistant.utils.llm import resolve_api_key, create_openai_client


def test_load_config_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("key: value\nn: 1\n", encoding="utf-8")
    assert load_config(str(p)) == {"key": "value", "n": 1}


def test_load_config_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"a": [1, 2]}', encoding="utf-8")
    assert load_config(str(p)) == {"a": [1, 2]}


def test_load_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.yaml"))


def test_load_config_unsupported_suffix_raises(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_save_load_json_roundtrip(tmp_path):
    p = tmp_path / "sub" / "data.json"  # 父目录不存在,应自动创建
    payload = {"a": [1, 2], "b": "中文"}
    save_json(payload, str(p))
    assert load_json(str(p)) == payload


def test_format_paper_info_contains_title():
    text = format_paper_info({"title": "My Paper", "authors": ["A"], "source": "arxiv"})
    assert "My Paper" in text
    assert "arxiv" in text


def test_resolve_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    assert resolve_api_key() == "sk-deepseek"


def test_resolve_api_key_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key({"api_keys": {"openai_api_key": "from-config"}}) == "from-config"


def test_create_openai_client_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert create_openai_client({}) is None
