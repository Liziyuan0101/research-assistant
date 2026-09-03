"""utils 模块测试:helpers(配置/JSON)与 llm(api-key 解析)。"""

import pytest

from research_assistant.utils.helpers import (
    load_config,
    save_json,
    load_json,
    format_paper_info,
    resolve_env_placeholders,
    resolve_device,
    tokenize,
)
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


def test_resolve_env_placeholders_nested(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-value")
    data = {"a": "${TEST_KEY}", "b": [1, "${TEST_KEY}"], "c": {"d": "${MISSING_KEY}"}}
    resolved = resolve_env_placeholders(data)
    assert resolved["a"] == "secret-value"
    assert resolved["b"] == [1, "secret-value"]
    assert resolved["c"]["d"] == ""  # 未设置的变量解析为空字符串


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_returns_valid():
    result = resolve_device("auto")
    assert result in ("cuda", "cpu")


def test_tokenize_handles_chinese():
    tokens = tokenize("锂电池剩余寿命预测")
    # 中文应被切出 token(而非切空)
    assert len(tokens) > 0
    assert all('一' <= c <= '鿿' for c in ''.join(tokens))


def test_tokenize_mixed_language():
    tokens = tokenize("lithium-ion battery 寿命预测")
    assert "lithium" in tokens
    assert "battery" in tokens
    assert any(t and '一' <= t[0] <= '鿿' for t in tokens)
