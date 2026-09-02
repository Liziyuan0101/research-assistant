"""
Helper utilities
"""

import json
import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    return config


def save_json(data: Any, output_path: str, indent: int = 2):
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        output_path: 输出路径
        indent: 缩进
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(input_path: str) -> Any:
    """
    加载JSON文件
    
    Args:
        input_path: 输入路径
        
    Returns:
        加载的数据
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def format_paper_info(paper: Dict) -> str:
    """
    格式化论文信息为可读文本
    
    Args:
        paper: 论文字典
        
    Returns:
        格式化的文本
    """
    lines = []
    lines.append(f"Title: {paper.get('title', 'N/A')}")
    lines.append(f"Authors: {', '.join(paper.get('authors', [])[:5])}")
    lines.append(f"Published: {paper.get('published', 'N/A')}")
    lines.append(f"Source: {paper.get('source', 'N/A')}")
    
    if 'abstract' in paper:
        lines.append(f"\nAbstract:\n{paper['abstract'][:500]}...")

    return '\n'.join(lines)


_ENV_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def resolve_env_placeholders(obj):
    """递归解析 ${VAR} 环境变量占位符(未设置的变量解析为空字符串)。"""
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), ''), obj)
    if isinstance(obj, dict):
        return {k: resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_placeholders(v) for v in obj]
    return obj


def resolve_device(device: str) -> str:
    """解析设备名:auto -> cuda(若可用)否则 cpu;其余原样返回。"""
    if device == 'auto':
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
    return device
