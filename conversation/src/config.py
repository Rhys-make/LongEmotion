"""
配置文件管理
"""
import yaml
import os
from pathlib import Path


def get_config():
    """
    读取配置文件
    
    Returns:
        dict: 配置字典
    """
    config_path = Path(__file__).parent.parent / "conf.d" / "config.yaml"
    
    if not config_path.exists():
        # 如果配置文件不存在，返回默认配置
        return {
            "openai": {"key": ""},
            "llama2": {"host": ""},
            "llama3": {"host": ""}
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

