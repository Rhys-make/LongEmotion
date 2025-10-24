"""
全局配置文件
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# 创建目录
for dir_path in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_CONFIGS = {
    "classification": {
        "model_name": "bert-base-chinese",
        "num_labels": 7,
        "max_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3
    },
    "detection": {
        "model_name": "bert-base-chinese",
        "num_emotions": 7,
        "max_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "threshold": 0.5
    },
    "conversation": {
        "model_name": "Qwen/Qwen2-1.5B",
        "max_length": 1024,
        "max_new_tokens": 256,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "num_epochs": 2
    },
    "summary": {
        "model_name": "google/mt5-base",
        "max_input_length": 1024,
        "max_output_length": 256,
        "learning_rate": 5e-5,
        "batch_size": 8,
        "num_epochs": 3
    },
    "qa": {
        "model_name": "bert-base-chinese",
        "max_length": 512,
        "learning_rate": 3e-5,
        "batch_size": 8,
        "num_epochs": 3
    }
}

# 情感标签
EMOTION_LABELS = [
    "happiness",   # 快乐
    "sadness",     # 悲伤
    "anger",       # 愤怒
    "fear",        # 恐惧
    "surprise",    # 惊讶
    "disgust",     # 厌恶
    "neutral"      # 中性
]

# API 配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "LongEmotion API",
    "description": "情感分析与生成的统一 API 接口",
    "version": "1.0.0"
}

# Hugging Face 配置
HF_CONFIG = {
    "dataset_name": "LongEmotion/LongEmotion",
    "cache_dir": str(DATA_DIR),
    "use_auth_token": None  # 如果需要认证，设置你的 token
}

# 训练配置
TRAINING_CONFIG = {
    "early_stopping_patience": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42
}

