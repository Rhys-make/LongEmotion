"""
Emotion Summary (ES) 任务配置文件
包含模型、训练、推理等所有配置参数
"""

import os
from pathlib import Path

# ============================================
# 路径配置
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for dir_path in [DATA_DIR, MODEL_DIR, SUBMISSION_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# 模型配置
# ============================================
# 推荐的生成式模型选择
MODEL_OPTIONS = {
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "bart-base": "facebook/bart-base",
    "bart-large": "facebook/bart-large",
    "bart-large-cnn": "facebook/bart-large-cnn",  # 专门训练用于摘要
    "longt5-base": "google/long-t5-tglobal-base",  # 推荐：专门处理长文本
    "longt5-large": "google/long-t5-tglobal-large",
    "led-base": "allenai/led-base-16384",  # 支持超长输入（16k tokens）
    "pegasus-large": "google/pegasus-large",  # 专门用于摘要任务
}

# 当前使用的模型（可根据需要修改）
# 对于长文本心理病理报告总结，推荐使用LongT5或LED
MODEL_NAME = MODEL_OPTIONS["longt5-base"]  # 默认使用LongT5-base（最适合长文本）
MODEL_CHECKPOINT_DIR = MODEL_DIR / "emotion_summary"  # 模型保存路径

# ============================================
# 数据配置
# ============================================
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# 数据文件夹（按类型分类存储）
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

# 数据集来源
DATASET_NAME = "dwxdaisy/LongEmotion"  # Hugging Face数据集名称
DATASET_SUBSET = "emotion_summary"  # ES任务子集

# 备注：如果数据集名称不正确，prepare_datasets.py会自动创建示例数据用于测试

# ============================================
# 总结任务配置
# ============================================
# 需要总结的5个方面
SUMMARY_ASPECTS = [
    "causes",                    # 原因
    "symptoms",                  # 症状
    "treatment_process",         # 治疗过程
    "illness_characteristics",   # 疾病特征
    "treatment_effects",         # 治疗效果
]

# 中文标签映射
ASPECT_LABELS_CN = {
    "causes": "原因",
    "symptoms": "症状",
    "treatment_process": "治疗过程",
    "illness_characteristics": "疾病特征",
    "treatment_effects": "治疗效果",
}

# ============================================
# 模型超参数
# ============================================
# 输入输出长度
# LongT5和LED可以处理更长的输入
MAX_INPUT_LENGTH = 4096      # 输入最大长度（长文本报告，LongT5支持16k）
MAX_OUTPUT_LENGTH = 512      # 输出最大长度（每个方面的总结）
MAX_TOTAL_OUTPUT_LENGTH = 512 * 5  # 总输出长度（5个方面）

# 如果使用T5-base或BART，建议将MAX_INPUT_LENGTH改为1024或2048

# 训练参数
BATCH_SIZE = 4               # 训练批次大小（根据显存调整）
EVAL_BATCH_SIZE = 8          # 评估批次大小
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
NUM_EPOCHS = 3               # 训练轮数
LEARNING_RATE = 5e-5         # 学习率
WEIGHT_DECAY = 0.01          # 权重衰减
WARMUP_STEPS = 500           # 预热步数
MAX_GRAD_NORM = 1.0          # 梯度裁剪

# 保存和日志
SAVE_STEPS = 500             # 每多少步保存一次
EVAL_STEPS = 500             # 每多少步评估一次
LOGGING_STEPS = 100          # 每多少步记录一次日志
SAVE_TOTAL_LIMIT = 3         # 最多保存多少个检查点

# ============================================
# 生成配置
# ============================================
# 解码策略
NUM_BEAMS = 4                # Beam Search宽度
DO_SAMPLE = False            # 是否使用采样
TEMPERATURE = 1.0            # 采样温度
TOP_K = 50                   # Top-K采样
TOP_P = 0.95                 # Top-P (nucleus) 采样
NO_REPEAT_NGRAM_SIZE = 3     # 避免重复的n-gram大小
LENGTH_PENALTY = 1.0         # 长度惩罚

# ============================================
# 优化和加速
# ============================================
FP16 = True                  # 是否使用混合精度训练（节省显存）
FP16_OPT_LEVEL = "O1"        # 混合精度级别
USE_GRADIENT_CHECKPOINTING = True  # 是否使用梯度检查点（节省显存）

# 分布式训练
LOCAL_RANK = -1              # 本地GPU编号
WORLD_SIZE = 1               # 总GPU数量

# ============================================
# 评估配置
# ============================================
# 评估指标
EVAL_METRICS = [
    "rouge1",
    "rouge2",
    "rougeL",
    "bleu",
]

# GPT-4o评估（如果使用）
USE_GPT4O_EVAL = False       # 是否使用GPT-4o评估（需要API key）
GPT4O_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT4O_MODEL = "gpt-4o"       # GPT模型版本
GPT4O_EVAL_ASPECTS = [
    "factual_consistency",   # 事实一致性
    "completeness",          # 完整性
    "clarity",               # 清晰度
]

# ============================================
# 提示词模板（Prompt Template）
# ============================================
# 用于指导模型生成结构化总结
PROMPT_TEMPLATE = """请从以下心理病理报告中总结出这5个方面的内容：

报告内容：
{context}

请分别总结：
1. 原因（Causes）：
2. 症状（Symptoms）：
3. 治疗过程（Treatment Process）：
4. 疾病特征（Illness Characteristics）：
5. 治疗效果（Treatment Effects）：
"""

# T5特定的前缀（用于任务标识）
T5_PREFIX = "summarize psychological report: "

# ============================================
# 提交文件配置
# ============================================
SUBMISSION_FILE = SUBMISSION_DIR / "submission.jsonl"
SUBMISSION_FORMAT = "jsonl"  # 提交文件格式

# ============================================
# 设备配置
# ============================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4              # DataLoader的工作进程数

# ============================================
# 随机种子
# ============================================
SEED = 42

# ============================================
# 调试模式
# ============================================
DEBUG_MODE = False           # 调试模式（使用小数据集）
DEBUG_SAMPLES = 100          # 调试模式下的样本数

# ============================================
# 打印配置信息
# ============================================
def print_config():
    """打印当前配置信息"""
    print("=" * 60)
    print("Emotion Summary (ES) 配置信息")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"设备: {DEVICE}")
    print(f"最大输入长度: {MAX_INPUT_LENGTH}")
    print(f"最大输出长度: {MAX_OUTPUT_LENGTH}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"混合精度: {FP16}")
    print(f"梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"总结方面: {', '.join(SUMMARY_ASPECTS)}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()

