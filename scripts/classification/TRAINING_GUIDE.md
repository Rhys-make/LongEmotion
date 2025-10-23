# 情绪分类训练教程

## 📋 目录
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [推理预测](#推理预测)
- [训练技巧](#训练技巧)
- [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn
```

### 2. 验证安装

```python
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 数据准备

### 1. 运行数据准备脚本

如果还没有准备数据：

```bash
cd C:\Users\25138\project\LongEmotion
python scripts/classification/prepare_datasets.py
```

### 2. 检查数据文件

确认以下文件已生成：
```
data/classification/
├── train.jsonl        (训练集，约43,411条)
├── validation.jsonl   (验证集，约5,427条)
└── test.jsonl        (测试集，约201条)
```

### 3. 数据格式

每行一个JSON对象：
```json
{
    "id": 0,
    "Context": "I love this new feature!",
    "Subject": "I",
    "Choices": ["admiration", "joy", "anger", ...],
    "Answer": "joy"
}
```

---

## 模型训练

### 快速开始（推荐配置）

```bash
python scripts/classification/train.py \
    --model_name roberta-large \
    --batch_size 8 \
    --num_epochs 5 \
    --learning_rate 2e-5 \
    --use_amp \
    --gradient_accumulation_steps 2 \
    --output_dir checkpoint/classification
```

### 完整参数说明

#### 必选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_data` | `data/classification/train.jsonl` | 训练数据路径 |
| `--val_data` | `data/classification/validation.jsonl` | 验证数据路径 |
| `--output_dir` | `checkpoint/classification` | 模型保存路径 |

#### 模型参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--model_name` | `roberta-large` | 预训练模型<br>• `roberta-large` (推荐)<br>• `roberta-base` (快速)<br>• `microsoft/deberta-v3-large` (最佳) |
| `--max_length` | `512` | 最大序列长度 |

#### 训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--batch_size` | `8` | 批次大小（根据GPU调整）|
| `--num_epochs` | `5` | 训练轮数 |
| `--learning_rate` | `2e-5` | 学习率 |
| `--warmup_ratio` | `0.1` | 预热比例 |
| `--weight_decay` | `0.01` | 权重衰减 |

#### 优化参数

| 参数 | 说明 |
|------|------|
| `--use_amp` | 启用混合精度训练（强烈推荐，节省50%显存）|
| `--gradient_accumulation_steps` | 梯度累积步数（显存不足时增加）|
| `--patience` | 早停耐心值（默认3）|
| `--seed` | 随机种子（默认42）|

### 不同配置示例

#### 配置1: 高性能（GPU 24GB+）

```bash
python scripts/classification/train.py \
    --model_name microsoft/deberta-v3-large \
    --batch_size 16 \
    --num_epochs 8 \
    --learning_rate 1e-5 \
    --use_amp \
    --patience 5
```

#### 配置2: 标准配置（GPU 16GB）⭐ 推荐

```bash
python scripts/classification/train.py --model_name roberta-large --batch_size 4 --num_epochs 5 --learning_rate 2e-5 --use_amp --gradient_accumulation_steps 4 --max_length 256 强
```
python scripts/classification/train.py --model_name roberta-base --batch_size 8 --num_epochs 5 --use_amp --gradient_accumulation_steps 2 一般


评估
# ✅ 使用 best_model
python scripts/classification/evaluate.py --model_path checkpoint/classification/best_model --data_path data/classification/validation.jsonl

推理
# ✅ 使用 best_model

python scripts/classification/inference.py --model_path checkpoint/classification/best_model --test_data data/classification/test.jsonl
继续训练
# 可以使用 latest_model
python scripts/classification/train.py --resume_from_checkpoint checkpoint/classification/latest_model --num_epochs 10


python scripts/classification/train.py --resume_from_checkpoint checkpoint/classification/latest_model --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 4 --use_amp --num_workers 0 --max_length 512