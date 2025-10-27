# QA（问答）任务

## 概述

LongEmotion 的 QA 任务是基于**长篇心理学文献**的问答任务。模型需要：
1. 理解长上下文（可能超过 4096 tokens）
2. 回答心理学领域的问题
3. 使用 **F1 分数** 评估答案质量

## 目录结构

```
scripts/qa/
├── README.md                  # 本文件
├── TRAINING_GUIDE.md          # 详细训练指南
├── prepare_datasets.py        # 数据准备和预处理
├── qa.py                      # QA 模型类和工具函数
├── train.py                   # 训练脚本
├── inference.py               # 推理脚本
├── evaluate.py                # 评估脚本
└── __pycache__/              # Python 缓存（自动生成）
```

## 数据格式

### 输入格式（训练/验证/测试）
```json
{
  "id": 0,
  "problem": "基于以下心理学文献，问题是：...",
  "context": "一段长篇心理学或社会科学文献...",
  "answer": "因为..."
}
```

### 输出格式（预测结果）
```json
{
  "id": 0,
  "predicted_answer": "基于文献内容，答案是..."
}
```

## 快速开始

### 1. 准备数据

```bash
# 下载并预处理数据集
python scripts/qa/prepare_datasets.py
```

这会生成：
- `data/train.jsonl` - 训练集（来自多个长上下文 QA 数据集）
- `data/validation.jsonl` - 验证集
- `data/test.jsonl` - 测试集（来自 LongEmotion）

### 2. 训练模型

```bash
# 基础训练（BERT-base）
python scripts/qa/train.py

# 使用 Longformer（推荐用于长上下文）
python scripts/qa/train.py \
  --model_name allenai/longformer-base-4096 \
  --max_length 2048 \
  --batch_size 2 \
  --gradient_accumulation_steps 8

# 使用生成式模型（如 Mistral）
python scripts/qa/train.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --model_type generative \
  --max_length 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 16
```

### 3. 推理

```bash
# 对测试集进行预测
python scripts/qa/inference.py \
  --model_path checkpoint/best_model \
  --test_data data/test.jsonl \
  --output_file result/Emotion_QA_Result.jsonl
```

### 4. 评估

```bash
# 评估模型性能（需要有标注答案）
python scripts/qa/evaluate.py \
  --predictions result/Emotion_QA_Result.jsonl \
  --ground_truth data/test.jsonl \
  --output_dir evaluation/qa
```

## 支持的模型

### 编码器模型（Encoder）
- `bert-base-uncased` - 基础 BERT（512 tokens）
- `allenai/longformer-base-4096` - 长文档专用（4096 tokens）✅ 推荐
- `google/bigbird-roberta-base` - 稀疏注意力（4096 tokens）

### 生成式模型（Generative）
- `mistralai/Mistral-7B-Instruct-v0.2` - 高质量生成
- `meta-llama/Llama-2-7b-chat-hf` - 对话优化
- `Qwen/Qwen2-7B-Instruct` - 中文优化

## F1 分数评估

我们使用与 SQuAD 相同的 F1 评估方法：
1. 将预测答案和标准答案分词
2. 计算词级别的精确率和召回率
3. 计算 F1 = 2 × (精确率 × 召回率) / (精确率 + 召回率)

## 训练技巧

1. **长上下文处理**：
   - 使用 Longformer 或 BigBird 处理超长文本
   - 使用滑动窗口策略分割文档
   - 考虑使用层次化注意力机制

2. **内存优化**：
   - 使用混合精度训练（`--use_amp`）
   - 使用梯度累积（`--gradient_accumulation_steps`）
   - 减小批次大小（`--batch_size 1`）
   - 使用梯度检查点（`--gradient_checkpointing`）

3. **提升性能**：
   - 在领域数据上继续预训练
   - 使用数据增强（回译、同义词替换）
   - 集成多个模型的预测结果

## 常见问题

### Q: 内存不足怎么办？
A: 尝试：
- 减小 `--batch_size` 到 1
- 增加 `--gradient_accumulation_steps`
- 使用 `--use_amp` 混合精度
- 减小 `--max_length`

### Q: 如何提升 F1 分数？
A: 可以：
- 使用更大的模型（如 Mistral-7B）
- 增加训练数据量
- 调整学习率和训练轮数
- 使用领域适应（在心理学语料上预训练）

### Q: 推理太慢怎么办？
A: 可以：
- 使用量化模型（8-bit 或 4-bit）
- 增加 `--batch_size`
- 使用 GPU 而非 CPU
- 使用模型蒸馏

## 参考资料

- [LongBench: 长上下文理解基准](https://huggingface.co/datasets/THUDM/LongBench)
- [Longformer: 长文档 Transformer](https://arxiv.org/abs/2004.05150)
- [SQuAD 评估指标](https://rajpurkar.github.io/SQuAD-explorer/)

