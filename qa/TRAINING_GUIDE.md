# QA（问答）任务训练指南

本指南将帮助你完成 LongEmotion QA 任务的完整训练流程。

## 📋 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [模型选择](#模型选择)
4. [训练模型](#训练模型)
5. [推理预测](#推理预测)
6. [评估结果](#评估结果)
7. [优化建议](#优化建议)
8. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

确保已安装以下依赖：

```bash
pip install torch transformers datasets accelerate
pip install matplotlib numpy tqdm
```

### 2. 检查 GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

如果没有 GPU，可以使用 CPU 训练，但速度会较慢。

---

## 数据准备

### 步骤 1：下载并预处理数据

运行数据准备脚本：

```bash
# 下载 LongEmotion QA 测试集 + 其他训练数据集
python scripts/qa/prepare_datasets.py
```

**可选参数：**

```bash
# 自定义数据来源
python scripts/qa/prepare_datasets.py \
  --datasets squad narrativeqa hotpotqa \
  --max_samples_per_dataset 10000 \
  --use_synthetic \
  --output_dir data/qa
```

**生成的文件：**
- `data/qa/train.jsonl` - 训练集（~15K+ 样本）
- `data/qa/validation.jsonl` - 验证集（~1.5K+ 样本）
- `data/qa/test.jsonl` - 测试集（来自 LongEmotion）

### 步骤 2：查看数据

```bash
# 查看前 3 条训练数据
head -n 3 data/qa/train.jsonl
```

**数据格式示例：**

```json
{
  "id": 0,
  "problem": "根据上述心理学文献，什么是认知失调理论？",
  "context": "认知失调理论是由心理学家莱昂·费斯廷格于1957年提出...",
  "answer": "认知失调理论是指当个体同时持有矛盾的认知时产生的不舒适心理状态。"
}
```

---

## 模型选择

根据你的硬件资源选择合适的模型：

### 选项 1：BERT-base（推荐入门）

**优点：** 快速、易用、资源需求低  
**缺点：** 上下文长度限制（512 tokens）

```bash
--model_name bert-base-uncased
--model_type extractive
--max_length 512
```

### 选项 2：Longformer（推荐长上下文）✅

**优点：** 支持 4096 tokens、专为长文档设计  
**缺点：** 训练较慢、显存需求较高

```bash
--model_name allenai/longformer-base-4096
--model_type extractive
--max_length 2048
```

### 选项 3：Mistral-7B（推荐高质量生成）

**优点：** 生成质量高、理解能力强  
**缺点：** 显存需求极高（需 16GB+ GPU）

```bash
--model_name mistralai/Mistral-7B-Instruct-v0.2
--model_type generative
--max_length 4096
```

### 选项 4：T5/mT5（seq2seq）

**优点：** 适合生成式任务、中文支持好  
**缺点：** 需要调整训练策略

```bash
--model_name google/mt5-base
--model_type seq2seq
--max_length 1024
```

---

## 训练模型

### 🚀 快速开始（BERT-base）

```bash
python scripts/qa/train.py \
  --model_name bert-base-uncased \
  --model_type extractive \
  --batch_size 8 \
  --num_epochs 3 \
  --learning_rate 3e-5
```

### 🔥 推荐配置（Longformer）

```bash
python scripts/qa/train.py \
  --model_name allenai/longformer-base-4096 \
  --model_type extractive \
  --max_length 2048 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --learning_rate 3e-5 \
  --use_amp \
  --patience 3
```

### 💻 低资源配置（小显存/CPU）

```bash
python scripts/qa/train.py \
  --model_name bert-base-uncased \
  --model_type extractive \
  --max_length 256 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_epochs 5 \
  --learning_rate 3e-5 \
  --use_amp \
  --num_workers 0
```

### 🎯 生成式模型（Mistral）

```bash
python scripts/qa/train.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --model_type generative \
  --max_length 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_epochs 2 \
  --learning_rate 2e-5 \
  --use_amp \
  --patience 5
```

### 📊 从检查点恢复训练

```bash
python scripts/qa/train.py \
  --resume_from_checkpoint checkpoint/qa/latest_model \
  --num_epochs 2
```

---

## 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model_name` | 预训练模型名称 | `allenai/longformer-base-4096` |
| `--model_type` | 模型类型 | `extractive` / `generative` |
| `--max_length` | 最大序列长度 | 512-4096 |
| `--batch_size` | 批次大小 | 2-8 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 4-16 |
| `--num_epochs` | 训练轮数 | 3-5 |
| `--learning_rate` | 学习率 | 2e-5 ~ 5e-5 |
| `--use_amp` | 混合精度训练 | 建议开启 |
| `--patience` | 早停耐心值 | 3 |

---

## 推理预测

### 对测试集进行预测

```bash
python scripts/qa/inference.py \
  --model_path checkpoint/qa/best_model \
  --test_data data/qa/test.jsonl \
  --output_file result/Emotion_QA_Result.jsonl \
  --model_type extractive
```

### 生成式模型推理

```bash
python scripts/qa/inference.py \
  --model_path checkpoint/qa/best_model \
  --test_data data/qa/test.jsonl \
  --output_file result/Emotion_QA_Result.jsonl \
  --model_type generative \
  --max_answer_length 256
```

### 查看预测结果

```bash
# 查看前 3 条预测
head -n 3 result/Emotion_QA_Result.jsonl
```

**输出格式：**

```json
{"id": 0, "predicted_answer": "认知失调理论是指当个体同时持有矛盾的认知时..."}
{"id": 1, "predicted_answer": "根据文献，情绪调节策略包括..."}
```

---

## 评估结果

### 评估模型性能

如果你有标注的测试集答案，可以运行：

```bash
python scripts/qa/evaluate.py \
  --predictions result/Emotion_QA_Result.jsonl \
  --ground_truth data/qa/test.jsonl \
  --output_dir evaluation/qa
```

### 评估输出

**生成的文件：**
- `evaluation/qa/evaluation_metrics.json` - JSON 格式指标
- `evaluation/qa/evaluation_report.txt` - 文本格式报告
- `evaluation/qa/detailed_predictions.jsonl` - 详细预测（含 F1）
- `evaluation/qa/f1_distribution.png` - F1 分数分布图
- `evaluation/qa/prediction_examples.txt` - 最佳/最差预测示例

**示例报告：**

```
============================================================
QA 模型评估报告
============================================================

样本数量: 500

F1 分数:
  平均值: 0.7234
  标准差: 0.1523
  最小值: 0.2100
  最大值: 1.0000
  中位数: 0.7450

精确匹配 (Exact Match):
  准确率: 0.4520
  匹配数量: 226/500
============================================================
```

---

## 优化建议

### 1. 提升 F1 分数

#### 策略 1：使用更大的模型

```bash
# 从 BERT-base 升级到 Longformer
python scripts/qa/train.py \
  --model_name allenai/longformer-base-4096 \
  --max_length 2048
```

#### 策略 2：增加训练数据

```bash
# 下载更多数据集
python scripts/qa/prepare_datasets.py \
  --datasets squad narrativeqa hotpotqa \
  --max_samples_per_dataset 20000
```

#### 策略 3：调整超参数

```bash
# 降低学习率，增加训练轮数
python scripts/qa/train.py \
  --learning_rate 2e-5 \
  --num_epochs 5 \
  --warmup_ratio 0.1
```

#### 策略 4：数据增强

在 `prepare_datasets.py` 中添加回译、同义词替换等数据增强技术。

### 2. 降低内存占用

#### 策略 1：混合精度训练

```bash
python scripts/qa/train.py --use_amp
```

#### 策略 2：梯度累积

```bash
# 等效 batch_size = 2 × 8 = 16
python scripts/qa/train.py \
  --batch_size 2 \
  --gradient_accumulation_steps 8
```

#### 策略 3：减小序列长度

```bash
python scripts/qa/train.py --max_length 256
```

#### 策略 4：梯度检查点（待实现）

在模型配置中启用 `gradient_checkpointing`。

### 3. 加速训练

#### 策略 1：使用更小的验证集

修改 `prepare_datasets.py` 中的 `val_ratio` 参数。

#### 策略 2：减少早停耐心值

```bash
python scripts/qa/train.py --patience 2
```

#### 策略 3：使用多 GPU（待实现）

使用 `accelerate` 或 `torch.nn.DataParallel`。

---

## 常见问题

### Q1: CUDA Out of Memory 错误

**解决方案：**

```bash
# 方案 1：减小 batch_size 和 max_length
python scripts/qa/train.py \
  --batch_size 1 \
  --max_length 256 \
  --gradient_accumulation_steps 16

# 方案 2：使用混合精度
python scripts/qa/train.py --use_amp

# 方案 3：使用 CPU（慢）
python scripts/qa/train.py --device cpu
```

### Q2: F1 分数很低（< 0.3）

**可能原因：**
1. 训练数据与测试数据分布不同（领域不匹配）
2. 模型容量不足
3. 训练不充分

**解决方案：**
1. 收集心理学领域的 QA 数据
2. 使用更大的模型（Longformer / Mistral）
3. 增加训练轮数

### Q3: 训练时间太长

**解决方案：**
1. 使用更小的模型（BERT-base）
2. 减少训练数据量
3. 使用 GPU 而非 CPU
4. 启用混合精度训练

### Q4: 抽取式 vs 生成式该选哪个？

| 模型类型 | 优点 | 缺点 | 适用场景 |
|---------|------|------|----------|
| **抽取式** | 快速、准确、答案来自原文 | 无法总结、答案必须在文中 | 答案在原文中明确出现 |
| **生成式** | 可总结、灵活、理解更深 | 慢、可能幻觉、需大模型 | 需要总结或推理的问答 |

**推荐：**
- 如果测试集答案都来自原文：使用**抽取式**（Longformer）
- 如果需要总结或推理：使用**生成式**（Mistral）

### Q5: 如何处理超长文档（> 4096 tokens）？

**解决方案：**

1. **使用 Longformer** - 支持最长 4096 tokens
2. **文档分块** - 将长文档切分为多个片段
3. **滑动窗口** - 使用重叠窗口提取答案
4. **层次化模型** - 先检索相关段落，再进行 QA

### Q6: 没有 GPU 能训练吗？

可以，但会很慢。建议：

```bash
# CPU 训练配置
python scripts/qa/train.py \
  --model_name bert-base-uncased \
  --batch_size 2 \
  --max_length 128 \
  --num_epochs 1 \
  --device cpu
```

或者使用 Google Colab 免费 GPU。

---

## 完整训练流程示例

### 方案 A：快速测试（10 分钟）

```bash
# 1. 准备少量数据
python scripts/qa/prepare_datasets.py \
  --datasets squad \
  --max_samples_per_dataset 1000

# 2. 快速训练
python scripts/qa/train.py \
  --model_name bert-base-uncased \
  --batch_size 8 \
  --num_epochs 1 \
  --max_length 256

# 3. 推理
python scripts/qa/inference.py \
  --model_path checkpoint/qa/best_model \
  --model_type extractive

# 4. 评估
python scripts/qa/evaluate.py \
  --predictions result/Emotion_QA_Result.jsonl \
  --ground_truth data/qa/test.jsonl
```

### 方案 B：完整训练（2-4 小时）

```bash
# 1. 准备完整数据
python scripts/qa/prepare_datasets.py \
  --datasets squad narrativeqa hotpotqa \
  --max_samples_per_dataset 10000 \
  --use_synthetic

# 2. 使用 Longformer 训练
python scripts/qa/train.py \
  --model_name allenai/longformer-base-4096 \
  --model_type extractive \
  --max_length 2048 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --learning_rate 3e-5 \
  --use_amp \
  --patience 3

# 3. 推理
python scripts/qa/inference.py \
  --model_path checkpoint/qa/best_model \
  --model_type extractive \
  --max_length 2048

# 4. 评估
python scripts/qa/evaluate.py \
  --predictions result/Emotion_QA_Result.jsonl \
  --ground_truth data/qa/test.jsonl
```

### 方案 C：最佳性能（8+ 小时，需大 GPU）

```bash
# 1. 准备最多数据
python scripts/qa/prepare_datasets.py \
  --datasets squad narrativeqa hotpotqa \
  --max_samples_per_dataset 20000 \
  --use_synthetic

# 2. 使用 Mistral-7B 训练
python scripts/qa/train.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --model_type generative \
  --max_length 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_epochs 2 \
  --learning_rate 2e-5 \
  --use_amp \
  --patience 5

# 3. 推理
python scripts/qa/inference.py \
  --model_path checkpoint/qa/best_model \
  --model_type generative \
  --max_length 4096 \
  --max_answer_length 256

# 4. 评估
python scripts/qa/evaluate.py \
  --predictions result/Emotion_QA_Result.jsonl \
  --ground_truth data/qa/test.jsonl
```

---

## 监控训练进度

### 查看训练历史

```python
import json
import matplotlib.pyplot as plt

# 加载训练历史
with open('checkpoint/qa/training_history.json', 'r') as f:
    history = json.load(f)

# 绘制损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history['val_f1'], label='Val F1', color='green')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 Score Curve')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
```

---

## 下一步

- 🔍 **优化模型**：尝试不同的模型架构和超参数
- 📊 **数据增强**：添加更多领域相关的训练数据
- 🚀 **模型集成**：结合多个模型的预测结果
- 📝 **错误分析**：分析评估报告中的失败案例
- 🎯 **领域适应**：在心理学语料上继续预训练

---

## 参考资料

- [Longformer 论文](https://arxiv.org/abs/2004.05150)
- [SQuAD 数据集](https://rajpurkar.github.io/SQuAD-explorer/)
- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/)
- [NarrativeQA 数据集](https://github.com/deepmind/narrativeqa)

---

如有问题，请查看 [README.md](README.md) 或提交 Issue。

祝训练顺利！🎉

