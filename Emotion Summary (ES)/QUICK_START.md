# Emotion Summary (ES) - 快速开始指南

## 📋 环境要求

- Python 3.8+
- CUDA（如果使用GPU）
- 虚拟环境已激活

## 🚀 快速开始（5步走）

### 步骤 1: 安装依赖

在项目根目录的虚拟环境中运行：

```bash
pip install torch transformers datasets evaluate rouge-score nltk accelerate scikit-learn tqdm matplotlib seaborn
```

或者使用项目根目录的requirements.txt（如果包含所有依赖）：

```bash
pip install -r ../requirements.txt
```

### 步骤 2: 下载数据

```bash
cd "Emotion Summary (ES)/scripts"
python prepare_datasets.py
```

这将：
- 从Hugging Face下载LongEmotion的ES任务数据集
- 处理并保存为JSONL格式到 `data/` 目录
- 如果下载失败，会创建示例数据用于测试

### 步骤 3: 训练模型

```bash
python train.py
```

默认配置：
- 模型: T5-base
- 训练轮数: 3
- 批次大小: 4
- 学习率: 5e-5

训练完成后，模型会保存到 `model/best_model/`

**注意**: 
- 训练可能需要较长时间（取决于数据量和硬件）
- 如果显存不足，可以在 `config/config.py` 中减小 `BATCH_SIZE` 或启用 `GRADIENT_ACCUMULATION_STEPS`

### 步骤 4: 生成总结（推理）

```bash
python inference.py
```

这将：
- 加载训练好的模型
- 对测试集生成5个方面的总结
- 保存结果到 `submission/submission.jsonl`

### 步骤 5: 评估结果

```bash
python evaluate.py
```

评估指标：
- ROUGE-1/2/L
- BLEU
- (可选) GPT-4o评估

## 📝 配置修改

所有配置都在 `config/config.py` 文件中：

```python
# 常用配置项
MODEL_NAME = "t5-base"              # 可改为 "t5-large", "facebook/bart-base" 等
BATCH_SIZE = 4                      # 根据显存调整
NUM_EPOCHS = 3                      # 训练轮数
LEARNING_RATE = 5e-5                # 学习率
MAX_INPUT_LENGTH = 2048             # 输入最大长度
MAX_OUTPUT_LENGTH = 512             # 输出最大长度
```

## 🔧 常见问题

### 1. 显存不足（CUDA out of memory）

修改 `config/config.py`:
```python
BATCH_SIZE = 2  # 减小批次大小
GRADIENT_ACCUMULATION_STEPS = 8  # 增加梯度累积
FP16 = True  # 启用混合精度
```

### 2. 数据集下载失败

- 检查网络连接
- 确认Hugging Face数据集名称是否正确
- 可以手动下载数据集并放到 `data/` 目录

### 3. 找不到模型文件

确保先运行训练脚本，或者从Hugging Face下载预训练模型放到 `model/best_model/`

### 4. 安装依赖失败

某些库可能需要特定版本：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install transformers==4.35.0
```

## 📊 预期结果

训练好的模型应该能够：
- 准确提取心理病理报告的关键信息
- 生成结构化的5方面总结
- ROUGE-L 分数 > 0.30（根据具体任务难度）

## 🎯 进阶使用

### 使用更大的模型

```python
# 在 config/config.py 中修改
MODEL_NAME = "t5-large"  # 或 "google/long-t5-tglobal-base"
```

### 调整生成策略

```python
# 在 config/config.py 中修改
NUM_BEAMS = 8  # 增加beam search宽度
LENGTH_PENALTY = 2.0  # 鼓励生成更长的文本
```

### 使用GPT-4o评估

```python
# 在 config/config.py 中修改
USE_GPT4O_EVAL = True
GPT4O_API_KEY = "your-api-key"
```

## 📞 技术支持

如有问题，请查看：
1. README.md - 完整文档
2. config/config.py - 所有配置参数说明
3. 各脚本文件中的注释

## ✅ 检查清单

- [ ] 虚拟环境已激活
- [ ] 依赖已安装
- [ ] 数据已下载
- [ ] 模型已训练
- [ ] 推理已完成
- [ ] 结果已评估
- [ ] 提交文件已生成

