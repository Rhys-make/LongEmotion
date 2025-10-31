# Emotion Summary (ES) - 情绪总结任务

## 📖 任务简介

本项目是LongEmotion比赛中的**情绪总结（Emotion Summary）**任务。模型需要从长文本心理病理报告中总结以下5个方面：

1. **原因 (Causes)** - 心理问题的成因
2. **症状 (Symptoms)** - 患者表现出的症状
3. **治疗过程 (Treatment Process)** - 治疗的具体过程
4. **疾病特征 (Illness Characteristics)** - 疾病的主要特征
5. **治疗效果 (Treatment Effects)** - 治疗的效果和结果

生成的总结将使用**GPT-4o**进行评估，评估维度包括：
- **事实一致性 (Factual Consistency)** - 与原文的一致性
- **完整性 (Completeness)** - 信息的完整程度
- **清晰度 (Clarity)** - 表达的清晰程度

---

## 📁 项目结构

```
Emotion Summary (ES)/
├── data/                   # 数据集目录
│   ├── train.jsonl        # 训练集
│   ├── validation.jsonl   # 验证集
│   └── test.jsonl         # 测试集
├── model/                  # 模型存储目录
│   └── best_model/        # 最佳模型检查点
├── scripts/                # 核心脚本
│   ├── prepare_datasets.py  # 数据准备脚本
│   ├── train.py            # 训练脚本
│   ├── inference.py        # 推理脚本
│   ├── evaluate.py         # 评估脚本
│   └── model.py            # 模型定义
├── config/                 # 配置文件
│   └── config.py          # 配置参数
├── submission/             # 提交文件
│   └── submission.jsonl   # 比赛提交文件
├── logs/                   # 训练日志
└── README.md              # 项目说明文档
```

---

## 🚀 快速开始

### 1. 环境准备

确保已激活虚拟环境并安装必要的依赖：

```bash
# 已在虚拟环境中，安装依赖
pip install torch transformers datasets evaluate rouge-score nltk accelerate scikit-learn tqdm
```

### 2. 数据准备

下载并处理LongEmotion的ES任务数据集：

```bash
cd scripts
python prepare_datasets.py
```

### 3. 模型训练

使用生成式模型（如T5/BART）进行训练：

```bash
python scripts/train.py
```

### 4. 推理生成

对测试集生成总结：

```bash
python scripts/inference.py
```

### 5. 评估

评估生成的总结质量：

```bash
python scripts/evaluate.py
```

---

## 📊 模型选择

本任务适合使用以下生成式模型：

- **T5 (Text-to-Text Transfer Transformer)** - 推荐用于摘要任务
- **BART (Bidirectional and Auto-Regressive Transformers)** - 适合长文本生成
- **LongT5** - 专门针对长文本优化
- **LED (Longformer Encoder Decoder)** - 支持超长文本

---

## 📝 数据格式

### 输入格式示例

```json
{
    "id": 0,
    "context": "长篇心理病理报告内容...",
    "reference_summary": {
        "causes": "参考答案：原因总结",
        "symptoms": "参考答案：症状总结",
        "treatment_process": "参考答案：治疗过程总结",
        "illness_characteristics": "参考答案：疾病特征总结",
        "treatment_effects": "参考答案：治疗效果总结"
    }
}
```

### 输出格式示例

```json
{
    "id": 0,
    "generated_summary": {
        "causes": "生成的原因总结",
        "symptoms": "生成的症状总结",
        "treatment_process": "生成的治疗过程总结",
        "illness_characteristics": "生成的疾病特征总结",
        "treatment_effects": "生成的治疗效果总结"
    }
}
```

---

## 🔧 配置说明

主要配置参数在 `config/config.py` 中：

- `MODEL_NAME`: 使用的预训练模型名称
- `MAX_INPUT_LENGTH`: 最大输入长度
- `MAX_OUTPUT_LENGTH`: 最大输出长度
- `BATCH_SIZE`: 训练批次大小
- `LEARNING_RATE`: 学习率
- `NUM_EPOCHS`: 训练轮数

---

## 📈 评估指标

- **ROUGE-1/2/L**: 衡量n-gram重叠度
- **BLEU**: 机器翻译质量指标
- **GPT-4o评分**: 事实一致性、完整性、清晰度

---

## ⚠️ 注意事项

1. **长文本处理**: 心理病理报告通常很长，需要注意模型的最大输入长度限制
2. **结构化输出**: 确保模型输出包含所有5个方面的总结
3. **显存管理**: 使用梯度累积和混合精度训练以节省显存
4. **评估成本**: GPT-4o评估可能产生API调用费用，建议先用ROUGE等指标进行初步评估

---

## 📞 技术栈

- **PyTorch**: 深度学习框架
- **Transformers**: Hugging Face模型库
- **Datasets**: 数据集处理
- **Accelerate**: 训练加速
- **ROUGE/NLTK**: 评估指标

---

## 📅 项目进度

- [x] 创建项目结构
- [ ] 数据集准备
- [ ] 模型设计与训练
- [ ] 推理与评估
- [ ] 提交结果

---

## 🎯 目标

实现高质量的心理病理报告总结，在GPT-4o评估中获得高分，确保：
- 总结与原文事实一致
- 涵盖所有关键信息
- 表达清晰简洁

