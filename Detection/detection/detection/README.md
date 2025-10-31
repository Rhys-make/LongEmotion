# Emotion Detection Model - LongEmotion

情感检测模型，基于BERT的中文情感分类器

## 📊 模型信息

- **基础模型**: bert-base-chinese
- **任务类型**: 6分类情感检测
- **验证准确率**: 91.47%
- **框架**: PyTorch + Transformers

## 🏷️ 情感类别

模型可以识别以下6种情感：
- `sadness` (悲伤)
- `joy` (快乐)
- `love` (爱)
- `anger` (愤怒)
- `fear` (恐惧)
- `surprise` (惊讶)

## 📁 文件说明

```
detection_hug/
├── model.pt                      # 模型权重文件
├── config.json                   # 模型配置
├── tokenizer_config.json         # 分词器配置
├── vocab.txt                     # 词表
├── special_tokens_map.json       # 特殊符号映射
├── detection_model.py            # 模型定义
├── inference_example.py          # 推理示例
└── README.md                     # 本文件
```

## 🚀 快速开始

### 环境要求

```bash
pip install torch transformers
```

### 基本使用

```python
import torch
from transformers import BertTokenizer
from detection_model import EmotionDetectionModel

# 1. 加载分词器
tokenizer = BertTokenizer.from_pretrained(".")

# 2. 加载模型
model = EmotionDetectionModel(
    model_name="bert-base-chinese",
    num_emotions=6
)
checkpoint = torch.load("model.pt", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

# 3. 预测
text = "我今天很开心！"
encoding = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
outputs = model(**encoding)
predicted_emotion = torch.argmax(outputs['logits'], dim=-1).item()

# 情感映射
emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
print(f"预测情感: {emotions[predicted_emotion]}")
```

### 使用推理脚本

```bash
python inference_example.py
```

## 📈 模型性能

- **数据集**: dair-ai/emotion (中文情感数据)
- **验证准确率**: 91.47%
- **平均置信度**: 89.27%
- **最大序列长度**: 512 tokens

## 🔧 技术细节

### 模型架构

```
EmotionDetectionModel
├── BERT Encoder (bert-base-chinese)
│   └── 768-dim hidden states
├── Dropout (p=0.1)
├── Linear Layer (768 → 384)
├── ReLU Activation
├── Dropout (p=0.1)
└── Output Layer (384 → 6)
```

### 训练参数

- **优化器**: AdamW
- **学习率**: 2e-5
- **批次大小**: 16
- **最大长度**: 512

## 📝 引用

如果您使用此模型，请注明：
```
LongEmotion Detection Model
- 基于 bert-base-chinese
- 训练于 dair-ai/emotion 数据集
```

## 📧 联系方式

如有问题，请通过项目仓库联系。

---

**License**: 遵循 bert-base-chinese 的许可协议
**Created**: 2025

