# 🎯 Qwen2-7B-Instruct 情绪总结模型

<div align="center">

**基于 Qwen2-7B-Instruct + LoRA 的长文本心理咨询摘要模型**

[![Model](https://img.shields.io/badge/Model-Qwen2--7B-blue)](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
[![Method](https://img.shields.io/badge/Method-LoRA-green)](https://github.com/microsoft/LoRA)
[![Language](https://img.shields.io/badge/Language-中文-red)](https://github.com/thu-coai/PsyQA)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

</div>

---

## 📌 项目简介

本项目为 **LongEmotion 比赛 - Emotion Summary (ES)** 任务提供完整的解决方案。

### 任务目标
从长篇心理咨询案例中生成专业的**经验与反思总结**，包括：
- 案例描述理解
- 咨询过程分析
- 专业经验反思

### 技术方案
- **模型**: Qwen2-7B-Instruct（阿里巴巴中文大模型）
- **微调方法**: LoRA 参数高效微调
- **优化**: 4-bit 量化 + 梯度检查点
- **数据**: PsyQA 中文心理问答数据集

---

## 🌟 核心特点

### ✅ 为什么选择 Qwen2？

| 特性 | 说明 |
|------|------|
| 🇨🇳 **中文优化** | 专为中文优化，心理咨询文本理解最强 |
| 📏 **长文本支持** | 支持 32K tokens，处理完整咨询案例 |
| 🎯 **指令遵循** | Instruct 版本，任务导向能力强 |
| 💾 **显存友好** | 4-bit + LoRA 仅需 12GB 显存 |
| ⚡ **训练高效** | LoRA 只训练 <1% 参数，快速收敛 |

### ✅ 技术亮点

```
📊 性能对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
               显存需求   训练时间
全量微调        40GB+     很慢
LoRA微调       20GB      中等
4-bit+LoRA     12GB      快  ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
# 使用国内镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_qwen2.txt
```

### 2️⃣ 开始训练

```bash
cd scripts

# Windows
start_training_qwen2.bat

# Linux/Mac
./start_training_qwen2.sh

# 或直接运行
python train_qwen2.py
```

### 3️⃣ 生成预测

```bash
python inference_qwen2.py --model_path ../model/qwen2_emotion_summary/final
```

---

## 📁 项目结构

```
Emotion Summary (ES)/
├── 📊 data/                          # 数据目录
│   ├── train/                        # 训练集 (85 samples)
│   ├── validation/                   # 验证集 (15 samples)
│   └── test/                         # 测试集
│
├── 🤖 model/                         # 模型目录
│   └── qwen2_emotion_summary/
│       └── final/                    # ⭐ 训练后的 LoRA 权重
│
├── 📝 scripts/                       # 脚本目录
│   ├── model_qwen2.py               # ⭐ Qwen2 模型封装
│   ├── train_qwen2.py               # ⭐ 训练脚本
│   ├── inference_qwen2.py           # ⭐ 推理脚本
│   ├── download_and_convert_psyqa.py # 数据转换
│   └── start_training_qwen2.*       # 启动脚本
│
├── 📚 文档/
│   ├── QWEN2_TRAINING_GUIDE.md      # ⭐ 训练指南
│   ├── DEPLOYMENT_GUIDE.md          # ⭐ 部署指南
│   ├── PROJECT_SUMMARY.md           # 项目总结
│   └── README_QWEN2.md              # 本文件
│
└── 📦 requirements_qwen2.txt         # 依赖列表
```

---

## 💡 使用示例

### 训练模型

```python
# scripts/train_qwen2.py
python train_qwen2.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --lora_r 64
```

### 推理预测

```python
from model_qwen2 import Qwen2EmotionSummaryModel

# 加载模型
model = Qwen2EmotionSummaryModel(
    model_name="Qwen/Qwen2-7B-Instruct",
    use_lora=True,
    use_4bit=True
)

# 加载 LoRA 权重
model.load_lora_weights("../model/qwen2_emotion_summary/final")

# 生成摘要
summary = model.generate_summary(
    case_description=["案例描述..."],
    consultation_process=["咨询过程..."]
)

print(summary)
```

---

## 📊 数据说明

### 数据来源
- **PsyQA**: 中文心理问答数据集
- **链接**: https://github.com/thu-coai/PsyQA
- **当前**: 100 个示例（已转换）
- **建议**: 申请完整数据集（20,000+ 样本）

### 数据格式

```json
{
  "id": 1,
  "case_description": [
    "来访者基本信息和主诉..."
  ],
  "consultation_process": [
    "咨询过程段落1",
    "咨询过程段落2",
    "..."
  ],
  "experience_and_reflection": "专业的经验与反思总结..."
}
```

---

## ⚙️ 配置参数

### 推荐配置（12GB 显存）

```python
model_name = "Qwen/Qwen2-7B-Instruct"
use_4bit = True              # 4-bit 量化
lora_r = 64                  # LoRA rank
lora_alpha = 16              # LoRA alpha
batch_size = 2               # 批次大小
gradient_accumulation = 8    # 梯度累积
max_length = 8192            # 最大长度
learning_rate = 2e-4         # 学习率
num_epochs = 3               # 训练轮数
```

### 更多显存（24GB+）

```python
batch_size = 4               # 增大批次
gradient_accumulation = 4    
max_length = 16384           # 更长上下文
lora_r = 128                 # 更高精度
```

### 显存不足（8GB）

```python
batch_size = 1               # 减小批次
gradient_accumulation = 16   # 增加累积
max_length = 4096            # 减小长度
lora_r = 32                  # 降低 rank
```

---

## 📈 性能预期

### 训练时间

| GPU | 单 Epoch | 3 Epochs |
|-----|----------|----------|
| RTX 4090 | 1-2h | 3-6h |
| RTX 3090 | 2-3h | 6-9h |
| RTX 3080 | 3-4h | 9-12h |
| V100 | 2-3h | 6-9h |

### 显存占用

| 配置 | 显存占用 |
|------|----------|
| 4-bit + LoRA (r=64) | ~12GB ⭐ |
| FP16 + LoRA (r=64) | ~20GB |
| Full Fine-tuning | ~40GB+ |

---

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| **[QWEN2_TRAINING_GUIDE.md](QWEN2_TRAINING_GUIDE.md)** | 详细训练教程 |
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | 完整部署指南 |
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | 项目完成总结 |

---

## 🐛 常见问题

### Q: 显存不足怎么办？

**A**: 调整参数
```bash
python train_qwen2.py --batch_size 1 --max_length 4096 --lora_r 32
```

### Q: 如何使用完整 PsyQA 数据集？

**A**: 联系作者申请
- 邮箱: thu-sunhao@foxmail.com
- 需要签署数据使用协议

### Q: 如何继续训练？

**A**: 指定 output_dir
```bash
python train_qwen2.py --output_dir ../model/qwen2_emotion_summary
```

### Q: 推理速度慢？

**A**: 使用量化推理
```python
model = Qwen2EmotionSummaryModel(use_4bit=True)
```

---

## 🔗 相关资源

- [Qwen2 官方仓库](https://github.com/QwenLM/Qwen2)
- [Qwen2-7B-Instruct 模型](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [PsyQA 数据集](https://github.com/thu-coai/PsyQA)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

---

## 📄 许可证

本项目遵循 Apache 2.0 许可证。

- Qwen2 模型遵循 Apache 2.0
- PsyQA 数据集需要签署使用协议

---

## 🙏 致谢

- **Qwen 团队**: 提供优秀的中文大模型
- **PsyQA 团队**: 提供高质量心理咨询数据集
- **Hugging Face**: 提供强大的 Transformers 和 PEFT 库

---

## 📞 联系方式

如有问题，请：
1. 查阅详细文档
2. 提交 GitHub Issue
3. 联系项目维护者

---

<div align="center">

**🎉 祝训练顺利！**

如果这个项目对你有帮助，请给个 ⭐ Star！

</div>

