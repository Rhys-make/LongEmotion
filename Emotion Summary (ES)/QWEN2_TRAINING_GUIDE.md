# Qwen2-7B-Instruct 微调指南
## Emotion Summary (ES) 任务

---

## 📋 目录

1. [模型选择理由](#模型选择理由)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
4. [开始训练](#开始训练)
5. [推理使用](#推理使用)
6. [参数说明](#参数说明)

---

## 🎯 模型选择理由

**Qwen2-7B-Instruct** 是最适合本项目的模型：

### 为什么选择 Qwen2？

| 特性 | 说明 |
|------|------|
| ✅ **中文优化** | 阿里巴巴专门优化的中文大模型，中文理解能力最强 |
| ✅ **长文本支持** | 支持最长 32K tokens，完美处理长篇心理咨询案例 |
| ✅ **PsyQA 匹配** | 与我们使用的 PsyQA 数据集完美匹配 |
| ✅ **指令遵循** | Instruct 版本经过指令微调，更适合任务导向 |
| ✅ **开源友好** | 完全开源，可商用，社区支持好 |
| ✅ **参数高效** | 7B 参数，配合 LoRA 微调，显存友好 |

### 与其他模型对比

- **vs LLaMA-3.1-8B**: Qwen2 中文能力更强，更适合中文心理咨询
- **vs Mistral-7B**: Qwen2 专门针对中文优化，PsyQA 效果更好
- **vs Seq2Seq模型** (T5/BART): 因果语言模型更适合生成长文本摘要

---

## 🛠 环境准备

### 1. 系统要求

- **GPU**: 推荐 16GB+ 显存（使用 4-bit 量化最低 12GB）
- **内存**: 16GB+
- **磁盘**: 50GB+ 可用空间
- **系统**: Windows / Linux / macOS

### 2. 安装依赖

```bash
# 进入 ES 目录
cd "Emotion Summary (ES)"

# 安装 Qwen2 相关依赖
pip install -r requirements_qwen2.txt

# 或者安装全部依赖
pip install -r requirements_es.txt
```

### 3. 验证环境

```python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

---

## 📊 数据准备

数据已经通过 `download_and_convert_psyqa.py` 准备好：

```
data/
├── train/
│   └── Emotion_Summary.jsonl      # 训练集 (85%)
├── validation/
│   └── Emotion_Summary.jsonl      # 验证集 (15%)
└── test/
    └── Emotion_Summary.jsonl       # 测试集
```

### 数据格式

```json
{
  "id": 1,
  "case_description": ["案例描述..."],
  "consultation_process": ["咨询过程1", "咨询过程2", ...],
  "experience_and_reflection": "经验与反思总结..."
}
```

---

## 🚀 开始训练

### 方式 1: 使用脚本（推荐）

#### Windows:
```bash
cd scripts
start_training_qwen2.bat
```

#### Linux/macOS:
```bash
cd scripts
chmod +x start_training_qwen2.sh
./start_training_qwen2.sh
```

### 方式 2: 直接运行 Python

```bash
cd scripts
python train_qwen2.py
```

### 方式 3: 自定义参数

```bash
python train_qwen2.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --num_epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --max_length 8192 \
    --lora_r 64
```

---

## 📈 训练过程

训练会自动执行以下步骤：

1. **加载模型** - 下载 Qwen2-7B-Instruct (首次运行)
2. **应用量化** - 4-bit 量化节省显存
3. **配置 LoRA** - 添加 LoRA 适配器层
4. **准备数据** - 加载训练和验证数据
5. **开始训练** - 进行微调训练
6. **保存模型** - 保存 LoRA 权重

### 预期时间

- **单个 epoch**: ~2-4小时 (取决于GPU)
- **总训练时间**: ~6-12小时 (3 epochs)

### 显存使用

- **4-bit + LoRA**: ~12GB
- **FP16 + LoRA**: ~20GB
- **Full Fine-tuning**: ~40GB+

---

## 🔮 推理使用

训练完成后，使用模型进行推理：

```python
from model_qwen2 import Qwen2EmotionSummaryModel

# 加载微调后的模型
model = Qwen2EmotionSummaryModel(
    model_name="Qwen/Qwen2-7B-Instruct",
    use_lora=True
)

# 加载 LoRA 权重
model.load_lora_weights("../model/qwen2_emotion_summary/final")

# 生成摘要
case_desc = ["来访者，女性，27岁..."]
consult_process = ["咨询过程1", "咨询过程2"]

summary = model.generate_summary(
    case_description=case_desc,
    consultation_process=consult_process
)

print(summary)
```

---

## ⚙️ 参数说明

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `Qwen/Qwen2-7B-Instruct` | 预训练模型名称 |
| `use_4bit` | `True` | 是否使用 4-bit 量化 |
| `use_lora` | `True` | 是否使用 LoRA 微调 |

### LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_r` | `64` | LoRA 秩（越大越精确，但显存占用更多） |
| `lora_alpha` | `16` | LoRA 缩放因子 |
| `lora_dropout` | `0.05` | LoRA dropout 率 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_epochs` | `3` | 训练轮数 |
| `batch_size` | `2` | 每个设备的批次大小 |
| `gradient_accumulation_steps` | `8` | 梯度累积步数 |
| `learning_rate` | `2e-4` | 学习率 |
| `max_length` | `8192` | 最大序列长度 |

### 有效批次大小计算

```
有效批次大小 = batch_size × gradient_accumulation_steps × GPU数量
默认: 2 × 8 × 1 = 16
```

---

## 📝 常见问题

### Q1: 显存不足怎么办？

**方案1**: 减小 `batch_size`
```bash
python train_qwen2.py --batch_size 1
```

**方案2**: 增加 `gradient_accumulation_steps`
```bash
python train_qwen2.py --batch_size 1 --gradient_accumulation_steps 16
```

**方案3**: 减小 `max_length`
```bash
python train_qwen2.py --max_length 4096
```

**方案4**: 减小 `lora_r`
```bash
python train_qwen2.py --lora_r 32
```

### Q2: 训练速度慢怎么办？

1. 确保使用 GPU (检查 CUDA)
2. 使用 4-bit 量化 (`--use_4bit`)
3. 启用梯度检查点 (默认开启)
4. 考虑使用更小的 `max_length`

### Q3: 如何继续训练？

```bash
python train_qwen2.py --output_dir ../model/qwen2_emotion_summary
# 模型会自动从最后一个 checkpoint 继续
```

### Q4: 如何评估模型？

使用 `inference.py` 脚本：

```bash
python inference.py --model_path ../model/qwen2_emotion_summary/final
```

---

## 📊 性能监控

训练过程中可以监控：

1. **Loss 曲线** - 查看 `output_dir/runs/` 使用 TensorBoard
2. **GPU 使用率** - 使用 `nvidia-smi` 或 `watch -n 1 nvidia-smi`
3. **训练日志** - 查看终端输出

```bash
# 启动 TensorBoard
tensorboard --logdir=../model/qwen2_emotion_summary/runs
```

---

## 🎉 总结

Qwen2-7B-Instruct + LoRA 方案的优势：

- ✅ **中文最优**: 专门优化的中文模型
- ✅ **长文本**: 支持 32K context
- ✅ **显存友好**: 4-bit + LoRA 仅需 12GB
- ✅ **训练快速**: LoRA 参数少，收敛快
- ✅ **效果出色**: PsyQA 数据集表现优异

---

## 📚 相关资源

- [Qwen2 官方仓库](https://github.com/QwenLM/Qwen2)
- [Qwen2 模型卡片](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

---

**祝训练顺利！** 🚀

