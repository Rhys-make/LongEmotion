# LongEmotion 比赛方案

这是一个完整的 LongEmotion 比赛解决方案，使用 FastAPI + Hugging Face Transformers + PyTorch 实现五个情感任务。

## 📋 比赛任务

1. **Emotion Classification** - 情感分类
2. **Emotion Detection** - 情感检测（多标签）
3. **Emotion Conversation** - 情感对话生成
4. **Emotion Summary** - 情感摘要生成
5. **Emotion QA** - 情感问答

## 🏗️ 项目结构

```
LongEmotion/
│
├── data/                          # 数据集缓存目录
│   ├── classification/
│   ├── detection/
│   ├── conversation/
│   ├── summary/
│   └── qa/
│
├── models/                        # 模型实现
│   ├── classification_model.py   # 分类模型
│   ├── detection_model.py        # 检测模型
│   ├── conversation_model.py     # 对话模型
│   ├── summary_model.py          # 摘要模型
│   └── qa_model.py               # 问答模型
│
├── utils/                         # 工具模块
│   ├── preprocess.py             # 文本预处理
│   ├── evaluator.py              # 评估指标
│   └── trainer.py                # 训练逻辑
│
├── api/                           # FastAPI 接口
│   └── main.py                   # API 服务入口
│
├── scripts/                       # 脚本
│   ├── download_dataset.py       # 数据集下载
│   ├── train_all.py              # 批量训练
│   └── inference_all.py          # 批量推理
│
├── checkpoints/                   # 模型检查点（训练后生成）
├── results/                       # 推理结果（推理后生成）
├── logs/                          # 训练日志
│
├── requirements.txt               # 依赖包
└── README.md                     # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

#### 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n longemotion python=3.10
conda activate longemotion

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖:**
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- FastAPI >= 0.104.0
- Datasets >= 2.14.0

### 2. 下载数据集

从 Hugging Face 下载 LongEmotion 数据集：

```bash
python scripts/download_dataset.py
```

**可选参数:**

```bash
# 下载特定任务
python scripts/download_dataset.py --tasks classification detection

# 指定缓存目录
python scripts/download_dataset.py --cache_dir ./data

# 列出已下载的数据集
python scripts/download_dataset.py --list

# 验证数据集完整性
python scripts/download_dataset.py --verify
```

### 3. 训练模型

#### 训练所有任务

```bash
python scripts/train_all.py
```

#### 训练特定任务

```bash
# 训练分类任务
python scripts/train_all.py --tasks classification

# 训练多个任务
python scripts/train_all.py --tasks classification detection summary

# 自定义参数
python scripts/train_all.py \
    --tasks classification \
    --num_epochs 5 \
    --batch_size 32 \
    --device cuda
```

**训练参数说明:**
- `--data_dir`: 数据目录（默认: `./data`）
- `--output_dir`: 模型输出目录（默认: `./checkpoints`）
- `--tasks`: 要训练的任务列表
- `--num_epochs`: 训练轮数（默认: 3）
- `--batch_size`: 批量大小（默认: 16）
- `--device`: 设备（`cuda` 或 `cpu`）

**训练特性:**
- ✅ 自动保存最佳模型
- ✅ 早停机制（3 轮不改进则停止）
- ✅ 学习率预热和衰减
- ✅ 梯度裁剪
- ✅ 训练历史记录

### 4. 生成推理结果

对测试集进行推理，生成提交文件：

```bash
python scripts/inference_all.py
```

**可选参数:**

```bash
# 推理特定任务
python scripts/inference_all.py --tasks classification detection

# 指定模型和数据路径
python scripts/inference_all.py \
    --checkpoint_dir ./checkpoints \
    --data_dir ./data \
    --output_dir ./results \
    --device cuda
```

**输出文件:**

推理完成后，会在 `results/` 目录生成以下文件：
- `classification_test.jsonl`
- `detection_test.jsonl`
- `conversation_test.jsonl`
- `summary_test.jsonl`
- `qa_test.jsonl`

### 5. 启动 API 服务

启动 FastAPI 服务，提供 REST API 接口：

```bash
cd api
python main.py
```

**可选参数:**

```bash
python main.py \
    --host 0.0.0.0 \
    --port 8000 \
    --checkpoint_dir ../checkpoints \
    --device cuda \
    --reload  # 开发模式自动重载
```

服务启动后：
- API 服务: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 交互式文档: http://localhost:8000/redoc

## 🔌 API 使用示例

### 情感分类

```python
import requests

response = requests.post(
    "http://localhost:8000/classify",
    json={"text": "今天天气真好，心情特别开心！"}
)

print(response.json())
# 输出:
# {
#     "label": 0,
#     "emotion": "happiness",
#     "confidence": 0.95,
#     "probabilities": {...}
# }
```

### 情感检测

```python
response = requests.post(
    "http://localhost:8000/detect",
    json={"text": "这部电影既让人感动又有点害怕"}
)

print(response.json())
# 输出:
# {
#     "emotions": [
#         {"emotion": "sadness", "score": 0.85},
#         {"emotion": "fear", "score": 0.72}
#     ],
#     "all_scores": {...}
# }
```

### 情感对话

```python
response = requests.post(
    "http://localhost:8000/conversation",
    json={
        "context": "我今天考试没考好，感觉很沮丧",
        "emotion": "happiness"  # 可选，指定回复的情感
    }
)

print(response.json())
# 输出:
# {
#     "response": "别灰心！失败是成功之母，下次一定能考好的！"
# }
```

### 情感摘要

```python
response = requests.post(
    "http://localhost:8000/summary",
    json={"text": "很长的文本内容..."}
)

print(response.json())
# 输出:
# {
#     "summary": "简短的摘要内容"
# }
```

### 情感问答

```python
response = requests.post(
    "http://localhost:8000/qa",
    json={
        "question": "主人公的情感是什么？",
        "context": "故事的上下文内容..."
    }
)

print(response.json())
# 输出:
# {
#     "answer": "快乐",
#     "confidence": 0.88
# }
```

## 🧠 模型架构

### 1. Emotion Classification
- **基础模型**: `bert-base-chinese`
- **架构**: BERT + 分类头
- **输出**: 7 类情感标签

### 2. Emotion Detection
- **基础模型**: `bert-base-chinese`
- **架构**: BERT + 多标签分类头
- **输出**: 多个情感的概率分布

### 3. Emotion Conversation
- **基础模型**: `Qwen2-1.5B` 或 `ChatGLM3-6B`
- **架构**: 因果语言模型
- **输出**: 生成式对话回复

### 4. Emotion Summary
- **基础模型**: `google/mt5-base`
- **架构**: Encoder-Decoder
- **输出**: 文本摘要

### 5. Emotion QA
- **基础模型**: `bert-base-chinese`
- **架构**: BERT + QA 头（span extraction）
- **输出**: 答案文本及位置

## 📊 评估指标

- **Classification**: Accuracy, F1-score, Precision, Recall
- **Detection**: Micro-F1, Macro-F1
- **Conversation**: ROUGE-1, ROUGE-2, ROUGE-L
- **Summary**: ROUGE-1, ROUGE-2, ROUGE-L
- **QA**: Exact Match, ROUGE

## 💡 训练技巧

### 超参数建议

| 任务 | 学习率 | 批量大小 | 训练轮数 | 最大长度 |
|-----|--------|---------|---------|---------|
| Classification | 2e-5 | 16-32 | 3-5 | 512 |
| Detection | 2e-5 | 16-32 | 3-5 | 512 |
| Conversation | 5e-5 | 4-8 | 2-3 | 1024 |
| Summary | 5e-5 | 8-16 | 3-5 | 1024 |
| QA | 3e-5 | 8-16 | 2-4 | 512 |

### GPU 内存优化

如果遇到显存不足：

```python
# 1. 减小批量大小
python scripts/train_all.py --batch_size 8

# 2. 使用梯度累积（修改 trainer.py）
# 3. 使用混合精度训练（torch.cuda.amp）
# 4. 使用 8-bit 量化加载大模型
```

### 对话和摘要模型

由于这两个任务需要较大的生成式模型，可以考虑：

1. **使用小模型**: Qwen2-1.5B, mT5-base
2. **LoRA 微调**: 仅训练少量参数
3. **提示工程**: 优化提示词模板

## 🔧 常见问题

### 1. 数据集下载失败

```bash
# 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后放到 data/ 目录
```

### 2. CUDA 内存不足

- 减小批量大小
- 使用梯度检查点
- 使用模型并行

### 3. 模型加载错误

确保模型路径正确：
```bash
ls checkpoints/classification/best_model/
# 应该包含 config.json, pytorch_model.bin 等文件
```

### 4. API 服务启动慢

模型会在首次调用时加载，可以：
- 预加载模型（修改 main.py）
- 使用模型缓存
- 减小模型大小

## 📝 提交指南

### 比赛提交要求

1. **推理结果**: 每个任务的 `test.jsonl` 文件
2. **模型仓库**: 上传模型到 Hugging Face
3. **代码仓库**: 上传代码到 GitHub/Hugging Face

### 上传模型到 Hugging Face

```python
from huggingface_hub import HfApi

api = HfApi()

# 上传分类模型
api.upload_folder(
    folder_path="./checkpoints/classification/best_model",
    repo_id="your-username/longemotion-classification",
    repo_type="model"
)
```

### 提交文件格式

每个 `test.jsonl` 文件格式示例：

```jsonl
{"text": "...", "label": 0, "emotion": "happiness", ...}
{"text": "...", "label": 1, "emotion": "sadness", ...}
```

## 🔬 扩展与改进

### 可能的改进方向

1. **数据增强**: 回译、同义词替换
2. **模型集成**: 多模型投票或融合
3. **对抗训练**: 提高模型鲁棒性
4. **多任务学习**: 共享编码器
5. **Prompt 工程**: 优化生成式任务的提示

### 添加新任务

1. 在 `models/` 下创建新模型类
2. 在 `scripts/train_all.py` 添加训练逻辑
3. 在 `scripts/inference_all.py` 添加推理逻辑
4. 在 `api/main.py` 添加 API 路由

## 📚 参考资源

- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **FastAPI 文档**: https://fastapi.tiangolo.com
- **PyTorch 文档**: https://pytorch.org/docs
- **LongEmotion 数据集**: https://huggingface.co/datasets/LongEmotion/LongEmotion

## 📄 许可证

MIT License

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

**祝你在 LongEmotion 比赛中取得好成绩！** 🏆
