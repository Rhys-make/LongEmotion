# 情绪分类数据集准备脚本

## 📖 简介

本脚本用于自动下载和格式化情绪分类任务所需的数据集，包括：
- **GoEmotions** 数据集（训练集和验证集）
- **LongEmotion** 数据集（测试集）

## 🚀 使用方法

### 1. 确保已安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `datasets` - Hugging Face 数据集库
- `tqdm` - 进度条显示
- `spacy` - 自然语言处理库（用于主语提取）

### 1.5. 下载 spacy 英文模型（首次运行会自动下载）

如果想手动安装：
```bash
python -m spacy download en_core_web_sm
```

脚本会在首次运行时自动检查并下载该模型。

### 2. 运行脚本

在项目根目录下运行：

```bash
python scripts/classification/prepare_datasets.py
```

或者：

```bash
cd scripts/classification
python prepare_datasets.py
```

### 3. 等待下载和处理

脚本会自动：
1. 从 Hugging Face 下载 GoEmotions 数据集
2. 将数据转换为统一格式
3. 下载 LongEmotion 数据集的测试集
4. 保存为 JSONL 格式

## 📂 输出文件

脚本执行完成后，会在 `../data/` 目录下生成以下文件：

```
../data/
├── train.jsonl       # 训练集（来自 GoEmotions）
├── validation.jsonl  # 验证集（来自 GoEmotions）
└── test.jsonl        # 测试集（来自 LongEmotion）
```

## 📋 数据格式

### 训练集和验证集格式

```json
{
    "id": 0,
    "Context": "I love this new feature!",
    "Subject": "unknown",
    "Choices": ["admiration", "amusement", "anger", "..."],
    "Answer": "admiration"
}
```

**注意**：`Subject` 字段使用 spacy 自动提取。如果提取失败或主语是代词，会返回 "unknown"。

### 测试集格式

测试集包含相同的字段，但可能不包含 `Answer` 字段（取决于 LongEmotion 数据集的结构）。

## 📊 GoEmotions 情绪类别

GoEmotions 包含 28 种情绪类别：
- admiration, amusement, anger, annoyance, approval
- caring, confusion, curiosity, desire, disappointment
- disapproval, disgust, embarrassment, excitement, fear
- gratitude, grief, joy, love, nervousness
- optimism, pride, realization, relief, remorse
- sadness, surprise, neutral

## ⚠️ 注意事项

1. **网络连接**: 确保能够访问 Hugging Face 网站
2. **磁盘空间**: 确保有足够的磁盘空间（约 100-200 MB）
3. **下载时间**: 首次下载可能需要几分钟，取决于网络速度
4. **多标签处理**: GoEmotions 是多标签数据集，脚本默认取第一个标签
5. **主语提取**: 使用 spacy 的依存句法分析提取主语，处理速度会相对较慢
6. **spacy 模型**: 首次运行会自动下载英文模型（约 12 MB）

## 🔧 自定义

如果需要修改数据处理逻辑，可以编辑脚本中的以下函数：
- `extract_subject()` - 调整主语提取逻辑
- `convert_go_emotions_to_format()` - 调整 GoEmotions 数据转换逻辑
- `download_and_process_long_emotion()` - 调整 LongEmotion 数据处理逻辑

### 主语提取逻辑

脚本使用 spacy 的依存句法分析来提取主语：
- 查找依存关系为 `nsubj`（主语）或 `nsubjpass`（被动语态主语）的词
- 包含修饰词（如复合名词、形容词等）
- 如果主语是代词（如 I, you, he, she 等）或过长（>50字符），返回 "unknown"

## 📞 问题排查

### 问题：无法下载数据集
- 检查网络连接
- 尝试使用 VPN 或代理
- 确认 Hugging Face 数据集地址是否正确

### 问题：内存不足
- 脚本使用流式处理，内存占用应该不大
- 如果仍然出现问题，可以分批处理数据

### 问题：LongEmotion 数据集下载失败
- 确认数据集名称是否正确
- 检查是否有权限访问该数据集
- 可以手动下载并放置到对应目录

