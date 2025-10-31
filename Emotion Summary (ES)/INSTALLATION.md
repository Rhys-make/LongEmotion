# Emotion Summary (ES) - 安装指南

## 📦 依赖安装

### 方法1: 使用ES任务专用依赖文件（推荐）

在虚拟环境中运行以下命令：

```bash
pip install torch transformers datasets evaluate rouge-score nltk accelerate scikit-learn tqdm matplotlib seaborn huggingface-hub pyyaml psutil
```

### 方法2: 使用requirements文件

```bash
pip install -r "Emotion Summary (ES)/requirements_es.txt"
```

### 方法3: 使用项目根目录的requirements.txt

如果项目根目录的requirements.txt已经包含所有依赖：

```bash
pip install -r requirements.txt
```

## 📋 详细依赖列表

### 核心库（必需）

```bash
# 深度学习框架
pip install torch>=2.0.0

# Transformers和数据集
pip install transformers>=4.35.0
pip install datasets>=2.14.0

# 评估指标
pip install evaluate>=0.4.0
pip install rouge-score>=0.1.2
pip install nltk>=3.8.0
pip install scikit-learn>=1.3.0

# 训练加速
pip install accelerate>=0.24.0

# 工具库
pip install tqdm>=4.65.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install huggingface-hub>=0.19.0
```

### 可选库

```bash
# 可视化
pip install matplotlib>=3.5.0
pip install seaborn>=0.12.0

# 配置文件支持
pip install pyyaml>=6.0

# 系统监控
pip install psutil>=5.9.0

# GPT-4o评估（如需要）
pip install openai>=1.0.0

# TensorBoard监控（如需要）
pip install tensorboard>=2.14.0
```

## 🔧 NLTK数据下载

首次使用NLTK时需要下载数据：

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

或在命令行中：

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🎯 PyTorch安装（根据CUDA版本）

### CUDA 11.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU版本

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## ✅ 验证安装

运行以下脚本验证安装是否成功：

```python
import torch
import transformers
import datasets
import evaluate
from rouge_score import rouge_scorer
import nltk

print("✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ Transformers version:", transformers.__version__)
print("✓ Datasets version:", datasets.__version__)
print("✓ Evaluate version:", evaluate.__version__)
print("✓ 所有依赖安装成功！")
```

保存为 `test_install.py` 并运行：

```bash
python test_install.py
```

## 🔍 常见问题

### 1. torch安装失败

- 确认CUDA版本
- 使用官方PyTorch安装命令（访问 pytorch.org）
- 如果在CPU上运行，安装CPU版本

### 2. transformers版本冲突

```bash
pip install --upgrade transformers
```

### 3. NLTK数据下载失败

手动下载并放到NLTK数据目录：
- Windows: `C:\Users\<用户名>\AppData\Roaming\nltk_data`
- Linux/Mac: `~/nltk_data`

### 4. 网络问题导致安装失败

使用国内镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers datasets
```

## 💾 磁盘空间需求

- **依赖库**: ~5GB
- **预训练模型**: 
  - T5-base: ~900MB
  - T5-large: ~3GB
  - LongT5-base: ~1GB
- **数据集**: ~500MB（根据实际大小）
- **总计**: 建议预留至少10GB空间

## 🚀 安装后下一步

1. 验证安装成功
2. 运行数据准备脚本
3. 开始训练模型

详见 `QUICK_START.md`

