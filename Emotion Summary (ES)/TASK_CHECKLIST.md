# 📋 Emotion Summary (ES) 任务完整清单

## 📊 任务概述
- **任务目标**: 为心理咨询案例生成经验与反思摘要
- **时间限制**: 4.3 小时训练时间
- **数据来源**: PsyQA 中文心理问答数据集
- **模型选择**: mT5-base (多语言 T5，适合中文摘要任务)

---

## ✅ 已完成工作

### 1. 数据准备 ✅
- **训练集**: 85 个样本
  - 路径: `data/train/Emotion_Summary.jsonl`
  - 来源: PsyQA 数据集 (85% 分割)
  
- **验证集**: 15 个样本
  - 路径: `data/validation/Emotion_Summary.jsonl`
  - 来源: PsyQA 数据集 (15% 分割)
  
- **测试集**: 150 个样本
  - 路径: `data/test/Emotion_Summary.jsonl`
  - 来源: 比赛提供的官方测试集

**数据格式** (所有数据集格式一致):
```json
{
  "id": 整数ID,
  "case_description": [字符串列表],
  "consultation_process": [字符串列表],
  "experience_and_reflection": "字符串"
}
```

### 2. 脚本开发 ✅

#### 核心脚本:
- ✅ `scripts/simple_train.py` - **简化训练脚本** (推荐使用)
  - 使用 Hugging Face Trainer API
  - 针对 mT5-base 优化
  - 自动处理编码、优化器等问题
  - 内存优化配置
  
- ✅ `scripts/download_and_convert_psyqa.py` - 数据下载和转换
  - 从 GitHub 克隆 PsyQA 数据集
  - 转换为 ES 任务格式
  - 自动分割训练/验证集
  - 生成合成的 `experience_and_reflection` 字段

#### 辅助脚本:
- ✅ `scripts/train.py` - 通用 Seq2Seq 训练脚本
- ✅ `scripts/train_qwen2.py` - Qwen2 专用训练脚本 (需要更多资源)
- ✅ `scripts/inference.py` - 推理脚本 (训练后使用)
- ✅ `scripts/evaluate.py` - 评估脚本 (训练后使用)

#### 工具脚本:
- ✅ `check_status.py` - 任务状态自查脚本

### 3. 配置文件 ✅
- ✅ `config/config.py` - 模型和训练配置

### 4. 环境准备 ✅
已安装依赖:
- ✅ `transformers` - Hugging Face 模型库
- ✅ `torch` - PyTorch 深度学习框架
- ✅ `datasets` - 数据集处理库
- ✅ `protobuf` - Protocol Buffers (tokenizer 依赖)
- ✅ `tiktoken` - Tokenizer 库
- ✅ `sentencepiece` - SentencePiece tokenizer

---

## 🎯 当前状态

### 训练准备状态: ✅ 100% 完成

所有必需条件已满足:
- ✅ 数据集已准备完毕 (训练集 + 验证集 + 测试集)
- ✅ 训练脚本已优化并测试
- ✅ 所有依赖已安装
- ✅ 模型目录已创建

---

## 📝 下一步操作 (立即执行)

### 开始训练

```powershell
# 步骤 1: 进入项目目录
cd "Emotion Summary (ES)"

# 步骤 2: 开始训练
python scripts/simple_train.py
```

### 预计训练时间: 2-3 小时

**训练参数** (已在 `simple_train.py` 中配置):
- 模型: `google/mt5-base`
- 训练轮数: 3 epochs
- 批次大小: 1 (带梯度累积 4 步，等效 batch_size=4)
- 输入长度: 512 tokens
- 输出长度: 256 tokens
- 学习率: 3e-4

**模型保存位置**:
- 训练过程检查点: `model/mt5_emotion_summary/checkpoint-*`
- 最终模型: `model/mt5_emotion_summary/final/`

---

## 📂 重要文件路径总结

### 数据文件
```
Emotion Summary (ES)/
├── data/
│   ├── train/
│   │   └── Emotion_Summary.jsonl          (85 样本)
│   ├── validation/
│   │   └── Emotion_Summary.jsonl          (15 样本)
│   └── test/
│       └── Emotion_Summary.jsonl          (150 样本，官方测试集)
```

### 脚本文件
```
Emotion Summary (ES)/
├── scripts/
│   ├── simple_train.py                     ⭐ 主训练脚本
│   ├── download_and_convert_psyqa.py       (数据准备)
│   ├── inference.py                        (推理脚本，训练后使用)
│   └── evaluate.py                         (评估脚本，训练后使用)
```

### 模型文件
```
Emotion Summary (ES)/
├── model/
│   └── mt5_emotion_summary/
│       ├── checkpoint-50/                  (训练过程保存点)
│       ├── checkpoint-100/
│       └── final/                          ⭐ 最终训练模型
```

### 配置和文档
```
Emotion Summary (ES)/
├── config/
│   └── config.py                           (配置文件)
├── check_status.py                         (状态检查脚本)
├── TASK_CHECKLIST.md                       ⭐ 本文档
└── README.md                               (项目说明)
```

---

## 🔧 已解决的问题

### 问题 1: 数据集选择
- **解决方案**: 选择 PsyQA 数据集，因为它是中文心理问答，与测试集格式最匹配

### 问题 2: 模型选择
- **初始计划**: Qwen2-7B-Instruct (7B 参数，需要 4-bit 量化)
- **问题**: 训练时间过长 (8-10 小时)，超出 4.3 小时限制
- **解决方案**: 改用 mT5-base (580M 参数)，更适合时间限制

### 问题 3: bitsandbytes 兼容性
- **问题**: Windows 系统不支持 bitsandbytes
- **解决方案**: 不使用 4-bit 量化，直接使用 FP32 训练 mT5-base

### 问题 4: 编码错误
- **问题**: `SyntaxError: Non-UTF-8 code`
- **解决方案**: 在脚本开头添加 `# -*- coding: utf-8 -*-`

### 问题 5: AdamW 导入错误
- **问题**: `ImportError: cannot import name 'AdamW' from 'transformers'`
- **解决方案**: 使用 `Seq2SeqTrainer` API，自动处理优化器

### 问题 6: TrainingArguments 参数错误
- **问题**: `evaluation_strategy` 参数在新版本中已更名
- **解决方案**: 使用 `eval_strategy` 替代 `evaluation_strategy`

### 问题 7: 缺少依赖
- **问题**: 缺少 `datasets`, `protobuf`, `tiktoken`, `sentencepiece`
- **解决方案**: 逐个安装所有依赖，使用国内镜像加速

### 问题 8: 内存优化
- **问题**: 训练可能消耗大量内存
- **解决方案**: 
  - 批次大小降为 1
  - 序列长度限制为 512/256
  - 禁用多进程数据加载
  - 使用梯度累积

---

## 📊 训练监控

### 训练过程中会看到:
```
======================================================================
开始训练 mT5 模型
======================================================================

加载模型: google/mt5-base
(首次运行会下载模型，约 2GB)

加载数据...
训练集: 85 样本
验证集: 15 样本

预处理数据...

开始训练...
预计时间: 2-3 小时（已优化内存使用）
----------------------------------------------------------------------

训练进度显示：
{'loss': 损失值, 'learning_rate': 学习率, 'epoch': 轮数}
[步数/总步数 剩余时间, 每秒迭代数]

每 50 步保存一次检查点
每 50 步评估一次验证集

----------------------------------------------------------------------

保存模型...

======================================================================
✅ 训练完成！
📁 模型保存在: model/mt5_emotion_summary/final
======================================================================
```

---

## 🚀 训练后步骤

### 1. 使用训练好的模型进行推理
```powershell
python scripts/inference.py
```

### 2. 评估模型性能
```powershell
python scripts/evaluate.py
```

### 3. 生成提交文件
(需要根据比赛要求创建提交脚本)

---

## 💡 关键决策总结

1. **数据集**: PsyQA (中文心理问答，100个样本)
   - 优点: 领域匹配，中文数据，格式适合
   - 缺点: 数据量小，需要生成合成的反思内容

2. **模型**: mT5-base (580M 参数)
   - 优点: 多语言支持（中文强），适合摘要，训练快速
   - 缺点: 比 7B 模型能力弱

3. **训练策略**: 标准 fine-tuning (无量化)
   - 优点: 简单稳定，兼容性好
   - 缺点: 内存占用较高

4. **时间分配**: 
   - 模型下载: 15-30 分钟 ✅ (已完成)
   - 模型训练: 2-3 小时 ⏳ (即将开始)
   - 推理测试: 30 分钟
   - 总计: ~4 小时 (符合 4.3 小时限制)

---

## ⚠️ 注意事项

1. **不要中断训练**: 训练过程中会自动保存检查点，但中断会浪费时间
2. **监控显存/内存**: 如果内存不足，训练会自动失败
3. **网络稳定**: 首次运行需要下载模型，确保网络稳定
4. **使用镜像源**: 如果下载慢，使用清华/阿里云镜像

---

## 📞 快速参考命令

```powershell
# 检查任务状态
python "Emotion Summary (ES)\check_status.py"

# 开始训练
cd "Emotion Summary (ES)"
python scripts/simple_train.py

# 训练完成后推理
python scripts/inference.py

# 查看模型文件
dir model\mt5_emotion_summary\final
```

---

**文档最后更新**: 2025-10-30
**任务状态**: ✅ 准备就绪，可以开始训练
**下一步**: 运行 `python scripts/simple_train.py` 开始训练

