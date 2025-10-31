# 🚀 开始训练 - 详细步骤

## ✅ 当前位置检查

您现在应该在:
```
C:\Users\xsz20\project(self)\LongEmotion\
```

## 📝 正确的训练命令

### 步骤 1: 进入 ES 目录
```powershell
cd "Emotion Summary (ES)"
```

### 步骤 2: 验证位置
```powershell
pwd
# 应该显示: C:\Users\xsz20\project(self)\LongEmotion\Emotion Summary (ES)
```

### 步骤 3: 开始训练
```powershell
python scripts/simple_train.py
```

## 🎯 完整命令（一次性执行）

如果您当前在 `LongEmotion` 目录:
```powershell
cd "Emotion Summary (ES)"; python scripts/simple_train.py
```

或者分两步:
```powershell
cd "Emotion Summary (ES)"
python scripts/simple_train.py
```

## ⏱️ 训练预期

- **数据集**: 16,485 训练样本 + 4,122 验证样本
- **轮数**: 5 epochs
- **预计时间**: 4-6 小时
- **保存位置**: `model/mt5_emotion_summary/final/`

## 📊 训练过程中会看到

```
======================================================================
开始训练 mT5 模型
======================================================================

加载模型: google/mt5-base
(首次运行会下载模型，约 2GB)

加载数据...
训练集: 16485 样本
验证集: 4122 样本

预处理数据...

开始训练...
预计时间: 4-6 小时（已优化长序列处理）
----------------------------------------------------------------------

[步数/总步数 剩余时间, 每秒迭代数]
{'loss': ..., 'learning_rate': ..., 'epoch': ...}

每 500 步保存检查点
每 500 步评估验证集

----------------------------------------------------------------------

✅ 训练完成！
📁 模型保存在: model/mt5_emotion_summary/final
======================================================================
```

## ⚠️ 如果遇到错误

### 错误 1: "No such file or directory"
- **原因**: 路径不对
- **解决**: 确保先 `cd "Emotion Summary (ES)"`

### 错误 2: "No module named 'xxx'"
- **原因**: 缺少依赖
- **解决**: `pip install datasets transformers torch`

### 错误 3: 内存不足
- **原因**: 显存/内存不够
- **解决**: 已优化参数，应该可以运行

## 🎯 训练完成后

模型会保存在:
```
Emotion Summary (ES)/model/mt5_emotion_summary/final/
```

然后可以进行推理:
```powershell
python scripts/inference.py
```

