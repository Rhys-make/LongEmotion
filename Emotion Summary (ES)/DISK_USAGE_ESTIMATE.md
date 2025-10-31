# 💾 磁盘占用估算

## 📊 详细占用分析

### 1. 模型文件大小

**mT5-base 基础信息**:
- 参数量: 580M (5.8亿参数)
- 单个模型文件: **~1.2GB** (FP32)
- 优化器状态: **~2.4GB** (Adam 需要存储动量和方差)

### 2. 训练过程中的占用

#### Checkpoint 保存策略
```python
save_steps = 500          # 每500步保存
save_total_limit = 2      # 只保留最近2个checkpoint
```

**单个 Checkpoint 包含**:
- `model.safetensors`: 1.2GB (模型权重)
- `optimizer.pt`: 2.4GB (优化器状态)
- `scheduler.pt`: 5MB (学习率调度器)
- `trainer_state.json`: 1MB (训练状态)
- `training_args.bin`: 1MB (训练参数)
- **单个checkpoint总计**: **~3.6GB**

**训练过程中**:
- 保留2个最近的checkpoint: 3.6GB × 2 = **7.2GB**
- 最终模型 (final): **1.2GB** (只有权重，无优化器状态)
- **峰值总计**: **8.4GB**

### 3. 数据集占用

```
data/train/Emotion_Summary.jsonl      ~150MB (16,485 样本)
data/validation/Emotion_Summary.jsonl  ~40MB (4,122 样本)
data/test/Emotion_Summary.jsonl        ~10MB (150 样本)
                                      --------
                                      ~200MB
```

### 4. Hugging Face 缓存

```
~/.cache/huggingface/hub/
├── models--google--mt5-base/         ~2.5GB (预训练模型)
└── datasets--facebook--empathetic/   ~7MB (数据集缓存)
                                      --------
                                      ~2.5GB
```

---

## 📦 总占用估算

### 训练过程中（峰值）

| 项目 | 大小 |
|------|------|
| 2个最新 checkpoints | 7.2GB |
| Final 模型 | 1.2GB |
| 数据集 | 0.2GB |
| HF 模型缓存 | 2.5GB |
| **总计** | **~11GB** |

### 训练完成后（清理后）

| 项目 | 大小 |
|------|------|
| Final 模型 | 1.2GB |
| 数据集 | 0.2GB |
| HF 模型缓存 | 2.5GB |
| **总计** | **~4GB** |

---

## 🗑️ 清理策略（节省空间）

### 方案 1: 只保留最终模型（推荐）

训练完成后，删除所有checkpoint：
```powershell
# 删除所有checkpoint，只保留final模型
Remove-Item -Recurse -Force "model/mt5_emotion_summary/checkpoint-*"
```

**节省空间**: 7.2GB → 只剩 1.2GB 模型
**总占用**: 约 **4GB**

### 方案 2: 压缩最终模型

如果不需要继续训练，可以删除训练相关文件：
```powershell
cd "model/mt5_emotion_summary/final"
Remove-Item "optimizer.pt"      # 删除优化器（如果存在）
Remove-Item "scheduler.pt"       # 删除调度器
Remove-Item "trainer_state.json" # 删除训练状态
Remove-Item "training_args.bin"  # 删除训练参数
```

**节省空间**: 额外节省 ~2.4GB
**总占用**: 约 **1.6GB**

### 方案 3: 清理 Hugging Face 缓存（激进）

如果确定不再需要重新下载模型：
```powershell
# 清理所有 HF 缓存
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub"
```

**节省空间**: 2.5GB
**注意**: 下次使用需要重新下载

---

## ⚠️ 空间不足时的优化策略

### 策略 1: 减少保存频率
修改 `simple_train.py`:
```python
save_steps = 1000,        # 从500改为1000
save_total_limit = 1,     # 从2改为1，只保留最后一个
```
**节省**: 3.6GB (少保存一个checkpoint)

### 策略 2: 实时清理旧checkpoint
在训练脚本中添加自动清理（已启用 `save_total_limit=2`）

### 策略 3: 不保存中间checkpoint
```python
save_strategy = "no",     # 不保存中间checkpoint
# 只在训练结束后保存final模型
```
**节省**: 7.2GB (但失去断点续训能力)

---

## 📏 您需要的最小空间

### 最低要求（训练+最终模型）
```
训练数据:           0.2GB
训练过程峰值:       8.4GB (临时)
HF缓存:            2.5GB
最终保留:          1.2GB (只保留final模型)
                  --------
安全余量:          +2GB
推荐最小空间:      14GB
```

### 推荐空间（含缓存）
```
推荐可用空间:      15-20GB
```

### 最终占用（训练后清理）
```
只保留模型:        约 4GB
压缩后:           约 1.6GB
```

---

## 🎯 建议

### 如果可用空间 > 15GB
✅ 直接训练，无需担心

### 如果可用空间 10-15GB
⚠️ 可以训练，但建议：
- 设置 `save_total_limit=1`
- 训练完立即清理checkpoint

### 如果可用空间 < 10GB
❌ 空间不足，建议：
1. 清理其他文件释放空间
2. 或使用更小的模型（如 t5-small）
3. 或使用云端训练服务

---

## 📝 空间检查命令

```powershell
# 检查当前目录可用空间
Get-PSDrive C | Select-Object Used,Free

# 检查 Emotion Summary 目录大小
Get-ChildItem "Emotion Summary (ES)" -Recurse | Measure-Object -Property Length -Sum
```

---

## 🚀 训练后自动清理脚本

创建 `cleanup_checkpoints.ps1`:
```powershell
# 训练完成后运行此脚本清理
Write-Host "清理旧的 checkpoints..."
Remove-Item -Recurse -Force "model/mt5_emotion_summary/checkpoint-*"
Write-Host "✅ 清理完成！只保留 final 模型"
Write-Host "节省空间: ~7GB"
```

使用方法：
```powershell
# 训练完成后
.\cleanup_checkpoints.ps1
```

