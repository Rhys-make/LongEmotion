# 内存优化训练指南

## 问题描述

如果你在运行训练脚本时遇到系统关机的问题，这通常是由于以下原因：

1. **内存/显存不足**：模型太大或批次大小过大
2. **系统资源耗尽**：训练过程中消耗了过多系统资源
3. **GPU显存溢出**：显存不足以加载模型和数据

## 解决方案

我们提供了三个不同级别的训练脚本来解决这个问题：

### 1. 系统资源检查

首先运行系统检查脚本，了解你的系统配置：

```bash
python classification/scripts/check_system.py
```

这个脚本会：
- 检查CPU、内存、GPU信息
- 根据系统配置推荐合适的训练参数
- 生成对应的训练命令

### 2. 超低内存训练脚本

如果你的系统资源有限，使用超低内存版本：

```bash
python classification/scripts/train_low_memory.py \
    --resume_from_checkpoint checkpoint/classification/latest_model \
    --num_epochs 1
```

**特点：**
- 批次大小：1
- 序列长度：128
- 梯度累积：16步
- 强制内存清理
- 使用混合精度训练

### 3. 安全训练脚本

如果你的系统资源中等，使用安全版本：

```bash
python classification/scripts/train_safe.py \
    --resume_from_checkpoint checkpoint/classification/latest_model \
    --num_epochs 1
```

**特点：**
- 批次大小：4
- 序列长度：256
- 梯度累积：4步
- 内存优化
- 错误处理

### 4. 原始训练脚本（不推荐）

原始脚本使用较大参数，容易导致内存问题：

```bash
# 不推荐 - 容易导致系统关机
python classification/scripts/train.py \
    --resume_from_checkpoint checkpoint/classification/latest_model \
    --num_epochs 1
```

## 参数说明

### 关键参数

| 参数 | 超低内存 | 安全版本 | 原始版本 | 说明 |
|------|----------|----------|----------|------|
| `--batch_size` | 1 | 4 | 8 | 批次大小，越小越省内存 |
| `--max_length` | 128 | 256 | 512 | 序列长度，越小越省内存 |
| `--gradient_accumulation_steps` | 16 | 4 | 1 | 梯度累积，越大越省内存 |
| `--model_name` | bert-base-chinese | bert-base-chinese | roberta-large | 模型大小 |

### 内存优化技术

1. **混合精度训练** (`--use_amp`)：减少显存使用
2. **梯度检查点**：用计算换内存
3. **梯度累积**：模拟大批次但使用小批次
4. **定期内存清理**：防止内存泄漏
5. **单线程数据加载** (`num_workers=0`)：避免多进程内存问题

## 使用建议

### 根据系统配置选择

**GPU显存 < 4GB 或 系统内存 < 8GB：**
```bash
python classification/scripts/train_low_memory.py \
    --resume_from_checkpoint checkpoint/classification/latest_model \
    --num_epochs 1
```

**GPU显存 4-8GB 或 系统内存 8-16GB：**
```bash
python classification/scripts/train_safe.py \
    --resume_from_checkpoint checkpoint/classification/latest_model \
    --num_epochs 1
```

**GPU显存 > 8GB 且 系统内存 > 16GB：**
```bash
python classification/scripts/train.py \
    --resume_from_checkpoint checkpoint/classification/latest_model \
    --num_epochs 1
```

### 进一步优化

如果仍然遇到内存问题，可以：

1. **减小批次大小到1**：
   ```bash
   --batch_size 1
   ```

2. **减小序列长度**：
   ```bash
   --max_length 64  # 或更小
   ```

3. **增加梯度累积**：
   ```bash
   --gradient_accumulation_steps 32
   ```

4. **使用CPU训练**：
   ```bash
   CUDA_VISIBLE_DEVICES='' python classification/scripts/train_low_memory.py
   ```

5. **使用更小的模型**：
   ```bash
   --model_name distilbert-base-chinese  # 更小的模型
   ```

## 监控训练过程

训练过程中注意观察：

1. **内存使用率**：不应超过90%
2. **GPU显存使用**：不应超过95%
3. **系统温度**：避免过热
4. **训练日志**：注意内存不足警告

## 常见错误及解决方案

### 1. CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**解决方案：**
- 减小批次大小
- 减小序列长度
- 增加梯度累积步数
- 使用混合精度训练

### 2. 系统关机
**解决方案：**
- 使用超低内存训练脚本
- 检查系统散热
- 关闭其他占用内存的程序
- 使用CPU训练

### 3. 训练速度慢
**解决方案：**
- 在保证不关机的前提下适当增加批次大小
- 使用GPU训练
- 减少梯度累积步数

## 性能对比

| 配置 | 内存使用 | 训练速度 | 稳定性 | 推荐场景 |
|------|----------|----------|--------|----------|
| 超低内存 | 最低 | 最慢 | 最高 | 资源受限 |
| 安全版本 | 中等 | 中等 | 高 | 一般使用 |
| 原始版本 | 最高 | 最快 | 低 | 资源充足 |

## 总结

选择合适的训练脚本和参数是避免系统关机问题的关键。建议：

1. 先运行系统检查脚本
2. 根据推荐选择训练脚本
3. 如果仍有问题，进一步减小参数
4. 监控训练过程，及时调整

记住：**稳定训练比快速训练更重要**！
