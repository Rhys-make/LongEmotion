# QA训练安全使用指南

## 快速开始

### 方法1：一键启动（推荐）
```bash
python qa/scripts/quick_start_safe.py
```

模型将保存到 `qa/checkpoint/` 目录下。
这个脚本会自动：
- 检查你的系统资源
- 选择最安全的配置
- 开始训练

### 方法2：手动选择配置

#### 超低内存配置（最安全）
```bash
python qa/scripts/train_low_memory.py \
    --model_name bert-base-chinese \
    --model_type extractive \
    --batch_size 1 \
    --max_length 128 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1 \
    --use_amp
```

#### CPU训练（如果GPU显存不足）
```bash
CUDA_VISIBLE_DEVICES='' python qa/scripts/train_low_memory.py \
    --model_name bert-base-chinese \
    --model_type extractive \
    --batch_size 1 \
    --max_length 128 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1
```

## 参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--model_name` | bert-base-chinese | 使用中文BERT模型 |
| `--model_type` | extractive | 抽取式QA，比生成式更省内存 |
| `--batch_size` | 1 | 最小批次大小 |
| `--max_length` | 128 | 序列长度，越小越省内存 |
| `--gradient_accumulation_steps` | 16 | 梯度累积，模拟大批次 |
| `--num_epochs` | 1 | 训练轮数，先试试1轮 |
| `--use_amp` | 是 | 混合精度训练，节省显存 |

## 如果仍然关机

### 进一步减小参数
```bash
python qa/scripts/train_low_memory.py \
    --model_name bert-base-chinese \
    --model_type extractive \
    --batch_size 1 \
    --max_length 64 \
    --gradient_accumulation_steps 32 \
    --num_epochs 1 \
    --use_amp
```

### 使用更小的模型
```bash
python qa/scripts/train_low_memory.py \
    --model_name distilbert-base-chinese \
    --model_type extractive \
    --batch_size 1 \
    --max_length 128 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1 \
    --use_amp
```

### 强制使用CPU
```bash
CUDA_VISIBLE_DEVICES='' python qa/scripts/train_low_memory.py \
    --model_name bert-base-chinese \
    --model_type extractive \
    --batch_size 1 \
    --max_length 64 \
    --gradient_accumulation_steps 32 \
    --num_epochs 1
```

## 监控训练

训练过程中注意观察：
1. **内存使用率**：不应超过85%
2. **GPU显存**：不应超过90%
3. **系统温度**：避免过热
4. **训练日志**：注意内存警告

## 常见错误

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**解决**：减小批次大小或使用CPU训练

### 系统关机
**解决**：使用更小的参数或CPU训练

### 训练速度极慢
**解决**：在保证不关机的前提下适当增加批次大小

## 总结

**最重要的原则**：稳定训练比快速训练更重要！

如果系统关机，所有训练进度都会丢失。建议：
1. 先使用最安全的配置
2. 确认可以稳定训练后，再逐步增加参数
3. 始终监控系统资源使用情况

## 联系支持

如果遇到问题，请提供：
1. 系统配置信息
2. 错误日志
3. 使用的训练命令
