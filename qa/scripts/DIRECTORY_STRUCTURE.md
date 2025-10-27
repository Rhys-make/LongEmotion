# QA训练目录结构说明

## 目录结构

```
qa/
├── checkpoint/                    # QA模型检查点目录
│   ├── best_model/               # 最佳模型
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── ...
│   ├── latest_model/             # 最新模型
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── ...
│   └── training_history.json     # 训练历史
├── data/                         # 训练数据
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── scripts/                      # 训练脚本
│   ├── train.py                  # 原始训练脚本
│   ├── train_low_memory.py       # 超低内存训练脚本
│   ├── quick_start_safe.py       # 一键启动脚本
│   └── ...
└── result/                       # 训练结果
```

## 模型保存位置

### 默认保存路径
- **最佳模型**: `qa/checkpoint/best_model/`
- **最新模型**: `qa/checkpoint/latest_model/`
- **训练历史**: `qa/checkpoint/training_history.json`

### 自定义保存路径
如果你想保存到其他位置，可以使用 `--output_dir` 参数：

```bash
python qa/scripts/train_low_memory.py \
    --output_dir /path/to/your/checkpoint \
    --model_name bert-base-chinese \
    --batch_size 1 \
    --max_length 128 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1 \
    --use_amp
```

## 从检查点恢复训练

### 从最新模型恢复
```bash
python qa/scripts/train_low_memory.py \
    --resume_from_checkpoint checkpoint/latest_model \
    --num_epochs 1
```

### 从最佳模型恢复
```bash
python qa/scripts/train_low_memory.py \
    --resume_from_checkpoint checkpoint/best_model \
    --num_epochs 1
```

### 从自定义路径恢复
```bash
python qa/scripts/train_low_memory.py \
    --resume_from_checkpoint /path/to/your/checkpoint \
    --num_epochs 1
```

## 注意事项

1. **目录权限**: 确保有写入权限
2. **磁盘空间**: 确保有足够的磁盘空间保存模型
3. **路径分隔符**: 在Windows上使用反斜杠 `\`，在Linux/Mac上使用正斜杠 `/`
4. **相对路径**: 脚本中的路径是相对于脚本所在目录的

## 清理检查点

如果磁盘空间不足，可以删除旧的检查点：

```bash
# 删除最新模型（保留最佳模型）
rm -rf checkpoint/latest_model

# 删除所有检查点（谨慎操作）
rm -rf checkpoint/*
```

