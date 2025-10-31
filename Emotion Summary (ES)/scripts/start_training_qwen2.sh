#!/bin/bash
# Qwen2-7B-Instruct 训练启动脚本

echo "================================"
echo "Qwen2-7B-Instruct 微调训练"
echo "Emotion Summary (ES) 任务"
echo "================================"
echo ""

# 检查 CUDA 是否可用
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✓ CUDA 可用"
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
else
    echo "⚠ CUDA 不可用，将使用 CPU（训练会非常慢）"
fi

echo ""
echo "开始训练..."
echo ""

# 运行训练脚本
python train_qwen2.py \
    --model_name "Qwen/Qwen2-7B-Instruct" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_length 8192 \
    --use_4bit \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05

echo ""
echo "================================"
echo "训练完成！"
echo "================================"

