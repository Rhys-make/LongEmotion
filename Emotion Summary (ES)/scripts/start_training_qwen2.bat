@echo off
REM Qwen2-7B-Instruct 训练启动脚本 (Windows)

echo ================================
echo Qwen2-7B-Instruct 微调训练
echo Emotion Summary (ES) 任务
echo ================================
echo.

REM 检查 CUDA
python -c "import torch; print('CUDA 可用' if torch.cuda.is_available() else '警告: CUDA 不可用')"
echo.

echo 开始训练...
echo.

REM 运行训练脚本
python train_qwen2.py ^
    --model_name "Qwen/Qwen2-7B-Instruct" ^
    --num_epochs 3 ^
    --batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --learning_rate 2e-4 ^
    --max_length 8192 ^
    --use_4bit ^
    --lora_r 64 ^
    --lora_alpha 16 ^
    --lora_dropout 0.05

echo.
echo ================================
echo 训练完成！
echo ================================
pause

