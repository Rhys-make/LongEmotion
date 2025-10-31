# -*- coding: utf-8 -*-
"""检查训练结果"""

import json
import os
from pathlib import Path

print("="*70)
print("🔍 训练结果检查")
print("="*70)

# 1. 检查模型文件
print("\n📁 1. 模型文件检查")
print("-"*70)

model_dir = Path("model/mt5_fast")
checkpoint_dir = model_dir / "checkpoint-1000"
final_dir = model_dir / "final"

if checkpoint_dir.exists():
    print(f"✅ Checkpoint目录存在: {checkpoint_dir}")
    checkpoint_files = list(checkpoint_dir.glob("*"))
    print(f"   文件数: {len(checkpoint_files)}")
    for f in checkpoint_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"   - {f.name}: {size_mb:.1f} MB")
else:
    print(f"❌ Checkpoint目录不存在")

print()

if final_dir.exists():
    print(f"✅ 最终模型目录存在: {final_dir}")
    final_files = list(final_dir.glob("*"))
    print(f"   文件数: {len(final_files)}")
    
    total_size = 0
    for f in final_files:
        size_mb = f.stat().st_size / (1024**2)
        total_size += size_mb
        print(f"   - {f.name}: {size_mb:.1f} MB")
    
    print(f"\n   总大小: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
else:
    print(f"❌ 最终模型目录不存在")

# 2. 检查训练状态
print("\n📊 2. 训练状态检查")
print("-"*70)

trainer_state_file = checkpoint_dir / "trainer_state.json"

if trainer_state_file.exists():
    with open(trainer_state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    print(f"✅ 训练状态文件存在")
    print(f"\n基本信息:")
    print(f"  总步数: {state['global_step']}")
    print(f"  训练轮数: {state['epoch']:.2f}")
    print(f"  最大步数: {state['max_steps']}")
    
    # 获取loss历史
    log_history = state['log_history']
    
    print(f"\n训练历史 (最后5条记录):")
    for log in log_history[-5:]:
        if 'loss' in log:
            print(f"  步 {log['step']}: loss={log['loss']:.4f}")
        elif 'eval_loss' in log:
            print(f"  步 {log['step']}: eval_loss={log['eval_loss']:.4f}")
    
    # 找到第一个和最后一个loss
    train_losses = [log for log in log_history if 'loss' in log]
    if len(train_losses) >= 2:
        first_loss = train_losses[0]['loss']
        last_loss = train_losses[-1]['loss']
        improvement = ((first_loss - last_loss) / first_loss) * 100
        
        print(f"\nLoss变化:")
        print(f"  初始loss: {first_loss:.4f}")
        print(f"  最终loss: {last_loss:.4f}")
        print(f"  改善: {improvement:.1f}%")
    
    # 检查是否有验证loss
    eval_losses = [log for log in log_history if 'eval_loss' in log]
    if eval_losses:
        print(f"\n验证集:")
        for log in eval_losses:
            print(f"  步 {log['step']}: eval_loss={log.get('eval_loss', 'N/A'):.4f}")
else:
    print(f"❌ 训练状态文件不存在")

# 3. 检查模型配置
print("\n⚙️  3. 模型配置检查")
print("-"*70)

config_file = final_dir / "config.json"
if config_file.exists():
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"✅ 模型配置存在")
    print(f"  模型类型: {config.get('model_type', 'N/A')}")
    print(f"  词汇表大小: {config.get('vocab_size', 'N/A'):,}")
    print(f"  隐藏层大小: {config.get('d_model', 'N/A')}")
    print(f"  注意力头数: {config.get('num_heads', 'N/A')}")
    print(f"  层数: {config.get('num_layers', 'N/A')}")
else:
    print(f"❌ 模型配置不存在")

# 4. 训练完成度评估
print("\n✅ 4. 训练完成度评估")
print("-"*70)

checks = {
    "模型文件完整": final_dir.exists() and len(list(final_dir.glob("*"))) >= 7,
    "Checkpoint保存": checkpoint_dir.exists(),
    "训练状态记录": trainer_state_file.exists(),
    "模型配置存在": config_file.exists(),
}

all_pass = all(checks.values())

for name, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")

if all_pass:
    print("\n🎉 训练成功完成！")
else:
    print("\n⚠️  训练可能未完全完成")

# 5. 下一步建议
print("\n" + "="*70)
print("📋 下一步操作")
print("="*70)

if all_pass:
    print("\n✅ 训练已完成，可以进行推理了！")
    print("\n需要做的事情:")
    print("  1. 创建推理脚本")
    print("  2. 对测试集进行推理")
    print("  3. 生成提交文件")
    print("\n准备创建推理脚本...")
else:
    print("\n⚠️  请先确保训练完成")

print("="*70)

