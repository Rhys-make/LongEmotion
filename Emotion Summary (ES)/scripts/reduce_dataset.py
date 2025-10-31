# -*- coding: utf-8 -*-
"""随机删除训练集和验证集各20%的数据以加快训练"""

import json
import random
from pathlib import Path

def reduce_dataset(input_file, output_file, keep_ratio=0.8):
    """保留指定比例的数据"""
    # 读取数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    original_count = len(data)
    
    # 随机打乱
    random.shuffle(data)
    
    # 保留80%
    keep_count = int(len(data) * keep_ratio)
    reduced_data = data[:keep_count]
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in reduced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return original_count, len(reduced_data)

def main():
    print("="*70)
    print("📉 减少数据集大小（保留80%）")
    print("="*70)
    
    random.seed(42)  # 设置随机种子保证可复现
    
    # 处理训练集
    print("\n处理训练集...")
    train_input = Path("data/train/Emotion_Summary.jsonl")
    train_output = Path("data/train/Emotion_Summary.jsonl.backup")
    
    # 备份原文件
    if train_input.exists():
        import shutil
        shutil.copy(train_input, train_output)
        print(f"  ✅ 已备份原文件: {train_output}")
    
    train_orig, train_new = reduce_dataset(train_input, train_input, keep_ratio=0.8)
    print(f"  原始: {train_orig:,} 样本")
    print(f"  保留: {train_new:,} 样本 (80%)")
    print(f"  删除: {train_orig - train_new:,} 样本 (20%)")
    
    # 处理验证集
    print("\n处理验证集...")
    val_input = Path("data/validation/Emotion_Summary.jsonl")
    val_output = Path("data/validation/Emotion_Summary.jsonl.backup")
    
    # 备份原文件
    if val_input.exists():
        shutil.copy(val_input, val_output)
        print(f"  ✅ 已备份原文件: {val_output}")
    
    val_orig, val_new = reduce_dataset(val_input, val_input, keep_ratio=0.8)
    print(f"  原始: {val_orig:,} 样本")
    print(f"  保留: {val_new:,} 样本 (80%)")
    print(f"  删除: {val_orig - val_new:,} 样本 (20%)")
    
    # 总结
    print("\n" + "="*70)
    print("✅ 数据集减少完成！")
    print("="*70)
    print(f"\n新的数据集大小:")
    print(f"  训练集: {train_new:,} 样本")
    print(f"  验证集: {val_new:,} 样本")
    print(f"  总计: {train_new + val_new:,} 样本")
    
    print(f"\n预计训练时间缩短: ~20%")
    print(f"\n如需恢复原数据:")
    print(f"  mv data/train/Emotion_Summary.jsonl.backup data/train/Emotion_Summary.jsonl")
    print(f"  mv data/validation/Emotion_Summary.jsonl.backup data/validation/Emotion_Summary.jsonl")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

