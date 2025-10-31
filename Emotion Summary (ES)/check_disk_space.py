# -*- coding: utf-8 -*-
"""检查磁盘空间是否足够训练"""

import shutil
import os
from pathlib import Path

def get_size_mb(path):
    """获取目录大小（MB）"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_size_mb(entry.path)
    except PermissionError:
        pass
    return total / (1024 * 1024)

def main():
    print("="*70)
    print("💾 磁盘空间检查")
    print("="*70)
    
    # 检查可用空间
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    
    print(f"\n📊 当前磁盘状态:")
    print(f"  总容量: {total_gb:.1f} GB")
    print(f"  已使用: {used_gb:.1f} GB ({used/total*100:.1f}%)")
    print(f"  可用空间: {free_gb:.1f} GB")
    
    # 检查现有文件占用
    print(f"\n📁 当前项目占用:")
    
    data_dir = Path("data")
    model_dir = Path("model")
    
    if data_dir.exists():
        data_size = get_size_mb(data_dir)
        print(f"  数据集: {data_size:.0f} MB")
    
    if model_dir.exists():
        model_size = get_size_mb(model_dir)
        print(f"  模型文件: {model_size:.0f} MB ({model_size/1024:.1f} GB)")
    
    # 估算训练所需空间
    print(f"\n📦 训练所需空间估算:")
    required_space = {
        "数据集": 0.2,
        "训练峰值 (2个checkpoint)": 7.2,
        "最终模型": 1.2,
        "HF缓存": 2.5,
        "安全余量": 2.0
    }
    
    total_required = 0
    for item, size in required_space.items():
        print(f"  {item}: {size} GB")
        total_required += size
    
    print(f"  {'-'*50}")
    print(f"  推荐最小空间: {total_required:.1f} GB")
    
    # 判断是否足够
    print(f"\n{'='*70}")
    if free_gb >= total_required:
        print(f"✅ 空间充足！")
        print(f"   可用: {free_gb:.1f} GB")
        print(f"   需要: {total_required:.1f} GB")
        print(f"   余量: {free_gb - total_required:.1f} GB")
    elif free_gb >= total_required * 0.8:
        print(f"⚠️  空间有点紧张，但应该可以")
        print(f"   可用: {free_gb:.1f} GB")
        print(f"   需要: {total_required:.1f} GB")
        print(f"   建议: 训练完立即清理checkpoint")
    else:
        print(f"❌ 空间不足！")
        print(f"   可用: {free_gb:.1f} GB")
        print(f"   需要: {total_required:.1f} GB")
        print(f"   缺少: {total_required - free_gb:.1f} GB")
        print(f"\n💡 建议:")
        print(f"   1. 清理不需要的文件")
        print(f"   2. 使用更小的模型 (t5-small)")
        print(f"   3. 修改脚本只保留1个checkpoint")
    
    # 训练后空间估算
    print(f"\n{'='*70}")
    print(f"📝 训练完成后磁盘占用:")
    print(f"  保留所有checkpoint: ~8.4 GB")
    print(f"  只保留final模型: ~1.2 GB")
    print(f"  完全清理后: ~0 GB (可删除模型)")
    
    print(f"\n🗑️  节省空间命令:")
    print(f'  Remove-Item -Recurse -Force "model/mt5_emotion_summary/checkpoint-*"')
    print(f"  (节省约 7 GB)")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()

