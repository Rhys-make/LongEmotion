# -*- coding: utf-8 -*-
"""任务自查脚本"""

import json
import os
from pathlib import Path

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    return exists

def check_jsonl_data(file_path, name):
    """检查JSONL数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        print(f"  📊 {name} 样本数: {len(data)}")
        
        if len(data) > 0:
            first_item = data[0]
            print(f"  🔑 字段: {list(first_item.keys())}")
            
            # 检查必需字段
            required_fields = ['id', 'case_description', 'consultation_process', 'experience_and_reflection']
            missing_fields = [f for f in required_fields if f not in first_item]
            
            if missing_fields:
                print(f"  ⚠️  缺少字段: {missing_fields}")
            else:
                print(f"  ✅ 所有必需字段完整")
        
        return True
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False

def main():
    print("="*70)
    print("🔍 Emotion Summary (ES) 任务自查报告")
    print("="*70)
    
    base_dir = Path("Emotion Summary (ES)")
    
    print("\n📁 1. 数据文件检查")
    print("-"*70)
    
    # 检查数据文件
    train_file = base_dir / "data/train/Emotion_Summary.jsonl"
    val_file = base_dir / "data/validation/Emotion_Summary.jsonl"
    test_file = base_dir / "data/test/Emotion_Summary.jsonl"
    
    train_ok = check_file_exists(train_file, "训练集")
    if train_ok:
        check_jsonl_data(train_file, "训练集")
    
    val_ok = check_file_exists(val_file, "验证集")
    if val_ok:
        check_jsonl_data(val_file, "验证集")
    
    test_ok = check_file_exists(test_file, "测试集")
    if test_ok:
        check_jsonl_data(test_file, "测试集")
    
    print("\n📜 2. 脚本文件检查")
    print("-"*70)
    
    scripts = [
        ("数据下载脚本", base_dir / "scripts/download_and_convert_psyqa.py"),
        ("简化训练脚本", base_dir / "scripts/simple_train.py"),
        ("通用训练脚本", base_dir / "scripts/train.py"),
        ("Qwen2训练脚本", base_dir / "scripts/train_qwen2.py"),
        ("推理脚本", base_dir / "scripts/inference.py"),
        ("评估脚本", base_dir / "scripts/evaluate.py"),
    ]
    
    for desc, path in scripts:
        check_file_exists(path, desc)
    
    print("\n⚙️  3. 配置文件检查")
    print("-"*70)
    
    config_file = base_dir / "config/config.py"
    check_file_exists(config_file, "配置文件")
    
    print("\n📦 4. 模型目录检查")
    print("-"*70)
    
    model_dir = base_dir / "model"
    if model_dir.exists():
        print(f"✅ 模型目录: {model_dir}")
        subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"  📂 已有子目录: {[d.name for d in subdirs]}")
        else:
            print(f"  📭 模型目录为空（训练后将保存模型）")
    else:
        print(f"⚠️  模型目录不存在，将自动创建")
    
    print("\n🎯 5. 训练准备状态")
    print("-"*70)
    
    checklist = {
        "数据集已准备": train_ok and val_ok and test_ok,
        "训练脚本完整": check_file_exists(base_dir / "scripts/simple_train.py", ""),
        "配置文件存在": check_file_exists(config_file, ""),
    }
    
    all_ready = all(checklist.values())
    
    for item, status in checklist.items():
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {item}")
    
    print("\n" + "="*70)
    if all_ready:
        print("✅ 所有准备工作已完成，可以开始训练！")
        print("\n📝 下一步操作:")
        print("   1. cd \"Emotion Summary (ES)\"")
        print("   2. python scripts/simple_train.py")
    else:
        print("⚠️  还有部分准备工作未完成，请检查上述项目")
    print("="*70)
    
    print("\n📊 6. 重要文件路径总结")
    print("-"*70)
    print(f"📁 项目根目录: {base_dir.absolute()}")
    print(f"📊 训练集: {train_file.absolute()}")
    print(f"📊 验证集: {val_file.absolute()}")
    print(f"📊 测试集: {test_file.absolute()}")
    print(f"🐍 训练脚本: {(base_dir / 'scripts/simple_train.py').absolute()}")
    print(f"💾 模型保存目录: {(base_dir / 'model/mt5_emotion_summary').absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()

