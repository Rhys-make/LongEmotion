# -*- coding: utf-8 -*-
"""
使用 Parquet 格式直接下载 Empathetic Dialogues 真实数据集
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import urllib.request
import pandas as pd

def download_and_convert():
    """下载真实的 Empathetic Dialogues 数据集"""
    
    print("="*70)
    print("下载 Empathetic Dialogues 完整数据集")
    print("="*70)
    
    # Parquet 文件 URL (从 Hugging Face datasets viewer 获取)
    base_url = "https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/main/data"
    
    train_url = f"{base_url}/train-00000-of-00001.parquet"
    val_url = f"{base_url}/validation-00000-of-00001.parquet"
    
    print("\n📥 正在下载训练集...")
    try:
        import requests
        
        # 下载训练集
        response = requests.get(train_url, timeout=300)
        train_parquet = "temp_train.parquet"
        with open(train_parquet, 'wb') as f:
            f.write(response.content)
        
        # 下载验证集
        print("\n📥 正在下载验证集...")
        response = requests.get(val_url, timeout=300)
        val_parquet = "temp_val.parquet"
        with open(val_parquet, 'wb') as f:
            f.write(response.content)
        
        # 读取 Parquet 文件
        print("\n📖 正在读取数据...")
        train_df = pd.read_parquet(train_parquet)
        val_df = pd.read_parquet(val_parquet)
        
        print(f"✅ 下载完成！")
        print(f"   训练集: {len(train_df)} 条记录")
        print(f"   验证集: {len(val_df)} 条记录")
        
        # 转换为字典列表
        train_data = train_df.to_dict('records')
        val_data = val_df.to_dict('records')
        
        # 组织对话
        print("\n🔄 正在组织对话...")
        train_conversations = organize_by_conversation(train_data)
        val_conversations = organize_by_conversation(val_data)
        
        print(f"✅ 训练集: {len(train_conversations)} 个完整对话")
        print(f"✅ 验证集: {len(val_conversations)} 个完整对话")
        
        # 合并并重新分割
        all_conversations = list(train_conversations.values()) + list(val_conversations.values())
        random.shuffle(all_conversations)
        
        # 80/20 分割
        split_idx = int(len(all_conversations) * 0.8)
        train_convs = all_conversations[:split_idx]
        val_convs = all_conversations[split_idx:]
        
        print(f"\n📊 重新分割:")
        print(f"   训练集: {len(train_convs)} 个对话 (80%)")
        print(f"   验证集: {len(val_convs)} 个对话 (20%)")
        
        # 转换为 ES 格式
        print("\n🔄 正在转换为 ES 格式...")
        train_es_data = [convert_to_es_format(conv, idx) for idx, conv in enumerate(train_convs)]
        val_es_data = [convert_to_es_format(conv, idx) for idx, conv in enumerate(val_convs)]
        
        # 保存数据
        save_data(train_es_data, val_es_data)
        
        # 清理临时文件
        import os
        os.remove(train_parquet)
        os.remove(val_parquet)
        
        print("\n✅ 成功！使用了真实的完整数据集！")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n尝试备用方案...")
        try_alternative_method()

def try_alternative_method():
    """备用方案：使用 datasets 库的旧版本或其他方法"""
    print("\n尝试使用 datasets 库加载...")
    
    try:
        from datasets import load_dataset
        import datasets
        
        # 尝试指定版本
        dataset = load_dataset(
            "facebook/empathetic_dialogues",
            revision="refs/convert/parquet"  # 使用 parquet 转换分支
        )
        
        print(f"✅ 加载成功！")
        print(f"   训练集: {len(dataset['train'])} 条记录")
        print(f"   验证集: {len(dataset['validation'])} 条记录")
        
        # 组织对话
        train_conversations = organize_by_conversation(dataset['train'])
        val_conversations = organize_by_conversation(dataset['validation'])
        
        all_conversations = list(train_conversations.values()) + list(val_conversations.values())
        random.shuffle(all_conversations)
        
        split_idx = int(len(all_conversations) * 0.8)
        train_convs = all_conversations[:split_idx]
        val_convs = all_conversations[split_idx:]
        
        print(f"\n📊 数据分割:")
        print(f"   训练集: {len(train_convs)} 个对话 (80%)")
        print(f"   验证集: {len(val_convs)} 个对话 (20%)")
        
        # 转换并保存
        train_es_data = [convert_to_es_format(conv, idx) for idx, conv in enumerate(train_convs)]
        val_es_data = [convert_to_es_format(conv, idx) for idx, conv in enumerate(val_convs)]
        
        save_data(train_es_data, val_es_data)
        
        print("\n✅ 成功！")
        
    except Exception as e2:
        print(f"❌ 备用方案也失败: {e2}")
        print("\n使用扩展的示例数据...")
        use_expanded_samples()

def use_expanded_samples():
    """创建更多的示例数据"""
    print("\n📝 创建扩展示例数据集（1000个对话）...")
    
    # 导入示例创建函数
    from download_empathetic_dialogues_v2 import create_sample_english_data, convert_to_es_format
    
    # 创建更多示例
    all_conversations = []
    for _ in range(10):  # 重复10次，得到1000个对话
        conversations = create_sample_english_data()
        all_conversations.extend(conversations)
    
    random.shuffle(all_conversations)
    
    split_idx = int(len(all_conversations) * 0.8)
    train_convs = all_conversations[:split_idx]
    val_convs = all_conversations[split_idx:]
    
    train_es_data = [convert_to_es_format(conv, idx) for idx, conv in enumerate(train_convs)]
    val_es_data = [convert_to_es_format(conv, idx) for idx, conv in enumerate(val_convs)]
    
    save_data(train_es_data, val_es_data)
    
    print(f"✅ 创建了扩展数据集:")
    print(f"   训练集: {len(train_es_data)} 个对话")
    print(f"   验证集: {len(val_es_data)} 个对话")

def organize_by_conversation(dataset):
    """按对话ID组织数据"""
    conversations = defaultdict(list)
    
    for item in dataset:
        conv_id = item['conv_id']
        conversations[conv_id].append(item)
    
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x['utterance_idx'])
    
    return conversations

def convert_to_es_format(conversation, item_id):
    """转换为 ES 格式"""
    from download_empathetic_dialogues_v2 import convert_to_es_format as convert_func
    return convert_func(conversation, item_id)

def save_data(train_es_data, val_es_data):
    """保存数据"""
    train_dir = Path("data/train")
    val_dir = Path("data/validation")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = train_dir / "Emotion_Summary.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_es_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    val_file = val_dir / "Emotion_Summary.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_es_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n💾 数据已保存:")
    print(f"   训练集: {train_file} ({len(train_es_data)} 样本)")
    print(f"   验证集: {val_file} ({len(val_es_data)} 样本)")

if __name__ == "__main__":
    random.seed(42)
    
    # 检查依赖
    try:
        import pandas
        import requests
    except ImportError:
        print("需要安装额外依赖:")
        print("pip install pandas requests")
        import sys
        sys.exit(1)
    
    download_and_convert()

