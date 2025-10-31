# -*- coding: utf-8 -*-
"""分批处理脚本 - 准备数据供AI分析"""

import json
from pathlib import Path

def create_batches(batch_size=10):
    """将测试数据分批"""
    test_file = Path("data/test/Emotion_Summary.jsonl")
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    # 分批
    batches = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        batches.append(batch)
    
    # 保存每批到单独文件供处理
    batch_dir = Path("data/batches")
    batch_dir.mkdir(exist_ok=True)
    
    for i, batch in enumerate(batches):
        batch_file = batch_dir / f"batch_{i+1:03d}.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"批次 {i+1}: {len(batch)} 样本 -> {batch_file}")
    
    print(f"\n总计: {len(batches)} 批次, {len(test_data)} 样本")
    return len(batches), len(test_data)

if __name__ == "__main__":
    print("="*80)
    print("🔄 准备分批数据")
    print("="*80)
    create_batches(batch_size=10)

