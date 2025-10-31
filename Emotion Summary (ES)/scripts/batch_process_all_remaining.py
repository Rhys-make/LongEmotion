# -*- coding: utf-8 -*-
"""
批量处理所有剩余样本 (ID 11-200)
这个脚本将读取所有测试数据，并为每个样本准备完整的上下文，
供AI助手进行深度语义分析
"""

import json
import sys

# 确保UTF-8输出
sys.stdout.reconfigure(encoding='utf-8')

def read_all_test_samples():
    """读取所有测试样本"""
    samples = []
    with open('data/test/Emotion_Summary.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def get_already_processed():
    """获取已处理的样本ID"""
    processed = set()
    
    for batch_file in ['ai_results_batch1.json', 'ai_results_batch2.json']:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch = json.load(f)
                for item in batch:
                    processed.add(item['id'])
        except FileNotFoundError:
            pass
    
    return processed

def main():
    print("\n" + "="*80)
    print("🚀 批量处理剩余样本 - 准备阶段")
    print("="*80)
    
    # 读取所有样本
    all_samples = read_all_test_samples()
    print(f"\n✓ 读取 {len(all_samples)} 个测试样本")
    
    # 获取已处理ID
    processed_ids = get_already_processed()
    print(f"✓ 已处理 {len(processed_ids)} 个样本")
    
    # 筛选待处理样本
    remaining = [s for s in all_samples if s['id'] not in processed_ids]
    print(f"✓ 待处理 {len(remaining)} 个样本\n")
    
    if len(remaining) == 0:
        print("🎉 所有样本已处理完成！")
        return
    
    # 保存待处理样本的完整信息（供AI分析使用）
    output_file = 'remaining_samples_for_ai.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in remaining:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"💾 已保存 {len(remaining)} 个待处理样本到: {output_file}")
    print(f"📝 样本ID范围: {min(s['id'] for s in remaining)} - {max(s['id'] for s in remaining)}")
    
    print("\n" + "="*80)
    print("✅ 准备完成！现在可以开始AI分析")
    print("="*80)
    print(f"\n📋 待处理样本数: {len(remaining)}")
    print(f"⏱️ 预计时间: 约 {len(remaining) * 2} 分钟 (每样本约2分钟)")
    print(f"💡 建议: 分批处理，每批10-20个样本\n")

if __name__ == '__main__':
    main()

