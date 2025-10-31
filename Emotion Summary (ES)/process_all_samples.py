# -*- coding: utf-8 -*-
"""
批量处理全部150个样本的AI深度分析
"""

import json
import sys
import time

# 确保UTF-8输出
sys.stdout.reconfigure(encoding='utf-8')

def read_test_data():
    """读取测试数据"""
    data = []
    with open('data/test/Emotion_Summary.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def ai_deep_analysis(sample):
    """
    对单个样本进行AI深度语义分析
    这个函数模拟AI分析过程，实际上我们会直接通过LLM来完成
    """
    sample_id = sample['id']
    case_desc = ' '.join(sample['case_description']) if isinstance(sample['case_description'], list) else sample['case_description']
    consultation = ' '.join(sample['consultation_process']) if isinstance(sample['consultation_process'], list) else sample['consultation_process']
    reflection = sample['experience_and_reflection']
    
    full_text = f"Case Description: {case_desc}\n\nConsultation Process: {consultation}\n\nReflection: {reflection}"
    
    # 这里返回一个占位符，实际处理会由我（AI助手）来完成
    return {
        "id": sample_id,
        "full_text": full_text,
        "case_description": case_desc,
        "consultation_process": consultation,
        "reflection": reflection
    }

def main():
    print("="*80)
    print("🚀 开始批量处理全部150个样本")
    print("="*80)
    
    # 读取测试数据
    print("\n📖 正在读取测试数据...")
    test_data = read_test_data()
    print(f"✓ 成功读取 {len(test_data)} 个样本")
    
    # 检查已处理的样本
    processed_ids = set()
    
    try:
        with open('ai_results_batch1.json', 'r', encoding='utf-8') as f:
            batch1 = json.load(f)
            for item in batch1:
                processed_ids.add(item['id'])
        print(f"✓ 已处理 batch1: {len(batch1)} 个样本")
    except:
        pass
    
    try:
        with open('ai_results_batch2.json', 'r', encoding='utf-8') as f:
            batch2 = json.load(f)
            for item in batch2:
                processed_ids.add(item['id'])
        print(f"✓ 已处理 batch2: {len(batch2)} 个样本")
    except:
        pass
    
    print(f"\n📊 已处理: {len(processed_ids)} 个样本")
    print(f"📊 待处理: {len(test_data) - len(processed_ids)} 个样本")
    
    # 提取待处理样本的关键信息
    remaining_samples = []
    for sample in test_data:
        if sample['id'] not in processed_ids:
            info = ai_deep_analysis(sample)
            remaining_samples.append(info)
    
    print(f"\n💾 准备输出待处理样本信息...")
    
    # 保存待处理样本信息
    with open('remaining_samples_info.json', 'w', encoding='utf-8') as f:
        json.dump(remaining_samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已保存 {len(remaining_samples)} 个待处理样本信息到 remaining_samples_info.json")
    print(f"\n🎯 这些样本需要通过AI进行深度语义分析")
    print(f"📝 样本ID范围: {min(s['id'] for s in remaining_samples)} - {max(s['id'] for s in remaining_samples)}")
    
    return len(remaining_samples)

if __name__ == '__main__':
    count = main()
    print(f"\n{'='*80}")
    print(f"✅ 准备工作完成！待处理样本数: {count}")
    print(f"{'='*80}\n")

