# -*- coding: utf-8 -*-
"""详细对比训练集和测试集的差异"""

import json
from pathlib import Path

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    train = load_data("data/train/Emotion_Summary.jsonl")
    test = load_data("data/test/Emotion_Summary.jsonl")
    
    print("="*70)
    print("🔍 训练集 vs 测试集 详细对比")
    print("="*70)
    
    # 计算统计数据
    def get_stats(data):
        case_lens = [len(' '.join(item['case_description'])) for item in data]
        consult_lens = [len(' '.join(item['consultation_process'])) for item in data]
        reflection_lens = [len(item.get('experience_and_reflection', '')) for item in data]
        
        consult_turns = [len(item['consultation_process']) for item in data]
        
        return {
            'case_len': (sum(case_lens)/len(case_lens), min(case_lens), max(case_lens)),
            'consult_len': (sum(consult_lens)/len(consult_lens), min(consult_lens), max(consult_lens)),
            'reflection_len': (sum(reflection_lens)/len(reflection_lens), min(reflection_lens), max(reflection_lens)),
            'consult_turns': (sum(consult_turns)/len(consult_turns), min(consult_turns), max(consult_turns))
        }
    
    train_stats = get_stats(train)
    test_stats = get_stats(test)
    
    print(f"\n📊 内容长度对比 (字符数):")
    print(f"\n{'指标':<30} {'训练集 (平均)':<20} {'测试集 (平均)':<20} {'差距倍数':<15}")
    print("-"*85)
    
    print(f"{'case_description 长度':<30} {train_stats['case_len'][0]:>10.0f} {test_stats['case_len'][0]:>20.0f} {test_stats['case_len'][0]/train_stats['case_len'][0]:>15.1f}x")
    print(f"{'consultation_process 长度':<30} {train_stats['consult_len'][0]:>10.0f} {test_stats['consult_len'][0]:>20.0f} {test_stats['consult_len'][0]/train_stats['consult_len'][0]:>15.1f}x")
    print(f"{'consultation_process 轮数':<30} {train_stats['consult_turns'][0]:>10.1f} {test_stats['consult_turns'][0]:>20.1f} {test_stats['consult_turns'][0]/train_stats['consult_turns'][0]:>15.1f}x")
    print(f"{'experience_and_reflection 长度':<30} {train_stats['reflection_len'][0]:>10.0f} {test_stats['reflection_len'][0]:>20.0f} {test_stats['reflection_len'][0]/train_stats['reflection_len'][0]:>15.1f}x")
    
    print(f"\n📝 示例对比:")
    print(f"\n【训练集示例 #1】")
    print(f"Case: {train[0]['case_description']}")
    print(f"Consultation ({len(train[0]['consultation_process'])} 轮):")
    for turn in train[0]['consultation_process'][:3]:
        print(f"  - {turn[:80]}...")
    print(f"Reflection ({len(train[0]['experience_and_reflection'])} 字符): {train[0]['experience_and_reflection'][:150]}...")
    
    print(f"\n{'='*70}")
    print(f"【测试集示例 #1】")
    print(f"Case: {test[0]['case_description'][0][:100]}...")
    print(f"Consultation ({len(test[0]['consultation_process'])} 轮):")
    for turn in test[0]['consultation_process'][:3]:
        print(f"  - {turn[:80]}...")
    if len(test[0]['consultation_process']) > 3:
        print(f"  ... 还有 {len(test[0]['consultation_process'])-3} 轮对话")
    print(f"Reflection ({len(test[0]['experience_and_reflection'])} 字符): {test[0]['experience_and_reflection'][:150]}...")
    
    print(f"\n{'='*70}")
    print("⚠️  关键发现")
    print(f"{'='*70}")
    
    print(f"\n1. 📏 长度差异:")
    print(f"   - 测试集的对话轮数是训练集的 {test_stats['consult_turns'][0]/train_stats['consult_turns'][0]:.1f} 倍")
    print(f"   - 测试集的对话内容是训练集的 {test_stats['consult_len'][0]/train_stats['consult_len'][0]:.1f} 倍")
    print(f"   - 测试集需要生成的反思是训练集的 {test_stats['reflection_len'][0]/train_stats['reflection_len'][0]:.1f} 倍")
    
    print(f"\n2. 🎯 内容类型:")
    print(f"   训练集: 简短的情绪对话 (Empathetic Dialogues)")
    print(f"   测试集: 深度心理咨询案例 (Clinical Psychology Cases)")
    
    print(f"\n3. ⚠️  潜在问题:")
    print(f"   - 训练集和测试集的**领域不完全匹配**")
    print(f"   - 训练集是日常情绪对话，测试集是专业心理咨询")
    print(f"   - 训练集的反思较短，测试集需要长篇深度反思")
    
    print(f"\n4. 💡 建议:")
    print(f"   ✅ 继续训练（数据量足够）")
    print(f"   ✅ 使用 mT5-base（擅长文本生成）")
    print(f"   ⚠️  可能需要调整 max_output_length（当前256，建议512-1024）")
    print(f"   ⚠️  预期：模型能学到基本的反思模式，但可能不如专业数据训练的效果")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()

