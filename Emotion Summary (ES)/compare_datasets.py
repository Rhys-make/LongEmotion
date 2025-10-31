# -*- coding: utf-8 -*-
"""对比训练集、验证集和测试集"""

import json
from pathlib import Path

def load_data(file_path):
    """加载JSONL数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def analyze_dataset(data, name):
    """分析数据集特征"""
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"{'='*70}")
    
    print(f"\n数量: {len(data):,} 个样本")
    
    if len(data) > 0:
        first = data[0]
        print(f"\n字段: {list(first.keys())}")
        
        # 检查语言
        case_desc = first.get('case_description', [''])[0]
        consult_proc = first.get('consultation_process', [''])[0] if first.get('consultation_process') else ''
        
        is_english = any(word in (case_desc + consult_proc).lower() for word in ['the', 'and', 'is', 'was', 'visitor', 'client'])
        language = "英文 (English)" if is_english else "中文 (Chinese)"
        print(f"\n语言: {language}")
        
        # 显示示例
        print(f"\ncase_description 示例:")
        print(f"  {case_desc[:100]}...")
        
        if consult_proc:
            print(f"\nconsultation_process 示例:")
            print(f"  {consult_proc[:100]}...")
        
        # 统计长度
        case_desc_lens = [len(item.get('case_description', [])) for item in data]
        consult_lens = [len(item.get('consultation_process', [])) for item in data]
        
        print(f"\n内容长度统计:")
        print(f"  case_description 段数 - 平均: {sum(case_desc_lens)/len(case_desc_lens):.1f}, 最小: {min(case_desc_lens)}, 最大: {max(case_desc_lens)}")
        print(f"  consultation_process 段数 - 平均: {sum(consult_lens)/len(consult_lens):.1f}, 最小: {min(consult_lens)}, 最大: {max(consult_lens)}")
        
        # 检查 experience_and_reflection
        has_reflection = 'experience_and_reflection' in first
        if has_reflection:
            reflection_lens = [len(item.get('experience_and_reflection', '')) for item in data]
            print(f"  experience_and_reflection 字符数 - 平均: {sum(reflection_lens)/len(reflection_lens):.0f}, 最小: {min(reflection_lens)}, 最大: {max(reflection_lens)}")
            print(f"\n✅ 包含 experience_and_reflection (训练目标)")
        else:
            print(f"\n❌ 不包含 experience_and_reflection (需要模型生成)")

def main():
    print("="*70)
    print("🔍 训练集、验证集、测试集完整对比")
    print("="*70)
    
    # 加载数据
    train_file = Path("data/train/Emotion_Summary.jsonl")
    val_file = Path("data/validation/Emotion_Summary.jsonl")
    test_file = Path("data/test/Emotion_Summary.jsonl")
    
    train_data = load_data(train_file)
    val_data = load_data(val_file)
    test_data = load_data(test_file)
    
    # 分析每个数据集
    analyze_dataset(train_data, "训练集 (Training Set)")
    analyze_dataset(val_data, "验证集 (Validation Set)")
    analyze_dataset(test_data, "测试集 (Test Set)")
    
    # 对比总结
    print(f"\n{'='*70}")
    print("📝 对比总结")
    print(f"{'='*70}")
    
    print(f"\n数量对比:")
    print(f"  训练集: {len(train_data):,} 样本")
    print(f"  验证集: {len(val_data):,} 样本")
    print(f"  测试集: {len(test_data):,} 样本")
    print(f"  训练/验证比例: {len(train_data)/len(val_data):.1f}:1")
    print(f"  训练集是测试集的: {len(train_data)/len(test_data):.1f} 倍")
    
    # 语言一致性检查
    train_sample = train_data[0]['case_description'][0]
    test_sample = test_data[0]['case_description'][0]
    
    train_is_english = any(word in train_sample.lower() for word in ['the', 'and', 'is', 'client', 'counselor'])
    test_is_english = any(word in test_sample.lower() for word in ['the', 'and', 'is', 'visitor'])
    
    print(f"\n语言一致性:")
    if train_is_english and test_is_english:
        print(f"  ✅ 训练集和测试集都是英文 - 完美匹配！")
    elif not train_is_english and not test_is_english:
        print(f"  ✅ 训练集和测试集都是中文 - 完美匹配！")
    else:
        print(f"  ❌ 警告：训练集和测试集语言不一致！")
        print(f"     训练集: {'英文' if train_is_english else '中文'}")
        print(f"     测试集: {'英文' if test_is_english else '中文'}")
    
    # 任务说明
    print(f"\n{'='*70}")
    print("🎯 训练任务")
    print(f"{'='*70}")
    print(f"\n任务类型: 文本生成 (Text Generation)")
    print(f"输入: case_description + consultation_process")
    print(f"输出: experience_and_reflection")
    print(f"\n模型目标:")
    print(f"  1. 学习从 {len(train_data):,} 个训练样本中理解咨询案例")
    print(f"  2. 在 {len(val_data):,} 个验证样本上评估性能")
    print(f"  3. 最终在 {len(test_data):,} 个测试样本上生成高质量的经验反思")
    
    # 数据充分性评估
    print(f"\n数据充分性评估:")
    if len(train_data) > 10000:
        print(f"  ✅ 训练数据充足 ({len(train_data):,} 样本) - 可以训练出良好的模型")
    elif len(train_data) > 1000:
        print(f"  ⚠️  训练数据中等 ({len(train_data):,} 样本) - 建议多训练几轮")
    else:
        print(f"  ❌ 训练数据较少 ({len(train_data):,} 样本) - 可能需要更多数据或数据增强")
    
    print(f"\n{'='*70}")
    print("✅ 对比完成！现在可以开始训练了！")
    print(f"{'='*70}")
    print(f"\n下一步: python scripts/simple_train.py")

if __name__ == "__main__":
    main()

