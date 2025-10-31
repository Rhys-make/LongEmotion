# -*- coding: utf-8 -*-
"""
随机抽取20个样本进行语义自查
对照测试集原文验证AI分析的准确性和合理性
"""

import json
import random
import sys

sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    """加载测试集和结果数据"""
    # 读取测试集
    test_data = {}
    with open('data/test/Emotion_Summary.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                test_data[item['id']] = item
    
    # 读取结果
    result_data = {}
    with open('results/Emotion_Summary_Result.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                result_data[item['id']] = item
    
    return test_data, result_data

def extract_key_info(text):
    """提取文本中的关键信息"""
    text_lower = text.lower()
    
    keywords = {
        'family': ['father', 'mother', 'parent', 'sister', 'brother', 'family', 'grandmother', 'grandfather', 'child', 'daughter', 'son'],
        'emotion': ['anxiety', 'depression', 'fear', 'anger', 'sad', 'happy', 'stress', 'worry', 'nervous', 'panic'],
        'symptom': ['headache', 'dizzy', 'pain', 'tired', 'insomnia', 'sleep', 'chest tightness', 'palpitation'],
        'behavior': ['avoid', 'withdraw', 'refuse', 'fight', 'argue', 'conflict', 'escape'],
        'therapy': ['hypnosis', 'cbt', 'therapy', 'counseling', 'session', 'treatment'],
        'relationship': ['marriage', 'divorce', 'husband', 'wife', 'boyfriend', 'girlfriend', 'friend', 'classmate']
    }
    
    found = {}
    for category, words in keywords.items():
        found[category] = [w for w in words if w in text_lower]
    
    return found

def check_consistency(test_item, result_item):
    """检查结果与原文的一致性"""
    
    # 合并原文所有内容
    case_desc = ' '.join(test_item['case_description']) if isinstance(test_item['case_description'], list) else test_item['case_description']
    consultation = ' '.join(test_item['consultation_process']) if isinstance(test_item['consultation_process'], list) else test_item['consultation_process']
    reflection = test_item['experience_and_reflection']
    
    original_text = f"{case_desc} {consultation} {reflection}".lower()
    
    # 提取原文关键信息
    original_keywords = extract_key_info(original_text)
    
    # 检查各个预测字段
    checks = {}
    
    for field in ['predicted_cause', 'predicted_symptoms', 'predicted_treatment_process', 
                  'predicted_illness_Characteristics', 'predicted_treatment_effect']:
        predicted_text = result_item[field]
        predicted_keywords = extract_key_info(predicted_text)
        
        # 计算关键词覆盖度
        coverage = {}
        for category in original_keywords:
            if original_keywords[category]:
                overlap = set(original_keywords[category]) & set(predicted_keywords[category])
                coverage[category] = len(overlap) / len(original_keywords[category]) if original_keywords[category] else 0
        
        checks[field] = {
            'length': len(predicted_text),
            'coverage': coverage,
            'avg_coverage': sum(coverage.values()) / len(coverage) if coverage else 0
        }
    
    return checks, original_keywords

def evaluate_sample(sample_id, test_data, result_data):
    """评估单个样本"""
    
    print(f"\n{'='*80}")
    print(f"📝 样本 ID: {sample_id}")
    print(f"{'='*80}")
    
    test_item = test_data[sample_id]
    result_item = result_data[sample_id]
    
    # 显示原文概要
    case_desc = test_item['case_description']
    if isinstance(case_desc, list):
        case_desc = ' '.join(case_desc)
    
    print(f"\n📖 原文概要（前200字符）:")
    print(f"  {case_desc[:200]}...")
    
    # 检查一致性
    checks, original_keywords = check_consistency(test_item, result_item)
    
    print(f"\n🔍 原文关键信息统计:")
    for category, words in original_keywords.items():
        if words:
            print(f"  - {category}: {len(words)} 个关键词")
    
    print(f"\n✅ 分析字段评估:")
    
    overall_scores = []
    
    for field, check in checks.items():
        avg_cov = check['avg_coverage']
        length = check['length']
        
        # 评分
        if avg_cov >= 0.6 and length >= 500:
            score = "🌟 优秀"
            score_val = 5
        elif avg_cov >= 0.4 and length >= 300:
            score = "✓ 良好"
            score_val = 4
        elif avg_cov >= 0.2 and length >= 200:
            score = "⚠ 及格"
            score_val = 3
        else:
            score = "✗ 需改进"
            score_val = 2
        
        overall_scores.append(score_val)
        
        field_name = field.replace('predicted_', '')
        print(f"  {field_name}:")
        print(f"    - 长度: {length} 字符")
        print(f"    - 关键词覆盖度: {avg_cov:.1%}")
        print(f"    - 评分: {score}")
    
    # 总体评分
    avg_score = sum(overall_scores) / len(overall_scores)
    
    if avg_score >= 4.5:
        overall = "🌟 优秀"
    elif avg_score >= 3.5:
        overall = "✓ 良好"
    elif avg_score >= 2.5:
        overall = "⚠ 及格"
    else:
        overall = "✗ 需改进"
    
    print(f"\n💯 总体评价: {overall} (平均分: {avg_score:.1f}/5)")
    
    return avg_score

def main():
    print("\n" + "="*80)
    print("🔍 AI分析语义质量自查")
    print("随机抽取20个样本与原文对照验证")
    print("="*80)
    
    # 加载数据
    print("\n📂 正在加载数据...")
    test_data, result_data = load_data()
    print(f"✓ 测试集: {len(test_data)} 个样本")
    print(f"✓ 结果集: {len(result_data)} 个样本")
    
    # 随机抽取20个样本
    all_ids = list(test_data.keys())
    random.seed(42)  # 固定随机种子，保证可复现
    sample_ids = random.sample(all_ids, min(20, len(all_ids)))
    sample_ids.sort()
    
    print(f"\n🎲 随机抽取的样本ID: {sample_ids}")
    
    # 逐个评估
    scores = []
    for sample_id in sample_ids:
        score = evaluate_sample(sample_id, test_data, result_data)
        scores.append(score)
    
    # 总体统计
    print(f"\n{'='*80}")
    print("📊 总体统计结果")
    print(f"{'='*80}")
    
    avg_score = sum(scores) / len(scores)
    
    print(f"\n✓ 检查样本数: {len(scores)}")
    print(f"✓ 平均评分: {avg_score:.2f}/5")
    print(f"✓ 最高评分: {max(scores):.2f}/5")
    print(f"✓ 最低评分: {min(scores):.2f}/5")
    
    # 评分分布
    excellent = sum(1 for s in scores if s >= 4.5)
    good = sum(1 for s in scores if 3.5 <= s < 4.5)
    pass_count = sum(1 for s in scores if 2.5 <= s < 3.5)
    poor = sum(1 for s in scores if s < 2.5)
    
    print(f"\n📈 评分分布:")
    print(f"  🌟 优秀 (≥4.5): {excellent} 个样本 ({excellent/len(scores)*100:.1f}%)")
    print(f"  ✓ 良好 (3.5-4.4): {good} 个样本 ({good/len(scores)*100:.1f}%)")
    print(f"  ⚠ 及格 (2.5-3.4): {pass_count} 个样本 ({pass_count/len(scores)*100:.1f}%)")
    print(f"  ✗ 需改进 (<2.5): {poor} 个样本 ({poor/len(scores)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    
    if avg_score >= 4.0:
        print("✅ 结论: AI分析质量优秀，与原文高度一致！")
    elif avg_score >= 3.0:
        print("✅ 结论: AI分析质量良好，基本符合原文内容。")
    elif avg_score >= 2.0:
        print("⚠️ 结论: AI分析质量及格，但仍有改进空间。")
    else:
        print("✗ 结论: AI分析质量不足，需要优化。")
    
    print(f"{'='*80}\n")
    
    return avg_score

if __name__ == '__main__':
    avg_score = main()
    sys.exit(0 if avg_score >= 3.0 else 1)

