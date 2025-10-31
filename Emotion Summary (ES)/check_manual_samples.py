# -*- coding: utf-8 -*-
"""
专门检查前10个手动分析样本的质量
"""

import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    """加载测试集和结果数据"""
    test_data = {}
    with open('data/test/Emotion_Summary.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                test_data[item['id']] = item
    
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
    
    case_desc = ' '.join(test_item['case_description']) if isinstance(test_item['case_description'], list) else test_item['case_description']
    consultation = ' '.join(test_item['consultation_process']) if isinstance(test_item['consultation_process'], list) else test_item['consultation_process']
    reflection = test_item['experience_and_reflection']
    
    original_text = f"{case_desc} {consultation} {reflection}".lower()
    original_keywords = extract_key_info(original_text)
    
    checks = {}
    
    for field in ['predicted_cause', 'predicted_symptoms', 'predicted_treatment_process', 
                  'predicted_illness_Characteristics', 'predicted_treatment_effect']:
        predicted_text = result_item[field]
        predicted_keywords = extract_key_info(predicted_text)
        
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
    
    return checks

def main():
    print("\n" + "="*80)
    print("🔍 检查前10个手动分析样本的质量")
    print("="*80)
    
    test_data, result_data = load_data()
    
    # 检查ID 1-10
    manual_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    scores = []
    
    for sample_id in manual_ids:
        if sample_id not in result_data:
            print(f"\n⚠️ 样本 {sample_id} 不存在于结果中")
            continue
        
        test_item = test_data[sample_id]
        result_item = result_data[sample_id]
        
        checks = check_consistency(test_item, result_item)
        
        field_scores = []
        for field, check in checks.items():
            avg_cov = check['avg_coverage']
            length = check['length']
            
            if avg_cov >= 0.6 and length >= 500:
                score_val = 5
            elif avg_cov >= 0.4 and length >= 300:
                score_val = 4
            elif avg_cov >= 0.2 and length >= 200:
                score_val = 3
            else:
                score_val = 2
            
            field_scores.append(score_val)
        
        avg_score = sum(field_scores) / len(field_scores)
        scores.append(avg_score)
        
        if avg_score >= 4.5:
            status = "🌟"
        elif avg_score >= 3.5:
            status = "✓"
        elif avg_score >= 2.5:
            status = "⚠"
        else:
            status = "✗"
        
        print(f"  样本 {sample_id:2d}: {status} {avg_score:.2f}/5")
    
    print(f"\n{'='*80}")
    print(f"📊 前10个样本统计")
    print(f"{'='*80}")
    print(f"  平均分: {sum(scores)/len(scores):.2f}/5")
    print(f"  最高分: {max(scores):.2f}/5")
    print(f"  最低分: {min(scores):.2f}/5")
    
    excellent = sum(1 for s in scores if s >= 4.5)
    good = sum(1 for s in scores if 3.5 <= s < 4.5)
    pass_count = sum(1 for s in scores if 2.5 <= s < 3.5)
    poor = sum(1 for s in scores if s < 2.5)
    
    print(f"\n  🌟 优秀: {excellent} 个")
    print(f"  ✓ 良好: {good} 个")
    print(f"  ⚠ 及格: {pass_count} 个")
    print(f"  ✗ 需改进: {poor} 个")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()

