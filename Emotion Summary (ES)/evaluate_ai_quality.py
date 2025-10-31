# -*- coding: utf-8 -*-
"""
评估AI分析质量的脚本
"""

import json
import sys

# 确保UTF-8输出
sys.stdout.reconfigure(encoding='utf-8')

def evaluate_quality(sample_id, data):
    """评估单个样本的质量"""
    print(f"\n{'='*80}")
    print(f"📊 样本 {sample_id} 质量评估")
    print(f"{'='*80}")
    
    scores = {}
    
    # 1. 病因分析质量
    cause = data['predicted_cause']
    cause_length = len(cause)
    cause_has_logic = '→' in cause or 'stemming from' in cause.lower() or 'rooted in' in cause.lower()
    cause_has_specifics = any(word in cause.lower() for word in ['father', 'mother', 'childhood', 'relationship', 'trauma', 'adoption', 'suicide', 'betrayal'])
    
    scores['cause'] = {
        'length': cause_length,
        'has_logic_chain': cause_has_logic,
        'has_specific_details': cause_has_specifics,
        'score': min(10, cause_length // 50 + (5 if cause_has_logic else 0) + (3 if cause_has_specifics else 0))
    }
    
    # 2. 症状描述质量
    symptoms = data['predicted_symptoms']
    symptoms_length = len(symptoms)
    symptoms_categories = sum([
        'physical' in symptoms.lower(),
        'emotional' in symptoms.lower(),
        'behavioral' in symptoms.lower(),
        'cognitive' in symptoms.lower(),
        'psychological' in symptoms.lower()
    ])
    
    scores['symptoms'] = {
        'length': symptoms_length,
        'categories_covered': symptoms_categories,
        'score': min(10, symptoms_length // 60 + symptoms_categories * 2)
    }
    
    # 3. 治疗过程质量
    treatment = data['predicted_treatment_process']
    treatment_length = len(treatment)
    treatment_has_phases = any(word in treatment.lower() for word in ['phase', 'session', 'stage', 'step', 'initial', 'breakthrough'])
    treatment_has_methods = any(word in treatment.lower() for word in ['hypnosis', 'cbt', 'therapy', 'intervention', 'technique', 'dialogue'])
    
    scores['treatment'] = {
        'length': treatment_length,
        'has_phases': treatment_has_phases,
        'has_methods': treatment_has_methods,
        'score': min(10, treatment_length // 70 + (4 if treatment_has_phases else 0) + (4 if treatment_has_methods else 0))
    }
    
    # 4. 疾病特征质量
    characteristics = data['predicted_illness_Characteristics']
    char_length = len(characteristics)
    char_has_analysis = 'key characteristics' in characteristics.lower() or 'characterized by' in characteristics.lower()
    
    scores['characteristics'] = {
        'length': char_length,
        'has_structured_analysis': char_has_analysis,
        'score': min(10, char_length // 60 + (5 if char_has_analysis else 0))
    }
    
    # 5. 治疗效果质量
    effect = data['predicted_treatment_effect']
    effect_length = len(effect)
    effect_domains = sum([
        'cognitive' in effect.lower() or 'insight' in effect.lower(),
        'emotional' in effect.lower(),
        'behavioral' in effect.lower(),
        'relationship' in effect.lower() or 'relational' in effect.lower()
    ])
    
    scores['effect'] = {
        'length': effect_length,
        'domains_covered': effect_domains,
        'score': min(10, effect_length // 60 + effect_domains * 2)
    }
    
    # 计算总分
    total_score = sum(s['score'] for s in scores.values())
    max_score = 50
    percentage = (total_score / max_score) * 100
    
    # 打印评估结果
    print(f"\n📝 详细评分：")
    print(f"  1. 病因分析 (Cause): {scores['cause']['score']}/10")
    print(f"     - 长度: {scores['cause']['length']} 字符")
    print(f"     - 逻辑链: {'✓' if scores['cause']['has_logic_chain'] else '✗'}")
    print(f"     - 具体细节: {'✓' if scores['cause']['has_specific_details'] else '✗'}")
    
    print(f"\n  2. 症状描述 (Symptoms): {scores['symptoms']['score']}/10")
    print(f"     - 长度: {scores['symptoms']['length']} 字符")
    print(f"     - 覆盖类别: {scores['symptoms']['categories_covered']}/5")
    
    print(f"\n  3. 治疗过程 (Treatment): {scores['treatment']['score']}/10")
    print(f"     - 长度: {scores['treatment']['length']} 字符")
    print(f"     - 分阶段: {'✓' if scores['treatment']['has_phases'] else '✗'}")
    print(f"     - 具体方法: {'✓' if scores['treatment']['has_methods'] else '✗'}")
    
    print(f"\n  4. 疾病特征 (Characteristics): {scores['characteristics']['score']}/10")
    print(f"     - 长度: {scores['characteristics']['length']} 字符")
    print(f"     - 结构化分析: {'✓' if scores['characteristics']['has_structured_analysis'] else '✗'}")
    
    print(f"\n  5. 治疗效果 (Effect): {scores['effect']['score']}/10")
    print(f"     - 长度: {scores['effect']['length']} 字符")
    print(f"     - 覆盖领域: {scores['effect']['domains_covered']}/4")
    
    print(f"\n{'─'*80}")
    print(f"💯 总分: {total_score}/{max_score} ({percentage:.1f}%)")
    
    if percentage >= 80:
        quality = "🌟 优秀 (Excellent)"
    elif percentage >= 60:
        quality = "✓ 良好 (Good)"
    elif percentage >= 40:
        quality = "⚠ 及格 (Pass)"
    else:
        quality = "✗ 需改进 (Needs Improvement)"
    
    print(f"📊 质量评级: {quality}")
    
    return total_score, max_score, percentage

def main():
    print("\n" + "="*80)
    print("🎯 AI深度分析质量评估")
    print("="*80)
    
    # 读取两个批次的结果
    all_results = []
    
    try:
        with open('ai_results_batch1.json', 'r', encoding='utf-8') as f:
            batch1 = json.load(f)
            all_results.extend(batch1)
            print(f"\n✓ 读取 batch1: {len(batch1)} 个样本")
    except Exception as e:
        print(f"\n✗ 读取 batch1 失败: {e}")
    
    try:
        with open('ai_results_batch2.json', 'r', encoding='utf-8') as f:
            batch2 = json.load(f)
            all_results.extend(batch2)
            print(f"✓ 读取 batch2: {len(batch2)} 个样本")
    except Exception as e:
        print(f"✗ 读取 batch2 失败: {e}")
    
    print(f"\n📦 总计: {len(all_results)} 个样本待评估\n")
    
    # 评估每个样本
    total_scores = []
    for result in all_results:
        score, max_score, percentage = evaluate_quality(result['id'], result)
        total_scores.append(percentage)
    
    # 总体评估
    print("\n" + "="*80)
    print("📊 总体质量评估")
    print("="*80)
    
    avg_percentage = sum(total_scores) / len(total_scores)
    print(f"\n平均分数: {avg_percentage:.1f}%")
    print(f"最高分: {max(total_scores):.1f}%")
    print(f"最低分: {min(total_scores):.1f}%")
    
    excellent_count = sum(1 for s in total_scores if s >= 80)
    good_count = sum(1 for s in total_scores if 60 <= s < 80)
    pass_count = sum(1 for s in total_scores if 40 <= s < 60)
    poor_count = sum(1 for s in total_scores if s < 40)
    
    print(f"\n质量分布:")
    print(f"  🌟 优秀 (≥80%): {excellent_count} 个样本")
    print(f"  ✓ 良好 (60-79%): {good_count} 个样本")
    print(f"  ⚠ 及格 (40-59%): {pass_count} 个样本")
    print(f"  ✗ 需改进 (<40%): {poor_count} 个样本")
    
    print("\n" + "="*80)
    if avg_percentage >= 70:
        print("✅ 结论: 质量评估通过！可以继续处理全部150个样本。")
        return True
    else:
        print("⚠️ 结论: 质量需要改进，建议优化方法后再继续。")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

