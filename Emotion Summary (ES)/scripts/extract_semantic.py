# -*- coding: utf-8 -*-
"""语义理解版提取 - V4最终版"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def clean_sentence(text):
    """清理句子，确保完整性"""
    text = text.strip()
    # 如果句子被截断，尝试补全
    if text and not text[-1] in '.!?。！？':
        # 查找最后一个完整句子
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > len(text) * 0.5:  # 如果有超过一半的完整内容
            text = text[:last_period+1]
    return text

def deduplicate_sentences(sentences):
    """去重：移除高度相似的句子"""
    unique = []
    for sent in sentences:
        # 检查是否与已有句子重复
        is_duplicate = False
        for existing in unique:
            # 如果两个句子有80%以上相同，认为是重复
            overlap = sum(1 for word in sent.split() if word in existing.split())
            if overlap / max(len(sent.split()), 1) > 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(sent)
    return unique

def extract_cause_semantic(case_desc, consult_process):
    """
    病因提取 - 完整逻辑链版本
    目标：关键事件 → 情感冲突 → 关系阻碍 → 心理转化
    """
    full_text = " ".join(case_desc + consult_process)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    cause_info = {
        'events': [],      # 关键事件
        'emotions': [],    # 情感冲突
        'relationships': [], # 关系问题
        'transformation': [] # 心理转化
    }
    
    # 1. 关键事件（家庭背景、重大事件）
    event_keywords = ['adopted', 'adoption', 'discovered', 'found out', 'after marriage', 
                     'biological', 'father passed', 'mother left', 'divorce', 'death']
    for sent in sentences:
        if any(kw in sent.lower() for kw in event_keywords):
            if len(sent) > 40:
                cause_info['events'].append(sent.strip())
    
    # 2. 情感冲突
    emotion_keywords = ['guilt', 'shame', 'anger', 'resentment', 'abandoned', 'rejected',
                       'confused', 'conflicted', 'irritated', 'frustrated']
    for sent in sentences:
        if any(kw in sent.lower() for kw in emotion_keywords):
            if len(sent) > 40:
                cause_info['emotions'].append(sent.strip())
    
    # 3. 关系阻碍
    relation_keywords = ['refused', 'opposed', 'unwilling', 'conflict with', 'disagreement',
                        'wife', 'husband', 'parents', 'family', 'divorce']
    for sent in sentences:
        if any(kw in sent.lower() for kw in relation_keywords):
            # 只保留涉及冲突/阻碍的
            if any(neg in sent.lower() for neg in ['refused', 'opposed', 'unwilling', 'conflict', 'disagree']):
                if len(sent) > 40:
                    cause_info['relationships'].append(sent.strip())
    
    # 4. 心理转化（如何导致当前问题）
    transform_keywords = ['led to', 'resulted in', 'caused', 'triggered', 'manifested as',
                         'developed', 'began to', 'started to']
    for sent in sentences:
        if any(kw in sent.lower() for kw in transform_keywords):
            if len(sent) > 40:
                cause_info['transformation'].append(sent.strip())
    
    # 去重并组织
    result_parts = []
    
    # 优先级：事件 → 情感 → 关系 → 转化
    for key in ['events', 'emotions', 'relationships', 'transformation']:
        items = deduplicate_sentences(cause_info[key][:2])  # 每类最多2句
        result_parts.extend(items)
    
    if result_parts:
        result = " ".join(result_parts)
        return clean_sentence(result[:600])
    
    # 备选
    return clean_sentence(" ".join(case_desc)[:400])

def extract_symptoms_semantic(case_desc, consult_process):
    """
    症状提取 - 完整分类版本
    目标：躯体症状 + 心理症状 + 行为症状
    """
    full_text = " ".join(case_desc + consult_process)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    symptom_categories = {
        'physical': [],    # 躯体症状
        'psychological': [], # 心理症状
        'behavioral': []   # 行为症状
    }
    
    # 1. 躯体症状
    physical_keywords = ['pain', 'discomfort', 'dizzy', 'nausea', 'throat', 'chest', 
                        'stomach', 'headache', 'insomnia', 'sleep', 'fatigue',
                        'nasopharynx', 'bowel', 'hemorrhoid']
    
    # 2. 心理症状
    psychological_keywords = ['anxiety', 'anxious', 'worry', 'worried', 'fear', 'afraid',
                             'panic', 'depress', 'sad', 'hopeless', 'obsess', 'compulsive']
    
    # 3. 行为症状
    behavioral_keywords = ['check', 'repeatedly', 'constantly', 'avoid', 'unable to',
                          'difficulty', 'search', 'looked up', 'diagnosed', 'self-diagnose']
    
    # 排除词（改善、恢复相关）
    exclude_keywords = ['improved', 'better', 'recovered', 'relief', 'no longer',
                       'stopped', 'able to', 'began to improve']
    
    for sent in sentences:
        sent_lower = sent.lower()
        
        # 如果包含排除词，跳过
        if any(ex in sent_lower for ex in exclude_keywords):
            continue
        
        if len(sent) < 40:
            continue
        
        # 分类
        if any(kw in sent_lower for kw in physical_keywords):
            symptom_categories['physical'].append(sent.strip())
        if any(kw in sent_lower for kw in psychological_keywords):
            symptom_categories['psychological'].append(sent.strip())
        if any(kw in sent_lower for kw in behavioral_keywords):
            symptom_categories['behavioral'].append(sent.strip())
    
    # 去重并组织
    result_parts = []
    
    # 每类选最相关的2句
    for category in ['physical', 'psychological', 'behavioral']:
        items = deduplicate_sentences(symptom_categories[category][:2])
        result_parts.extend(items)
    
    if result_parts:
        result = " ".join(result_parts)
        return clean_sentence(result[:700])
    
    # 备选
    return clean_sentence(" ".join(case_desc[:2])[:500])

def extract_treatment_semantic(case_desc, consult_process):
    """
    治疗过程提取 - 专业治疗为主
    目标：专业干预步骤（排除自行尝试、通用理论）
    """
    full_text = " ".join(consult_process)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    treatment_info = {
        'professional': [],  # 专业治疗
        'techniques': [],    # 具体技术
        'process': []        # 治疗步骤
    }
    
    # 专业治疗关键词
    professional_keywords = ['therapist', 'counselor', 'psychologist', 'conducted',
                           'guided', 'facilitated', 'explored', 'addressed', 'helped',
                           'session', 'consultation']
    
    # 具体技术
    technique_keywords = ['hypnosis', 'hypnotherapy', 'imagery', 'dialogue', 'cognitive',
                         'behavioral', 'exposure', 'mindfulness', 'relaxation']
    
    # 排除词（自行尝试、通用理论）
    exclude_keywords = ['tried', 'attempted', 'without effect', 'ineffective', 'failed',
                       'generally', 'typically', 'usually', 'medication can only']
    
    for sent in sentences:
        sent_lower = sent.lower()
        
        # 排除自行尝试和通用理论
        if any(ex in sent_lower for ex in exclude_keywords):
            continue
        
        if len(sent) < 40:
            continue
        
        # 优先专业治疗
        if any(kw in sent_lower for kw in technique_keywords):
            treatment_info['techniques'].append(sent.strip())
        elif any(kw in sent_lower for kw in professional_keywords):
            treatment_info['professional'].append(sent.strip())
    
    # 去重并组织
    result_parts = []
    
    # 优先技术描述
    for key in ['techniques', 'professional']:
        items = deduplicate_sentences(treatment_info[key][:3])
        result_parts.extend(items)
    
    if result_parts:
        result = " ".join(result_parts)
        return clean_sentence(result[:700])
    
    # 备选
    return clean_sentence(" ".join(consult_process[1:3])[:600])

def extract_characteristics_semantic(case_desc, consult_process):
    """
    疾病特征提取 - 归纳总结版本
    目标：本质属性 + 症状规律（非重复症状）
    """
    full_text = " ".join(case_desc + consult_process)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    char_info = {
        'nature': [],     # 本质属性（心理性、非器质性）
        'patterns': [],   # 表现规律（波动性、交替性）
        'mechanisms': []  # 心理机制（防御、强迫）
    }
    
    # 本质属性
    nature_keywords = ['psychological', 'mental', 'emotional', 'not physical',
                      'no organic', 'psychosomatic', 'disorder', 'condition']
    
    # 表现规律
    pattern_keywords = ['fluctuate', 'vary', 'depend on', 'influenced by', 'when',
                       'attention', 'distract', 'focus', 'disappear', 'reappear']
    
    # 心理机制
    mechanism_keywords = ['defense', 'mechanism', 'cope', 'avoid', 'transfer',
                         'obsessive', 'compulsive', 'anxiety', 'hypochondria']
    
    # 排除词（具体症状描述）
    exclude_keywords = ['throat', 'stomach', 'pain', 'discomfort', 'specific symptom']
    
    for sent in sentences:
        sent_lower = sent.lower()
        
        if len(sent) < 40:
            continue
        
        # 归纳性描述，非具体症状
        if any(kw in sent_lower for kw in nature_keywords):
            char_info['nature'].append(sent.strip())
        if any(kw in sent_lower for kw in pattern_keywords):
            char_info['patterns'].append(sent.strip())
        if any(kw in sent_lower for kw in mechanism_keywords):
            char_info['mechanisms'].append(sent.strip())
    
    # 去重并组织
    result_parts = []
    
    for key in ['nature', 'patterns', 'mechanisms']:
        items = deduplicate_sentences(char_info[key][:2])
        result_parts.extend(items)
    
    if result_parts:
        result = " ".join(result_parts)
        return clean_sentence(result[:600])
    
    # 备选
    return clean_sentence(" ".join(case_desc)[:400])

def extract_effect_semantic(case_desc, consult_process):
    """
    治疗效果提取 - 实际改善版本
    目标：情绪/行为/认知改善（排除失败尝试、通用信息）
    """
    full_text = " ".join(consult_process)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    effect_info = {
        'emotional': [],   # 情绪改善
        'behavioral': [],  # 行为变化
        'cognitive': []    # 认知改变
    }
    
    # 情绪改善
    emotional_keywords = ['felt better', 'relief', 'relieved', 'happy', 'smiled', 'smile',
                         'calm', 'peaceful', 'relaxed', 'burden lifted', 'changed person']
    
    # 行为变化
    behavioral_keywords = ['no longer', 'stopped', 'able to', 'began to', 'started to',
                          'returned to', 'resumed', 'improved']
    
    # 认知改变
    cognitive_keywords = ['realized', 'understood', 'recognized', 'insight', 'aware',
                         'acknowledged', 'accepted']
    
    # 排除词（失败尝试、通用理论）
    exclude_keywords = ['tried but failed', 'without effect', 'ineffective', 'unable to',
                       'generally', 'typically', 'medication can only', 'hoped to']
    
    # 优先从后半部分提取
    second_half_start = len(sentences) // 2
    
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        
        # 排除失败和通用信息
        if any(ex in sent_lower for ex in exclude_keywords):
            continue
        
        if len(sent) < 40:
            continue
        
        # 后半部分权重更高
        weight = 2 if i >= second_half_start else 1
        
        if any(kw in sent_lower for kw in emotional_keywords):
            effect_info['emotional'].append((weight, sent.strip()))
        if any(kw in sent_lower for kw in behavioral_keywords):
            # 确保是积极变化
            if any(pos in sent_lower for pos in ['better', 'improved', 'able', 'began', 'started']):
                effect_info['behavioral'].append((weight, sent.strip()))
        if any(kw in sent_lower for kw in cognitive_keywords):
            effect_info['cognitive'].append((weight, sent.strip()))
    
    # 按权重排序并去重
    result_parts = []
    
    for key in ['emotional', 'behavioral', 'cognitive']:
        # 排序
        items = sorted(effect_info[key], key=lambda x: x[0], reverse=True)
        sents = [item[1] for item in items[:2]]
        sents = deduplicate_sentences(sents)
        result_parts.extend(sents)
    
    if result_parts:
        result = " ".join(result_parts)
        return clean_sentence(result[:600])
    
    # 备选
    return clean_sentence(" ".join(consult_process[-2:])[:500])

def main():
    print("="*80)
    print("🧠 语义理解版提取 - V4最终版")
    print("="*80)
    
    test_file = Path("data/test/Emotion_Summary.jsonl")
    print(f"\n📊 加载测试数据: {test_file}")
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"✅ 测试样本数: {len(test_data)}")
    
    print(f"\n💡 V4改进要点:")
    print(f"  1. 病因: 完整逻辑链（事件→情感→关系→转化）")
    print(f"  2. 症状: 完整分类（躯体+心理+行为），去重")
    print(f"  3. 治疗: 专业干预为主（排除自行尝试）")
    print(f"  4. 特征: 归纳总结（非重复症状）")
    print(f"  5. 效果: 实际改善（排除失败和通用信息）")
    
    print(f"\n🔮 开始语义提取...\n")
    results = []
    
    for item in tqdm(test_data, desc="提取进度"):
        item_id = item.get("id", 0)
        case_desc = item.get("case_description", [])
        consult_process = item.get("consultation_process", [])
        
        result = {
            "id": item_id,
            "predicted_cause": extract_cause_semantic(case_desc, consult_process),
            "predicted_symptoms": extract_symptoms_semantic(case_desc, consult_process),
            "predicted_treatment_process": extract_treatment_semantic(case_desc, consult_process),
            "predicted_illness_Characteristics": extract_characteristics_semantic(case_desc, consult_process),
            "predicted_treatment_effect": extract_effect_semantic(case_desc, consult_process)
        }
        
        results.append(result)
    
    # 备份并保存
    output_file = Path("results/Emotion_Summary_Result.jsonl")
    
    if output_file.exists():
        backup_file = Path("results/Emotion_Summary_Result_v3_precise.jsonl")
        import shutil
        shutil.copy(output_file, backup_file)
        print(f"\n📦 V3版本已备份到: {backup_file}")
    
    print(f"\n💾 保存V4结果到: {output_file} (覆盖)")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ 完成！共生成 {len(results)} 条结果")
    
    # 显示示例
    print(f"\n📝 示例输出 (样本 1):")
    print("="*80)
    sample = results[0]
    
    fields = [
        ("predicted_cause", "病因"),
        ("predicted_symptoms", "症状"),
        ("predicted_treatment_process", "治疗过程"),
        ("predicted_illness_Characteristics", "疾病特征"),
        ("predicted_treatment_effect", "治疗效果")
    ]
    
    for field, name in fields:
        content = sample[field]
        print(f"\n【{name}】({len(content)} 字符)")
        print(f"{content[:200]}...")
    
    # 质量检查
    print(f"\n" + "="*80)
    print(f"🔍 质量检查")
    print("="*80)
    
    contents = [sample[field] for field, _ in fields]
    unique = len(set(contents))
    print(f"\n1. 重复度: {unique}/5 唯一")
    
    lengths = [len(c) for c in contents]
    print(f"2. 平均长度: {sum(lengths)/len(lengths):.0f} 字符")
    
    # 检查关键信息
    full_text = " ".join(contents)
    checks = {
        "收养背景": any(kw in full_text.lower() for kw in ['adopt', 'biological']),
        "情感冲突": any(kw in full_text.lower() for kw in ['anger', 'guilt', 'conflict']),
        "催眠治疗": any(kw in full_text.lower() for kw in ['hypnosis', 'imagery']),
        "症状分类": any(kw in full_text.lower() for kw in ['throat', 'anxiety', 'check']),
        "治疗改善": any(kw in full_text.lower() for kw in ['smile', 'relief', 'better'])
    }
    
    print(f"\n3. 关键信息覆盖:")
    for key, value in checks.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key}")
    
    print("\n" + "="*80)
    print("✅ V4语义提取完成！")
    print(f"📁 提交文件: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()

