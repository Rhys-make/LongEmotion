# -*- coding: utf-8 -*-
"""精确提取脚本 - 基于五维度核心信息提炼指南"""

import json
import re
from pathlib import Path
from tqdm import tqdm

def extract_cause_precise(case_desc, consult_process):
    """
    病因：深层心理冲突、未解决的情感矛盾或关键现实事件
    重点：关键事件→情感冲突→关系阻碍→心理转化
    """
    
    full_text = " ".join(case_desc + consult_process)
    
    # 关键词：深层原因、背景事件、情感冲突
    cause_patterns = [
        # 家庭背景
        r'(adopted|adoption|biological parent|father|mother|family background|childhood|abuse|divorce|death|loss)[^.!?。！？]{0,200}[.!?。！？]',
        # 情感冲突
        r'(conflict|guilt|shame|anger|resentment|abandonment|rejection|betray)[^.!?。！？]{0,200}[.!?。！？]',
        # 关键事件
        r'(after|when|following|since|because of|due to|result from|stem from|trigger)[^.!?。！？]{0,200}[.!?。！？]',
        # 关系问题
        r'(relationship|marriage|wife|husband|refused|opposed|conflict with)[^.!?。！？]{0,200}[.!?。！？]',
    ]
    
    cause_sentences = []
    for pattern in cause_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            sent = match.group(0).strip()
            if len(sent) > 30 and sent not in cause_sentences:
                cause_sentences.append(sent)
    
    # 优先选择包含"深层原因"的句子
    if cause_sentences:
        # 按相关性排序（包含多个关键词的优先）
        scored = []
        for sent in cause_sentences:
            score = sum([
                2 if any(kw in sent.lower() for kw in ['adopted', 'biological', 'father', 'mother']) else 0,
                2 if any(kw in sent.lower() for kw in ['guilt', 'shame', 'conflict', 'abandonment']) else 0,
                1 if any(kw in sent.lower() for kw in ['because', 'due to', 'result']) else 0,
            ])
            scored.append((score, sent))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        result = " ".join([s[1] for s in scored[:4]])
        
        if len(result) > 600:
            return result[:600].rsplit('.', 1)[0] + "."
        return result
    
    # 备选：从咨询过程中提取涉及病因分析的部分
    analysis_keywords = ['cause', 'reason', 'origin', 'root', 'stem', 'underlying']
    for text in consult_process:
        if any(kw in text.lower() for kw in analysis_keywords):
            if len(text) > 400:
                return text[:400].rsplit('.', 1)[0] + "."
            return text
    
    # 最后备选
    return " ".join(case_desc)[:400].rsplit('.', 1)[0] + "."

def extract_symptoms_precise(case_desc, consult_process):
    """
    症状：具体的躯体/心理/行为表现
    重点：具体可感知的症状，排除恢复过程
    """
    
    full_text = " ".join(case_desc + consult_process)
    
    # 症状关键词
    symptom_patterns = [
        # 躯体症状
        r'(pain|discomfort|dizzy|nausea|throat|chest|stomach|head|sleep)[^.!?。！？]{0,200}[.!?。！？]',
        # 心理症状
        r'(anxiety|depress|worry|fear|panic|obsess|compulsive|avoid)[^.!?。！？]{0,200}[.!?。！？]',
        # 行为症状
        r'(check|search|avoid|unable to|difficulty|repeatedly|constantly)[^.!?。！？]{0,200}[.!?。！？]',
        # 症状描述
        r'(symptom|experience|feel|suffer|complain)[^.!?。！？]{0,200}[.!?。！？]',
    ]
    
    symptom_sentences = []
    for pattern in symptom_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            sent = match.group(0).strip()
            # 排除"恢复"相关的句子
            if len(sent) > 30 and sent not in symptom_sentences:
                if not any(kw in sent.lower() for kw in ['improved', 'better', 'recovered', 'relief']):
                    symptom_sentences.append(sent)
    
    if symptom_sentences:
        # 按症状相关性排序
        scored = []
        for sent in symptom_sentences:
            score = sum([
                2 if any(kw in sent.lower() for kw in ['anxiety', 'depression', 'obsessive']) else 0,
                2 if any(kw in sent.lower() for kw in ['pain', 'discomfort', 'throat', 'chest']) else 0,
                1 if any(kw in sent.lower() for kw in ['repeatedly', 'constantly', 'unable']) else 0,
            ])
            scored.append((score, sent))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        result = " ".join([s[1] for s in scored[:5]])
        
        if len(result) > 700:
            return result[:700].rsplit('.', 1)[0] + "."
        return result
    
    # 备选：从案例描述中提取
    for text in case_desc:
        if any(kw in text.lower() for kw in ['symptom', 'experience', 'suffer', 'feel']):
            if len(text) > 500:
                return text[:500].rsplit('.', 1)[0] + "."
            return text
    
    return " ".join(case_desc[:2])[:500].rsplit('.', 1)[0] + "."

def extract_treatment_precise(case_desc, consult_process):
    """
    治疗过程：针对疾病的干预步骤
    重点：具体治疗方法，排除通用理论
    """
    
    full_text = " ".join(consult_process)
    
    # 治疗关键词
    treatment_patterns = [
        # 具体治疗技术
        r'(hypnosis|hypnotherapy|cognitive|behavioral|imagery|dialogue|exposure)[^.!?。！？]{0,250}[.!?。！？]',
        # 治疗过程
        r'(session|consultation|therapy|treatment|intervention|technique)[^.!?。！？]{0,250}[.!?。！？]',
        # 咨询师行为
        r'(counselor|therapist|guided|conducted|explored|helped|addressed)[^.!?。！？]{0,250}[.!?。！？]',
        # 治疗步骤
        r'(first|initially|then|next|following|during|after the)[^.!?。！？]{0,250}[.!?。！？]',
    ]
    
    treatment_sentences = []
    for pattern in treatment_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            sent = match.group(0).strip()
            if len(sent) > 30 and sent not in treatment_sentences:
                treatment_sentences.append(sent)
    
    if treatment_sentences:
        # 优先包含具体技术的句子
        scored = []
        for sent in treatment_sentences:
            score = sum([
                3 if any(kw in sent.lower() for kw in ['hypnosis', 'imagery', 'cognitive', 'behavioral']) else 0,
                2 if any(kw in sent.lower() for kw in ['guided', 'conducted', 'explored']) else 0,
                1 if any(kw in sent.lower() for kw in ['session', 'therapy', 'treatment']) else 0,
            ])
            scored.append((score, sent))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        result = " ".join([s[1] for s in scored[:4]])
        
        if len(result) > 700:
            return result[:700].rsplit('.', 1)[0] + "."
        return result
    
    # 备选：提取咨询过程的中间部分
    if len(consult_process) > 2:
        middle = " ".join(consult_process[1:3])
        if len(middle) > 600:
            return middle[:600].rsplit('.', 1)[0] + "."
        return middle
    
    return " ".join(consult_process[:2])[:600].rsplit('.', 1)[0] + "."

def extract_characteristics_precise(case_desc, consult_process):
    """
    疾病特征：本质属性与表现规律
    重点：归纳总结，提炼规律
    """
    
    full_text = " ".join(case_desc + consult_process)
    
    # 特征关键词
    char_patterns = [
        # 疾病性质
        r'(psychological|mental|emotional|disorder|condition|nature)[^.!?。！？]{0,200}[.!?。！？]',
        # 表现规律
        r'(pattern|characteristic|feature|tend to|typically|repeatedly)[^.!?。！？]{0,200}[.!?。！？]',
        # 症状规律
        r'(when|whenever|fluctuate|variable|depend on|influenced by)[^.!?。！？]{0,200}[.!?。！？]',
        # 心理机制
        r'(defense|mechanism|cope|avoid|anxiety|obsessive|compulsive)[^.!?。！？]{0,200}[.!?。！？]',
    ]
    
    char_sentences = []
    for pattern in char_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            sent = match.group(0).strip()
            if len(sent) > 30 and sent not in char_sentences:
                char_sentences.append(sent)
    
    if char_sentences:
        scored = []
        for sent in char_sentences:
            score = sum([
                2 if any(kw in sent.lower() for kw in ['psychological', 'mental', 'disorder']) else 0,
                2 if any(kw in sent.lower() for kw in ['pattern', 'characteristic', 'typically']) else 0,
                1 if any(kw in sent.lower() for kw in ['fluctuate', 'depend', 'influenced']) else 0,
            ])
            scored.append((score, sent))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        result = " ".join([s[1] for s in scored[:4]])
        
        if len(result) > 600:
            return result[:600].rsplit('.', 1)[0] + "."
        return result
    
    # 备选：从案例分析部分提取
    for text in consult_process:
        if any(kw in text.lower() for kw in ['analysis', 'characteristic', 'nature', 'disorder']):
            if len(text) > 500:
                return text[:500].rsplit('.', 1)[0] + "."
            return text
    
    return " ".join(case_desc)[:400].rsplit('.', 1)[0] + "."

def extract_effect_precise(case_desc, consult_process):
    """
    治疗效果：治疗后的具体改善
    重点：实际变化，排除预期效果
    """
    
    full_text = " ".join(consult_process)
    
    # 效果关键词
    effect_patterns = [
        # 情绪改善
        r'(felt better|relief|improved|happy|smile|calm|peaceful)[^.!?。！？]{0,200}[.!?。！？]',
        # 行为变化
        r'(no longer|stopped|able to|began to|started to)[^.!?。！？]{0,200}[.!?。！？]',
        # 认知变化
        r'(realized|understood|recognized|insight|aware)[^.!?。！？]{0,200}[.!?。！？]',
        # 效果描述
        r'(after|following|result|outcome|effect|progress)[^.!?。！？]{0,200}[.!?。！？]',
    ]
    
    effect_sentences = []
    
    # 优先从后半部分提取（通常包含治疗效果）
    second_half = consult_process[len(consult_process)//2:]
    text_to_search = " ".join(second_half)
    
    for pattern in effect_patterns:
        matches = re.finditer(pattern, text_to_search, re.IGNORECASE)
        for match in matches:
            sent = match.group(0).strip()
            if len(sent) > 30 and sent not in effect_sentences:
                effect_sentences.append(sent)
    
    # 如果后半部分没找到，再查找前半部分
    if not effect_sentences:
        for pattern in effect_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                sent = match.group(0).strip()
                if len(sent) > 30 and sent not in effect_sentences:
                    effect_sentences.append(sent)
    
    if effect_sentences:
        scored = []
        for sent in effect_sentences:
            score = sum([
                3 if any(kw in sent.lower() for kw in ['relief', 'improved', 'happy', 'smile']) else 0,
                2 if any(kw in sent.lower() for kw in ['no longer', 'able to', 'began']) else 0,
                1 if any(kw in sent.lower() for kw in ['realized', 'understood', 'insight']) else 0,
            ])
            scored.append((score, sent))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        result = " ".join([s[1] for s in scored[:4]])
        
        if len(result) > 600:
            return result[:600].rsplit('.', 1)[0] + "."
        return result
    
    # 备选：提取最后部分
    if consult_process:
        last_part = consult_process[-1]
        if len(last_part) > 500:
            return last_part[:500].rsplit('.', 1)[0] + "."
        return last_part
    
    return "The treatment showed positive effects on the client's condition."

def main():
    print("="*80)
    print("🎯 精确提取 - 基于五维度核心信息提炼指南")
    print("="*80)
    
    # 加载测试数据
    test_file = Path("data/test/Emotion_Summary.jsonl")
    print(f"\n📊 加载测试数据: {test_file}")
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"✅ 测试样本数: {len(test_data)}")
    
    print(f"\n💡 提取策略（基于五维度指南）:")
    print(f"  1. 病因: 深层心理冲突、情感矛盾、关键事件")
    print(f"  2. 症状: 具体躯体/心理/行为表现（排除恢复过程）")
    print(f"  3. 治疗过程: 具体干预步骤（排除通用理论）")
    print(f"  4. 疾病特征: 本质属性与表现规律（归纳总结）")
    print(f"  5. 治疗效果: 实际改善（排除预期效果）")
    
    # 进行提取
    print(f"\n🔮 开始精确提取...\n")
    results = []
    
    for item in tqdm(test_data, desc="提取进度"):
        item_id = item.get("id", 0)
        case_desc = item.get("case_description", [])
        consult_process = item.get("consultation_process", [])
        
        result = {
            "id": item_id,
            "predicted_cause": extract_cause_precise(case_desc, consult_process),
            "predicted_symptoms": extract_symptoms_precise(case_desc, consult_process),
            "predicted_treatment_process": extract_treatment_precise(case_desc, consult_process),
            "predicted_illness_Characteristics": extract_characteristics_precise(case_desc, consult_process),
            "predicted_treatment_effect": extract_effect_precise(case_desc, consult_process)
        }
        
        results.append(result)
    
    # 备份并保存
    output_file = Path("results/Emotion_Summary_Result.jsonl")
    
    if output_file.exists():
        backup_file = Path("results/Emotion_Summary_Result_v2_rules.jsonl")
        import shutil
        shutil.copy(output_file, backup_file)
        print(f"\n📦 旧版本已备份到: {backup_file}")
    
    print(f"\n💾 保存结果到: {output_file} (覆盖)")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ 完成！共生成 {len(results)} 条结果")
    
    # 显示示例
    print(f"\n📝 示例输出 (样本 1 - 34岁男性疑病症案例):")
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
    
    # 检查关键词覆盖
    keywords = ["adopt", "hypnosis", "imagery", "anxiety", "relief"]
    full_text = " ".join(contents).lower()
    found = [kw for kw in keywords if kw in full_text]
    print(f"3. 关键词覆盖: {len(found)}/{len(keywords)} ({', '.join(found)})")
    
    print("\n" + "="*80)
    print("✅ 精确提取完成！")
    print(f"📁 提交文件: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()

