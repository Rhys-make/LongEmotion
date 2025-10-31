# -*- coding: utf-8 -*-
"""基于规则的智能提取脚本 - 直接从案例中提取5个字段"""

import json
import re
from pathlib import Path
from tqdm import tqdm

def extract_cause(case_desc, consult_process):
    """提取病因"""
    
    full_text = " ".join(case_desc + consult_process)
    
    # 关键词和模式
    cause_keywords = [
        "cause", "reason", "origin", "stem", "root", "trigger", 
        "because", "due to", "result from", "attributed to",
        "背景", "原因", "诱因", "导致", "由于",
        "childhood", "family", "past", "experience", "trauma",
        "father", "mother", "parent", "marriage", "divorce",
        "adoption", "adopted", "biological"
    ]
    
    # 查找包含这些关键词的句子
    sentences = re.split(r'[.!?。！？]', full_text)
    relevant_sentences = []
    
    for sent in sentences:
        if any(kw in sent.lower() for kw in cause_keywords):
            sent = sent.strip()
            if len(sent) > 20:  # 过滤太短的句子
                relevant_sentences.append(sent)
    
    # 如果找到相关句子，组合它们
    if relevant_sentences:
        # 取前3-5个最相关的句子
        result = ". ".join(relevant_sentences[:5])
        if len(result) > 600:
            result = result[:600] + "..."
        return result
    
    # 如果没找到，返回案例描述的摘要
    case_text = " ".join(case_desc)
    if len(case_text) > 400:
        return case_text[:400] + "..."
    return case_text

def extract_symptoms(case_desc, consult_process):
    """提取症状"""
    
    full_text = " ".join(case_desc + consult_process)
    
    # 症状关键词
    symptom_keywords = [
        "symptom", "suffer", "feel", "experience", "complaint",
        "anxiety", "depres", "insomnia", "sleep", "stress",
        "pain", "discomfort", "worry", "fear", "panic",
        "obsess", "compuls", "avoid", "withdraw",
        "症状", "表现", "感到", "焦虑", "抑郁", "失眠"
    ]
    
    sentences = re.split(r'[.!?。！？]', full_text)
    relevant_sentences = []
    
    for sent in sentences:
        if any(kw in sent.lower() for kw in symptom_keywords):
            sent = sent.strip()
            if len(sent) > 20:
                relevant_sentences.append(sent)
    
    if relevant_sentences:
        result = ". ".join(relevant_sentences[:6])
        if len(result) > 700:
            result = result[:700] + "..."
        return result
    
    # 备选：从咨询过程开头提取
    if consult_process:
        first_part = consult_process[0]
        if len(first_part) > 400:
            return first_part[:400] + "..."
        return first_part
    
    return "The client presented with psychological distress requiring counseling support."

def extract_treatment_process(case_desc, consult_process):
    """提取治疗过程"""
    
    full_text = " ".join(consult_process)  # 主要从咨询过程提取
    
    # 治疗关键词
    treatment_keywords = [
        "treatment", "therapy", "counseling", "intervention",
        "technique", "method", "approach", "session",
        "hypnosis", "hypnotherapy", "cognitive", "behavioral",
        "guided imagery", "dialogue", "exploration",
        "治疗", "咨询", "疗法", "干预", "催眠"
    ]
    
    sentences = re.split(r'[.!?。！？]', full_text)
    relevant_sentences = []
    
    for sent in sentences:
        if any(kw in sent.lower() for kw in treatment_keywords):
            sent = sent.strip()
            if len(sent) > 20:
                relevant_sentences.append(sent)
    
    if relevant_sentences:
        result = ". ".join(relevant_sentences[:6])
        if len(result) > 700:
            result = result[:700] + "..."
        return result
    
    # 备选：取咨询过程中间部分
    if len(consult_process) > 2:
        mid_text = " ".join(consult_process[1:3])
        if len(mid_text) > 500:
            return mid_text[:500] + "..."
        return mid_text
    
    return "The counselor employed therapeutic techniques to address the client's concerns."

def extract_illness_characteristics(case_desc, consult_process):
    """提取疾病特征"""
    
    full_text = " ".join(case_desc + consult_process)
    
    # 特征关键词
    char_keywords = [
        "characteristic", "feature", "pattern", "nature",
        "disorder", "condition", "illness", "problem",
        "psychological", "mental", "emotional", "behavioral",
        "persistent", "recurrent", "chronic", "acute",
        "特征", "特点", "性质", "模式"
    ]
    
    sentences = re.split(r'[.!?。！？]', full_text)
    relevant_sentences = []
    
    for sent in sentences:
        if any(kw in sent.lower() for kw in char_keywords):
            sent = sent.strip()
            if len(sent) > 20:
                relevant_sentences.append(sent)
    
    if relevant_sentences:
        result = ". ".join(relevant_sentences[:5])
        if len(result) > 600:
            result = result[:600] + "..."
        return result
    
    # 备选：从案例描述提取
    if case_desc:
        desc_text = " ".join(case_desc)
        if len(desc_text) > 400:
            return desc_text[:400] + "..."
        return desc_text
    
    return "The case exhibits typical characteristics of psychological distress."

def extract_treatment_effect(case_desc, consult_process):
    """提取治疗效果"""
    
    full_text = " ".join(consult_process)  # 主要从咨询过程提取
    
    # 效果关键词
    effect_keywords = [
        "effect", "result", "outcome", "improvement", "progress",
        "better", "improved", "relief", "recovered", "change",
        "successful", "effective", "helpful", "beneficial",
        "after", "following", "post", "finally",
        "效果", "改善", "好转", "恢复", "进步"
    ]
    
    sentences = re.split(r'[.!?。！？]', full_text)
    relevant_sentences = []
    
    # 优先查找文本后半部分（通常包含效果）
    mid_point = len(sentences) // 2
    for sent in sentences[mid_point:]:
        if any(kw in sent.lower() for kw in effect_keywords):
            sent = sent.strip()
            if len(sent) > 20:
                relevant_sentences.append(sent)
    
    # 如果后半部分没找到，再查找前半部分
    if not relevant_sentences:
        for sent in sentences[:mid_point]:
            if any(kw in sent.lower() for kw in effect_keywords):
                sent = sent.strip()
                if len(sent) > 20:
                    relevant_sentences.append(sent)
    
    if relevant_sentences:
        result = ". ".join(relevant_sentences[:5])
        if len(result) > 600:
            result = result[:600] + "..."
        return result
    
    # 备选：取咨询过程最后部分
    if consult_process:
        last_text = consult_process[-1]
        if len(last_text) > 400:
            return last_text[:400] + "..."
        return last_text
    
    return "The treatment showed positive effects on the client's condition."

def main():
    print("="*80)
    print("🔧 基于规则的智能提取 - 直接从案例提取5个字段")
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
    
    print(f"\n💡 提取策略:")
    print(f"  1. 病因: 从案例中查找包含原因、背景、诱因的句子")
    print(f"  2. 症状: 查找包含症状、表现、感受的句子")
    print(f"  3. 治疗过程: 从咨询过程中提取治疗方法和技术")
    print(f"  4. 疾病特征: 查找包含特征、性质、模式的句子")
    print(f"  5. 治疗效果: 从咨询过程后半部分提取改善和效果")
    print(f"\n⚡ 速度: ~1-2秒/样本 (无需模型推理)")
    
    # 进行提取
    print(f"\n🔮 开始提取...\n")
    results = []
    
    for item in tqdm(test_data, desc="提取进度"):
        # 获取ID和输入
        item_id = item.get("id", 0)
        case_desc = item.get("case_description", [])
        consult_process = item.get("consultation_process", [])
        
        # 提取5个字段
        result = {
            "id": item_id,
            "predicted_cause": extract_cause(case_desc, consult_process),
            "predicted_symptoms": extract_symptoms(case_desc, consult_process),
            "predicted_treatment_process": extract_treatment_process(case_desc, consult_process),
            "predicted_illness_Characteristics": extract_illness_characteristics(case_desc, consult_process),
            "predicted_treatment_effect": extract_treatment_effect(case_desc, consult_process)
        }
        
        results.append(result)
    
    # 备份旧文件并保存新结果
    output_file = Path("results/Emotion_Summary_Result.jsonl")
    
    if output_file.exists():
        backup_file = Path("results/Emotion_Summary_Result_v1_model.jsonl")
        import shutil
        shutil.copy(output_file, backup_file)
        print(f"\n📦 模型版本已备份到: {backup_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存结果到: {output_file} (覆盖)")
    
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
        print(f"{content[:150]}...")
    
    # 质量检查
    print(f"\n" + "="*80)
    print(f"🔍 质量检查")
    print("="*80)
    
    # 1. 检查重复
    contents = [sample[field] for field, _ in fields]
    unique = len(set(contents))
    print(f"\n1. 重复度检查:")
    print(f"   唯一内容数: {unique}/5")
    if unique == 5:
        print(f"   ✅ 所有字段内容不同")
    else:
        print(f"   ⚠️  有 {5-unique} 个重复")
    
    # 2. 检查长度
    lengths = [len(c) for c in contents]
    avg_len = sum(lengths) / len(lengths)
    print(f"\n2. 内容长度:")
    print(f"   平均: {avg_len:.0f} 字符")
    print(f"   范围: {min(lengths)}-{max(lengths)} 字符")
    
    # 3. 检查关键词（以第一个样本为例）
    keywords = ["adopt", "Shanxi", "hypochondria", "nasopharynx", "throat", 
                "hemorrhoid", "hypnosis", "imagery", "marriage", "wife"]
    
    full_text = " ".join(contents).lower()
    found = [kw for kw in keywords if kw.lower() in full_text]
    
    print(f"\n3. 案例细节提取 (样本1关键词):")
    print(f"   提取关键词: {len(found)}/{len(keywords)}")
    if found:
        print(f"   找到: {', '.join(found)}")
        if len(found) >= 5:
            print(f"   ✅ 成功提取案例细节")
        else:
            print(f"   ⚠️  提取了部分细节")
    
    print("\n" + "="*80)
    print("✅ 提取完成！")
    print(f"📁 提交文件: {output_file}")
    print(f"📋 方法: 基于规则的智能提取")
    print("="*80)

if __name__ == "__main__":
    main()

