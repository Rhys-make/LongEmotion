# -*- coding: utf-8 -*-
"""改进版推理脚本 - 提取具体案例信息"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def create_improved_prompt(case_desc, consult_process, field_name):
    """创建改进的prompt，强调提取具体信息"""
    
    # 将列表转为文本
    case_text = " ".join(case_desc) if isinstance(case_desc, list) else case_desc
    consult_text = " ".join(consult_process) if isinstance(consult_process, list) else consult_process
    
    # 增加长度限制，保留更多细节
    case_text = case_text[:1500]
    consult_text = consult_text[:3500]
    
    # 根据字段创建详细的提取式prompt
    prompts = {
        "cause": f"""Extract and summarize the ROOT CAUSE of the psychological problem from this case.
Focus on: underlying issues, family background, past experiences, triggering events.
Provide SPECIFIC DETAILS from the case.

Case: {case_text}

Consultation: {consult_text}

Extracted Cause (provide specific details):""",
        
        "symptoms": f"""Extract and list the SPECIFIC SYMPTOMS displayed by the client.
Include: physical symptoms, behavioral symptoms, psychological symptoms, emotional patterns.
Provide DETAILED descriptions from the case.

Case: {case_text}

Consultation: {consult_text}

Extracted Symptoms (list specific details):""",
        
        "treatment": f"""Extract and describe the TREATMENT PROCESS used in this case.
Include: therapeutic methods, interventions, techniques, session activities.
Provide SPECIFIC steps and approaches mentioned.

Case: {case_text}

Consultation: {consult_text}

Extracted Treatment Process (describe specific methods):""",
        
        "characteristics": f"""Extract the CHARACTERISTICS of this psychological condition.
Include: nature of symptoms, patterns, psychological mechanisms, diagnostic features.
Provide SPECIFIC observations from the case.

Case: {case_text}

Consultation: {consult_text}

Extracted Illness Characteristics (describe specific features):""",
        
        "effect": f"""Extract and describe the TREATMENT OUTCOMES and effects.
Include: emotional changes, behavioral improvements, insights gained, overall progress.
Provide SPECIFIC results mentioned in the case.

Case: {case_text}

Consultation: {consult_text}

Extracted Treatment Effect (describe specific outcomes):"""
    }
    
    return prompts.get(field_name, "")

def generate_single_field(model, tokenizer, prompt, max_length=400):
    """生成单个字段，使用更好的参数"""
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        max_length=1024,  # 增加输入长度
        truncation=True,
        padding=False,
        return_tensors="pt"
    )
    
    # 生成 - 使用更好的参数
    outputs = model.generate(
        **inputs,
        max_length=max_length,  # 增加输出长度
        min_length=50,  # 最小长度
        num_beams=8,  # 增加beam search
        early_stopping=True,
        no_repeat_ngram_size=3,  # 避免重复
        length_penalty=1.2,  # 鼓励生成更长的输出
        repetition_penalty=1.5,  # 惩罚重复
        do_sample=False,  # 使用贪婪解码
    )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text.strip()

def main():
    print("="*80)
    print("🔧 改进版推理 - 提取具体案例信息")
    print("="*80)
    
    # 加载模型
    model_path = "model/mt5_fast/final"
    print(f"\n📦 加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    print(f"✅ 模型加载完成")
    
    # 加载测试数据
    test_file = Path("data/test/Emotion_Summary.jsonl")
    print(f"\n📊 加载测试数据: {test_file}")
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"✅ 测试样本数: {len(test_data)}")
    
    # 改进说明
    print(f"\n💡 改进要点:")
    print(f"  1. 输入长度: 512 → 1024 tokens (保留更多上下文)")
    print(f"  2. 输出长度: 256 → 400 tokens (允许更详细的输出)")
    print(f"  3. Beam搜索: 4 → 8 (提高生成质量)")
    print(f"  4. Prompt改进: 强调'extract specific details from case'")
    print(f"  5. 重复惩罚: repetition_penalty=1.5 (减少模板化)")
    
    print(f"\n⏱️  预计时间: ~60-80 分钟")
    print(f"    (每个样本5个字段 × 150样本 × 25秒/字段)")
    
    # 进行推理
    print(f"\n🔮 开始推理...\n")
    results = []
    
    for item in tqdm(test_data, desc="样本进度"):
        # 获取ID
        item_id = item.get("id", 0)
        
        # 获取输入
        case_desc = item.get("case_description", [])
        consult_process = item.get("consultation_process", [])
        
        # 生成5个字段
        result = {"id": item_id}
        
        # 1. 病因
        prompt = create_improved_prompt(case_desc, consult_process, "cause")
        result["predicted_cause"] = generate_single_field(model, tokenizer, prompt, max_length=400)
        
        # 2. 症状
        prompt = create_improved_prompt(case_desc, consult_process, "symptoms")
        result["predicted_symptoms"] = generate_single_field(model, tokenizer, prompt, max_length=400)
        
        # 3. 治疗过程
        prompt = create_improved_prompt(case_desc, consult_process, "treatment")
        result["predicted_treatment_process"] = generate_single_field(model, tokenizer, prompt, max_length=400)
        
        # 4. 疾病特征
        prompt = create_improved_prompt(case_desc, consult_process, "characteristics")
        result["predicted_illness_Characteristics"] = generate_single_field(model, tokenizer, prompt, max_length=400)
        
        # 5. 治疗效果
        prompt = create_improved_prompt(case_desc, consult_process, "effect")
        result["predicted_treatment_effect"] = generate_single_field(model, tokenizer, prompt, max_length=400)
        
        results.append(result)
    
    # 保存结果
    output_file = Path("results/Emotion_Summary_Result.jsonl")
    # 备份旧文件
    if output_file.exists():
        backup_file = Path("results/Emotion_Summary_Result_old.jsonl")
        output_file.rename(backup_file)
        print(f"\n📦 旧文件已备份到: {backup_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存结果到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ 完成！共生成 {len(results)} 条结果")
    
    # 显示示例
    print(f"\n📝 示例输出 (样本 1):")
    print("="*80)
    sample = results[0]
    
    fields = ["predicted_cause", "predicted_symptoms", "predicted_treatment_process", 
              "predicted_illness_Characteristics", "predicted_treatment_effect"]
    
    for field in fields:
        content = sample[field]
        print(f"\n【{field}】")
        print(f"长度: {len(content)} 字符")
        print(f"内容: {content[:200]}...")
    
    # 验证改进
    print(f"\n" + "="*80)
    print(f"🔍 质量检查")
    print("="*80)
    
    # 1. 检查重复
    contents = [sample[field] for field in fields]
    unique = len(set(contents))
    print(f"\n1. 重复度检查:")
    print(f"   唯一内容数: {unique}/5")
    if unique == 5:
        print(f"   ✅ 所有字段内容不同")
    else:
        print(f"   ⚠️  仍有 {5-unique} 个重复")
    
    # 2. 检查长度
    lengths = [len(c) for c in contents]
    avg_len = sum(lengths) / len(lengths)
    print(f"\n2. 内容长度:")
    print(f"   平均: {avg_len:.0f} 字符")
    print(f"   范围: {min(lengths)}-{max(lengths)} 字符")
    if avg_len > 280:
        print(f"   ✅ 比之前更详细 (之前平均280字符)")
    
    # 3. 检查关键词
    keywords = ["adopt", "Shanxi", "hypochondria", "nasopharynx", "throat", 
                "hemorrhoid", "hypnosis", "imagery", "marriage", "wife"]
    
    full_text = " ".join(contents).lower()
    found = [kw for kw in keywords if kw.lower() in full_text]
    
    print(f"\n3. 案例细节提取:")
    print(f"   提取关键词: {len(found)}/{len(keywords)}")
    if found:
        print(f"   找到: {', '.join(found[:5])}...")
        if len(found) >= 5:
            print(f"   ✅ 提取了较多案例细节")
        else:
            print(f"   ⚠️  仍需改进")
    else:
        print(f"   ❌ 未提取到案例细节")
    
    print("\n" + "="*80)
    print("✅ 推理完成！")
    print(f"📁 提交文件: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()

