# -*- coding: utf-8 -*-
"""比赛格式推理脚本 - 分别生成5个字段"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def create_field_prompt(case_desc, consult_process, field_name):
    """为每个字段创建专门的prompt"""
    
    # 将列表转为文本
    case_text = " ".join(case_desc) if isinstance(case_desc, list) else case_desc
    consult_text = " ".join(consult_process) if isinstance(consult_process, list) else consult_process
    
    # 截断
    case_text = case_text[:400]
    consult_text = consult_text[:800]
    
    # 根据字段类型创建不同的prompt
    prompts = {
        "cause": f"Based on this psychological case, summarize the CAUSE of the problem:\n\nCase: {case_text}\n\nConsultation: {consult_text}\n\nCause:",
        
        "symptoms": f"Based on this psychological case, summarize the SYMPTOMS:\n\nCase: {case_text}\n\nConsultation: {consult_text}\n\nSymptoms:",
        
        "treatment": f"Based on this psychological case, summarize the TREATMENT PROCESS:\n\nCase: {case_text}\n\nConsultation: {consult_text}\n\nTreatment:",
        
        "characteristics": f"Based on this psychological case, summarize the ILLNESS CHARACTERISTICS:\n\nCase: {case_text}\n\nConsultation: {consult_text}\n\nCharacteristics:",
        
        "effect": f"Based on this psychological case, summarize the TREATMENT EFFECT:\n\nCase: {case_text}\n\nConsultation: {consult_text}\n\nEffect:"
    }
    
    return prompts.get(field_name, "")

def generate_single_field(model, tokenizer, prompt, max_length=128):
    """生成单个字段"""
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding=False,
        return_tensors="pt"
    )
    
    # 生成
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.7,
    )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text.strip()

def main():
    print("="*80)
    print("🏆 生成比赛要求格式的推理结果")
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
    print(f"\n⚠️  注意: 每个样本需要生成5个字段，总共 {len(test_data) * 5} 次模型调用")
    print(f"   预计时间: {len(test_data) * 5 * 3 // 60} 分钟\n")
    
    # 进行推理
    print(f"🔮 开始推理...\n")
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
        prompt = create_field_prompt(case_desc, consult_process, "cause")
        result["predicted_cause"] = generate_single_field(model, tokenizer, prompt)
        
        # 2. 症状
        prompt = create_field_prompt(case_desc, consult_process, "symptoms")
        result["predicted_symptoms"] = generate_single_field(model, tokenizer, prompt)
        
        # 3. 治疗过程
        prompt = create_field_prompt(case_desc, consult_process, "treatment")
        result["predicted_treatment_process"] = generate_single_field(model, tokenizer, prompt)
        
        # 4. 疾病特征
        prompt = create_field_prompt(case_desc, consult_process, "characteristics")
        result["predicted_illness_Characteristics"] = generate_single_field(model, tokenizer, prompt)
        
        # 5. 治疗效果
        prompt = create_field_prompt(case_desc, consult_process, "effect")
        result["predicted_treatment_effect"] = generate_single_field(model, tokenizer, prompt)
        
        results.append(result)
    
    # 保存结果
    output_file = Path("results/Emotion_Summary_Result.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存结果到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ 完成！共生成 {len(results)} 条结果")
    
    # 显示示例
    print(f"\n📝 示例输出 (前2条):")
    print("-"*80)
    for i, result in enumerate(results[:2], 1):
        print(f"\n【样本 {i}】 ID: {result['id']}")
        print(f"  病因: {result['predicted_cause'][:100]}...")
        print(f"  症状: {result['predicted_symptoms'][:100]}...")
        print(f"  治疗过程: {result['predicted_treatment_process'][:100]}...")
        print(f"  疾病特征: {result['predicted_illness_Characteristics'][:100]}...")
        print(f"  治疗效果: {result['predicted_treatment_effect'][:100]}...")
    
    # 验证格式
    print(f"\n🔍 格式验证:")
    required_fields = ["id", "predicted_cause", "predicted_symptoms", 
                      "predicted_treatment_process", "predicted_illness_Characteristics", 
                      "predicted_treatment_effect"]
    
    all_valid = True
    for i, r in enumerate(results):
        missing = [f for f in required_fields if f not in r]
        empty = [f for f in required_fields if f in r and not r[f]]
        
        if missing:
            print(f"  ❌ 样本 {r['id']}: 缺少字段 {missing}")
            all_valid = False
        elif empty:
            print(f"  ⚠️  样本 {r['id']}: 空字段 {empty}")
    
    if all_valid:
        print(f"  ✅ 所有样本格式正确")
    
    # 统计生成长度
    print(f"\n📊 生成内容统计:")
    for field in required_fields[1:]:  # 跳过id
        lengths = [len(r[field]) for r in results]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        print(f"  {field}: 平均 {avg_len:.0f} 字符 (范围: {min(lengths)}-{max(lengths)})")
    
    print("\n" + "="*80)
    print("✅ 推理完成！")
    print(f"📁 提交文件: {output_file}")
    print(f"📋 格式: 符合比赛要求 ✓")
    print("="*80)

if __name__ == "__main__":
    main()

