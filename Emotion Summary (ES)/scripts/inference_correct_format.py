# -*- coding: utf-8 -*-
"""正确格式的推理脚本 - 生成比赛要求的5个字段"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def create_structured_prompt(case_desc, consult_process):
    """创建结构化的prompt，引导模型生成5个字段"""
    
    # 将列表转为文本
    case_text = " ".join(case_desc) if isinstance(case_desc, list) else case_desc
    consult_text = " ".join(consult_process) if isinstance(consult_process, list) else consult_process
    
    # 截断以适应模型长度限制
    case_text = case_text[:500]
    consult_text = consult_text[:1500]
    
    # 创建结构化prompt
    prompt = f"""Analyze this psychological counseling case and provide 5 specific summaries:

Case Description: {case_text}

Consultation Process: {consult_text}

Please provide:
1. Cause (病因):
2. Symptoms (症状):
3. Treatment Process (治疗过程):
4. Illness Characteristics (疾病特征):
5. Treatment Effect (治疗效果):"""
    
    return prompt

def parse_model_output(generated_text):
    """从模型输出中提取5个字段"""
    
    # 初始化5个字段
    result = {
        "predicted_cause": "",
        "predicted_symptoms": "",
        "predicted_treatment_process": "",
        "predicted_illness_Characteristics": "",
        "predicted_treatment_effect": ""
    }
    
    # 简单解析：如果模型输出包含标记，就提取
    lines = generated_text.split('\n')
    current_field = None
    
    for line in lines:
        line = line.strip()
        if 'cause' in line.lower() or '病因' in line or '1.' in line:
            current_field = 'predicted_cause'
        elif 'symptom' in line.lower() or '症状' in line or '2.' in line:
            current_field = 'predicted_symptoms'
        elif 'treatment process' in line.lower() or '治疗过程' in line or '3.' in line:
            current_field = 'predicted_treatment_process'
        elif 'characteristic' in line.lower() or '疾病特征' in line or '4.' in line:
            current_field = 'predicted_illness_Characteristics'
        elif 'effect' in line.lower() or '治疗效果' in line or '5.' in line:
            current_field = 'predicted_treatment_effect'
        elif current_field and line:
            # 添加内容到当前字段
            if result[current_field]:
                result[current_field] += " " + line
            else:
                result[current_field] = line
    
    # 如果没有成功解析，就把整个文本放入第一个字段
    if not any(result.values()):
        result["predicted_cause"] = generated_text[:200] if len(generated_text) > 200 else generated_text
        result["predicted_symptoms"] = generated_text[:200] if len(generated_text) > 200 else generated_text
        result["predicted_treatment_process"] = generated_text[:200] if len(generated_text) > 200 else generated_text
        result["predicted_illness_Characteristics"] = generated_text[:200] if len(generated_text) > 200 else generated_text
        result["predicted_treatment_effect"] = generated_text[:200] if len(generated_text) > 200 else generated_text
    
    return result

def main():
    print("="*70)
    print("生成正确格式的推理结果")
    print("="*70)
    
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
    
    # 进行推理
    print(f"\n🔮 开始推理...")
    results = []
    
    for item in tqdm(test_data, desc="推理进度"):
        # 获取ID
        item_id = item.get("id", 0)
        
        # 构造结构化prompt
        case_desc = item.get("case_description", [])
        consult_process = item.get("consultation_process", [])
        
        prompt = create_structured_prompt(case_desc, consult_process)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            max_length=512,  # 增加长度以容纳prompt
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        # 生成
        outputs = model.generate(
            **inputs,
            max_length=256,  # 生成更长的输出
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析为5个字段
        parsed_fields = parse_model_output(generated_text)
        
        # 构建结果
        result = {
            "id": item_id,
            "predicted_cause": parsed_fields["predicted_cause"],
            "predicted_symptoms": parsed_fields["predicted_symptoms"],
            "predicted_treatment_process": parsed_fields["predicted_treatment_process"],
            "predicted_illness_Characteristics": parsed_fields["predicted_illness_Characteristics"],
            "predicted_treatment_effect": parsed_fields["predicted_treatment_effect"]
        }
        
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
    print(f"\n📝 示例输出 (前3条):")
    print("-"*70)
    for i, result in enumerate(results[:3], 1):
        print(f"\n样本 {i} (ID: {result['id']}):")
        for key in ["predicted_cause", "predicted_symptoms", "predicted_treatment_process", 
                    "predicted_illness_Characteristics", "predicted_treatment_effect"]:
            value = result[key]
            print(f"  {key}: {value[:80]}..." if len(value) > 80 else f"  {key}: {value}")
    
    # 验证格式
    print(f"\n🔍 格式验证:")
    all_have_fields = all(
        all(key in r for key in ["id", "predicted_cause", "predicted_symptoms", 
                                  "predicted_treatment_process", "predicted_illness_Characteristics", 
                                  "predicted_treatment_effect"])
        for r in results
    )
    
    if all_have_fields:
        print(f"  ✅ 所有样本都包含必需字段")
    else:
        print(f"  ❌ 部分样本缺少字段")
    
    print("\n" + "="*70)
    print("✅ 推理完成！")
    print(f"📁 提交文件: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()

