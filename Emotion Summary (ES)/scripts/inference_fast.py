# -*- coding: utf-8 -*-
"""快速推理脚本 - 对测试集进行推理"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def main():
    print("="*70)
    print("推理测试集")
    print("="*70)
    
    # 加载模型
    model_path = "model/mt5_fast/final"
    print(f"\n加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    print(f"模型加载完成")
    
    # 加载测试数据
    test_file = Path("data/test/Emotion_Summary.jsonl")
    print(f"\n加载测试数据: {test_file}")
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"测试样本数: {len(test_data)}")
    
    # 进行推理
    print(f"\n开始推理...")
    results = []
    
    for item in tqdm(test_data, desc="推理进度"):
        # 构造输入
        case_desc = " ".join(item.get("case_description", []))[:200]
        consult = " ".join(item.get("consultation_process", []))[:300]
        input_text = case_desc + " " + consult
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=128,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        # 生成
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 保存结果
        result = {
            "id": item.get("id", ""),
            "case_description": item.get("case_description", []),
            "consultation_process": item.get("consultation_process", []),
            "experience_and_reflection": generated_text
        }
        results.append(result)
    
    # 保存结果
    output_file = Path("results/test_predictions.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存结果到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"完成！共生成 {len(results)} 条结果")
    
    # 显示示例
    print(f"\n示例输出 (前3条):")
    print("-"*70)
    for i, result in enumerate(results[:3], 1):
        print(f"\n样本 {i} (ID: {result['id']}):")
        print(f"  输入长度: {len(' '.join(result['case_description'] + result['consultation_process']))} 字符")
        print(f"  生成长度: {len(result['experience_and_reflection'])} 字符")
        print(f"  生成内容: {result['experience_and_reflection'][:200]}...")
    
    print("\n" + "="*70)
    print("推理完成！")
    print(f"结果保存在: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()

