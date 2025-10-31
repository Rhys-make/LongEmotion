# -*- coding: utf-8 -*-
"""格式对比分析"""

import json

print("="*80)
print("📋 提交格式对比分析")
print("="*80)

# 比赛要求的格式
required_format = {
    "id": 0,
    "predicted_cause": "...........",
    "predicted_symptoms": "...........",
    "predicted_treatment_process": "...........",
    "predicted_illness_Characteristics": "...........",
    "predicted_treatment_effect": "..........."
}

# 当前生成的格式（示例）
current_format_sample = {
    "id": 1,
    "case_description": ["..."],
    "consultation_process": ["..."],
    "experience_and_reflection": "This case presents..."
}

print("\n❌ 当前生成的格式 (错误):")
print("-"*80)
for key in current_format_sample.keys():
    print(f"  - {key}")

print("\n✅ 比赛要求的格式 (正确):")
print("-"*80)
for key in required_format.keys():
    print(f"  - {key}")

print("\n🔍 关键区别:")
print("-"*80)
print("当前格式:")
print("  包含原始输入 (case_description, consultation_process)")
print("  生成单一文本 (experience_and_reflection)")
print()
print("要求格式:")
print("  只有 id")
print("  需要生成 5 个独立的预测字段:")
print("    1. predicted_cause - 病因")
print("    2. predicted_symptoms - 症状") 
print("    3. predicted_treatment_process - 治疗过程")
print("    4. predicted_illness_Characteristics - 疾病特征")
print("    5. predicted_treatment_effect - 治疗效果")

print("\n" + "="*80)
print("💡 解决方案")
print("="*80)
print("\n需要修改模型推理方式:")
print("  1. 输入: case_description + consultation_process")
print("  2. 输出: 5个独立的字段，而不是一个长文本")
print("  3. 使用结构化的prompt引导模型生成5个字段")
print()
print("方案选择:")
print("  方案1: 修改prompt，让模型一次生成5个字段（推荐）")
print("  方案2: 分5次调用模型，每次生成一个字段")
print("  方案3: 生成后用NLP提取5个字段")

print("\n" + "="*80)

