# -*- coding: utf-8 -*-
"""问题分析"""

import json

print("="*80)
print("❌ 生成结果问题分析")
print("="*80)

# 读取生成的结果
with open("results/Emotion_Summary_Result.jsonl", 'r', encoding='utf-8') as f:
    results = [json.loads(line) for line in f if line.strip()]

# 分析第一个样本
sample = results[0]

print(f"\n【样本 1】")
print(f"ID: {sample['id']}")
print("-"*80)

fields = ["predicted_cause", "predicted_symptoms", "predicted_treatment_process", 
          "predicted_illness_Characteristics", "predicted_treatment_effect"]

print("\n🔍 检查5个字段内容:")
for i, field in enumerate(fields, 1):
    content = sample[field]
    print(f"\n{i}. {field}:")
    print(f"   长度: {len(content)} 字符")
    print(f"   内容: {content[:150]}...")

# 检查重复度
print("\n"+"="*80)
print("⚠️  重复度分析")
print("="*80)

contents = [sample[field] for field in fields]
unique_contents = set(contents)

print(f"\n总字段数: {len(contents)}")
print(f"唯一内容数: {len(unique_contents)}")

if len(unique_contents) < len(contents):
    print(f"\n❌ 发现重复！有 {len(contents) - len(unique_contents)} 个字段内容完全相同")
    
    # 检查哪些字段相同
    from collections import Counter
    counter = Counter(contents)
    for content, count in counter.items():
        if count > 1:
            print(f"\n重复内容 (出现{count}次):")
            print(f"  {content[:100]}...")
            print(f"  相关字段:", [fields[i] for i, c in enumerate(contents) if c == content])
else:
    print(f"\n✅ 所有字段内容不同")

# 检查是否包含案例细节
print("\n"+"="*80)
print("🔍 案例细节提取检查")
print("="*80)

# 关键词列表（从测试集第一个样本）
keywords = [
    "adopt", "adoption", "Shanxi", "hypochondria", 
    "nasopharynx", "throat", "hemorrhoid", "hypnosis",
    "guided imagery", "marriage", "wife", "biological parent"
]

print(f"\n检查是否包含案例关键词:")
found_keywords = []
for keyword in keywords:
    found = any(keyword.lower() in sample[field].lower() for field in fields)
    status = "✅" if found else "❌"
    print(f"  {status} {keyword}")
    if found:
        found_keywords.append(keyword)

print(f"\n提取关键词数: {len(found_keywords)}/{len(keywords)}")

if len(found_keywords) < 3:
    print(f"\n❌ 严重问题：几乎没有提取到案例具体信息！")
else:
    print(f"\n✅ 提取了部分案例信息")

# 分析问题原因
print("\n"+"="*80)
print("📋 问题原因分析")
print("="*80)

print("\n1. 模型训练不匹配:")
print("   - 训练数据: Empathetic Dialogues (情感对话)")
print("   - 目标任务: 信息提取 (从案例中提取结构化信息)")
print("   - 结论: 模型没有学会提取信息，只会生成模板化文本")

print("\n2. Prompt设计问题:")
print("   - 当前prompt太简单，只是要求 'summarize the CAUSE'")
print("   - 模型不理解需要从输入中提取具体细节")
print("   - 需要更明确的指令")

print("\n3. 序列长度限制:")
print("   - 输入: 512 tokens (可能不够容纳完整案例)")
print("   - 输出: 256 tokens (限制了详细程度)")
print("   - 实际输出: ~280 字符 (约70 tokens)")

print("\n"+"="*80)
print("💡 解决方案")
print("="*80)

print("\n方案1: 改进Prompt (推荐)")
print("  - 使用更明确的提取式prompt")
print("  - 在prompt中给出示例")
print("  - 强调 'extract specific details from the case'")

print("\n方案2: 增加序列长度")
print("  - 输入: 512 → 1024 tokens")
print("  - 输出: 256 → 512 tokens")
print("  - 缺点: 速度慢5-10倍")

print("\n方案3: 使用更好的生成参数")
print("  - 增加 num_beams: 4 → 8")
print("  - 调整 temperature 和 top_p")
print("  - 使用 do_sample=True")

print("\n方案4: 后处理优化")
print("  - 检测重复内容")
print("  - 使用规则提取关键信息")
print("  - 结合关键词匹配")

print("\n"+"="*80)
print("推荐: 先尝试方案1（改进Prompt） + 方案2（增加序列长度）")
print("="*80)

