# -*- coding: utf-8 -*-
"""
恢复AI深度分析结果
从batch文件重新合并到最终结果文件
"""

import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("🔄 恢复AI深度分析结果")
print("="*80 + "\n")

# 读取所有batch文件
batch_files = [
    "ai_results_batch1.json",
    "ai_results_batch2.json",
    "ai_results_batch3.json",
    "ai_results_batch4.json",
    "ai_results_batch5.json"
]

all_results = {}

for batch_file in batch_files:
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
            print(f"✓ 读取 {batch_file}: {len(batch_data)} 样本")
            
            # batch_data是数组，需要转换为字典
            for item in batch_data:
                sample_id = str(item['id'])
                all_results[sample_id] = item
    except FileNotFoundError:
        print(f"⚠️ {batch_file} 不存在，跳过")
    except Exception as e:
        print(f"❌ 读取 {batch_file} 失败: {e}")

print(f"\n📊 合并结果: 总共 {len(all_results)} 个样本\n")

# 转换为JSONL格式
output_file = "results/Emotion_Summary_Result.jsonl"

# 备份现有文件
import os
import shutil
if os.path.exists(output_file):
    backup_file = "results/Emotion_Summary_Result_model_backup.jsonl"
    shutil.copy(output_file, backup_file)
    print(f"✓ 备份现有文件到: {backup_file}\n")

# 写入新文件
with open(output_file, 'w', encoding='utf-8') as f:
    # 按ID排序
    for sample_id in sorted(all_results.keys(), key=lambda x: int(x)):
        result = all_results[sample_id]
        
        # 构建输出格式
        output = {
            "id": int(sample_id),
            "predicted_cause": result.get("predicted_cause", ""),
            "predicted_symptoms": result.get("predicted_symptoms", ""),
            "predicted_treatment_process": result.get("predicted_treatment_process", ""),
            "predicted_illness_Characteristics": result.get("predicted_illness_Characteristics", ""),
            "predicted_treatment_effect": result.get("predicted_treatment_effect", "")
        }
        
        json.dump(output, f, ensure_ascii=False)
        f.write('\n')

print(f"✅ AI深度分析结果已恢复到: {output_file}")

# 验证
with open(output_file, 'r', encoding='utf-8') as f:
    data = [json.loads(l) for l in f if l.strip()]

print(f"\n✅ 验证结果:")
print(f"   - 样本总数: {len(data)}")
print(f"   - ID范围: {min(d['id'] for d in data)} - {max(d['id'] for d in data)}")

# 显示第一个样本
sample1 = data[0]
print(f"\n📊 样本预览（ID=1）:")
for field in sample1.keys():
    if field == 'id':
        print(f"   {field}: {sample1[field]}")
    else:
        content = sample1[field]
        preview = content[:150] + "..." if len(content) > 150 else content
        print(f"   {field}:")
        print(f"      {preview}\n")

print("="*80)
print("🎉 恢复完成！")
print("="*80 + "\n")

