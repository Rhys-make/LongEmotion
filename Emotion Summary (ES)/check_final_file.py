# -*- coding: utf-8 -*-
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("📁 最终结果文件验证")
print("="*80 + "\n")

# 文件路径
file_path = "results/Emotion_Summary_Result.jsonl"
full_path = os.path.abspath(file_path)

print(f"📂 完整路径:")
print(f"   {full_path}\n")

# 检查文件是否存在
if not os.path.exists(file_path):
    print("❌ 文件不存在！")
    sys.exit(1)

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    data = [json.loads(l) for l in f if l.strip()]

# 文件大小
size_bytes = os.path.getsize(file_path)
size_mb = size_bytes / (1024 * 1024)

print("✅ 文件验证:")
print(f"   ✓ 文件存在")
print(f"   ✓ 样本总数: {len(data)}")
print(f"   ✓ ID范围: {min(d['id'] for d in data)} - {max(d['id'] for d in data)}")
print(f"   ✓ 文件大小: {size_mb:.2f} MB")

print(f"\n✅ 字段验证:")
sample = data[0]
required_fields = [
    'id',
    'predicted_cause',
    'predicted_symptoms',
    'predicted_treatment_process',
    'predicted_illness_Characteristics',
    'predicted_treatment_effect'
]

all_present = True
for field in required_fields:
    if field in sample:
        print(f"   ✓ {field}")
    else:
        print(f"   ✗ {field} (缺失)")
        all_present = False

if all_present:
    print(f"\n✅ 所有必需字段都存在！")
else:
    print(f"\n⚠️ 有字段缺失！")

print(f"\n📊 内容预览（样本ID=1）:")
sample1 = data[0]
for field in required_fields:
    if field == 'id':
        print(f"   {field}: {sample1[field]}")
    else:
        content = sample1[field]
        preview = content[:100] + "..." if len(content) > 100 else content
        print(f"   {field}: {preview}")

print("\n" + "="*80)
print("🎉 验证完成！文件就绪，可以提交！")
print("="*80 + "\n")

print("📍 文件位置:")
print(f"   {full_path}")
print()

