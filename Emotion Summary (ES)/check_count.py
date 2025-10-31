# -*- coding: utf-8 -*-
"""检查文件行数"""

import json

# 读取测试集
with open('data/test/Emotion_Summary.jsonl', 'r', encoding='utf-8') as f:
    test_lines = f.readlines()
    test_data = [json.loads(l) for l in test_lines if l.strip()]

# 读取结果文件
with open('results/Emotion_Summary_Result.jsonl', 'r', encoding='utf-8') as f:
    result_lines = f.readlines()
    result_data = [json.loads(l) for l in result_lines if l.strip()]

print("="*60)
print("文件行数统计")
print("="*60)

print(f"\n测试集 (data/test/Emotion_Summary.jsonl):")
print(f"  总行数: {len(test_lines)}")
print(f"  有效样本数: {len(test_data)}")

print(f"\n结果文件 (results/Emotion_Summary_Result.jsonl):")
print(f"  总行数: {len(result_lines)}")
print(f"  有效样本数: {len(result_data)}")

print(f"\n测试集ID范围:")
test_ids = [t['id'] for t in test_data]
print(f"  最小ID: {min(test_ids)}")
print(f"  最大ID: {max(test_ids)}")
print(f"  前5个ID: {test_ids[:5]}")
print(f"  后5个ID: {test_ids[-5:]}")

print(f"\n结果文件ID范围:")
result_ids = [r['id'] for r in result_data]
print(f"  最小ID: {min(result_ids)}")
print(f"  最大ID: {max(result_ids)}")
print(f"  前5个ID: {result_ids[:5]}")
print(f"  后5个ID: {result_ids[-5:]}")

print(f"\n匹配检查:")
if len(test_data) == len(result_data):
    print(f"  ✅ 数量匹配: 都是 {len(test_data)} 个样本")
else:
    print(f"  ❌ 数量不匹配:")
    print(f"     测试集: {len(test_data)}")
    print(f"     结果: {len(result_data)}")

# 检查ID是否完全对应
missing_ids = set(test_ids) - set(result_ids)
extra_ids = set(result_ids) - set(test_ids)

if missing_ids:
    print(f"  ⚠️  缺少的ID: {sorted(missing_ids)}")
if extra_ids:
    print(f"  ⚠️  多余的ID: {sorted(extra_ids)}")
if not missing_ids and not extra_ids:
    print(f"  ✅ 所有ID完全匹配")

print("\n" + "="*60)

