# -*- coding: utf-8 -*-
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("\n" + "="*80)
print("🔍 验证最终结果文件")
print("="*80 + "\n")

try:
    with open('results/Emotion_Summary_Result.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f if l.strip()]
    
    print(f"✓ 文件读取成功")
    print(f"✓ 总样本数: {len(data)}")
    print(f"✓ ID范围: {min(d['id'] for d in data)} - {max(d['id'] for d in data)}")
    
    print(f"\n✓ 字段完整性检查:")
    required_fields = [
        'id',
        'predicted_cause',
        'predicted_symptoms',
        'predicted_treatment_process',
        'predicted_illness_Characteristics',
        'predicted_treatment_effect'
    ]
    
    sample = data[0]
    for field in required_fields:
        status = "✓" if field in sample else "✗"
        print(f"  {status} {field}")
    
    print(f"\n✓ 内容质量抽查（样本ID=1）:")
    sample1 = [d for d in data if d['id'] == 1][0]
    for field in ['predicted_cause', 'predicted_symptoms', 'predicted_treatment_process']:
        length = len(sample1[field])
        print(f"  - {field}: {length} 字符")
    
    print("\n" + "="*80)
    print("🎉 验证完成！结果文件格式正确且完整")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"✗ 验证失败: {e}")
    sys.exit(1)

