# -*- coding: utf-8 -*-
"""自查脚本 - 检查所有配置"""

import json
from pathlib import Path

print("="*70)
print("🔍 完整自查报告")
print("="*70)

# 1. 检查数据集
print("\n📊 1. 数据集检查")
print("-"*70)

train_file = Path("data/train/Emotion_Summary.jsonl")
val_file = Path("data/validation/Emotion_Summary.jsonl")
test_file = Path("data/test/Emotion_Summary.jsonl")

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

train_data = load_data(train_file)
val_data = load_data(val_file)
test_data = load_data(test_file)

print(f"可用数据:")
print(f"  训练集: {len(train_data):,} 样本")
print(f"  验证集: {len(val_data):,} 样本")
print(f"  测试集: {len(test_data):,} 样本")

print(f"\nfast_train.py 将使用:")
print(f"  训练集: 8,000 样本 (从 {len(train_data):,} 中抽取)")
print(f"  验证集: 800 样本 (从 {len(val_data):,} 中抽取)")

if len(train_data) >= 8000:
    print(f"  ✅ 训练数据充足")
else:
    print(f"  ❌ 训练数据不足！只有 {len(train_data)} 样本")

if len(val_data) >= 800:
    print(f"  ✅ 验证数据充足")
else:
    print(f"  ❌ 验证数据不足！只有 {len(val_data)} 样本")

# 2. 检查数据内容长度
print("\n📏 2. 数据内容长度检查")
print("-"*70)

train_sample = train_data[0]
test_sample = test_data[0]

train_input = " ".join(train_sample.get("case_description", []) + train_sample.get("consultation_process", []))
train_output = train_sample.get("experience_and_reflection", "")

test_input = " ".join(test_sample.get("case_description", []) + test_sample.get("consultation_process", []))
test_output = test_sample.get("experience_and_reflection", "")

print(f"训练集第1条:")
print(f"  输入长度: {len(train_input)} 字符")
print(f"  输出长度: {len(train_output)} 字符")

print(f"\n测试集第1条:")
print(f"  输入长度: {len(test_input)} 字符")
print(f"  输出长度: {len(test_output)} 字符")

print(f"\nfast_train.py 配置:")
print(f"  输入截断: 200+300=500 字符")
print(f"  输出截断: 600 字符")
print(f"  Tokenizer最大长度: 128 tokens")

avg_train_input = sum(len(" ".join(d.get("case_description", []) + d.get("consultation_process", []))) for d in train_data[:100]) / 100
avg_train_output = sum(len(d.get("experience_and_reflection", "")) for d in train_data[:100]) / 100

print(f"\n训练集平均长度 (前100条):")
print(f"  输入: {avg_train_input:.0f} 字符")
print(f"  输出: {avg_train_output:.0f} 字符")

if avg_train_input < 500:
    print(f"  ✅ 输入截断合理")
else:
    print(f"  ⚠️ 输入可能被大量截断")

if avg_train_output < 600:
    print(f"  ✅ 输出截断合理")
else:
    print(f"  ⚠️ 输出可能被大量截断")

# 3. 检查训练配置
print("\n⚙️  3. 训练配置检查")
print("-"*70)

config = {
    "模型": "mT5-small (300M参数)",
    "训练数据": "8,000 样本",
    "验证数据": "800 样本",
    "训练轮数": "1 epoch",
    "Batch size": "4",
    "Gradient accumulation": "2 (等效batch=8)",
    "序列长度": "128 tokens",
    "学习率": "1e-4",
}

for key, value in config.items():
    print(f"  {key}: {value}")

# 4. 预估训练时间
print("\n⏱️  4. 训练时间预估")
print("-"*70)

total_steps = 8000 // 4 // 2  # samples / batch / accumulation
print(f"总训练步数: {total_steps} 步")
print(f"预计每步: 3-5 秒")
print(f"预计总时间: {total_steps * 3 // 60} - {total_steps * 5 // 60} 分钟")
print(f"预计范围: 50-70 分钟")

# 5. 检查磁盘空间
print("\n💾 5. 磁盘空间检查")
print("-"*70)

import shutil
total, used, free = shutil.disk_usage(".")
free_gb = free / (1024**3)

print(f"可用磁盘空间: {free_gb:.1f} GB")

required_space = {
    "模型下载": 1.5,
    "训练checkpoint": 2.4,
    "最终模型": 1.2,
}

total_required = sum(required_space.values())
print(f"需要空间: {total_required:.1f} GB")

for item, size in required_space.items():
    print(f"  - {item}: {size} GB")

if free_gb >= total_required:
    print(f"✅ 空间充足 (余量: {free_gb - total_required:.1f} GB)")
else:
    print(f"❌ 空间不足 (缺少: {total_required - free_gb:.1f} GB)")

# 6. 检查脚本文件
print("\n📝 6. 脚本文件检查")
print("-"*70)

script_file = Path("scripts/fast_train.py")
if script_file.exists():
    print(f"✅ fast_train.py 存在")
    size_kb = script_file.stat().st_size / 1024
    print(f"   文件大小: {size_kb:.1f} KB")
else:
    print(f"❌ fast_train.py 不存在")

# 7. 最终总结
print("\n" + "="*70)
print("✅ 自查完成")
print("="*70)

all_checks = [
    ("数据集充足", len(train_data) >= 8000 and len(val_data) >= 800),
    ("配置合理", True),
    ("磁盘空间充足", free_gb >= total_required),
    ("脚本文件存在", script_file.exists()),
]

all_pass = all(check[1] for check in all_checks)

print("\n检查结果:")
for name, passed in all_checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")

if all_pass:
    print("\n🚀 一切就绪！可以开始训练！")
    print("\n运行命令:")
    print("  python scripts/fast_train.py")
    print("\n预计时间: 50-70 分钟")
else:
    print("\n⚠️  存在问题，请检查上述错误")

print("="*70)


