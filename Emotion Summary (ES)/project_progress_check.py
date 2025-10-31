# -*- coding: utf-8 -*-
"""比赛任务进度全面自查"""

import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("🏆 比赛任务进度全面自查")
print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ========== 1. 数据准备检查 ==========
print("\n" + "="*80)
print("📊 1. 数据准备检查")
print("="*80)

train_file = Path("data/train/Emotion_Summary.jsonl")
val_file = Path("data/validation/Emotion_Summary.jsonl")
test_file = Path("data/test/Emotion_Summary.jsonl")

data_status = {}

for name, file_path in [("训练集", train_file), ("验证集", val_file), ("测试集", test_file)]:
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        # 检查第一条数据的格式
        sample = data[0] if data else {}
        has_id = "id" in sample
        has_case = "case_description" in sample
        has_consult = "consultation_process" in sample
        has_reflection = "experience_and_reflection" in sample
        
        data_status[name] = {
            "exists": True,
            "count": len(data),
            "format_ok": has_id and has_case and has_consult,
            "has_reflection": has_reflection
        }
        
        print(f"\n{name}:")
        print(f"  ✅ 文件存在: {file_path}")
        print(f"  📝 样本数量: {len(data):,}")
        print(f"  🔍 格式检查:")
        print(f"     - id: {'✅' if has_id else '❌'}")
        print(f"     - case_description: {'✅' if has_case else '❌'}")
        print(f"     - consultation_process: {'✅' if has_consult else '❌'}")
        print(f"     - experience_and_reflection: {'✅' if has_reflection else '❌'}")
    else:
        data_status[name] = {"exists": False}
        print(f"\n{name}: ❌ 文件不存在")

# ========== 2. 模型训练检查 ==========
print("\n" + "="*80)
print("🤖 2. 模型训练检查")
print("="*80)

model_dir = Path("model/mt5_fast/final")
checkpoint_dir = Path("model/mt5_fast/checkpoint-1000")

training_status = {}

if model_dir.exists():
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    
    if not missing_files:
        # 读取训练状态
        trainer_state = checkpoint_dir / "trainer_state.json"
        if trainer_state.exists():
            with open(trainer_state, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            log_history = state['log_history']
            train_losses = [log for log in log_history if 'loss' in log]
            
            training_status = {
                "completed": True,
                "steps": state['global_step'],
                "epochs": state['epoch'],
                "initial_loss": train_losses[0]['loss'] if train_losses else None,
                "final_loss": train_losses[-1]['loss'] if train_losses else None,
            }
            
            print(f"\n✅ 模型训练完成")
            print(f"  📁 模型路径: {model_dir}")
            print(f"  📊 训练状态:")
            print(f"     - 总步数: {training_status['steps']}")
            print(f"     - 训练轮数: {training_status['epochs']:.2f}")
            print(f"     - 初始Loss: {training_status['initial_loss']:.4f}")
            print(f"     - 最终Loss: {training_status['final_loss']:.4f}")
            print(f"     - Loss改善: {((training_status['initial_loss'] - training_status['final_loss']) / training_status['initial_loss'] * 100):.1f}%")
            
            # 模型大小
            model_file = model_dir / "model.safetensors"
            size_gb = model_file.stat().st_size / (1024**3)
            print(f"  💾 模型大小: {size_gb:.2f} GB")
        else:
            training_status = {"completed": False, "reason": "缺少训练状态文件"}
            print(f"\n⚠️  模型存在但缺少训练状态文件")
    else:
        training_status = {"completed": False, "reason": f"缺少文件: {missing_files}"}
        print(f"\n⚠️  模型不完整，缺少: {', '.join(missing_files)}")
else:
    training_status = {"completed": False, "reason": "模型目录不存在"}
    print(f"\n❌ 模型未训练")

# ========== 3. 推理结果检查 ==========
print("\n" + "="*80)
print("🔮 3. 推理结果检查")
print("="*80)

results_file = Path("results/test_predictions.jsonl")

inference_status = {}

if results_file.exists():
    with open(results_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f if line.strip()]
    
    # 检查格式
    if results:
        sample = results[0]
        has_all_fields = all(k in sample for k in ["id", "case_description", "consultation_process", "experience_and_reflection"])
        
        # 检查是否所有测试样本都有结果
        test_count = data_status.get("测试集", {}).get("count", 0)
        all_processed = len(results) == test_count
        
        # 统计生成内容长度
        gen_lengths = [len(r.get("experience_and_reflection", "")) for r in results]
        avg_length = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0
        min_length = min(gen_lengths) if gen_lengths else 0
        max_length = max(gen_lengths) if gen_lengths else 0
        
        inference_status = {
            "completed": True,
            "count": len(results),
            "format_ok": has_all_fields,
            "all_processed": all_processed,
            "avg_length": avg_length,
            "min_length": min_length,
            "max_length": max_length
        }
        
        print(f"\n✅ 推理完成")
        print(f"  📁 结果文件: {results_file}")
        print(f"  📊 推理统计:")
        print(f"     - 结果数量: {len(results)}")
        print(f"     - 测试集数量: {test_count}")
        print(f"     - 完整性: {'✅ 全部处理' if all_processed else '❌ 不完整'}")
        print(f"     - 格式正确: {'✅' if has_all_fields else '❌'}")
        print(f"  📏 生成内容长度:")
        print(f"     - 平均: {avg_length:.0f} 字符")
        print(f"     - 最短: {min_length} 字符")
        print(f"     - 最长: {max_length} 字符")
        
        # 显示几个示例
        print(f"\n  📝 示例输出 (前3条):")
        for i, result in enumerate(results[:3], 1):
            gen_text = result.get("experience_and_reflection", "")
            print(f"\n     样本 {i} (ID: {result.get('id', 'N/A')}):")
            print(f"       生成长度: {len(gen_text)} 字符")
            print(f"       内容预览: {gen_text[:100]}...")
    else:
        inference_status = {"completed": False, "reason": "结果文件为空"}
        print(f"\n⚠️  结果文件存在但为空")
else:
    inference_status = {"completed": False, "reason": "结果文件不存在"}
    print(f"\n❌ 推理未完成")

# ========== 4. 提交文件检查 ==========
print("\n" + "="*80)
print("📤 4. 提交文件检查")
print("="*80)

# 检查结果文件是否符合提交格式
submission_ready = False

if results_file.exists() and inference_status.get("completed", False):
    print(f"\n✅ 提交文件已生成: {results_file}")
    print(f"\n  📋 提交文件信息:")
    print(f"     - 文件路径: {results_file}")
    print(f"     - 文件大小: {results_file.stat().st_size / 1024:.1f} KB")
    print(f"     - 样本数量: {len(results)}")
    
    # 检查是否所有样本都有非空的 experience_and_reflection
    empty_count = sum(1 for r in results if not r.get("experience_and_reflection", "").strip())
    if empty_count == 0:
        submission_ready = True
        print(f"     - 完整性: ✅ 所有样本都有生成内容")
    else:
        print(f"     - 完整性: ⚠️  有 {empty_count} 个样本缺少生成内容")
else:
    print(f"\n❌ 提交文件未准备好")

# ========== 5. 整体进度总结 ==========
print("\n" + "="*80)
print("📈 5. 整体进度总结")
print("="*80)

tasks = [
    ("数据准备", all(s.get("exists", False) for s in data_status.values())),
    ("模型训练", training_status.get("completed", False)),
    ("测试集推理", inference_status.get("completed", False)),
    ("提交文件准备", submission_ready),
]

completed_tasks = sum(1 for _, status in tasks if status)
total_tasks = len(tasks)
progress = (completed_tasks / total_tasks) * 100

print(f"\n整体进度: {completed_tasks}/{total_tasks} ({progress:.0f}%)")
print(f"\n任务清单:")
for task_name, completed in tasks:
    status = "✅ 完成" if completed else "❌ 未完成"
    print(f"  {status} - {task_name}")

# ========== 6. 下一步建议 ==========
print("\n" + "="*80)
print("🎯 6. 下一步操作建议")
print("="*80)

if progress == 100:
    print("\n🎉 恭喜！所有任务已完成！")
    print(f"\n📤 可以提交结果了：")
    print(f"   文件路径: {results_file}")
    print(f"   样本数量: {inference_status['count']}")
    print(f"\n💡 提交前建议:")
    print(f"   1. 检查几个样本的生成质量")
    print(f"   2. 确认格式符合比赛要求")
    print(f"   3. 备份结果文件")
else:
    print("\n还需要完成的任务:")
    for task_name, completed in tasks:
        if not completed:
            print(f"  ❌ {task_name}")
    
    if not training_status.get("completed", False):
        print(f"\n下一步: 开始模型训练")
        print(f"  命令: python scripts/fast_train.py")
    elif not inference_status.get("completed", False):
        print(f"\n下一步: 运行推理")
        print(f"  命令: python scripts/inference_fast.py")

# ========== 7. 性能评估 ==========
print("\n" + "="*80)
print("⚡ 7. 性能评估")
print("="*80)

if training_status.get("completed", False) and inference_status.get("completed", False):
    print(f"\n模型性能:")
    print(f"  训练Loss改善: {((training_status['initial_loss'] - training_status['final_loss']) / training_status['initial_loss'] * 100):.1f}%")
    print(f"  最终Loss: {training_status['final_loss']:.4f}")
    
    print(f"\n生成质量指标:")
    print(f"  平均生成长度: {inference_status['avg_length']:.0f} 字符")
    print(f"  长度范围: {inference_status['min_length']} - {inference_status['max_length']} 字符")
    
    # 简单质量评估
    if inference_status['avg_length'] < 50:
        print(f"  ⚠️  生成内容较短，可能质量不佳")
    elif inference_status['avg_length'] > 1000:
        print(f"  ⚠️  生成内容较长，可能包含冗余")
    else:
        print(f"  ✅ 生成长度合理")

print("\n" + "="*80)
print("✅ 自查完成")
print("="*80)

