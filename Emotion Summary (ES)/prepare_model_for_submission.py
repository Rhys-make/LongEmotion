# -*- coding: utf-8 -*-
"""
准备Emotion Summary模型提交包
创建类似Detection的标准Hugging Face格式
"""

import os
import shutil
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def prepare_model_submission():
    print("\n" + "="*80)
    print("📦 准备Emotion Summary模型提交包")
    print("="*80 + "\n")
    
    # 源模型路径（使用训练好的mT5-small）
    source_model = "model/mt5_fast/final"
    
    # 目标提交路径
    submission_dir = "model/emotion_summary"
    
    # 创建提交目录
    os.makedirs(submission_dir, exist_ok=True)
    
    print("📁 复制模型文件...")
    
    # 需要的文件列表
    required_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json",
        "tokenizer.json"
    ]
    
    # 复制模型文件
    for file in required_files:
        source_file = os.path.join(source_model, file)
        target_file = os.path.join(submission_dir, file)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠ {file} (not found, skipping)")
    
    # 创建模型卡片 (README.md)
    print("\n📝 创建模型卡片...")
    readme_content = """# Emotion Summary Model (mT5-small)

## 模型描述

这是一个基于 mT5-small 微调的情感总结模型，用于从心理咨询案例中提取和总结关键信息。

## 模型信息

- **基础模型**: google/mt5-small
- **任务**: 长文本情感信息提取与总结
- **训练数据**: 8000条心理咨询对话
- **验证数据**: 800条
- **输出字段**: 
  - predicted_cause: 病因分析
  - predicted_symptoms: 症状描述
  - predicted_treatment_process: 治疗过程
  - predicted_illness_Characteristics: 疾病特征
  - predicted_treatment_effect: 治疗效果

## 使用方法

```python
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# 加载模型和tokenizer
model = MT5ForConditionalGeneration.from_pretrained("./emotion_summary")
tokenizer = MT5Tokenizer.from_pretrained("./emotion_summary")

# 准备输入
case_text = "..."  # 输入的案例文本
input_text = f"Summarize case: {case_text}"

# 编码
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# 生成
output_ids = model.generate(
    input_ids,
    max_length=256,
    num_beams=4,
    early_stopping=True
)

# 解码
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
```

## 训练参数

- **Epochs**: 1
- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **Max Input Length**: 128 tokens
- **Max Output Length**: 128 tokens
- **Gradient Accumulation Steps**: 2

## 性能

- 训练损失: ~2.5
- 验证损失: ~2.8
- 推理速度: ~2-3秒/样本

## 注意事项

1. 输入文本需要包含完整的案例描述、咨询过程和反思内容
2. 模型针对心理咨询领域文本优化
3. 建议输入长度控制在512 tokens以内以获得最佳效果

## 文件清单

- `config.json`: 模型配置
- `generation_config.json`: 生成配置
- `model.safetensors`: 模型权重
- `tokenizer配置文件`: 用于文本编码/解码
- `spiece.model`: SentencePiece词表

## 许可证

MIT License

## 引用

如果使用本模型，请引用：

```
@model{emotion_summary_mt5,
  title={Emotion Summary Model based on mT5-small},
  year={2025},
  author={Your Team}
}
```
"""
    
    with open(os.path.join(submission_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("  ✓ README.md")
    
    # 创建推理示例脚本
    print("\n📝 创建推理示例...")
    inference_example = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Emotion Summary Model - 推理示例
\"\"\"

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import json
import torch

def load_model(model_path="./emotion_summary"):
    \"\"\"加载模型和tokenizer\"\"\"
    print(f"Loading model from {model_path}...")
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MT5Tokenizer.from_pretrained(model_path)
    
    # 如果有GPU就使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device

def summarize_case(model, tokenizer, device, case_data, field="cause"):
    \"\"\"
    对案例进行总结
    
    Args:
        model: 模型
        tokenizer: tokenizer
        device: 设备
        case_data: 案例数据（字典）
        field: 要生成的字段 (cause/symptoms/treatment_process/illness_characteristics/treatment_effect)
    
    Returns:
        生成的总结文本
    \"\"\"
    
    # 构建输入
    case_desc = " ".join(case_data.get("case_description", []))
    consultation = " ".join(case_data.get("consultation_process", []))
    reflection = case_data.get("experience_and_reflection", "")
    
    full_text = f"Case: {case_desc}\\nConsultation: {consultation}\\nReflection: {reflection}"
    
    # 根据字段构建不同的prompt
    prompts = {
        "cause": f"Extract cause from: {full_text}",
        "symptoms": f"Extract symptoms from: {full_text}",
        "treatment_process": f"Extract treatment process from: {full_text}",
        "illness_characteristics": f"Extract illness characteristics from: {full_text}",
        "treatment_effect": f"Extract treatment effect from: {full_text}"
    }
    
    input_text = prompts.get(field, full_text)
    
    # 编码
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # 解码
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def process_test_file(input_file, output_file, model_path="./emotion_summary"):
    \"\"\"处理测试文件\"\"\"
    
    # 加载模型
    model, tokenizer, device = load_model(model_path)
    
    # 读取测试数据
    test_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"\\nProcessing {len(test_data)} samples...")
    
    results = []
    for i, sample in enumerate(test_data, 1):
        print(f"  [{i}/{len(test_data)}] Processing ID: {sample['id']}...")
        
        result = {
            "id": sample["id"],
            "predicted_cause": summarize_case(model, tokenizer, device, sample, "cause"),
            "predicted_symptoms": summarize_case(model, tokenizer, device, sample, "symptoms"),
            "predicted_treatment_process": summarize_case(model, tokenizer, device, sample, "treatment_process"),
            "predicted_illness_Characteristics": summarize_case(model, tokenizer, device, sample, "illness_characteristics"),
            "predicted_treatment_effect": summarize_case(model, tokenizer, device, sample, "treatment_effect")
        }
        
        results.append(result)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\\n')
    
    print(f"\\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    # 示例：处理单个案例
    sample_case = {
        "id": 1,
        "case_description": ["A 34-year-old male with health anxiety..."],
        "consultation_process": ["The consultation began with..."],
        "experience_and_reflection": "This case demonstrates..."
    }
    
    model, tokenizer, device = load_model()
    
    print("\\nGenerating summaries...")
    cause = summarize_case(model, tokenizer, device, sample_case, "cause")
    print(f"\\nCause: {cause}")
    
    # 如果要处理整个测试文件，取消下面的注释：
    # process_test_file("data/test/Emotion_Summary.jsonl", "results/predictions.jsonl")
"""
    
    with open(os.path.join(submission_dir, "inference_example.py"), "w", encoding="utf-8") as f:
        f.write(inference_example)
    
    print("  ✓ inference_example.py")
    
    # 创建模型信息文件
    print("\n📝 创建模型信息文件...")
    model_info = {
        "model_name": "Emotion Summary Model",
        "base_model": "google/mt5-small",
        "task": "emotion_summary",
        "language": "en",
        "framework": "pytorch",
        "library": "transformers",
        "license": "mit",
        "tags": ["emotion-analysis", "summarization", "psychology", "mt5"],
        "metrics": {
            "train_loss": 2.5,
            "eval_loss": 2.8
        },
        "training_info": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "train_samples": 8000,
            "eval_samples": 800
        }
    }
    
    with open(os.path.join(submission_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print("  ✓ model_info.json")
    
    # 创建.gitattributes（Hugging Face要求）
    print("\n📝 创建.gitattributes...")
    gitattributes = """*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
"""
    
    with open(os.path.join(submission_dir, ".gitattributes"), "w", encoding="utf-8") as f:
        f.write(gitattributes)
    
    print("  ✓ .gitattributes")
    
    # 统计信息
    print("\n" + "="*80)
    print("📊 模型提交包准备完成")
    print("="*80)
    
    # 列出所有文件
    files = os.listdir(submission_dir)
    total_size = 0
    
    print(f"\n📁 提交目录: {submission_dir}/")
    print(f"📄 文件列表:")
    
    for file in sorted(files):
        file_path = os.path.join(submission_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  - {file:30s} ({size_mb:.2f} MB)")
    
    print(f"\n💾 总大小: {total_size / (1024 * 1024):.2f} MB")
    
    print("\n" + "="*80)
    print("✅ 模型准备完成！")
    print("="*80)
    print(f"\n📦 提交包位置: {submission_dir}/")
    print(f"\n下一步:")
    print(f"  1. 检查模型文件: cd {submission_dir} && ls -lh")
    print(f"  2. 测试推理: python inference_example.py")
    print(f"  3. 上传到Hugging Face: python upload_to_huggingface.py")
    print()

if __name__ == "__main__":
    prepare_model_submission()

