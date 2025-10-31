# -*- coding: utf-8 -*-
"""极速训练脚本 - 优化到极致"""

import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

# 加载数据
def load_data(file_path, max_samples=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    return data

# 数据预处理
def preprocess_function(examples, tokenizer):
    inputs = []
    targets = []
    
    for ex in examples:
        # 输入：案例描述 + 咨询过程（截断）
        case_desc = " ".join(ex.get("case_description", []))[:200]  # 限制长度
        consult = " ".join(ex.get("consultation_process", []))[:300]  # 限制长度
        input_text = case_desc + " " + consult
        inputs.append(input_text)
        
        # 输出：经验与反思（截断）
        reflection = ex.get("experience_and_reflection", "")[:600]  # 限制长度
        targets.append(reflection)
    
    # 极短的序列长度以加快速度
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)
    labels = tokenizer(targets, max_length=128, truncation=True, padding=False)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 主函数
def main():
    print("="*60)
    print("⚡ 极速训练模式 - mT5-small")
    print("="*60)
    
    # 使用更小更快的模型
    model_name = "google/mt5-small"  # 300M 参数，比 base (580M) 快一倍
    
    print(f"\n🚀 加载模型: {model_name}")
    print("   (更小更快的模型)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 加载指定数量的数据
    print("\n📊 加载数据...")
    train_file = Path("data/train/Emotion_Summary.jsonl")
    val_file = Path("data/validation/Emotion_Summary.jsonl")
    
    # 训练集8000条，验证集800条
    train_data = load_data(train_file, max_samples=8000)
    val_data = load_data(val_file, max_samples=800)
    
    print(f"✅ 训练集: {len(train_data)} 样本（限制）")
    print(f"✅ 验证集: {len(val_data)} 样本（限制）")
    
    # 预处理
    print("\n🔄 预处理数据...")
    
    train_processed = []
    for item in train_data:
        result = preprocess_function([item], tokenizer)
        train_processed.append({
            'input_ids': result['input_ids'][0],
            'attention_mask': result['attention_mask'][0],
            'labels': result['labels'][0]
        })
    
    val_processed = []
    for item in val_data:
        result = preprocess_function([item], tokenizer)
        val_processed.append({
            'input_ids': result['input_ids'][0],
            'attention_mask': result['attention_mask'][0],
            'labels': result['labels'][0]
        })
    
    train_dataset = Dataset.from_list(train_processed)
    val_dataset = Dataset.from_list(val_processed)
    
    # 极速训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir="model/mt5_fast",
        num_train_epochs=1,  # 只训练1轮
        per_device_train_batch_size=4,  # 增加batch size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # 等效batch_size=8
        learning_rate=1e-4,  # 更高的学习率
        weight_decay=0.01,
        logging_steps=100,
        save_steps=2000,  # 根据数据量调整
        eval_steps=2000,
        eval_strategy="steps",
        save_total_limit=1,
        predict_with_generate=False,  # 关闭生成评估以加快速度
        fp16=False,
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        warmup_steps=50,
        max_grad_norm=1.0,
    )
    
    # 创建 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    
    # 开始训练
    print("\n⚡ 开始极速训练...")
    print("优化配置:")
    print("  - 模型: mT5-small (300M, 比base快一倍)")
    print("  - 训练数据: 8000 样本")
    print("  - 验证数据: 800 样本")
    print("  - 轮数: 1 epoch")
    print("  - Batch: 4")
    print("  - 序列长度: 128 tokens")
    print("\n预计时间: 45-60 分钟")
    print("-"*60)
    
    trainer.train()
    
    # 保存模型
    print("\n💾 保存模型...")
    trainer.save_model("model/mt5_fast/final")
    tokenizer.save_pretrained("model/mt5_fast/final")
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print(f"📁 模型保存在: model/mt5_fast/final")
    print("="*60)

if __name__ == "__main__":
    main()

