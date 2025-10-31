# -*- coding: utf-8 -*-
"""简单训练脚本 - 使用 Trainer API"""

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
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# 数据预处理
def preprocess_function(examples, tokenizer):
    # 合并输入
    inputs = []
    targets = []
    
    for ex in examples:
        # 输入：案例描述 + 咨询过程
        input_text = " ".join(ex.get("case_description", []) + ex.get("consultation_process", []))
        inputs.append(input_text)
        
        # 输出：经验与反思
        targets.append(ex.get("experience_and_reflection", ""))
    
    # Tokenize（根据训练集实际长度优化）
    # 训练集平均: 输入~400字符(~150 tokens), 输出~1500字符(~400 tokens)
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding=False)
    labels = tokenizer(targets, max_length=256, truncation=True, padding=False)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 主函数
def main():
    print("="*60)
    print("开始训练 mT5 模型")
    print("="*60)
    
    # 模型和参数
    model_name = "google/mt5-base"
    
    print(f"\n加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 加载数据
    print("\n加载数据...")
    train_file = Path("data/train/Emotion_Summary.jsonl")
    val_file = Path("data/validation/Emotion_Summary.jsonl")
    
    train_data = load_data(train_file)
    val_data = load_data(val_file)
    
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    # 转换为 Dataset
    print("\n预处理数据...")
    
    # 处理训练集
    train_processed = []
    for item in train_data:
        result = preprocess_function([item], tokenizer)
        # 展开字典
        train_processed.append({
            'input_ids': result['input_ids'][0],
            'attention_mask': result['attention_mask'][0],
            'labels': result['labels'][0]
        })
    
    # 处理验证集
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
    
    # 训练参数（优化速度）
    training_args = Seq2SeqTrainingArguments(
        output_dir="model/mt5_emotion_summary",
        num_train_epochs=3,  # 减少到3轮（数据减少20%后足够）
        per_device_train_batch_size=2,  # 增加到2（加快训练）
        per_device_eval_batch_size=2,  # 增加到2
        gradient_accumulation_steps=4,  # 减少到4步，等效batch_size=8
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,  # 减少日志频率
        save_steps=1000,  # 减少保存频率以加快速度
        eval_steps=1000,
        eval_strategy="steps",
        save_total_limit=1,  # 只保留1个checkpoint节省空间
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        warmup_steps=200,
        max_grad_norm=1.0,  # 添加梯度裁剪
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
    print("\n开始训练...")
    print("预计时间: 2-3 小时（已优化内存使用）")
    print("-"*60)
    
    trainer.train()
    
    # 保存模型
    print("\n保存模型...")
    trainer.save_model("model/mt5_emotion_summary/final")
    tokenizer.save_pretrained("model/mt5_emotion_summary/final")
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("📁 模型保存在: model/mt5_emotion_summary/final")
    print("="*60)

if __name__ == "__main__":
    main()