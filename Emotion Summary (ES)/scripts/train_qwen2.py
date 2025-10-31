"""
Qwen2-7B-Instruct 训练脚本 - Emotion Summary (ES) 任务
使用 LoRA 进行参数高效微调
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from model_qwen2 import Qwen2EmotionSummaryModel


class Qwen2EmotionSummaryDataset(Dataset):
    """
    Qwen2 情绪总结数据�?
    """
    
    def __init__(
        self,
        data_file: Path,
        tokenizer,
        max_length: int = 8192,
    ):
        """
        初始化数据集
        
        Args:
            data_file: 数据文件路径（JSONL格式�?
            tokenizer: Qwen2 tokenizer
            max_length: 最大序列长�?
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        print(f"加载数据: {data_file}")
        self.samples = self.load_data(data_file)
        print(f"  加载�?{len(self.samples)} 个样�?)
    
    def load_data(self, data_file: Path) -> List[Dict]:
        """加载JSONL数据"""
        samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line.strip())
                    samples.append(sample)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 提取数据
        case_description = sample.get("case_description", [])
        consultation_process = sample.get("consultation_process", [])
        experience_and_reflection = sample.get("experience_and_reflection", "")
        
        # 构建完整的训练文本（使用对话模板�?
        case_text = "\n".join(case_description) if isinstance(case_description, list) else case_description
        process_text = "\n".join(consultation_process) if isinstance(consultation_process, list) else consultation_process
        
        input_text = f"""【案例描述�?
{case_text}

【咨询过程�?
{process_text}"""
        
        # 使用 chat template
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的心理咨询师。请根据提供的心理咨询案例，生成深入、专业的经验与反思总结�?
            },
            {
                "role": "user",
                "content": f"请对以下心理咨询案例进行深入分析和反思：\n\n{input_text}"
            },
            {
                "role": "assistant",
                "content": experience_and_reflection
            }
        ]
        
        # 应用 chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 准备训练数据
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # labels �?input_ids 相同（因果语言模型�?
        labels = input_ids.clone()
        
        # �?padding token �?label 设为 -100（不计算损失�?
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_qwen2_model(
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    train_file: Path = None,
    val_file: Path = None,
    output_dir: Path = None,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 8192,
    use_4bit: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """
    训练 Qwen2 模型
    
    Args:
        model_name: 模型名称
        train_file: 训练数据文件
        val_file: 验证数据文件
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习�?
        max_length: 最大序列长�?
        use_4bit: 是否使用4-bit量化
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """
    
    print("\n" + "=" * 70)
    print("Qwen2-7B-Instruct 微调训练 - Emotion Summary 任务")
    print("=" * 70 + "\n")
    
    # 默认路径
    if train_file is None:
        train_file = Path(__file__).parent.parent / "data" / "train" / "Emotion_Summary.jsonl"
    if val_file is None:
        val_file = Path(__file__).parent.parent / "data" / "validation" / "Emotion_Summary.jsonl"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "model" / "qwen2_emotion_summary"
    
    print(f"训练数据: {train_file}")
    print(f"验证数据: {val_file}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 检查文件存�?
    if not train_file.exists():
        print(f"错误: 训练文件不存�? {train_file}")
        return
    
    # 初始化模�?
    print("步骤 1/4: 初始化模�?)
    print("-" * 70)
    model_wrapper = Qwen2EmotionSummaryModel(
        model_name=model_name,
        max_input_length=max_length,
        use_4bit=use_4bit,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    # 准备数据�?
    print("\n步骤 2/4: 准备数据�?)
    print("-" * 70)
    train_dataset = Qwen2EmotionSummaryDataset(
        data_file=train_file,
        tokenizer=model_wrapper.tokenizer,
        max_length=max_length,
    )
    
    val_dataset = None
    if val_file.exists():
        val_dataset = Qwen2EmotionSummaryDataset(
            data_file=val_file,
            tokenizer=model_wrapper.tokenizer,
            max_length=max_length,
        )
    
    # 配置训练参数
    print("\n步骤 3/4: 配置训练参数")
    print("-" * 70)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=not use_4bit,  # 4-bit量化时不使用fp16
        bf16=False,
        logging_steps=10,
        save_steps=100,
        eval_steps=100 if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",  # 不使用wandb�?
        remove_unused_columns=False,
        gradient_checkpointing=True,  # 节省显存
    )
    
    print(f"  训练轮数: {num_epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  梯度累积步数: {gradient_accumulation_steps}")
    print(f"  有效批次大小: {batch_size * gradient_accumulation_steps}")
    print(f"  学习�? {learning_rate}")
    print(f"  最大序列长�? {max_length}")
    print(f"  4-bit量化: {use_4bit}")
    print(f"  梯度检查点: True (节省显存)")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=model_wrapper.tokenizer,
        mlm=False,  # 因果语言模型不使用MLM
    )
    
    # 创建 Trainer
    print("\n步骤 4/4: 开始训�?)
    print("-" * 70)
    trainer = Trainer(
        model=model_wrapper.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 开始训�?
    print("\n🚀 训练开�?..\n")
    trainer.train()
    
    # 保存最终模�?
    print("\n保存模型...")
    final_model_path = output_dir / "final"
    model_wrapper.model.save_pretrained(final_model_path)
    model_wrapper.tokenizer.save_pretrained(final_model_path)
    
    print("\n" + "=" * 70)
    print("�?训练完成�?)
    print("=" * 70)
    print(f"模型已保存到: {final_model_path}")
    print()
    
    return model_wrapper, trainer


def main():
    """主函�?""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练 Qwen2-7B-Instruct 用于情绪总结任务")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct", help="模型名称")
    parser.add_argument("--train_file", type=str, default=None, help="训练数据文件")
    parser.add_argument("--val_file", type=str, default=None, help="验证数据文件")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习�?)
    parser.add_argument("--max_length", type=int, default=8192, help="最大序列长�?)
    parser.add_argument("--use_4bit", action="store_true", default=True, help="使用4-bit量化")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    args = parser.parse_args()
    
    # 转换路径
    train_file = Path(args.train_file) if args.train_file else None
    val_file = Path(args.val_file) if args.val_file else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # 开始训�?
    train_qwen2_model(
        model_name=args.model_name,
        train_file=train_file,
        val_file=val_file,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    main()

