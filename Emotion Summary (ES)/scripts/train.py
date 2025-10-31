"""
训练脚本 - Emotion Summary (ES) 任务
训练情绪总结模型
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    MODEL_NAME,
    TRAIN_FILE,
    VALIDATION_FILE,
    MODEL_CHECKPOINT_DIR,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    MAX_GRAD_NORM,
    SAVE_STEPS,
    EVAL_STEPS,
    LOGGING_STEPS,
    DEVICE,
    SEED,
    FP16,
    SUMMARY_ASPECTS,
    T5_PREFIX,
)


class EmotionSummaryDataset(Dataset):
    """
    情绪总结数据�?
    """
    
    def __init__(
        self,
        data_file: Path,
        tokenizer,
        max_input_length: int,
        max_output_length: int,
        is_test: bool = False
    ):
        """
        初始化数据集
        
        Args:
            data_file: 数据文件路径
            tokenizer: 分词�?
            max_input_length: 最大输入长�?
            max_output_length: 最大输出长�?
            is_test: 是否为测试集
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.is_test = is_test
        
        # 加载数据
        self.samples = self.load_data(data_file)
        
    def load_data(self, data_file: Path) -> List[Dict]:
        """加载JSONL数据"""
        samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 输入文本
        context = sample.get("context", "")
        input_text = f"{T5_PREFIX}{context}"
        
        # Tokenize输入
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 如果不是测试集，准备目标文本
        if not self.is_test and "reference_summary" in sample:
            # 将所有方面的总结合并为一个文�?
            reference = sample["reference_summary"]
            target_parts = []
            
            for aspect in SUMMARY_ASPECTS:
                if aspect in reference:
                    target_parts.append(f"{aspect}: {reference[aspect]}")
            
            target_text = " | ".join(target_parts)
            
            # Tokenize目标
            targets = self.tokenizer(
                target_text,
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 准备labels�?100会被忽略�?
            labels = targets["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": labels.squeeze(),
                "id": sample.get("id", idx)
            }
        else:
            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "id": sample.get("id", idx)
            }


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    gradient_accumulation_steps,
    max_grad_norm
):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        # 移动到设�?
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        total_loss += loss.item()
        
        # 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 更新进度�?
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, eval_loader, device):
    """
    评估模型
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(eval_loader)
    return avg_loss


def main():
    """
    主训练函�?
    """
    print("=" * 60)
    print("Emotion Summary (ES) 模型训练")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # 加载tokenizer和模�?
    print(f"\n加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    
    # 创建数据�?
    print("\n加载数据�?..")
    train_dataset = EmotionSummaryDataset(
        TRAIN_FILE,
        tokenizer,
        MAX_INPUT_LENGTH,
        MAX_OUTPUT_LENGTH,
        is_test=False
    )
    
    val_dataset = EmotionSummaryDataset(
        VALIDATION_FILE,
        tokenizer,
        MAX_INPUT_LENGTH,
        MAX_OUTPUT_LENGTH,
        is_test=False
    )
    
    print(f"训练�? {len(train_dataset)} 样本")
    print(f"验证�? {len(val_dataset)} 样本")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Windows上设置为0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # 优化器和调度�?
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # 训练循环
    print(f"\n开始训�?..")
    print(f"总轮�? {NUM_EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"学习�? {LEARNING_RATE}")
    print(f"设备: {DEVICE}")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print('=' * 60)
        
        # 训练
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            DEVICE,
            epoch,
            GRADIENT_ACCUMULATION_STEPS,
            MAX_GRAD_NORM
        )
        
        print(f"\n训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss = evaluate(model, val_loader, DEVICE)
        print(f"验证损失: {val_loss:.4f}")
        
        # 保存最佳模�?
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\n�?发现更好的模型！保存�?{MODEL_CHECKPOINT_DIR}")
            MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_CHECKPOINT_DIR)
            tokenizer.save_pretrained(MODEL_CHECKPOINT_DIR)
    
    print("\n" + "=" * 60)
    print("训练完成�?)
    print(f"最佳验证损�? {best_val_loss:.4f}")
    print(f"模型已保存到: {MODEL_CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

