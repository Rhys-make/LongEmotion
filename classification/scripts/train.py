#!/usr/bin/env python3
"""
情绪分类模型训练脚本
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import logging
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from classification.scripts.classification import (
    EmotionClassifier,
    create_dataloaders
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子设置为: {seed}")


def calculate_accuracy(outputs, labels):
    """计算准确率"""
    preds = torch.argmax(outputs.logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0
        
        return self.early_stop


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    scaler=None,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False
):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="训练中")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # 将数据移到设备上
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播（使用混合精度）
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            # 梯度累积
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * gradient_accumulation_steps
        accuracy = calculate_accuracy(outputs, labels)
        total_correct += accuracy * labels.size(0)
        total_samples += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'acc': accuracy,
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy


def evaluate(model, val_loader, device):
    """在验证集上评估"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证中")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            accuracy = calculate_accuracy(outputs, labels)
            total_correct += accuracy * labels.size(0)
            total_samples += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': accuracy
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy


def train(args):
    """主训练函数"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    logger.info(f"加载模型: {args.model_name}")
    
    # 首先创建数据加载器以获取标签数量
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_loader, val_loader, label2id, id2label = create_dataloaders(
        train_path=args.train_data,
        val_path=args.val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    logger.info(f"标签数量: {len(label2id)}")
    logger.info(f"训练样本: {len(train_loader.dataset)}")
    logger.info(f"验证样本: {len(val_loader.dataset)}")
    
    # 保存标签映射
    with open(output_dir / 'label2id.json', 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(output_dir / 'id2label.json', 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    
    # 初始化模型
    classifier = EmotionClassifier(
        model_name=args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        device=device
    )
    model = classifier.get_model()
    
    # 如果指定了检查点，从检查点加载
    if args.resume_from_checkpoint:
        logger.info(f"从检查点恢复: {args.resume_from_checkpoint}")
        classifier.load(args.resume_from_checkpoint)
        model = classifier.get_model()
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"总训练步数: {num_training_steps}")
    logger.info(f"预热步数: {num_warmup_steps}")
    
    # 混合精度训练
    scaler = GradScaler() if args.use_amp else None
    
    # 早停
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # 训练循环
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*60}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=args.use_amp
        )
        
        logger.info(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.info(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / 'best_model'
            classifier.save(str(best_model_path))
            logger.info(f"✅ 保存最佳模型 (验证准确率: {val_acc:.4f})")
        
        # 保存最新模型
        latest_model_path = output_dir / 'latest_model'
        classifier.save(str(latest_model_path))
        
        # 早停检查
        if early_stopping(val_acc):
            logger.info(f"早停触发！最佳验证准确率: {best_val_acc:.4f}")
            break
    
    # 保存训练历史
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("训练完成！")
    logger.info(f"最佳验证准确率: {best_val_acc:.4f}")
    logger.info(f"模型保存在: {output_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='情绪分类模型训练')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, default='..classification/data/train.jsonl',
                        help='训练数据路径')
    parser.add_argument('--val_data', type=str, default='..classification/data/validation.jsonl',
                        help='验证数据路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='roberta-large',
                        help='预训练模型名称')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='预热比例')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='../checkpoint/classification',
                        help='模型输出目录')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='从检查点恢复训练')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    
    args = parser.parse_args()
    
    # 打印配置
    logger.info("训练配置:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # 开始训练
    train(args)


if __name__ == "__main__":
    main()

