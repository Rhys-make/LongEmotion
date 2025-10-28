#!/usr/bin/env python3
"""
QA 模型训练脚本
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import logging
import argparse
from pathlib import Path
import psutil
import gc
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qa.qa import (
    QAModel,
    create_dataloaders,
    calculate_f1
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


def check_system_resources():
    """检查系统资源"""
    memory = psutil.virtual_memory()
    logger.info(f"系统内存: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    if memory.percent > 85:
        logger.warning(f"⚠️  系统内存使用率过高: {memory.percent:.1f}%")
        logger.warning("建议关闭其他程序或减少batch_size")
        return False
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_used = torch.cuda.memory_allocated(0)
        gpu_free = gpu_memory - gpu_used
        gpu_free_gb = gpu_free / 1024**3
        
        logger.info(f"GPU显存: {gpu_used/1024**3:.1f}GB / {gpu_memory/1024**3:.1f}GB (可用: {gpu_free_gb:.1f}GB)")
        
        if gpu_free_gb < 2.0:
            logger.warning(f"⚠️  GPU显存不足: 仅剩{gpu_free_gb:.1f}GB")
            return False
    
    return True


def safe_cleanup():
    """安全清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0
        
        return self.early_stop


def train_epoch_extractive(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    scaler=None,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False
):
    """训练一个 epoch（抽取式 QA）"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="训练中")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # 每10个step检查一次内存
        if step % 10 == 0:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.error(f"❌ 内存使用率过高: {memory.percent:.1f}%，停止训练")
                raise RuntimeError("内存不足，训练停止")
        
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            # 前向传播
            if use_amp and scaler is not None:
                with autocast('cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                    
                scaler.scale(loss).backward()
                
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
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            progress_bar.set_postfix({
                'loss': loss.item() * gradient_accumulation_steps,
                'lr': scheduler.get_last_lr()[0]
            })
            
            # 每5个step清理一次GPU缓存
            if step % 5 == 0:
                safe_cleanup()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"❌ GPU显存不足: {e}")
                safe_cleanup()
                raise RuntimeError("GPU显存不足，请减少batch_size")
            else:
                raise e
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_generative(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    scaler=None,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False
):
    """训练一个 epoch（生成式 QA）"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="训练中")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        if use_amp and scaler is not None:
            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
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
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_extractive(model, val_loader, tokenizer, device):
    """在验证集上评估（抽取式 QA）"""
    model.eval()
    total_loss = 0
    all_f1_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证中")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 计算 F1 分数
            start_pred = torch.argmax(outputs.start_logits, dim=1)
            end_pred = torch.argmax(outputs.end_logits, dim=1)
            
            for i in range(len(input_ids)):
                # 预测答案
                pred_start = start_pred[i].item()
                pred_end = end_pred[i].item()
                if pred_end < pred_start:
                    pred_end = pred_start
                
                pred_tokens = input_ids[i][pred_start:pred_end+1]
                pred_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                
                # 真实答案
                true_start = start_positions[i].item()
                true_end = end_positions[i].item()
                true_tokens = input_ids[i][true_start:true_end+1]
                true_answer = tokenizer.decode(true_tokens, skip_special_tokens=True)
                
                # 计算 F1
                f1 = calculate_f1(pred_answer, true_answer)
                all_f1_scores.append(f1)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'f1': np.mean(all_f1_scores) if all_f1_scores else 0.0
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
    
    return avg_loss, avg_f1


def evaluate_generative(model, val_loader, tokenizer, device, max_answer_length=256):
    """在验证集上评估（生成式 QA）"""
    model.eval()
    total_loss = 0
    all_f1_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证中")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 计算损失
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            
            # 生成答案
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_answer_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
            
            for i in range(len(input_ids)):
                # 解码预测答案
                pred_answer = tokenizer.decode(generated[i], skip_special_tokens=True)
                
                # 解码真实答案
                true_labels = labels[i]
                true_labels = true_labels[true_labels != -100]
                true_answer = tokenizer.decode(true_labels, skip_special_tokens=True)
                
                # 计算 F1
                f1 = calculate_f1(pred_answer, true_answer)
                all_f1_scores.append(f1)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'f1': np.mean(all_f1_scores) if all_f1_scores else 0.0
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
    
    return avg_loss, avg_f1


def train(args):
    """主训练函数"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 检查系统资源
    logger.info("检查系统资源...")
    if not check_system_resources():
        logger.error("❌ 系统资源不足，无法开始训练")
        logger.info("建议:")
        logger.info("  - 关闭其他程序释放内存")
        logger.info("  - 减少batch_size (建议1-2)")
        logger.info("  - 增加gradient_accumulation_steps (建议8-16)")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    logger.info(f"加载模型: {args.model_name}")
    
    # 设置离线模式
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    qa_model = QAModel(
        model_name=args.model_name,
        model_type=args.model_type,
        max_length=args.max_length,
        device=device
    )
    
    model = qa_model.get_model()
    tokenizer = qa_model.get_tokenizer()
    
    # 如果指定了检查点，从检查点加载
    if args.resume_from_checkpoint:
        logger.info(f"从检查点恢复: {args.resume_from_checkpoint}")
        qa_model.load(args.resume_from_checkpoint)
        model = qa_model.get_model()
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        train_path=args.train_data,
        val_path=args.val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_type=args.model_type,
        num_workers=args.num_workers
    )
    
    logger.info(f"训练样本: {len(train_loader.dataset)}")
    logger.info(f"验证样本: {len(val_loader.dataset)}")
    
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
    scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
    
    # 早停
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    
    # 训练循环
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*60}")
        
        # 训练
        if args.model_type == "extractive":
            train_loss = train_epoch_extractive(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                scaler=scaler,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                use_amp=args.use_amp
            )
        else:
            train_loss = train_epoch_generative(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                scaler=scaler,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                use_amp=args.use_amp
            )
        
        logger.info(f"训练 - Loss: {train_loss:.4f}")
        
        # 验证
        if args.model_type == "extractive":
            val_loss, val_f1 = evaluate_extractive(model, val_loader, tokenizer, device)
        else:
            val_loss, val_f1 = evaluate_generative(model, val_loader, tokenizer, device)
        
        logger.info(f"验证 - Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = output_dir / 'best_model'
            qa_model.save(str(best_model_path))
            logger.info(f"✅ 保存最佳模型 (验证 F1: {val_f1:.4f})")
        
        # 保存最新模型
        latest_model_path = output_dir / 'latest_model'
        qa_model.save(str(latest_model_path))
        
        # 早停检查
        if early_stopping(val_f1):
            logger.info(f"早停触发！最佳验证 F1: {best_val_f1:.4f}")
            break
    
    # 保存训练历史
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("训练完成！")
    logger.info(f"最佳验证 F1: {best_val_f1:.4f}")
    logger.info(f"模型保存在: {output_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='QA 模型训练')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, default='../data/train.jsonl',
                        help='训练数据路径')
    parser.add_argument('--val_data', type=str, default='../data/validation.jsonl',
                        help='验证数据路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='预训练模型名称')
    parser.add_argument('--model_type', type=str, default='extractive',
                        choices=['extractive', 'generative', 'seq2seq'],
                        help='模型类型')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小 (移动GPU建议1-2)')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='预热比例')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='梯度累积步数')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='../checkpoint',
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

