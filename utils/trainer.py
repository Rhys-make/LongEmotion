"""
通用训练逻辑封装
提供统一的训练、验证、保存逻辑
"""
import os
import json
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from utils.evaluator import get_evaluator


class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        output_dir: str = "./checkpoints",
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        early_stopping_patience: int = 3,
        metric_for_best_model: str = "loss",
        greater_is_better: bool = False
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_dataloader: 训练数据加载器
            eval_dataloader: 验证数据加载器
            output_dir: 输出目录
            learning_rate: 学习率
            num_epochs: 训练轮数
            device: 设备
            early_stopping_patience: 早停耐心值
            metric_for_best_model: 用于选择最佳模型的指标
            greater_is_better: 指标越大越好
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        
        # 初始化优化器
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # 初始化学习率调度器
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # 评估器
        self.evaluator = get_evaluator()
        
        # 训练状态
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.patience_counter = 0
        self.global_step = 0
        
    def train(self, compute_metrics_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        执行训练
        
        Args:
            compute_metrics_fn: 自定义指标计算函数
            
        Returns:
            训练历史
        """
        print(f"开始训练，设备: {self.device}")
        print(f"训练轮数: {self.num_epochs}")
        print(f"总训练步数: {len(self.train_dataloader) * self.num_epochs}")
        
        history = {
            'train_loss': [],
            'eval_metrics': []
        }
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*50}")
            
            # 训练阶段
            train_loss = self._train_epoch()
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            if self.eval_dataloader is not None:
                eval_metrics = self._eval_epoch(compute_metrics_fn)
                history['eval_metrics'].append(eval_metrics)
                
                # 检查是否需要保存最佳模型
                current_metric = eval_metrics.get(self.metric_for_best_model, train_loss)
                
                if self._is_better_metric(current_metric):
                    print(f"✓ 发现更好的模型！{self.metric_for_best_model}: {current_metric:.4f}")
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self.save_model("best_model")
                else:
                    self.patience_counter += 1
                    print(f"✗ 模型未改进，耐心计数: {self.patience_counter}/{self.early_stopping_patience}")
                
                # 早停检查
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n早停触发！在 epoch {epoch + 1} 停止训练")
                    break
        
        # 保存最终模型
        self.save_model("final_model")
        
        # 保存训练历史
        with open(self.output_dir / "training_history.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"\n训练完成！模型已保存到 {self.output_dir}")
        return history
    
    def _train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="训练中")
        
        for batch in progress_bar:
            # 将数据移到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_dataloader)
        print(f"平均训练损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def _eval_epoch(self, compute_metrics_fn: Optional[Callable] = None) -> Dict[str, float]:
        """验证一个 epoch"""
        self.model.eval()
        total_loss = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="验证中"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                total_loss += loss.item()
                
                # 收集预测和标签
                if hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy().tolist())
                    
                    if 'labels' in batch:
                        all_labels.extend(batch['labels'].cpu().numpy().tolist())
        
        avg_loss = total_loss / len(self.eval_dataloader)
        metrics = {'loss': avg_loss}
        
        # 计算其他指标
        if compute_metrics_fn and all_predictions and all_labels:
            computed_metrics = compute_metrics_fn(all_predictions, all_labels)
            metrics.update(computed_metrics)
        
        print(f"验证指标: {metrics}")
        return metrics
    
    def _is_better_metric(self, current_metric: float) -> bool:
        """判断当前指标是否更好"""
        if self.greater_is_better:
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric
    
    def save_model(self, name: str = "model"):
        """保存模型"""
        save_path = self.output_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), save_path / "model.pt")
        
        print(f"模型已保存到 {save_path}")
    
    @classmethod
    def load_model(cls, model_class, model_path: str, device: str = "cuda"):
        """加载模型"""
        if hasattr(model_class, 'from_pretrained'):
            model = model_class.from_pretrained(model_path)
        else:
            model = model_class()
            model.load_state_dict(torch.load(Path(model_path) / "model.pt"))
        
        return model.to(device)

