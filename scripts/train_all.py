"""
批量训练所有任务
统一训练入口
"""
import os
import sys
from pathlib import Path
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.classification_model import EmotionClassificationModel
from models.detection_model import EmotionDetectionModelWrapper
from models.summary_model import EmotionSummaryModel
from models.qa_model import EmotionQAModel
from utils.trainer import UnifiedTrainer
from utils.evaluator import get_evaluator


class TaskTrainer:
    """任务训练器"""
    
    def __init__(
        self,
        data_dir: str = "./data",
        output_dir: str = "./checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化训练器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            device: 设备
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.evaluator = get_evaluator()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_classification(self, num_epochs: int = 3, batch_size: int = 16):
        """训练分类任务"""
        print("\n" + "="*60)
        print("训练任务: 情感分类 (Emotion Classification)")
        print("="*60)
        
        # 加载数据
        task_dir = self.data_dir / "classification"
        train_dataset = load_from_disk(str(task_dir / "train"))
        eval_dataset = load_from_disk(str(task_dir / "validation"))
        
        # 初始化模型
        model = EmotionClassificationModel(device=self.device)
        
        # 准备数据加载器
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = torch.tensor([item['label'] for item in batch])
            
            encoding = model.preprocess(texts)
            encoding['labels'] = labels.to(self.device)
            
            return encoding
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        # 训练
        trainer = UnifiedTrainer(
            model=model.model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            output_dir=str(self.output_dir / "classification"),
            num_epochs=num_epochs,
            device=self.device,
            metric_for_best_model="f1",
            greater_is_better=True
        )
        
        history = trainer.train(
            compute_metrics_fn=self.evaluator.compute_classification_metrics
        )
        
        print("✓ 分类任务训练完成！")
        return history
    
    def train_detection(self, num_epochs: int = 3, batch_size: int = 16):
        """训练检测任务"""
        print("\n" + "="*60)
        print("训练任务: 情感检测 (Emotion Detection)")
        print("="*60)
        
        # 加载数据
        task_dir = self.data_dir / "detection"
        train_dataset = load_from_disk(str(task_dir / "train"))
        eval_dataset = load_from_disk(str(task_dir / "validation"))
        
        # 初始化模型
        model_wrapper = EmotionDetectionModelWrapper(device=self.device)
        
        # 准备数据加载器
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            # 假设 emotions 是多标签二进制向量
            labels = torch.tensor([item['emotions'] for item in batch])
            
            encoding = model_wrapper.preprocess(texts)
            encoding['labels'] = labels.to(self.device)
            
            return encoding
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        # 训练
        trainer = UnifiedTrainer(
            model=model_wrapper.model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            output_dir=str(self.output_dir / "detection"),
            num_epochs=num_epochs,
            device=self.device,
            metric_for_best_model="f1_macro",
            greater_is_better=True
        )
        
        history = trainer.train(
            compute_metrics_fn=lambda preds, labels: self.evaluator.compute_detection_metrics(
                [[int(p > 0.5) for p in pred] for pred in preds],
                labels
            )
        )
        
        print("✓ 检测任务训练完成！")
        return history
    
    def train_summary(self, num_epochs: int = 3, batch_size: int = 8):
        """训练摘要任务"""
        print("\n" + "="*60)
        print("训练任务: 情感摘要 (Emotion Summary)")
        print("="*60)
        
        # 加载数据
        task_dir = self.data_dir / "summary"
        train_dataset = load_from_disk(str(task_dir / "train"))
        eval_dataset = load_from_disk(str(task_dir / "validation"))
        
        # 初始化模型
        model = EmotionSummaryModel(device=self.device)
        
        # 准备数据加载器
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            summaries = [item['summary'] for item in batch]
            
            # 编码输入
            inputs = model.preprocess(texts)
            
            # 编码目标
            with model.tokenizer.as_target_tokenizer():
                labels = model.tokenizer(
                    summaries,
                    padding=True,
                    truncation=True,
                    max_length=model.max_output_length,
                    return_tensors="pt"
                )['input_ids'].to(self.device)
            
            inputs['labels'] = labels
            
            return inputs
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        # 训练
        trainer = UnifiedTrainer(
            model=model.model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            output_dir=str(self.output_dir / "summary"),
            num_epochs=num_epochs,
            device=self.device,
            learning_rate=5e-5
        )
        
        history = trainer.train()
        
        print("✓ 摘要任务训练完成！")
        return history
    
    def train_qa(self, num_epochs: int = 3, batch_size: int = 8):
        """训练问答任务"""
        print("\n" + "="*60)
        print("训练任务: 情感问答 (Emotion QA)")
        print("="*60)
        
        # 加载数据
        task_dir = self.data_dir / "qa"
        train_dataset = load_from_disk(str(task_dir / "train"))
        eval_dataset = load_from_disk(str(task_dir / "validation"))
        
        # 初始化模型
        model = EmotionQAModel(device=self.device)
        
        # 准备数据加载器
        def collate_fn(batch):
            questions = [item['question'] for item in batch]
            contexts = [item['context'] for item in batch]
            
            encoding = model.preprocess(questions, contexts)
            
            # 添加答案位置（如果有）
            if 'start_position' in batch[0]:
                encoding['start_positions'] = torch.tensor([item['start_position'] for item in batch]).to(self.device)
                encoding['end_positions'] = torch.tensor([item['end_position'] for item in batch]).to(self.device)
            
            return encoding
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        # 训练
        trainer = UnifiedTrainer(
            model=model.model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            output_dir=str(self.output_dir / "qa"),
            num_epochs=num_epochs,
            device=self.device,
            learning_rate=3e-5
        )
        
        history = trainer.train()
        
        print("✓ 问答任务训练完成！")
        return history
    
    def train_all(self, tasks: list = None, **kwargs):
        """训练所有任务"""
        all_tasks = {
            "classification": self.train_classification,
            "detection": self.train_detection,
            "summary": self.train_summary,
            "qa": self.train_qa
        }
        
        if tasks is None:
            tasks = list(all_tasks.keys())
        
        results = {}
        
        print("\n" + "="*60)
        print(f"开始训练 {len(tasks)} 个任务")
        print("="*60)
        
        for task in tasks:
            if task in all_tasks:
                try:
                    history = all_tasks[task](**kwargs)
                    results[task] = {"status": "success", "history": history}
                except Exception as e:
                    print(f"\n✗ {task} 训练失败: {e}")
                    results[task] = {"status": "failed", "error": str(e)}
            else:
                print(f"\n✗ 未知任务: {task}")
        
        print("\n" + "="*60)
        print("训练总结")
        print("="*60)
        
        for task, result in results.items():
            status = "✓" if result['status'] == 'success' else "✗"
            print(f"{status} {task}: {result['status']}")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练 LongEmotion 任务")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="数据目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="输出目录"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="要训练的任务 (classification detection summary qa)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批量大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = TaskTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # 训练
    results = trainer.train_all(
        tasks=args.tasks,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )

