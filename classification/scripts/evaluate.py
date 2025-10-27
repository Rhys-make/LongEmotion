#!/usr/bin/env python3
"""
情绪分类模型评估脚本
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from classification.scripts.classification import EmotionDataset, EmotionClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_labels(model_path: str, device: str):
    """加载模型和标签映射"""
    # 加载标签映射
    label2id_path = Path(model_path).parent / 'label2id.json'
    id2label_path = Path(model_path).parent / 'id2label.json'
    
    if label2id_path.exists():
        with open(label2id_path, 'r', encoding='utf-8') as f:
            label2id = json.load(f)
        with open(id2label_path, 'r', encoding='utf-8') as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
    else:
        logger.warning("未找到标签映射文件，将从模型配置中读取")
        label2id = None
        id2label = None
    
    # 加载模型
    classifier = EmotionClassifier(
        model_name=model_path,
        device=device
    )
    classifier.load(model_path)
    
    # 如果标签映射未找到，从模型配置中获取
    if label2id is None:
        model = classifier.get_model()
        label2id = model.config.label2id
        id2label = model.config.id2label
    
    return classifier, label2id, id2label


def evaluate_model(
    model_path: str,
    data_path: str,
    batch_size: int = 16,
    max_length: int = 512,
    device: str = None,
    output_dir: str = None
):
    """评估模型性能"""
    
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"使用设备: {device}")
    logger.info(f"评估数据: {data_path}")
    logger.info(f"模型路径: {model_path}")
    
    # 加载模型
    classifier, label2id, id2label = load_model_and_labels(model_path, device)
    model = classifier.get_model()
    tokenizer = classifier.get_tokenizer()
    
    logger.info(f"标签数量: {len(label2id)}")
    
    # 加载数据
    dataset = EmotionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id,
        is_test=False
    )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"评估样本数: {len(dataset)}")
    
    # 评估
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="评估中")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro'
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("评估结果:")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f} (macro)")
    logger.info(f"Recall:    {recall:.4f} (macro)")
    logger.info(f"F1-Score:  {f1:.4f} (macro)")
    logger.info(f"{'='*60}")
    
    # 详细的分类报告
    logger.info("\n分类报告:")
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=4
    )
    logger.info(f"\n{report}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\n混淆矩阵形状: {cm.shape}")
    
    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存指标
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'num_samples': len(all_labels)
        }
        
        with open(output_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 保存详细报告
        with open(output_path / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        # 保存混淆矩阵
        np.save(output_path / 'confusion_matrix.npy', cm)
        
        # 绘制混淆矩阵（如果类别数不太多）
        if len(id2label) <= 30:
            try:
                plt.figure(figsize=(12, 10))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names
                )
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(output_path / 'confusion_matrix.png', dpi=300)
                logger.info(f"混淆矩阵图已保存到: {output_path / 'confusion_matrix.png'}")
            except Exception as e:
                logger.warning(f"绘制混淆矩阵失败: {e}")
        
        # 保存预测结果
        predictions = []
        for i, (pred, label, logit) in enumerate(zip(all_preds, all_labels, all_logits)):
            predictions.append({
                'id': i,
                'true_label': id2label[int(label)],
                'predicted_label': id2label[int(pred)],
                'correct': bool(pred == label),
                'confidence': float(np.max(np.exp(logit) / np.sum(np.exp(logit))))
            })
        
        with open(output_path / 'predictions.json', 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n评估结果已保存到: {output_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    parser = argparse.ArgumentParser(description='评估情绪分类模型')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（例如: checkpoint/classification/best_model）')
    parser.add_argument('--data_path', type=str, default='../data/validation.jsonl',
                        help='评估数据路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)，默认自动选择')
    parser.add_argument('--output_dir', type=str, default='evaluation/classification',
                        help='结果输出目录')
    
    args = parser.parse_args()
    
    # 执行评估
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

