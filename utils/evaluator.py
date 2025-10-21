"""
评估指标计算模块
提供各种任务的评估指标计算
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    f1_score
)


class Evaluator:
    """评估器基类"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def compute_classification_metrics(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> Dict[str, float]:
        """
        计算分类任务指标
        
        Args:
            predictions: 预测标签
            labels: 真实标签
            
        Returns:
            指标字典
        """
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def compute_detection_metrics(
        self,
        predictions: List[List[int]],
        labels: List[List[int]]
    ) -> Dict[str, float]:
        """
        计算检测任务指标（多标签分类）
        
        Args:
            predictions: 预测标签列表
            labels: 真实标签列表
            
        Returns:
            指标字典
        """
        # 转换为二维数组
        y_pred = np.array(predictions)
        y_true = np.array(labels)
        
        # 计算每个标签的F1
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        return {
            'f1_micro': float(f1_micro),
            'f1_macro': float(f1_macro)
        }
    
    def compute_generation_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算生成任务指标（ROUGE等）
        
        Args:
            predictions: 预测文本
            references: 参考文本
            
        Returns:
            指标字典
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1': float(np.mean(rouge1_scores)),
                'rouge2': float(np.mean(rouge2_scores)),
                'rougeL': float(np.mean(rougeL_scores))
            }
        except ImportError:
            print("警告: rouge_score 未安装，返回默认值")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }
    
    def compute_qa_metrics(
        self,
        predictions: List[str],
        labels: List[str]
    ) -> Dict[str, float]:
        """
        计算问答任务指标
        
        Args:
            predictions: 预测答案
            labels: 真实答案
            
        Returns:
            指标字典
        """
        # 精确匹配率
        exact_match = sum(
            1 for pred, label in zip(predictions, labels)
            if pred.strip() == label.strip()
        ) / len(predictions)
        
        # 使用 ROUGE 作为辅助指标
        rouge_metrics = self.compute_generation_metrics(predictions, labels)
        
        return {
            'exact_match': float(exact_match),
            **rouge_metrics
        }


def get_evaluator() -> Evaluator:
    """获取评估器实例"""
    return Evaluator()

