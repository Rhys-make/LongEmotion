#!/usr/bin/env python3
"""
QA 模型评估脚本

使用 F1 分数评估模型性能
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.qa.qa import calculate_f1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """
    计算精确匹配分数
    
    Returns:
        1.0 如果完全匹配，否则 0.0
    """
    def normalize_answer(s):
        import string
        import re
        
        s = s.lower()
        s = ''.join(ch if ch not in string.punctuation else ' ' for ch in s)
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ' '.join(s.split())
        
        return s
    
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def evaluate(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    评估预测结果
    
    Args:
        predictions: 预测结果 [{"id": 0, "predicted_answer": "..."}, ...]
        ground_truth: 标准答案 [{"id": 0, "answer": "..."}, ...]
    
    Returns:
        评估指标字典
    """
    # 创建 ID 到答案的映射
    pred_dict = {item['id']: item['predicted_answer'] for item in predictions}
    truth_dict = {item['id']: item.get('answer', '') for item in ground_truth}
    
    # 计算指标
    f1_scores = []
    em_scores = []
    
    for sample_id in truth_dict:
        if sample_id not in pred_dict:
            logger.warning(f"样本 {sample_id} 缺少预测结果")
            continue
        
        pred = pred_dict[sample_id]
        truth = truth_dict[sample_id]
        
        # 计算 F1
        f1 = calculate_f1(pred, truth)
        f1_scores.append(f1)
        
        # 计算 EM
        em = calculate_exact_match(pred, truth)
        em_scores.append(em)
    
    # 汇总结果
    results = {
        'num_samples': len(f1_scores),
        'f1': {
            'mean': float(np.mean(f1_scores)),
            'std': float(np.std(f1_scores)),
            'min': float(np.min(f1_scores)),
            'max': float(np.max(f1_scores)),
            'median': float(np.median(f1_scores))
        },
        'exact_match': {
            'mean': float(np.mean(em_scores)),
            'count': int(np.sum(em_scores))
        }
    }
    
    return results, f1_scores, em_scores


def save_evaluation_report(
    results: Dict,
    predictions: List[Dict],
    ground_truth: List[Dict],
    f1_scores: List[float],
    output_dir: Path
):
    """保存评估报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存 JSON 格式的指标
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ 评估指标已保存: {metrics_path}")
    
    # 2. 保存文本格式的报告
    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("QA 模型评估报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"样本数量: {results['num_samples']}\n\n")
        
        f.write("F1 分数:\n")
        f.write(f"  平均值: {results['f1']['mean']:.4f}\n")
        f.write(f"  标准差: {results['f1']['std']:.4f}\n")
        f.write(f"  最小值: {results['f1']['min']:.4f}\n")
        f.write(f"  最大值: {results['f1']['max']:.4f}\n")
        f.write(f"  中位数: {results['f1']['median']:.4f}\n\n")
        
        f.write("精确匹配 (Exact Match):\n")
        f.write(f"  准确率: {results['exact_match']['mean']:.4f}\n")
        f.write(f"  匹配数量: {results['exact_match']['count']}/{results['num_samples']}\n\n")
        
        f.write("="*60 + "\n")
    
    logger.info(f"✅ 评估报告已保存: {report_path}")
    
    # 3. 保存详细预测结果（带 F1 分数）
    pred_dict = {item['id']: item['predicted_answer'] for item in predictions}
    truth_dict = {item['id']: item.get('answer', '') for item in ground_truth}
    
    detailed_path = output_dir / 'detailed_predictions.jsonl'
    with open(detailed_path, 'w', encoding='utf-8') as f:
        for idx, (sample_id, f1) in enumerate(zip(truth_dict.keys(), f1_scores)):
            item = {
                'id': sample_id,
                'predicted_answer': pred_dict.get(sample_id, ''),
                'ground_truth': truth_dict[sample_id],
                'f1_score': f1
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ 详细预测已保存: {detailed_path}")
    
    # 4. 绘制 F1 分数分布图
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(f1_scores, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.title('F1 Score Distribution')
        plt.axvline(results['f1']['mean'], color='red', linestyle='--', 
                    label=f"Mean: {results['f1']['mean']:.4f}")
        plt.axvline(results['f1']['median'], color='green', linestyle='--',
                    label=f"Median: {results['f1']['median']:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = output_dir / 'f1_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ F1 分布图已保存: {plot_path}")
    except Exception as e:
        logger.warning(f"绘制分布图失败: {e}")
    
    # 5. 保存最佳和最差的预测示例
    pred_with_f1 = list(zip(range(len(f1_scores)), f1_scores))
    pred_with_f1.sort(key=lambda x: x[1], reverse=True)
    
    examples_path = output_dir / 'prediction_examples.txt'
    with open(examples_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("最佳预测示例 (Top 5)\n")
        f.write("="*60 + "\n\n")
        
        for i, (idx, f1) in enumerate(pred_with_f1[:5], 1):
            sample_id = list(truth_dict.keys())[idx]
            f.write(f"示例 {i} (F1: {f1:.4f}):\n")
            f.write(f"  ID: {sample_id}\n")
            f.write(f"  预测: {pred_dict[sample_id][:200]}...\n")
            f.write(f"  真实: {truth_dict[sample_id][:200]}...\n\n")
        
        f.write("="*60 + "\n")
        f.write("最差预测示例 (Bottom 5)\n")
        f.write("="*60 + "\n\n")
        
        for i, (idx, f1) in enumerate(pred_with_f1[-5:], 1):
            sample_id = list(truth_dict.keys())[idx]
            f.write(f"示例 {i} (F1: {f1:.4f}):\n")
            f.write(f"  ID: {sample_id}\n")
            f.write(f"  预测: {pred_dict[sample_id][:200]}...\n")
            f.write(f"  真实: {truth_dict[sample_id][:200]}...\n\n")
    
    logger.info(f"✅ 预测示例已保存: {examples_path}")


def main():
    parser = argparse.ArgumentParser(description='评估 QA 模型')
    
    parser.add_argument('--predictions', type=str, required=True,
                        help='预测结果文件路径（JSONL 格式）')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='标准答案文件路径（JSONL 格式）')
    parser.add_argument('--output_dir', type=str, default='evaluation/qa',
                        help='评估结果输出目录')
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info(f"加载预测结果: {args.predictions}")
    predictions = load_jsonl(args.predictions)
    
    logger.info(f"加载标准答案: {args.ground_truth}")
    ground_truth = load_jsonl(args.ground_truth)
    
    logger.info(f"预测样本数: {len(predictions)}")
    logger.info(f"真实样本数: {len(ground_truth)}")
    
    # 评估
    logger.info("开始评估...")
    results, f1_scores, em_scores = evaluate(predictions, ground_truth)
    
    # 打印结果
    logger.info("\n" + "="*60)
    logger.info("评估结果:")
    logger.info("="*60)
    logger.info(f"样本数量: {results['num_samples']}")
    logger.info(f"平均 F1: {results['f1']['mean']:.4f} (±{results['f1']['std']:.4f})")
    logger.info(f"精确匹配: {results['exact_match']['mean']:.4f} ({results['exact_match']['count']}/{results['num_samples']})")
    logger.info("="*60)
    
    # 保存评估报告
    output_dir = Path(args.output_dir)
    save_evaluation_report(results, predictions, ground_truth, f1_scores, output_dir)
    
    logger.info(f"\n✅ 评估完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

