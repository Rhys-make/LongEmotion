#!/usr/bin/env python3
"""
QA 模型推理脚本

对测试集进行预测，输出格式：
{"id": 0, "predicted_answer": "..."}
{"id": 1, "predicted_answer": "..."}
...
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.qa.qa import QAModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_path: str):
    """加载测试数据"""
    samples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info(f"加载了 {len(samples)} 条测试样本")
    return samples


def predict_batch(
    qa_model: QAModel,
    samples: list,
    batch_size: int = 8,
    max_answer_length: int = 256
):
    """
    批量预测
    
    Args:
        qa_model: QA 模型
        samples: 样本列表
        batch_size: 批次大小
        max_answer_length: 最大答案长度
    
    Returns:
        预测结果列表
    """
    predictions = []
    
    # 单个处理（为了保证质量）
    progress_bar = tqdm(samples, desc="预测中")
    for sample in progress_bar:
        sample_id = sample.get('id', 0)
        problem = sample.get('problem', '')
        context = sample.get('context', '')
        
        # 预测答案
        predicted_answer = qa_model.predict(
            problem=problem,
            context=context,
            max_answer_length=max_answer_length
        )
        
        predictions.append({
            'id': sample_id,
            'predicted_answer': predicted_answer
        })
        
        # 更新进度条
        progress_bar.set_postfix({
            'id': sample_id,
            'answer_len': len(predicted_answer)
        })
    
    return predictions


def save_predictions(predictions: list, output_path: str):
    """保存预测结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ 预测结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='QA 模型推理')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（例如：checkpoint/qa/best_model）')
    parser.add_argument('--test_data', type=str, default='data/qa/test.jsonl',
                        help='测试数据路径')
    parser.add_argument('--output_file', type=str, default='result/Emotion_QA_Result.jsonl',
                        help='输出文件路径')
    parser.add_argument('--model_type', type=str, default='extractive',
                        choices=['extractive', 'generative', 'seq2seq'],
                        help='模型类型')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--max_answer_length', type=int, default=256,
                        help='最大答案长度')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小（暂不支持批量处理）')
    parser.add_argument('--device', type=str, default=None,
                        help='设备（cuda 或 cpu）')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"从 {args.model_path} 加载模型...")
    
    # 首先尝试从模型目录推断模型名称
    try:
        # 读取 config.json 获取模型信息
        config_path = Path(args.model_path) / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config.get('_name_or_path', args.model_path)
    except:
        model_name = args.model_path
    
    qa_model = QAModel(
        model_name=model_name,
        model_type=args.model_type,
        max_length=args.max_length,
        device=device
    )
    
    # 加载训练好的权重
    qa_model.load(args.model_path)
    
    logger.info("模型加载完成")
    
    # 加载测试数据
    test_samples = load_test_data(args.test_data)
    
    # 预测
    predictions = predict_batch(
        qa_model=qa_model,
        samples=test_samples,
        batch_size=args.batch_size,
        max_answer_length=args.max_answer_length
    )
    
    # 保存结果
    save_predictions(predictions, args.output_file)
    
    # 打印一些示例
    logger.info("\n预测示例:")
    for i, pred in enumerate(predictions[:3]):
        logger.info(f"\n样本 {pred['id']}:")
        logger.info(f"  问题: {test_samples[i].get('problem', '')[:100]}...")
        logger.info(f"  预测答案: {pred['predicted_answer'][:200]}...")
    
    logger.info(f"\n{'='*60}")
    logger.info("推理完成！")
    logger.info(f"总样本数: {len(predictions)}")
    logger.info(f"结果保存在: {args.output_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

