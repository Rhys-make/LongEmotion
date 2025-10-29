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

from qa.qa import QAModel

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
    
    parser.add_argument('--model_path', type=str, default='../qa/best_model',
                        help='模型路径（默认：../qa/best_model）')
    parser.add_argument('--test_data', type=str, default='../qa/data/test.jsonl',
                        help='测试数据路径')
    parser.add_argument('--output_file', type=str, default='../qa/result/Emotion_QA_Result.jsonl',
                        help='输出文件路径')
    parser.add_argument('--model_type', type=str, default='extractive',
                        choices=['extractive', 'generative', 'seq2seq'],
                        help='模型类型')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--doc_stride', type=int, default=128,
                        help='长上下文滑窗重叠步长')
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
    
    # 加载模型（稳健路径解析：qa/ 为基准 -> CWD -> 绝对路径）
    raw_path = Path(args.model_path)
    module_root = Path(__file__).parent.parent  # qa/
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((module_root / raw_path).resolve())
        candidates.append((Path.cwd() / raw_path).resolve())
    
    resolved_model_path = None
    for p in candidates:
        if p.exists():
            resolved_model_path = p
            break
    if resolved_model_path is None:
        # 最后再试原始绝对化
        p = raw_path.resolve()
        if p.exists():
            resolved_model_path = p
        else:
            raise FileNotFoundError(
                f"未找到模型目录，请检查 --model_path。尝试路径: "
                f"{[str(c) for c in candidates + [p]]}")
    
    logger.info(f"从 {resolved_model_path} 加载模型...")
    model_name = str(resolved_model_path)
    
    qa_model = QAModel(
        model_name=model_name,
        model_type=args.model_type,
        max_length=args.max_length,
        device=device,
        doc_stride=args.doc_stride
    )
    
    # 加载训练好的权重（使用解析后的绝对路径）
    qa_model.load(str(resolved_model_path))
    
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

