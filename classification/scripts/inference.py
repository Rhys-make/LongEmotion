#!/usr/bin/env python3
"""
情绪分类推理脚本
支持从每个样本的Choices中选择最匹配的答案
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

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from classification.scripts.classification import EmotionClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str):
    """加载模型"""
    label2id_path = Path(model_path).parent / 'label2id.json'
    id2label_path = Path(model_path).parent / 'id2label.json'
    
    if label2id_path.exists():
        with open(label2id_path, 'r', encoding='utf-8') as f:
            label2id = json.load(f)
        with open(id2label_path, 'r', encoding='utf-8') as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
    else:
        label2id = None
        id2label = None
    
    classifier = EmotionClassifier(
        model_name=model_path,
        device=device
    )
    classifier.load(model_path)
    
    if label2id is None:
        model = classifier.get_model()
        label2id = model.config.label2id
        id2label = model.config.id2label
    
    return classifier, label2id, id2label


def predict_from_choices(
    model,
    tokenizer,
    context: str,
    subject: str,
    choices: list,
    trained_label2id: dict,
    trained_id2label: dict,
    device: str,
    max_length: int = 512
):
    """
    从给定的choices中选择最匹配的答案
    
    策略：
    1. 将choices中的每个选项分解为单独的情绪
    2. 用模型预测所有基础情绪的概率
    3. 对于组合情绪，取各个情绪概率的平均值
    4. 选择得分最高的choice
    """
    
    # 构建输入
    choices_str = ', '.join(choices)
    input_text = f"Subject: {subject}. Context: {context}. Choices: {choices_str}"
    
    # Tokenize
    encoding = tokenizer(
        input_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 获取模型预测
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits[0]  # [num_labels]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    
    # 为每个choice计算得分
    choice_scores = []
    
    for choice in choices:
        # 分解组合情绪（用 & 或 and 分隔）
        emotions = []
        for sep in [' & ', ' and ', '&']:
            if sep in choice:
                emotions = [e.strip().lower() for e in choice.split(sep)]
                break
        
        if not emotions:
            # 单个情绪
            emotions = [choice.strip().lower()]
        
        # 获取每个情绪的概率
        emotion_probs = []
        for emotion in emotions:
            if emotion in trained_label2id:
                label_id = trained_label2id[emotion]
                emotion_probs.append(probs[label_id])
            else:
                # 如果情绪不在训练集中，给一个很低的概率
                emotion_probs.append(0.001)
        
        # 组合情绪的得分：取平均值
        if emotion_probs:
            score = np.mean(emotion_probs)
        else:
            score = 0.001
        
        choice_scores.append(score)
    
    # 选择得分最高的
    best_idx = np.argmax(choice_scores)
    predicted_choice = choices[best_idx]
    confidence = choice_scores[best_idx]
    
    return predicted_choice, confidence, choice_scores


def predict(
    model_path: str,
    test_data_path: str,
    output_path: str,
    device: str = None,
    max_length: int = 512
):
    """
    测试集推理
    自动从每个样本的Choices中选择最匹配的答案
    """
    
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"使用设备: {device}")
    logger.info(f"测试数据: {test_data_path}")
    logger.info(f"模型路径: {model_path}")
    
    # 加载模型
    classifier, label2id, id2label = load_model(model_path, device)
    model = classifier.get_model()
    tokenizer = classifier.get_tokenizer()
    
    logger.info(f"训练集标签数量: {len(label2id)}")
    logger.info(f"标签列表: {list(label2id.keys())[:10]}...")
    
    # 读取测试数据
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    logger.info(f"测试样本数: {len(test_data)}")
    
    # 逐个预测
    predictions = []
    
    for example in tqdm(test_data, desc="推理中"):
        sample_id = example.get('id', len(predictions))
        context = example.get('Context', example.get('context', ''))
        subject = example.get('Subject', example.get('subject', 'unknown'))
        choices = example.get('Choices', example.get('choices', []))
        
        if not choices:
            logger.warning(f"样本 {sample_id} 没有 Choices 字段")
            predicted_label = "neutral"
            confidence = 0.0
        else:
            # 从choices中选择
            predicted_label, confidence, all_scores = predict_from_choices(
                model=model,
                tokenizer=tokenizer,
                context=context,
                subject=subject,
                choices=choices,
                trained_label2id=label2id,
                trained_id2label=id2label,
                device=device,
                max_length=max_length
            )
        
        predictions.append({
            'id': sample_id,
            'predicted_label': predicted_label
        })
    
    # 保存结果为JSONL格式（每行一个JSON）
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"推理完成！")
    logger.info(f"预测样本数: {len(predictions)}")
    logger.info(f"结果已保存到: {output_path}")
    logger.info(f"{'='*60}")
    
    # 打印前几个预测示例
    logger.info("\n预测示例（前5条）:")
    for i, pred in enumerate(predictions[:5]):
        logger.info(f"  {i+1}. ID: {pred['id']}, Label: {pred['predicted_label']}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='情绪分类推理（从Choices中选择答案）')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（例如: checkpoint/classification/best_model）')
    parser.add_argument('--test_data', type=str, default='..classification/data/test.jsonl',
                        help='测试数据路径')
    parser.add_argument('--output_path', type=str, default='..classification/result/Emotion_Classification_Result.jsonl',
                        help='输出文件路径')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)，默认自动选择')
    
    args = parser.parse_args()
    
    # 执行推理
    predict(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_path=args.output_path,
        device=args.device,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
