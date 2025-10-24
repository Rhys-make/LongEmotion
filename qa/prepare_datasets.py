#!/usr/bin/env python3
"""
QA 数据集准备和预处理脚本

功能：
1. 下载 LongEmotion 的 QA 测试集
2. 下载并整合其他长上下文 QA 数据集（NarrativeQA、HotpotQA 等）
3. 统一格式并保存为 JSONL
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm import tqdm
import re

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """清洗文本"""
    if not text:
        return ""
    
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留基本标点）
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    
    return text.strip()


def truncate_context(text: str, max_tokens: int = 16000) -> str:
    """
    截断上下文到指定长度
    注意：这里简单按字符数估算，实际应该用 tokenizer
    """
    # 粗略估算：1 token ≈ 4 字符
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def download_longemtion_qa_test(output_path: Path) -> int:
    """
    下载 LongEmotion 的 QA 测试集
    
    Returns:
        样本数量
    """
    logger.info("下载 LongEmotion QA 测试集...")
    
    try:
        # 从 HuggingFace 加载数据集
        dataset = load_dataset("LongEmotion/LongEmotion", "qa", split="test")
        
        # 转换格式并保存
        samples = []
        for idx, item in enumerate(tqdm(dataset, desc="处理测试集")):
            sample = {
                "id": idx,
                "problem": item.get("problem", ""),
                "context": clean_text(item.get("context", "")),
                "answer": item.get("answer", "")
            }
            samples.append(sample)
        
        # 保存为 JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ 测试集已保存：{output_path}（{len(samples)} 条）")
        return len(samples)
    
    except Exception as e:
        logger.error(f"下载失败：{e}")
        logger.info("将创建一个示例测试集...")
        
        # 创建示例数据
        samples = [
            {
                "id": 0,
                "problem": "根据上述心理学文献，什么是认知失调理论？",
                "context": "认知失调理论（Cognitive Dissonance Theory）是由心理学家莱昂·费斯廷格于1957年提出的。该理论认为，当个体同时持有两个或多个相互矛盾的认知（如信念、态度、价值观等）时，会产生一种不舒适的心理状态，称为认知失调。为了减少这种不适，个体会采取各种策略来协调这些矛盾的认知...",
                "answer": "认知失调理论是指当个体同时持有矛盾的认知时产生的不舒适心理状态，个体会采取策略来减少这种不适。"
            }
        ]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        return len(samples)


def convert_narrativeqa(max_samples: int = 5000) -> List[Dict]:
    """
    转换 NarrativeQA 数据集
    
    NarrativeQA 是一个基于故事和书籍的阅读理解数据集
    """
    logger.info("下载 NarrativeQA 数据集...")
    
    try:
        dataset = load_dataset("narrativeqa", split="train")
        
        samples = []
        for idx, item in enumerate(tqdm(dataset, desc="处理 NarrativeQA")):
            if idx >= max_samples:
                break
            
            # 获取文档内容（可能很长）
            context = item.get("document", {}).get("text", "")
            if not context:
                continue
            
            # 获取问题和答案
            question = item.get("question", {}).get("text", "")
            answers = item.get("answers", [])
            if not question or not answers:
                continue
            
            # 使用第一个答案
            answer = answers[0].get("text", "") if isinstance(answers[0], dict) else answers[0]
            
            sample = {
                "problem": clean_text(question),
                "context": truncate_context(clean_text(context)),
                "answer": clean_text(answer)
            }
            samples.append(sample)
        
        logger.info(f"✅ NarrativeQA: {len(samples)} 条")
        return samples
    
    except Exception as e:
        logger.warning(f"NarrativeQA 下载失败：{e}")
        return []


def convert_squad(max_samples: int = 10000) -> List[Dict]:
    """
    转换 SQuAD 数据集（虽然不是长上下文，但质量高）
    """
    logger.info("下载 SQuAD v2 数据集...")
    
    try:
        dataset = load_dataset("squad_v2", split="train")
        
        samples = []
        for idx, item in enumerate(tqdm(dataset, desc="处理 SQuAD")):
            if idx >= max_samples:
                break
            
            context = item.get("context", "")
            question = item.get("question", "")
            answers = item.get("answers", {}).get("text", [])
            
            if not context or not question or not answers:
                continue
            
            sample = {
                "problem": clean_text(question),
                "context": clean_text(context),
                "answer": clean_text(answers[0])
            }
            samples.append(sample)
        
        logger.info(f"✅ SQuAD: {len(samples)} 条")
        return samples
    
    except Exception as e:
        logger.warning(f"SQuAD 下载失败：{e}")
        return []


def convert_hotpotqa(max_samples: int = 5000) -> List[Dict]:
    """
    转换 HotpotQA 数据集（多跳推理）
    """
    logger.info("下载 HotpotQA 数据集...")
    
    try:
        dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
        
        samples = []
        for idx, item in enumerate(tqdm(dataset, desc="处理 HotpotQA")):
            if idx >= max_samples:
                break
            
            # 合并所有支持段落作为上下文
            contexts = item.get("context", {}).get("sentences", [])
            if not contexts:
                continue
            
            # 拼接所有段落
            full_context = " ".join([" ".join(sents) for sents in contexts if sents])
            
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not question or not answer:
                continue
            
            sample = {
                "problem": clean_text(question),
                "context": truncate_context(clean_text(full_context)),
                "answer": clean_text(answer)
            }
            samples.append(sample)
        
        logger.info(f"✅ HotpotQA: {len(samples)} 条")
        return samples
    
    except Exception as e:
        logger.warning(f"HotpotQA 下载失败：{e}")
        return []


def create_synthetic_psychology_qa(num_samples: int = 1000) -> List[Dict]:
    """
    创建合成的心理学问答数据
    
    注意：这是简化版本，实际应该使用 LLM 生成更真实的数据
    """
    logger.info("生成合成心理学问答数据...")
    
    # 心理学主题模板
    topics = [
        ("认知心理学", "认知失调、记忆、注意力、决策"),
        ("发展心理学", "儿童发展、青少年心理、成人发展"),
        ("社会心理学", "从众行为、群体动力学、态度改变"),
        ("临床心理学", "抑郁症、焦虑症、治疗方法"),
        ("积极心理学", "幸福感、心流体验、韧性"),
    ]
    
    samples = []
    for i in range(num_samples):
        topic, keywords = topics[i % len(topics)]
        
        sample = {
            "problem": f"请解释{topic}中的核心概念。",
            "context": f"{topic}是心理学的重要分支，主要研究{keywords}等方面。该领域的研究对理解人类行为和心理过程具有重要意义。",
            "answer": f"{topic}主要关注{keywords}等核心概念的研究和应用。"
        }
        samples.append(sample)
    
    logger.info(f"✅ 合成数据: {len(samples)} 条")
    return samples


def split_train_val(samples: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """
    划分训练集和验证集
    """
    import random
    random.shuffle(samples)
    
    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    return train_samples, val_samples


def save_jsonl(samples: List[Dict], output_path: Path):
    """
    保存为 JSONL 格式
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加 ID
    for idx, sample in enumerate(samples):
        if "id" not in sample:
            sample["id"] = idx
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ 已保存：{output_path}（{len(samples)} 条）")


def main():
    parser = argparse.ArgumentParser(description='准备 QA 数据集')
    
    parser.add_argument('--output_dir', type=str, default='data/qa',
                        help='输出目录')
    parser.add_argument('--max_samples_per_dataset', type=int, default=5000,
                        help='每个数据集的最大样本数')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='是否使用合成心理学数据')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['squad', 'narrativeqa', 'hotpotqa'],
                        help='要使用的数据集列表')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 下载 LongEmotion 测试集
    test_path = output_dir / 'test.jsonl'
    download_longemtion_qa_test(test_path)
    
    # 2. 收集训练数据
    all_samples = []
    
    if 'squad' in args.datasets:
        all_samples.extend(convert_squad(args.max_samples_per_dataset))
    
    if 'narrativeqa' in args.datasets:
        all_samples.extend(convert_narrativeqa(args.max_samples_per_dataset))
    
    if 'hotpotqa' in args.datasets:
        all_samples.extend(convert_hotpotqa(args.max_samples_per_dataset))
    
    if args.use_synthetic:
        all_samples.extend(create_synthetic_psychology_qa(1000))
    
    logger.info(f"\n总样本数：{len(all_samples)}")
    
    # 3. 划分训练集和验证集
    train_samples, val_samples = split_train_val(all_samples, args.val_ratio)
    
    # 4. 保存
    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'validation.jsonl'
    
    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)
    
    # 5. 统计信息
    logger.info("\n" + "="*60)
    logger.info("数据准备完成！")
    logger.info(f"训练集：{train_path}（{len(train_samples)} 条）")
    logger.info(f"验证集：{val_path}（{len(val_samples)} 条）")
    logger.info(f"测试集：{test_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

