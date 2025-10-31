"""
评估脚本 - Emotion Summary (ES) 任务
评估生成的总结质量
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    VALIDATION_FILE,
    SUBMISSION_FILE,
    SUMMARY_ASPECTS,
    USE_GPT4O_EVAL,
    GPT4O_API_KEY,
)


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict:
    """
    计算ROUGE分数
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
    
    Returns:
        ROUGE分数字典
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            for key in scores.keys():
                scores[key].append(score[key].fmeasure)
        
        # 计算平均值
        avg_scores = {
            key: np.mean(values) for key, values in scores.items()
        }
        
        return avg_scores
        
    except ImportError:
        print("警告: rouge_score库未安装，跳过ROUGE评估")
        print("可以运行: pip install rouge-score")
        return {}


def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    计算BLEU分数
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
    
    Returns:
        BLEU分数
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smooth = SmoothingFunction()
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                smoothing_function=smooth.method1
            )
            bleu_scores.append(score)
        
        return np.mean(bleu_scores)
        
    except ImportError:
        print("警告: nltk库未安装，跳过BLEU评估")
        print("可以运行: pip install nltk")
        return 0.0


def evaluate_gpt4o(predictions: Dict, references: Dict, api_key: str) -> Dict:
    """
    使用GPT-4o评估（可选）
    
    Args:
        predictions: 预测总结
        references: 参考总结
        api_key: OpenAI API密钥
    
    Returns:
        评估分数
    """
    if not api_key:
        print("警告: 未设置GPT-4o API密钥，跳过GPT-4o评估")
        return {}
    
    try:
        import openai
        
        # TODO: 实现GPT-4o评估逻辑
        print("GPT-4o评估功能暂未实现")
        return {}
        
    except ImportError:
        print("警告: openai库未安装，跳过GPT-4o评估")
        return {}


def evaluate_multi_aspect_summaries(
    generated_file: Path,
    reference_file: Path
) -> Dict:
    """
    评估多方面总结
    
    Args:
        generated_file: 生成的总结文件
        reference_file: 参考总结文件
    
    Returns:
        评估结果
    """
    print("=" * 60)
    print("Emotion Summary (ES) 评估")
    print("=" * 60)
    
    # 加载数据
    print(f"\n加载生成结果: {generated_file}")
    generated_data = load_jsonl(generated_file)
    
    print(f"加载参考答案: {reference_file}")
    reference_data = load_jsonl(reference_file)
    
    # 确保数据对齐
    assert len(generated_data) == len(reference_data), "数据长度不匹配"
    
    # 按方面评估
    results = {}
    
    for aspect in SUMMARY_ASPECTS:
        print(f"\n评估方面: {aspect}")
        
        predictions = []
        references = []
        
        for gen, ref in zip(generated_data, reference_data):
            # 提取该方面的总结
            gen_summary = gen.get("generated_summary", {}).get(aspect, "")
            ref_summary = ref.get("reference_summary", {}).get(aspect, "")
            
            predictions.append(gen_summary)
            references.append(ref_summary)
        
        # 计算ROUGE
        rouge_scores = calculate_rouge_scores(predictions, references)
        
        # 计算BLEU
        bleu_score = calculate_bleu_score(predictions, references)
        
        results[aspect] = {
            "rouge": rouge_scores,
            "bleu": bleu_score
        }
        
        # 打印结果
        print(f"  ROUGE-1: {rouge_scores.get('rouge1', 0):.4f}")
        print(f"  ROUGE-2: {rouge_scores.get('rouge2', 0):.4f}")
        print(f"  ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}")
        print(f"  BLEU: {bleu_score:.4f}")
    
    # 计算平均分数
    print("\n" + "=" * 60)
    print("平均分数:")
    print("=" * 60)
    
    avg_rouge1 = np.mean([
        results[aspect]["rouge"].get("rouge1", 0)
        for aspect in SUMMARY_ASPECTS
    ])
    avg_rouge2 = np.mean([
        results[aspect]["rouge"].get("rouge2", 0)
        for aspect in SUMMARY_ASPECTS
    ])
    avg_rougeL = np.mean([
        results[aspect]["rouge"].get("rougeL", 0)
        for aspect in SUMMARY_ASPECTS
    ])
    avg_bleu = np.mean([
        results[aspect]["bleu"]
        for aspect in SUMMARY_ASPECTS
    ])
    
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    print(f"BLEU: {avg_bleu:.4f}")
    
    # GPT-4o评估（如果启用）
    if USE_GPT4O_EVAL:
        print("\n" + "=" * 60)
        print("GPT-4o评估:")
        print("=" * 60)
        gpt4o_scores = evaluate_gpt4o(generated_data, reference_data, GPT4O_API_KEY)
    
    return results


def main():
    """
    主函数
    """
    # 检查文件
    if not SUBMISSION_FILE.exists():
        print(f"错误: 生成文件不存在: {SUBMISSION_FILE}")
        print("请先运行 inference.py")
        return
    
    if not VALIDATION_FILE.exists():
        print(f"错误: 验证文件不存在: {VALIDATION_FILE}")
        print("请先运行 prepare_datasets.py")
        return
    
    # 评估
    results = evaluate_multi_aspect_summaries(
        SUBMISSION_FILE,
        VALIDATION_FILE
    )
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

