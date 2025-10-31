"""
推理脚本 - Emotion Summary (ES) 任务
对测试集生成情绪总结
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import torch
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    MODEL_CHECKPOINT_DIR,
    TEST_FILE,
    SUBMISSION_FILE,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    NUM_BEAMS,
    LENGTH_PENALTY,
    NO_REPEAT_NGRAM_SIZE,
    DEVICE,
    SUMMARY_ASPECTS,
    T5_PREFIX,
)

from model import EmotionSummaryModel


def load_test_data(test_file: Path) -> List[Dict]:
    """
    加载测试数据
    
    Args:
        test_file: 测试文件路径
    
    Returns:
        测试样本列表
    """
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            samples.append(sample)
    return samples


def parse_multi_aspect_summary(summary_text: str) -> Dict[str, str]:
    """
    解析多方面总结文本
    
    Args:
        summary_text: 生成的总结文本（格式: "aspect1: text1 | aspect2: text2"）
    
    Returns:
        解析后的字典
    """
    result = {}
    
    # 按 | 分割
    parts = summary_text.split(" | ")
    
    for part in parts:
        if ": " in part:
            aspect, text = part.split(": ", 1)
            aspect = aspect.strip()
            if aspect in SUMMARY_ASPECTS:
                result[aspect] = text.strip()
    
    # 确保所有方面都有值
    for aspect in SUMMARY_ASPECTS:
        if aspect not in result:
            result[aspect] = ""
    
    return result


def run_inference(
    model: EmotionSummaryModel,
    test_samples: List[Dict],
    output_file: Path
):
    """
    运行推理
    
    Args:
        model: 模型实例
        test_samples: 测试样本
        output_file: 输出文件路径
    """
    print(f"\n开始推理...")
    print(f"测试样本数: {len(test_samples)}")
    
    results = []
    
    for sample in tqdm(test_samples, desc="生成总结"):
        sample_id = sample.get("id", 0)
        context = sample.get("context", "")
        
        # 方法1: 生成整体总结（然后解析）
        # summary_text = model.generate_summary(
        #     context,
        #     num_beams=NUM_BEAMS,
        #     length_penalty=LENGTH_PENALTY,
        #     no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE
        # )
        # generated_summary = parse_multi_aspect_summary(summary_text)
        
        # 方法2: 分别生成各个方面的总结（推荐）
        generated_summary = model.generate_multi_aspect_summary(
            context,
            aspects=SUMMARY_ASPECTS,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE
        )
        
        # 构建结果
        result = {
            "id": sample_id,
            "generated_summary": generated_summary
        }
        
        results.append(result)
    
    # 保存结果
    print(f"\n保存结果到: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ 推理完成！共生成 {len(results)} 个总结")
    
    # 显示示例
    if results:
        print("\n示例结果:")
        sample_result = results[0]
        print(f"ID: {sample_result['id']}")
        for aspect, summary in sample_result['generated_summary'].items():
            print(f"  {aspect}: {summary[:100]}...")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("Emotion Summary (ES) 推理")
    print("=" * 60)
    
    # 检查模型是否存在
    if not MODEL_CHECKPOINT_DIR.exists():
        print(f"\n错误: 模型目录不存在: {MODEL_CHECKPOINT_DIR}")
        print("请先训练模型或下载预训练模型")
        return
    
    # 加载模型
    print(f"\n加载模型: {MODEL_CHECKPOINT_DIR}")
    model = EmotionSummaryModel()
    model.load_model(MODEL_CHECKPOINT_DIR)
    
    # 加载测试数据
    print(f"\n加载测试数据: {TEST_FILE}")
    
    if not TEST_FILE.exists():
        print(f"错误: 测试文件不存在: {TEST_FILE}")
        print("请先运行 prepare_datasets.py")
        return
    
    test_samples = load_test_data(TEST_FILE)
    
    # 运行推理
    run_inference(model, test_samples, SUBMISSION_FILE)
    
    print("\n" + "=" * 60)
    print("推理完成！")
    print(f"提交文件: {SUBMISSION_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

