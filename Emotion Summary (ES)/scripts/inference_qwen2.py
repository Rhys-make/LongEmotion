"""
Qwen2 推理脚本 - 生成测试集预测
"""

import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from model_qwen2 import Qwen2EmotionSummaryModel


def load_test_data(test_file: Path):
    """加载测试数据"""
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                samples.append(sample)
    return samples


def generate_predictions(
    model,
    test_samples,
    output_file: Path = None
):
    """
    生成预测结果
    
    Args:
        model: Qwen2EmotionSummaryModel 实例
        test_samples: 测试样本列表
        output_file: 输出文件路径
    """
    print("\n开始生成预测...")
    print(f"测试样本数: {len(test_samples)}")
    
    results = []
    
    for sample in tqdm(test_samples, desc="生成中"):
        sample_id = sample['id']
        case_description = sample.get('case_description', [])
        consultation_process = sample.get('consultation_process', [])
        
        try:
            # 生成摘要
            summary = model.generate_summary(
                case_description=case_description,
                consultation_process=consultation_process,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # 保存结果
            result = {
                'id': sample_id,
                'experience_and_reflection': summary
            }
            
            # 如果有参考答案，也保存下来用于对比
            if 'experience_and_reflection' in sample:
                result['reference'] = sample['experience_and_reflection']
            
            results.append(result)
            
        except Exception as e:
            print(f"\n错误: 样本 {sample_id} 生成失败: {e}")
            # 添加空结果
            results.append({
                'id': sample_id,
                'experience_and_reflection': '',
                'error': str(e)
            })
    
    # 保存结果
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\n✓ 预测结果已保存到: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen2 模型推理")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../model/qwen2_emotion_summary/final",
        help="模型路径（LoRA权重）"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2-7B-Instruct",
        help="基础模型名称"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../data/test/Emotion_Summary.jsonl",
        help="测试数据文件"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../submission/predictions.jsonl",
        help="输出文件"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="最大输入长度"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="使用4-bit量化"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Qwen2 情绪总结推理")
    print("=" * 70)
    print(f"基础模型: {args.base_model}")
    print(f"LoRA权重: {args.model_path}")
    print(f"测试文件: {args.test_file}")
    print(f"输出文件: {args.output_file}")
    print()
    
    # 检查文件存在
    test_file = Path(args.test_file)
    if not test_file.exists():
        print(f"错误: 测试文件不存在: {test_file}")
        return
    
    # 加载模型
    print("步骤 1/3: 加载模型")
    print("-" * 70)
    model = Qwen2EmotionSummaryModel(
        model_name=args.base_model,
        max_input_length=args.max_length,
        use_4bit=args.use_4bit,
        use_lora=True
    )
    
    # 加载 LoRA 权重
    lora_path = Path(args.model_path)
    if lora_path.exists():
        print(f"\n加载 LoRA 权重: {lora_path}")
        model.load_lora_weights(lora_path)
    else:
        print(f"\n警告: LoRA 权重不存在: {lora_path}")
        print("将使用未微调的基础模型")
    
    # 加载测试数据
    print("\n步骤 2/3: 加载测试数据")
    print("-" * 70)
    test_samples = load_test_data(test_file)
    print(f"加载了 {len(test_samples)} 个测试样本")
    
    # 生成预测
    print("\n步骤 3/3: 生成预测")
    print("-" * 70)
    results = generate_predictions(
        model=model,
        test_samples=test_samples,
        output_file=Path(args.output_file)
    )
    
    # 显示示例
    print("\n" + "=" * 70)
    print("示例预测 (第1个样本):")
    print("=" * 70)
    if results:
        example = results[0]
        print(f"ID: {example['id']}")
        print(f"\n生成的摘要:")
        print(example['experience_and_reflection'][:500] + "...")
        
        if 'reference' in example:
            print(f"\n参考摘要:")
            print(example['reference'][:500] + "...")
    
    print("\n" + "=" * 70)
    print("✓ 推理完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

