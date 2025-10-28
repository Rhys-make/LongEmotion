"""
最终推理脚本 - 使用已训练好的模型
"""
import sys
from pathlib import Path

# 添加scripts目录到路径，以便导入inference_longemotion
sys.path.append(str(Path(__file__).parent))

from inference_longemotion import LongEmotionInference

def main():
    print("="*80)
    print("LongEmotion 测试集推理 - 使用已训练模型")
    print("="*80)
    
    # 配置 - 使用Detection文件夹内的相对路径
    # 获取Detection文件夹的根目录
    detection_root = Path(__file__).parent.parent
    model_path = detection_root / "model" / "best_model.pt"
    test_file = detection_root / "test_data" / "test.jsonl"
    output_file = detection_root / "submission" / "predictions.jsonl"
    output_detailed = detection_root / "submission" / "predictions_detailed.json"
    
    print(f"\n模型: {model_path}")
    print(f"测试集: {test_file} (136个样本)")
    print(f"输出: {output_file}")
    
    # 创建推理器
    print("\n[1/2] 加载模型...")
    inference = LongEmotionInference(
        model_path=str(model_path),
        device="cpu",  # 使用CPU，更稳定
        max_length=512,
        batch_size=16
    )
    
    # 执行推理
    print("\n[2/2] 开始推理...")
    inference.inference_longemotion_test(
        test_file=str(test_file),
        output_file=str(output_file),
        output_detailed=str(output_detailed)
    )
    
    print("\n" + "="*80)
    print("[完成] 推理结束！")
    print("="*80)

if __name__ == "__main__":
    main()

