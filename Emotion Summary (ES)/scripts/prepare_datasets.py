"""
数据准备脚本 - Emotion Summary (ES) 任务
下载并处理LongEmotion的ES任务数据集
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    DATA_DIR, 
    DATASET_NAME, 
    DATASET_SUBSET,
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    SUMMARY_ASPECTS
)


def download_longemotion_es_data():
    """
    从Hugging Face下载LongEmotion的ES任务数据集
    """
    try:
        from datasets import load_dataset
        
        print(f"正在下载数据集: {DATASET_NAME}/{DATASET_SUBSET}")
        print("这可能需要几分钟时间...")
        
        # 下载数据集
        dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)
        
        print(f"数据集下载完成！")
        print(f"训练集: {len(dataset.get('train', []))} 样本")
        print(f"验证集: {len(dataset.get('validation', []))} 样本")
        print(f"测试集: {len(dataset.get('test', []))} 样本")
        
        return dataset
        
    except Exception as e:
        print(f"下载数据集失败: {e}")
        print("请检查：")
        print("1. 网络连接是否正常")
        print("2. 数据集名称是否正确")
        print("3. 是否有权限访问该数据集")
        return None


def process_and_save_data(dataset):
    """
    处理并保存数据集为JSONL格式
    
    Args:
        dataset: Hugging Face数据集对象
    """
    if dataset is None:
        print("数据集为空，跳过处理")
        return
    
    # 确保数据目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理各个分割
    splits = {
        'train': (dataset.get('train'), TRAIN_FILE),
        'validation': (dataset.get('validation'), VALIDATION_FILE),
        'test': (dataset.get('test'), TEST_FILE)
    }
    
    for split_name, (split_data, output_file) in splits.items():
        if split_data is None:
            print(f"跳过 {split_name} 分割（不存在）")
            continue
            
        print(f"\n处理 {split_name} 数据集...")
        processed_samples = []
        
        for idx, sample in enumerate(tqdm(split_data, desc=f"处理{split_name}")):
            # 构建处理后的样本
            processed_sample = {
                "id": sample.get("id", idx),
                "context": sample.get("context", sample.get("text", "")),
            }
            
            # 如果有参考答案（训练集和验证集）
            if "summary" in sample or any(aspect in sample for aspect in SUMMARY_ASPECTS):
                reference_summary = {}
                
                # 尝试不同的字段名称
                if "summary" in sample and isinstance(sample["summary"], dict):
                    reference_summary = sample["summary"]
                else:
                    for aspect in SUMMARY_ASPECTS:
                        if aspect in sample:
                            reference_summary[aspect] = sample[aspect]
                
                if reference_summary:
                    processed_sample["reference_summary"] = reference_summary
            
            processed_samples.append(processed_sample)
        
        # 保存为JSONL
        print(f"保存到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in processed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✓ {split_name} 数据集已保存: {len(processed_samples)} 样本")


def verify_data_format():
    """
    验证数据格式是否正确
    """
    print("\n验证数据格式...")
    
    for file_path, split_name in [
        (TRAIN_FILE, "训练集"),
        (VALIDATION_FILE, "验证集"),
        (TEST_FILE, "测试集")
    ]:
        if not file_path.exists():
            print(f"⚠ {split_name} 文件不存在: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\n{split_name}: {len(lines)} 样本")
            
            # 显示第一个样本
            if lines:
                first_sample = json.loads(lines[0])
                print(f"示例样本结构:")
                print(f"  - id: {first_sample.get('id')}")
                print(f"  - context长度: {len(first_sample.get('context', ''))} 字符")
                if "reference_summary" in first_sample:
                    print(f"  - 参考总结方面: {list(first_sample['reference_summary'].keys())}")


def create_dummy_data():
    """
    创建示例数据（用于测试）
    当无法下载真实数据时使用
    """
    print("\n创建示例数据...")
    
    dummy_samples = [
        {
            "id": 0,
            "context": "这是一个示例心理病理报告。患者因工作压力过大导致焦虑症状...",
            "reference_summary": {
                "causes": "工作压力过大",
                "symptoms": "焦虑、失眠、注意力不集中",
                "treatment_process": "认知行为疗法，每周一次，持续3个月",
                "illness_characteristics": "广泛性焦虑障碍",
                "treatment_effects": "症状明显改善，能够正常工作"
            }
        }
    ]
    
    # 保存示例数据
    for file_path in [TRAIN_FILE, VALIDATION_FILE, TEST_FILE]:
        with open(file_path, 'w', encoding='utf-8') as f:
            for sample in dummy_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"✓ 已创建示例数据: {file_path}")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("Emotion Summary (ES) 数据准备")
    print("=" * 60)
    
    # 尝试下载数据集
    dataset = download_longemotion_es_data()
    
    if dataset is not None:
        # 处理并保存数据
        process_and_save_data(dataset)
        
        # 验证数据格式
        verify_data_format()
    else:
        print("\n数据集下载失败，是否创建示例数据用于测试？(y/n)")
        # 自动创建示例数据
        create_dummy_data()
        print("\n注意: 当前使用的是示例数据，请稍后替换为真实数据集")
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

