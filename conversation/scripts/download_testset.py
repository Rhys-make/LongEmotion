"""
下载 LongEmotion 测试集
从 Hugging Face 下载 LongEmotion/LongEmotion 数据集的 emotion_conversation 子集
"""
import argparse
import json
import sys
from pathlib import Path
from datasets import load_dataset

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_emotion_conversation_dataset(
    output_file: str = "data/longemotion_testset.json",
    split: str = "default",
    max_samples: int = None
):
    """
    下载 emotion_conversation 数据集
    
    Args:
        output_file: 输出文件路径
        split: 数据集分割（default, train, test, validation）
        max_samples: 最大样本数（None表示全部）
    """
    print("=" * 60)
    print("LongEmotion 测试集下载器")
    print("=" * 60)
    print(f"数据集: LongEmotion/LongEmotion")
    print(f"子集: emotion_conversation")
    print(f"分割: {split}")
    print("=" * 60)
    
    # 加载数据集
    print("\n正在从 Hugging Face 下载数据集...")
    try:
        # 尝试使用 subset 参数
        try:
            if split and split != "default":
                dataset = load_dataset(
                    "LongEmotion/LongEmotion",
                    subset="emotion_conversation",
                    split=split,
                    trust_remote_code=True
                )
            else:
                # 如果是 default，尝试加载整个数据集
                dataset_dict = load_dataset(
                    "LongEmotion/LongEmotion",
                    subset="emotion_conversation",
                    trust_remote_code=True
                )
                # 如果有多个分割，使用 default 或第一个
                if isinstance(dataset_dict, dict):
                    if "default" in dataset_dict:
                        dataset = dataset_dict["default"]
                    elif "train" in dataset_dict:
                        dataset = dataset_dict["train"]
                    else:
                        dataset = list(dataset_dict.values())[0]
                else:
                    dataset = dataset_dict
        except Exception as e1:
            print(f"使用 subset 参数加载失败: {e1}")
            print("尝试直接加载数据集...")
            # 尝试直接加载
            dataset_dict = load_dataset(
                "LongEmotion/LongEmotion",
                split=split if split != "default" else None,
                trust_remote_code=True
            )
            if isinstance(dataset_dict, dict):
                # 如果有多个分割，尝试找到 emotion_conversation
                if "emotion_conversation" in dataset_dict:
                    dataset = dataset_dict["emotion_conversation"]
                elif "default" in dataset_dict:
                    dataset = dataset_dict["default"]
                else:
                    dataset = list(dataset_dict.values())[0]
            else:
                dataset = dataset_dict
        
        print(f"✅ 数据集加载成功")
        print(f"数据集列名: {dataset.column_names}")
        print(f"数据集大小: {len(dataset)} 条记录")
        
        # 限制样本数
        if max_samples and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"限制为前 {len(dataset)} 条记录")
        
        # 显示第一条样本的结构
        if len(dataset) > 0:
            print("\n第一条样本结构:")
            sample = dataset[0]
            print(f"  ID: {sample.get('id', 'N/A')}")
            print(f"  conversation_history 长度: {len(str(sample.get('conversation_history', '')))} 字符")
            print(f"  所有字段: {list(sample.keys())}")
        
        # 转换为列表格式
        print("\n正在转换数据格式...")
        data_list = []
        for idx, item in enumerate(dataset):
            data_item = {
                "id": item.get('id', idx),
                "conversation_history": item.get('conversation_history', '')
            }
            data_list.append(data_item)
        
        # 保存为 JSON 文件
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n正在保存到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 下载完成！")
        print(f"📁 保存位置: {output_path}")
        print(f"📊 总记录数: {len(data_list)}")
        
        return data_list
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="下载 LongEmotion 测试集")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/longemotion_testset.json",
        help="输出文件路径（默认: data/longemotion_testset.json）"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="default",
        help="数据集分割（default, train, test, validation，默认: default）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数（可选，None表示全部）"
    )
    
    args = parser.parse_args()
    
    download_emotion_conversation_dataset(
        output_file=args.output_file,
        split=args.split,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

