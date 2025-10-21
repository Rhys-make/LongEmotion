"""
数据集下载脚本
从 Hugging Face 下载 LongEmotion 数据集
"""
import os
import json
import shutil
from pathlib import Path
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download


def download_dataset(
    dataset_name: str = "LongEmotion/LongEmotion",
    cache_dir: str = "./data",
    tasks: list = None
):
    """
    下载数据集（更新版）
    
    Args:
        dataset_name: 数据集名称
        cache_dir: 缓存目录
        tasks: 要下载的任务列表，None 表示下载全部
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载数据集: {dataset_name}")
    print(f"缓存目录: {cache_dir}")
    
    # 任务到文件的映射
    task_file_mapping = {
        "classification": "Emotion Classification/Emotion_Classification.jsonl",
        "detection": "Emotion Detection/Emotion_Detection.jsonl",
        "conversation": "Emotion Conversation/Conversations_Long.jsonl",
        "summary": "Emotion Summary/Emotion_Summary.jsonl",
        "qa": "Emotion QA/Emotion QA.jsonl"
    }
    
    all_tasks = list(task_file_mapping.keys())
    if tasks is None:
        tasks = all_tasks

    success_count = 0
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"下载任务: {task}")
        print(f"{'='*60}")
        
        try:
            file_path = task_file_mapping.get(task)
            if not file_path:
                print(f"✗ 未知任务: {task}")
                continue
            
            # 每个任务的目标目录
            task_dir = cache_path / task
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # 下载文件到本地缓存
            print(f"下载文件: {file_path}")
            local_file = hf_hub_download(
                repo_id=dataset_name,
                filename=file_path,
                repo_type="dataset"
            )
            
            # 移动文件到目标任务目录
            dest_file = task_dir / Path(file_path).name
            shutil.copy(local_file, dest_file)
            print(f"✓ 文件保存到 {dest_file}")
            
            # 读取 JSONL 数据
            data_items = []
            with open(dest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data_items.append(json.loads(line))
            
            print(f"✓ 读取到 {len(data_items)} 条数据")
            
            if not data_items:
                print(f"✗ 数据文件为空")
                continue

            print(f"  - 字段示例: {list(data_items[0].keys())}")
            
            # 创建 Dataset 对象
            dataset = Dataset.from_list(data_items)
            
            # 简单划分 8:1:1
            total = len(dataset)
            train_size = int(0.8 * total)
            val_size = int(0.1 * total)
            
            train_dataset = dataset.select(range(train_size))
            val_dataset = dataset.select(range(train_size, train_size + val_size))
            test_dataset = dataset.select(range(train_size + val_size, total))
            
            # 保存数据
            train_dataset.save_to_disk(str(task_dir / "train"))
            val_dataset.save_to_disk(str(task_dir / "validation"))
            test_dataset.save_to_disk(str(task_dir / "test"))
            
            print(f"✓ {task} 数据集保存完成")
            print(f"  - train: {len(train_dataset)} 条")
            print(f"  - val:   {len(val_dataset)} 条")
            print(f"  - test:  {len(test_dataset)} 条")
            
            success_count += 1
        
        except Exception as e:
            print(f"✗ {task} 下载失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"数据集下载完成！成功: {success_count}/{len(tasks)}")
    print(f"{'='*60}")


# 其余 list_downloaded_datasets 与 verify_dataset 保持原样即可


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 LongEmotion 数据集")
    parser.add_argument("--dataset", type=str, default="LongEmotion/LongEmotion", help="数据集名称")
    parser.add_argument("--cache_dir", type=str, default="./data", help="缓存目录")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="要下载的任务 (classification detection conversation summary qa)")
    parser.add_argument("--list", action="store_true", help="列出已下载的数据集")
    parser.add_argument("--verify", action="store_true", help="验证数据集完整性")
    
    args = parser.parse_args()
    
    if args.list:
        from main import list_downloaded_datasets
        list_downloaded_datasets(args.cache_dir)
    elif args.verify:
        from main import verify_dataset
        verify_dataset(args.cache_dir)
    else:
        download_dataset(args.dataset, args.cache_dir, args.tasks)