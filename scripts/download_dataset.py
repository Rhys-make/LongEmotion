"""
数据集下载脚本
从 Hugging Face 下载 LongEmotion 数据集
"""
import os
from pathlib import Path
from datasets import load_dataset


def download_dataset(
    dataset_name: str = "LongEmotion/LongEmotion",
    cache_dir: str = "./data",
    tasks: list = None
):
    """
    下载数据集
    
    Args:
        dataset_name: 数据集名称
        cache_dir: 缓存目录
        tasks: 要下载的任务列表，None 表示下载全部
    """
    # 创建缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载数据集: {dataset_name}")
    print(f"缓存目录: {cache_dir}")
    
    # 定义所有任务
    all_tasks = [
        "classification",
        "detection",
        "conversation",
        "summary",
        "qa"
    ]
    
    if tasks is None:
        tasks = all_tasks
    
    # 下载每个任务的数据
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"下载任务: {task}")
        print(f"{'='*60}")
        
        try:
            # 尝试加载数据集的特定配置
            # 注意：实际配置名称可能需要根据数据集结构调整
            dataset = load_dataset(
                dataset_name,
                name=task,  # 配置名称
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            print(f"✓ {task} 任务数据集下载成功！")
            print(f"  - 包含分割: {list(dataset.keys())}")
            
            # 显示数据集信息
            for split in dataset.keys():
                print(f"  - {split}: {len(dataset[split])} 条样本")
                
                # 显示一个样本
                if len(dataset[split]) > 0:
                    print(f"  - 样本示例: {list(dataset[split][0].keys())}")
            
            # 保存数据集到本地
            task_dir = cache_path / task
            task_dir.mkdir(parents=True, exist_ok=True)
            
            for split in dataset.keys():
                dataset[split].save_to_disk(str(task_dir / split))
            
            print(f"✓ {task} 数据集已保存到 {task_dir}")
            
        except Exception as e:
            print(f"✗ {task} 任务下载失败: {e}")
            print(f"尝试使用默认配置下载...")
            
            try:
                # 如果特定配置失败，尝试加载默认配置
                dataset = load_dataset(
                    dataset_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                
                print(f"✓ 使用默认配置下载成功")
                print(f"  - 可用配置: {dataset.keys()}")
                
            except Exception as e2:
                print(f"✗ 默认配置也失败: {e2}")
                print("请检查数据集名称和网络连接")
    
    print(f"\n{'='*60}")
    print("数据集下载完成！")
    print(f"{'='*60}")


def list_downloaded_datasets(cache_dir: str = "./data"):
    """
    列出已下载的数据集
    
    Args:
        cache_dir: 缓存目录
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"缓存目录不存在: {cache_dir}")
        return
    
    print(f"\n已下载的数据集 (位于 {cache_dir}):")
    print(f"{'='*60}")
    
    for task_dir in cache_path.iterdir():
        if task_dir.is_dir():
            print(f"\n任务: {task_dir.name}")
            
            for split_dir in task_dir.iterdir():
                if split_dir.is_dir():
                    # 尝试加载数据集信息
                    try:
                        from datasets import load_from_disk
                        dataset = load_from_disk(str(split_dir))
                        print(f"  - {split_dir.name}: {len(dataset)} 条样本")
                    except:
                        print(f"  - {split_dir.name}: (无法读取)")


def verify_dataset(cache_dir: str = "./data"):
    """
    验证数据集完整性
    
    Args:
        cache_dir: 缓存目录
    """
    from datasets import load_from_disk
    
    cache_path = Path(cache_dir)
    tasks = ["classification", "detection", "conversation", "summary", "qa"]
    splits = ["train", "validation", "test"]
    
    print(f"\n验证数据集完整性...")
    print(f"{'='*60}")
    
    all_valid = True
    
    for task in tasks:
        print(f"\n任务: {task}")
        task_valid = True
        
        for split in splits:
            split_path = cache_path / task / split
            
            if split_path.exists():
                try:
                    dataset = load_from_disk(str(split_path))
                    print(f"  ✓ {split}: {len(dataset)} 条样本")
                except Exception as e:
                    print(f"  ✗ {split}: 加载失败 ({e})")
                    task_valid = False
            else:
                print(f"  ✗ {split}: 不存在")
                task_valid = False
        
        if not task_valid:
            all_valid = False
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✓ 所有数据集验证通过！")
    else:
        print("✗ 部分数据集缺失或损坏")
    
    return all_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 LongEmotion 数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        default="LongEmotion/LongEmotion",
        help="数据集名称"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data",
        help="缓存目录"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="要下载的任务 (classification detection conversation summary qa)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出已下载的数据集"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证数据集完整性"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_downloaded_datasets(args.cache_dir)
    elif args.verify:
        verify_dataset(args.cache_dir)
    else:
        download_dataset(
            dataset_name=args.dataset,
            cache_dir=args.cache_dir,
            tasks=args.tasks
        )

