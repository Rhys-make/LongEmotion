"""
LongEmotion 数据集加载器
从 Hugging Face 加载 LangAGI-Lab/cactus 数据集
"""
from datasets import load_dataset
from typing import List, Dict, Optional
import json
from pathlib import Path


class LongEmotionDatasetLoader:
    """LongEmotion 数据集加载器"""
    
    def __init__(self, dataset_name: str = "LangAGI-Lab/cactus"):
        """
        初始化数据集加载器
        
        Args:
            dataset_name: Hugging Face 数据集名称
        """
        self.dataset_name = dataset_name
    
    def load_dataset(self, split: Optional[str] = None):
        """
        加载数据集
        
        Args:
            split: 数据集分割（train, validation, test, 或 None 表示全部）
            
        Returns:
            数据集对象
        """
        try:
            if split:
                dataset = load_dataset(
                    self.dataset_name,
                    split=split,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    trust_remote_code=True
                )
            return dataset
        except Exception as e:
            raise ValueError(f"无法加载数据集 {self.dataset_name}: {e}")
    
    def convert_to_intake_format(self, dataset, max_samples: Optional[int] = None):
        """
        将数据集转换为输入格式
        
        Args:
            dataset: Hugging Face 数据集
            max_samples: 最大样本数（None表示全部）
            
        Returns:
            转换后的数据列表
        """
        converted_data = []
        
        # 限制样本数
        if max_samples and hasattr(dataset, '__len__'):
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # 遍历数据集
        for idx, item in enumerate(dataset):
            # 获取数据项的所有键
            item_dict = dict(item) if hasattr(item, 'keys') else item
            
            # 提取 ID
            item_id = item_dict.get('id', item_dict.get('ID', idx))
            
            # 提取客户信息和咨询原因
            # 根据数据集的实际结构提取字段
            client_information = (
                item_dict.get('client_information', '') or
                item_dict.get('client_info', '') or
                item_dict.get('Client', '') or
                f"客户ID: {item_id}"
            )
            
            reason_counseling = (
                item_dict.get('reason_counseling', '') or
                item_dict.get('reason', '') or
                item_dict.get('Reason', '') or
                item_dict.get('problem', '') or
                "需要心理咨询和支持"
            )
            
            # 提取 CBT 计划
            cbt_plan = (
                item_dict.get('cbt_plan', '') or
                item_dict.get('CBT_plan', '') or
                item_dict.get('plan', '') or
                "基于认知行为理论，帮助客户识别和改变负面思维模式，建立积极的应对策略"
            )
            
            # 提取对话历史（如果存在）
            history = item_dict.get('history', item_dict.get('conversation_history', ''))
            
            converted_item = {
                "id": item_id,
                "client_information": client_information,
                "reason_counseling": reason_counseling,
                "cbt_plan": cbt_plan,
                "history": history if history else ""
            }
            
            converted_data.append(converted_item)
        
        return converted_data
    
    def save_as_json(self, data: List[Dict], output_path: str):
        """
        将数据保存为 JSON 文件
        
        Args:
            data: 数据列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到: {output_path} ({len(data)} 条记录)")


def load_longemotion_test_set(
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    split: str = "test"
):
    """
    加载测试集并转换为输入格式
    
    Args:
        output_file: 输出文件路径（可选，如果提供则保存为JSON）
        max_samples: 最大样本数
        split: 数据集分割（test, train, validation）
        
    Returns:
        转换后的数据列表
    """
    loader = LongEmotionDatasetLoader()
    
    print(f"正在从 Hugging Face 加载数据集...")
    dataset = loader.load_dataset(split=split)
    
    print(f"数据集加载成功，共 {len(dataset)} 条记录")
    print(f"数据集列名: {dataset.column_names}")
    
    # 显示第一条样本的示例
    if len(dataset) > 0:
        print("\n第一条样本示例:")
        print(dataset[0])
    
    # 转换为输入格式
    print("\n正在转换为输入格式...")
    converted_data = loader.convert_to_intake_format(dataset, max_samples=max_samples)
    
    # 保存文件
    if output_file:
        loader.save_as_json(converted_data, output_file)
    
    return converted_data


if __name__ == "__main__":
    # 示例用法
    data = load_longemotion_test_set(
        output_file="data/longemotion_test.json",
        max_samples=10,  # 只加载前10条作为测试
        split="test"
    )
    
    print(f"\n转换完成，共 {len(data)} 条记录")

