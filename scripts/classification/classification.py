#!/usr/bin/env python3
"""
情绪分类模型和数据集定义
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    """情绪分类数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        label2id: Optional[Dict[str, int]] = None,
        is_test: bool = False
    ):
        """
        初始化数据集
        
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            label2id: 标签到ID的映射（训练时自动构建，验证/测试时传入）
            is_test: 是否为测试集（测试集没有 Answer 字段）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        # 读取数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"加载了 {len(self.data)} 条数据从 {data_path}")
        
        # 构建或使用标签映射
        if label2id is None:
            # 自动从数据中提取所有唯一标签（用于训练集）
            all_labels = set()
            for item in self.data:
                if 'Choices' in item:
                    all_labels.update(item['Choices'])
                if 'Answer' in item:
                    all_labels.add(item['Answer'])
            
            self.label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
            logger.info(f"构建标签映射: {len(self.label2id)} 个类别")
        else:
            self.label2id = label2id
        
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        subject = item.get('Subject', 'unknown')
        context = item.get('Context', '')
        choices = item.get('Choices', [])
        
        # 格式化输入：[CLS] Subject: {Subject}. Context: {Context}. Choices: {Choices}
        choices_str = ', '.join(choices) if choices else ''
        input_text = f"Subject: {subject}. Context: {context}. Choices: {choices_str}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 准备返回数据
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'id': item.get('id', idx)
        }
        
        # 如果不是测试集，添加标签
        if not self.is_test and 'Answer' in item:
            answer = item['Answer']
            if answer in self.label2id:
                result['labels'] = torch.tensor(self.label2id[answer], dtype=torch.long)
            else:
                # 如果标签不在映射中，跳过（或使用默认值）
                logger.warning(f"标签 '{answer}' 不在 label2id 中，使用 0 作为默认值")
                result['labels'] = torch.tensor(0, dtype=torch.long)
        
        return result


class EmotionClassifier:
    """情绪分类模型封装"""
    
    def __init__(
        self,
        model_name: str = "roberta-large",
        num_labels: int = None,
        label2id: Dict[str, int] = None,
        id2label: Dict[int, str] = None,
        device: str = None
    ):
        """
        初始化模型
        
        Args:
            model_name: 预训练模型名称
            num_labels: 标签数量
            label2id: 标签到ID的映射
            id2label: ID到标签的映射
            device: 设备（cuda/cpu）
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 初始化模型
        if num_labels is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.model.to(self.device)
        logger.info(f"模型加载到 {self.device}")
        logger.info(f"模型: {model_name}, 标签数: {num_labels}")
    
    def get_model(self):
        """获取模型实例"""
        return self.model
    
    def get_tokenizer(self):
        """获取tokenizer"""
        return self.tokenizer
    
    def save(self, save_path: str):
        """保存模型和tokenizer"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"模型已保存到 {save_path}")
    
    def load(self, load_path: str):
        """加载模型和tokenizer"""
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        logger.info(f"模型已从 {load_path} 加载")


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0
):
    """
    创建训练和验证数据加载器
    
    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, label2id, id2label
    """
    from torch.utils.data import DataLoader
    
    # 创建训练集
    train_dataset = EmotionDataset(
        train_path,
        tokenizer,
        max_length=max_length,
        is_test=False
    )
    
    # 使用训练集的标签映射创建验证集
    val_dataset = EmotionDataset(
        val_path,
        tokenizer,
        max_length=max_length,
        label2id=train_dataset.label2id,
        is_test=False
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, train_dataset.label2id, train_dataset.id2label


if __name__ == "__main__":
    # 测试代码
    print("测试数据集和模型...")
    
    # 测试数据集
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    dataset = EmotionDataset(
        "data/classification/train.jsonl",
        tokenizer,
        max_length=256
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"标签数量: {len(dataset.label2id)}")
    print(f"标签映射: {list(dataset.label2id.keys())[:5]}...")
    
    # 测试一个样本
    sample = dataset[0]
    print(f"\n样本键: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Label: {sample.get('labels', 'N/A')}")

