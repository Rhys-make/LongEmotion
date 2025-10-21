"""
情感分类模型
Emotion Classification Task
使用 BERT 进行多分类
"""
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from typing import Dict, List, Optional


class EmotionClassificationModel:
    """情感分类模型封装"""
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        num_labels: int = 7,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化分类模型
        
        Args:
            model_name: 预训练模型名称
            num_labels: 分类标签数量
            max_length: 最大序列长度
            device: 设备
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
        # 情感标签映射
        self.label_names = [
            "happiness",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust",
            "neutral"
        ]
    
    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        预处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            编码后的输入
        """
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            labels: 标签（训练时使用）
            
        Returns:
            模型输出
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def predict(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            
        Returns:
            预测结果列表
        """
        self.model.eval()
        
        inputs = self.preprocess(texts)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                'text': texts[i],
                'label': int(pred),
                'emotion': self.label_names[int(pred)],
                'confidence': float(probs[pred]),
                'probabilities': {
                    self.label_names[j]: float(probs[j])
                    for j in range(self.num_labels)
                }
            })
        
        return results
    
    def save(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"分类模型已保存到 {save_path}")
    
    @classmethod
    def load(cls, model_path: str, device: str = "cuda"):
        """加载模型"""
        instance = cls.__new__(cls)
        instance.device = device
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        instance.max_length = 512
        instance.num_labels = instance.model.config.num_labels
        instance.label_names = [
            "happiness", "sadness", "anger", "fear",
            "surprise", "disgust", "neutral"
        ]
        print(f"分类模型已从 {model_path} 加载")
        return instance

