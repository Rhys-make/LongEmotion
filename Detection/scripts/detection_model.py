"""
情感检测模型
Emotion Detection Task
多标签分类任务
"""
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    AutoModel,
    AutoTokenizer
)
from typing import Dict, List, Optional


class EmotionDetectionModel(nn.Module):
    """情感检测模型（多标签分类）"""
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        num_emotions: int = 7,
        dropout: float = 0.1
    ):
        """
        初始化检测模型
        
        Args:
            model_name: 预训练模型名称
            num_emotions: 情感类别数量
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 多标签分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotions)
        )
        
        self.num_emotions = num_emotions
    
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
            labels: 标签（多标签，形状为 [batch_size, num_emotions]）
            
        Returns:
            模型输出
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用 [CLS] token 的表示
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 分类
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # 使用 BCE Loss 进行多标签分类
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {
            'loss': loss,
            'logits': logits
        }


class EmotionDetectionModelWrapper:
    """情感检测模型封装器"""
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        num_emotions: int = 7,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5
    ):
        """
        初始化
        
        Args:
            model_name: 预训练模型名称
            num_emotions: 情感类别数量
            max_length: 最大序列长度
            device: 设备
            threshold: 分类阈值
        """
        self.model_name = model_name
        self.num_emotions = num_emotions
        self.max_length = max_length
        self.device = device
        self.threshold = threshold
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        self.model = EmotionDetectionModel(
            model_name=model_name,
            num_emotions=num_emotions
        ).to(device)
        
        # 情感标签名称
        self.emotion_names = [
            "happiness",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust",
            "neutral"
        ]
    
    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """预处理文本"""
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
    
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
            logits = outputs['logits']
            probabilities = torch.sigmoid(logits)  # 多标签使用 sigmoid
        
        results = []
        for i, probs in enumerate(probabilities):
            # 获取超过阈值的情感
            detected_emotions = []
            emotion_scores = {}
            
            for j, prob in enumerate(probs):
                emotion_scores[self.emotion_names[j]] = float(prob)
                if prob >= self.threshold:
                    detected_emotions.append({
                        'emotion': self.emotion_names[j],
                        'score': float(prob)
                    })
            
            results.append({
                'text': texts[i],
                'emotions': detected_emotions,
                'all_scores': emotion_scores
            })
        
        return results
    
    def save(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), f"{save_path}/model.pt")
        
        # 保存分词器
        self.tokenizer.save_pretrained(save_path)
        
        # 保存配置
        import json
        config = {
            'model_name': self.model_name,
            'num_emotions': self.num_emotions,
            'max_length': self.max_length,
            'threshold': self.threshold
        }
        with open(f"{save_path}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"检测模型已保存到 {save_path}")
    
    @classmethod
    def load(cls, model_path: str, device: str = "cuda"):
        """加载模型"""
        import json
        
        # 加载配置
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        instance = cls(
            model_name=config['model_name'],
            num_emotions=config['num_emotions'],
            max_length=config['max_length'],
            device=device,
            threshold=config.get('threshold', 0.5)
        )
        
        # 加载权重
        instance.model.load_state_dict(
            torch.load(f"{model_path}/model.pt", map_location=device)
        )
        
        print(f"检测模型已从 {model_path} 加载")
        return instance

