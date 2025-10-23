"""
情感摘要生成模型
Emotion Summary Task
使用 T5/BART 进行摘要生成
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig
)
from typing import List, Dict


class EmotionSummaryModel:
    """情感摘要生成模型"""
    
    def __init__(
        self,
        model_name: str = "google/mt5-base",  # 支持中文的 mT5
        max_input_length: int = 1024,
        max_output_length: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化摘要模型
        
        Args:
            model_name: 预训练模型名称
            max_input_length: 最大输入长度
            max_output_length: 最大输出长度
            device: 设备
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # 生成配置
        self.generation_config = GenerationConfig(
            max_length=max_output_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        预处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            编码后的输入
        """
        # 添加任务前缀
        processed_texts = [f"summarize: {text}" for text in texts]
        
        encoding = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def generate_summary(self, text: str) -> str:
        """
        生成单个摘要
        
        Args:
            text: 输入文本
            
        Returns:
            生成的摘要
        """
        self.model.eval()
        
        # 预处理
        inputs = self.preprocess([text])
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.generation_config.max_length,
                num_beams=self.generation_config.num_beams,
                length_penalty=self.generation_config.length_penalty,
                early_stopping=self.generation_config.early_stopping,
                no_repeat_ngram_size=self.generation_config.no_repeat_ngram_size
            )
        
        # 解码
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary.strip()
    
    def batch_generate(self, texts: List[str], batch_size: int = 4) -> List[Dict[str, str]]:
        """
        批量生成摘要
        
        Args:
            texts: 文本列表
            batch_size: 批量大小
            
        Returns:
            生成结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 预处理
            inputs = self.preprocess(batch_texts)
            
            # 生成
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.generation_config.max_length,
                    num_beams=self.generation_config.num_beams,
                    length_penalty=self.generation_config.length_penalty,
                    early_stopping=self.generation_config.early_stopping,
                    no_repeat_ngram_size=self.generation_config.no_repeat_ngram_size
                )
            
            # 解码
            summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 收集结果
            for text, summary in zip(batch_texts, summaries):
                results.append({
                    'text': text,
                    'summary': summary.strip()
                })
        
        return results
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ):
        """
        前向传播（用于训练）
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            labels: 标签
            
        Returns:
            模型输出
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def save(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"摘要模型已保存到 {save_path}")
    
    @classmethod
    def load(cls, model_path: str, device: str = "cuda"):
        """加载模型"""
        instance = cls.__new__(cls)
        instance.device = device
        instance.max_input_length = 1024
        instance.max_output_length = 256
        
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        
        instance.generation_config = GenerationConfig(
            max_length=256,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        print(f"摘要模型已从 {model_path} 加载")
        return instance

