"""
情感问答模型
Emotion QA Task
基于 BERT 的抽取式问答
"""
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    AutoModelForQuestionAnswering,
    AutoTokenizer
)
from typing import List, Dict, Tuple


class EmotionQAModel:
    """情感问答模型"""
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化问答模型
        
        Args:
            model_name: 预训练模型名称
            max_length: 最大序列长度
            device: 设备
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    
    def preprocess(
        self,
        questions: List[str],
        contexts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        预处理问答对
        
        Args:
            questions: 问题列表
            contexts: 上下文列表
            
        Returns:
            编码后的输入
        """
        encoding = self.tokenizer(
            questions,
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def extract_answer(
        self,
        question: str,
        context: str
    ) -> Dict[str, any]:
        """
        从上下文中抽取答案
        
        Args:
            question: 问题
            context: 上下文
            
        Returns:
            答案信息
        """
        self.model.eval()
        
        # 编码
        inputs = self.tokenizer(
            question,
            context,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        
        # 获取最佳答案位置
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()
        
        # 确保 end >= start
        if end_idx < start_idx:
            end_idx = start_idx
        
        # 解码答案
        input_ids = inputs['input_ids'][0]
        answer_tokens = input_ids[start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # 计算置信度
        start_prob = torch.softmax(start_logits, dim=1)[0, start_idx].item()
        end_prob = torch.softmax(end_logits, dim=1)[0, end_idx].item()
        confidence = (start_prob + end_prob) / 2
        
        return {
            'question': question,
            'context': context,
            'answer': answer.strip(),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'confidence': confidence
        }
    
    def batch_extract(
        self,
        questions: List[str],
        contexts: List[str]
    ) -> List[Dict[str, any]]:
        """
        批量抽取答案
        
        Args:
            questions: 问题列表
            contexts: 上下文列表
            
        Returns:
            答案列表
        """
        results = []
        
        # 逐个处理（QA任务通常不适合大批量）
        for question, context in zip(questions, contexts):
            result = self.extract_answer(question, context)
            results.append(result)
        
        return results
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None
    ):
        """
        前向传播（用于训练）
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            start_positions: 答案起始位置
            end_positions: 答案结束位置
            
        Returns:
            模型输出
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
    
    def save(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"问答模型已保存到 {save_path}")
    
    @classmethod
    def load(cls, model_path: str, device: str = "cuda"):
        """加载模型"""
        instance = cls.__new__(cls)
        instance.device = device
        instance.max_length = 512
        
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
        
        print(f"问答模型已从 {model_path} 加载")
        return instance


class GenerativeQAModel:
    """生成式问答模型（基于 T5/BART）"""
    
    def __init__(
        self,
        model_name: str = "google/mt5-base",
        max_input_length: int = 512,
        max_output_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化生成式问答模型
        
        Args:
            model_name: 预训练模型名称
            max_input_length: 最大输入长度
            max_output_length: 最大输出长度
            device: 设备
        """
        from transformers import AutoModelForSeq2SeqLM
        
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    def generate_answer(self, question: str, context: str) -> str:
        """生成答案"""
        self.model.eval()
        
        # 格式化输入
        input_text = f"question: {question} context: {context}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                early_stopping=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

