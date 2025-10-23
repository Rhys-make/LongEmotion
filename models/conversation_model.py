"""
情感对话生成模型
Emotion Conversation Task
基于 ChatGLM 或 Qwen 的对话生成
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from typing import List, Dict, Optional


class EmotionConversationModel:
    """情感对话生成模型"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B",  # 可替换为 THUDM/chatglm3-6b
        max_length: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False
    ):
        """
        初始化对话模型
        
        Args:
            model_name: 预训练模型名称
            max_length: 最大序列长度
            device: 设备
            load_in_8bit: 是否使用8bit量化
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=load_in_8bit
        )
        
        if not load_in_8bit and device != "cuda":
            self.model = self.model.to(device)
        
        # 生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    def format_prompt(self, context: str, emotion: Optional[str] = None) -> str:
        """
        格式化提示词
        
        Args:
            context: 对话上下文
            emotion: 目标情感（可选）
            
        Returns:
            格式化后的提示
        """
        if emotion:
            prompt = f"作为一个理解情感的对话助手，请基于以下上下文，以'{emotion}'的情感回复：\n\n上下文：{context}\n\n回复："
        else:
            prompt = f"作为一个理解情感的对话助手，请根据以下上下文进行回复：\n\n上下文：{context}\n\n回复："
        
        return prompt
    
    def generate_response(
        self,
        context: str,
        emotion: Optional[str] = None,
        max_new_tokens: int = 256
    ) -> str:
        """
        生成对话回复
        
        Args:
            context: 对话上下文
            emotion: 目标情感
            max_new_tokens: 最大生成长度
            
        Returns:
            生成的回复
        """
        self.model.eval()
        
        # 格式化输入
        prompt = self.format_prompt(context, emotion)
        
        # 编码
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                do_sample=self.generation_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def batch_generate(
        self,
        contexts: List[str],
        emotions: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        批量生成回复
        
        Args:
            contexts: 上下文列表
            emotions: 情感列表
            
        Returns:
            生成结果列表
        """
        if emotions is None:
            emotions = [None] * len(contexts)
        
        results = []
        for context, emotion in zip(contexts, emotions):
            response = self.generate_response(context, emotion)
            results.append({
                'context': context,
                'emotion': emotion,
                'response': response
            })
        
        return results
    
    def save(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"对话模型已保存到 {save_path}")
    
    @classmethod
    def load(cls, model_path: str, device: str = "cuda", load_in_8bit: bool = False):
        """加载模型"""
        instance = cls.__new__(cls)
        instance.device = device
        instance.max_length = 1024
        
        instance.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        
        instance.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=load_in_8bit
        )
        
        instance.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=instance.tokenizer.pad_token_id,
            eos_token_id=instance.tokenizer.eos_token_id
        )
        
        print(f"对话模型已从 {model_path} 加载")
        return instance

