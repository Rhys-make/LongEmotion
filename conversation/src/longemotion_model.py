"""
LongEmotion 模型加载器
从 Hugging Face 加载 LangAGI-Lab/camel 模型
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from src.llm import LLM


class LongEmotionModel(LLM):
    """LongEmotion 模型实现（基于 Camel 模型）"""
    
    def __init__(
        self,
        model_name: str = "LangAGI-Lab/camel",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False
    ):
        """
        初始化 LongEmotion 模型
        
        Args:
            model_name: 模型名称（默认 LangAGI-Lab/camel）
            device: 设备（cuda 或 cpu）
            load_in_8bit: 是否使用8bit量化
        """
        self.model_name = model_name
        self.device = device
        
        print(f"正在加载 LongEmotion 模型: {model_name}")
        
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
        
        print(f"LongEmotion 模型加载完成")
    
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        self.model.eval()
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()

