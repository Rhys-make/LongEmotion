"""
LLM抽象类和具体实现
"""
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from src.config import get_config


class LLM(ABC):
    """LLM抽象基类"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示
            
        Returns:
            生成的文本
        """
        pass


class ChatGPT(LLM):
    """ChatGPT实现"""
    
    def __init__(self):
        config = get_config()
        api_key = config.get('openai', {}).get('key', '')
        
        if not api_key:
            raise ValueError("OpenAI API key not found in config.yaml")
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
    
    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)


class LLama2(LLM):
    """Llama2实现（通过vLLM服务器）"""
    
    def __init__(self):
        config = get_config()
        host = config.get('llama2', {}).get('host', '')
        
        if not host:
            raise ValueError("Llama2 host not found in config.yaml")
        
        self.llm = ChatOpenAI(
            model="meta-llama/Llama-2-7b-chat-hf",
            temperature=0.7,
            base_url=host,
            api_key="dummy"  # vLLM不需要真实的key
        )
    
    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)


class LLama3(LLM):
    """Llama3实现（通过vLLM服务器）"""
    
    def __init__(self):
        config = get_config()
        host = config.get('llama3', {}).get('host', '')
        
        if not host:
            raise ValueError("Llama3 host not found in config.yaml")
        
        self.llm = ChatOpenAI(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=0.7,
            base_url=host,
            api_key="dummy"  # vLLM不需要真实的key
        )
    
    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

