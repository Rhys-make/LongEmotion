"""
咨询师代理类
"""
from abc import ABC, abstractmethod
from typing import List, Dict
from langchain.prompts import PromptTemplate
from pathlib import Path
from src.llm import LLM
from src.factory import LLMFactory


class CounselorAgent(ABC):
    """咨询师代理基类"""
    
    def __init__(self, llm_type: str):
        """
        初始化咨询师代理
        
        Args:
            llm_type: LLM类型
        """
        self.llm_type = llm_type
        self.llm: LLM = LLMFactory.get_llm(llm_type)
        self.language = "english"  # 默认英语
        self.prompt_template: PromptTemplate = None
    
    def load_prompt(self, filename: str) -> str:
        """
        加载提示模板文件
        
        Args:
            filename: 提示文件名
            
        Returns:
            提示文本
        """
        prompt_path = Path(__file__).parent.parent / "prompts" / filename
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @abstractmethod
    def generate(self, history: List[Dict]) -> str:
        """
        生成咨询回复
        
        Args:
            history: 对话历史，格式为 [{"role": "client"/"counselor", "message": "..."}]
            
        Returns:
            生成的回复
        """
        pass


class CactusCounselorAgent(CounselorAgent):
    """Cactus咨询师代理（基于认知行为理论）"""
    
    def __init__(self, llm_type: str):
        super().__init__(llm_type)
        self.language = "english"  # 可以改为 "chinese"
        
        # 加载提示模板
        # 特殊处理某些 llm_type
        if llm_type == "longemotion":
            prompt_file = "agent_cactus_longemotion.txt"
        elif llm_type == "cbt":
            prompt_file = "agent_cactus_cbt.txt"
        else:
            prompt_file = f"agent_cactus_{llm_type}.txt"
        prompt_text = self.load_prompt(prompt_file)
        
        # 解析提示模板中的变量
        input_variables = ["client_information", "reason_counseling", "cbt_plan", "history"]
        
        self.prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=prompt_text
        )
    
    def format_history(self, history: List[Dict]) -> str:
        """
        格式化对话历史
        
        Args:
            history: 对话历史
            
        Returns:
            格式化后的历史字符串
        """
        formatted = []
        for message in history:
            role = message.get('role', 'unknown').capitalize()
            msg = message.get('message', '')
            formatted.append(f"{role}: {msg}")
        return '\n'.join(formatted)
    
    def generate(
        self,
        history: List[Dict],
        client_information: str = "",
        reason_counseling: str = "",
        cbt_plan: str = ""
    ) -> str:
        """
        生成咨询回复
        
        Args:
            history: 对话历史
            client_information: 客户信息
            reason_counseling: 咨询原因
            cbt_plan: CBT计划
        """
        # 格式化历史
        formatted_history = self.format_history(history)
        
        # 构建提示
        prompt = self.prompt_template.format(
            client_information=client_information,
            reason_counseling=reason_counseling,
            cbt_plan=cbt_plan,
            history=formatted_history
        )
        
        # 生成回复
        response = self.llm.generate(prompt)
        return response

