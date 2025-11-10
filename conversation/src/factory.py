"""
工厂类：用于创建LLM和CounselorAgent实例
"""
from src.llm import LLM, ChatGPT, LLama2, LLama3
from src.agent import CactusCounselorAgent, CounselorAgent
from src.longemotion_model import LongEmotionModel
from src.cbt_model import CBTModel


class LLMFactory:
    """LLM工厂类"""
    
    @staticmethod
    def get_llm(llm_type: str) -> LLM:
        """
        根据类型获取LLM实例
        
        Args:
            llm_type: LLM类型（chatgpt, llama2, llama3, longemotion, cbt）
            
        Returns:
            LLM实例
            
        Raises:
            ValueError: 不支持的LLM类型
        """
        if llm_type == "chatgpt":
            return ChatGPT()
        elif llm_type == "llama2":
            return LLama2()
        elif llm_type == "llama3":
            return LLama3()
        elif llm_type == "longemotion":
            return LongEmotionModel()
        elif llm_type == "cbt":
            return CBTModel()
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")


class CounselorFactory:
    """咨询师代理工厂类"""
    
    @staticmethod
    def get_counselor(counselor_type: str, llm_type: str) -> CounselorAgent:
        """
        根据类型获取咨询师代理实例
        
        Args:
            counselor_type: 咨询师类型（cactus等）
            llm_type: LLM类型
            
        Returns:
            CounselorAgent实例
            
        Raises:
            ValueError: 不支持的咨询师类型
        """
        if counselor_type == "cactus":
            return CactusCounselorAgent(llm_type)
        else:
            raise ValueError(f"Unsupported counselor type: {counselor_type}")

