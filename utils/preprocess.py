"""
文本预处理工具模块
提供通用的文本清洗、分词等功能
"""
import re
from typing import List, Dict, Any


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.emotion_labels = {
            "happiness": 0,
            "sadness": 1,
            "anger": 2,
            "fear": 3,
            "surprise": 4,
            "disgust": 5,
            "neutral": 6
        }
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 去除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 去除特殊字符（保留中文、英文、数字和常用标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）《》\s]', '', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    def process_classification_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        处理分类任务数据
        
        Args:
            examples: 批量样本
            
        Returns:
            处理后的样本
        """
        texts = [self.clean_text(text) for text in examples.get('text', [])]
        labels = examples.get('label', [])
        
        return {
            'text': texts,
            'label': labels
        }
    
    def process_detection_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        处理检测任务数据
        
        Args:
            examples: 批量样本
            
        Returns:
            处理后的样本
        """
        texts = [self.clean_text(text) for text in examples.get('text', [])]
        emotions = examples.get('emotions', [])
        
        return {
            'text': texts,
            'emotions': emotions
        }
    
    def process_conversation_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        处理对话任务数据
        
        Args:
            examples: 批量样本
            
        Returns:
            处理后的样本
        """
        contexts = [self.clean_text(ctx) for ctx in examples.get('context', [])]
        responses = [self.clean_text(resp) for resp in examples.get('response', [])]
        
        return {
            'context': contexts,
            'response': responses
        }
    
    def process_summary_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        处理摘要任务数据
        
        Args:
            examples: 批量样本
            
        Returns:
            处理后的样本
        """
        texts = [self.clean_text(text) for text in examples.get('text', [])]
        summaries = [self.clean_text(summ) for summ in examples.get('summary', [])]
        
        return {
            'text': texts,
            'summary': summaries
        }
    
    def process_qa_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        处理问答任务数据
        
        Args:
            examples: 批量样本
            
        Returns:
            处理后的样本
        """
        contexts = [self.clean_text(ctx) for ctx in examples.get('context', [])]
        questions = [self.clean_text(q) for q in examples.get('question', [])]
        answers = examples.get('answer', [])
        
        return {
            'context': contexts,
            'question': questions,
            'answer': answers
        }


def get_preprocessor() -> TextPreprocessor:
    """获取预处理器实例"""
    return TextPreprocessor()

