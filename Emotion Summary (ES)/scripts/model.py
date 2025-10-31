"""
模型定义 - Emotion Summary (ES) 任务
定义用于情绪总结的生成式模型
支持：LongT5, LED, BART, T5, Pegasus等
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    LongT5ForConditionalGeneration,
    LEDForConditionalGeneration,
    PegasusForConditionalGeneration,
)

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    DEVICE,
    SUMMARY_ASPECTS,
    T5_PREFIX,
)


class EmotionSummaryModel:
    """
    情绪总结模型封装类
    支持多种Seq2Seq模型：LongT5, LED, BART, T5, Pegasus
    """
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_output_length: int = MAX_OUTPUT_LENGTH,
        device: str = None
    ):
        """
        初始化模型
        
        Args:
            model_name: 预训练模型名称
            max_input_length: 最大输入长度
            max_output_length: 最大输出长度
            device: 设备（cuda/cpu）
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device or DEVICE
        
        print(f"正在加载模型: {model_name}")
        print(f"设备: {self.device}")
        print(f"最大输入长度: {max_input_length}")
        print(f"最大输出长度: {max_output_length}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 根据模型类型选择合适的类
        # 自动检测模型类型并加载
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # 检测模型类型
        self.model_type = self._detect_model_type()
        
        print(f"✓ 模型加载完成")
        print(f"模型类型: {self.model_type}")
        print(f"模型参数量: {self.model.num_parameters():,}")
    
    def _detect_model_type(self) -> str:
        """检测模型类型"""
        model_class = self.model.__class__.__name__
        
        if "LongT5" in model_class:
            return "longt5"
        elif "LED" in model_class:
            return "led"
        elif "BART" in model_class:
            return "bart"
        elif "T5" in model_class:
            return "t5"
        elif "Pegasus" in model_class:
            return "pegasus"
        else:
            return "unknown"
    
    def prepare_input(self, context: str, aspect: Optional[str] = None) -> Dict:
        """
        准备模型输入
        
        Args:
            context: 输入的心理病理报告
            aspect: 要总结的具体方面（可选）
        
        Returns:
            tokenized输入
        """
        # 根据模型类型和方面构建提示词
        if aspect:
            # 针对特定方面的总结
            if self.model_type in ["longt5", "t5"]:
                prompt = f"summarize {aspect}: {context}"
            elif self.model_type == "bart":
                prompt = f"Summarize the {aspect} from this report: {context}"
            elif self.model_type == "led":
                prompt = f"Summarize {aspect}: {context}"
            elif self.model_type == "pegasus":
                prompt = context  # Pegasus不需要前缀
            else:
                prompt = f"summarize {aspect}: {context}"
        else:
            # 整体总结
            if self.model_type in ["longt5", "t5"]:
                prompt = f"{T5_PREFIX}{context}"
            else:
                prompt = context
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def generate_summary(
        self,
        context: str,
        aspect: Optional[str] = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        **kwargs
    ) -> str:
        """
        生成总结
        
        Args:
            context: 输入文本
            aspect: 要总结的方面
            num_beams: beam search宽度
            length_penalty: 长度惩罚
            no_repeat_ngram_size: 避免重复的n-gram大小
        
        Returns:
            生成的总结文本
        """
        # 准备输入
        inputs = self.prepare_input(context, aspect)
        
        # 生成
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
                **kwargs
            )
        
        # 解码
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
    
    def generate_multi_aspect_summary(
        self,
        context: str,
        aspects: List[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        生成多方面总结
        
        Args:
            context: 输入文本
            aspects: 要总结的方面列表
        
        Returns:
            各个方面的总结字典
        """
        aspects = aspects or SUMMARY_ASPECTS
        summaries = {}
        
        for aspect in aspects:
            summary = self.generate_summary(context, aspect=aspect, **kwargs)
            summaries[aspect] = summary
        
        return summaries
    
    def save_model(self, save_path: Path):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        print(f"保存模型到: {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print("✓ 模型保存完成")
    
    def load_model(self, load_path: Path):
        """
        加载模型
        
        Args:
            load_path: 加载路径
        """
        print(f"从 {load_path} 加载模型")
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        
        print("✓ 模型加载完成")


def test_model():
    """
    测试模型
    """
    print("=" * 60)
    print("测试 Emotion Summary 模型")
    print("=" * 60)
    
    # 创建模型实例
    model = EmotionSummaryModel()
    
    # 测试文本
    test_context = """
    The patient is a 35-year-old female who presented with symptoms of severe 
    anxiety and depression. She reported that her symptoms began approximately 
    six months ago following a traumatic work-related incident. Treatment 
    involved cognitive behavioral therapy and medication. After three months 
    of treatment, the patient showed significant improvement.
    """
    
    print("\n输入文本:")
    print(test_context)
    
    print("\n生成多方面总结:")
    summaries = model.generate_multi_aspect_summary(test_context)
    
    for aspect, summary in summaries.items():
        print(f"\n{aspect}:")
        print(f"  {summary}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_model()

