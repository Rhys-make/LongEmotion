"""
Qwen2-7B-Instruct 模型定义 - Emotion Summary (ES) 任务
使用 LoRA 进行高效微调，适合长文本心理咨询摘要任务
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config.config import DEVICE


class Qwen2EmotionSummaryModel:
    """
    基于 Qwen2-7B-Instruct 的情绪总结模型
    使用 LoRA 进行参数高效微调
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        max_input_length: int = 8192,
        max_output_length: int = 2048,
        device: str = None,
        use_4bit: bool = True,
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        """
        初始化 Qwen2 模型
        
        Args:
            model_name: 预训练模型名称
            max_input_length: 最大输入长度（Qwen2支持32K）
            max_output_length: 最大输出长度
            device: 设备（cuda/cpu）
            use_4bit: 是否使用4-bit量化（节省显存）
            use_lora: 是否使用LoRA微调
            lora_r: LoRA的秩
            lora_alpha: LoRA的alpha参数
            lora_dropout: LoRA的dropout
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device or DEVICE
        self.use_4bit = use_4bit
        self.use_lora = use_lora
        
        print("=" * 60)
        print(f"正在加载 Qwen2 模型: {model_name}")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"最大输入长度: {max_input_length}")
        print(f"最大输出长度: {max_output_length}")
        print(f"4-bit量化: {use_4bit}")
        print(f"LoRA微调: {use_lora}")
        
        # 配置量化参数（如果使用）
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("使用 4-bit 量化加载模型...")
        else:
            quantization_config = None
        
        # 加载 tokenizer
        print("加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",  # 重要：对于因果语言模型
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        print("加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # 自动分配到可用设备
            trust_remote_code=True,
            torch_dtype=torch.float16 if not use_4bit else None,
        )
        
        # 配置 LoRA
        if use_lora:
            print("\n配置 LoRA...")
            print(f"  - LoRA rank (r): {lora_r}")
            print(f"  - LoRA alpha: {lora_alpha}")
            print(f"  - LoRA dropout: {lora_dropout}")
            
            # 准备模型用于k-bit训练
            if use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA配置
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],  # Qwen2的attention和MLP层
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # 应用 LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            # 打印可训练参数
            self.model.print_trainable_parameters()
        
        print("\n✓ 模型加载完成！")
        print("=" * 60)
    
    def build_prompt(
        self,
        case_description: List[str],
        consultation_process: List[str],
        for_training: bool = False,
        experience_and_reflection: str = None
    ) -> str:
        """
        构建 Qwen2 的输入提示词
        
        Args:
            case_description: 案例描述
            consultation_process: 咨询过程
            for_training: 是否用于训练（包含标签）
            experience_and_reflection: 经验与反思（仅训练时）
        
        Returns:
            格式化的提示词
        """
        # 拼接案例描述
        case_text = "\n".join(case_description) if isinstance(case_description, list) else case_description
        
        # 拼接咨询过程
        process_text = "\n".join(consultation_process) if isinstance(consultation_process, list) else consultation_process
        
        # 构建输入文本
        input_text = f"""【案例描述】
{case_text}

【咨询过程】
{process_text}"""
        
        # 使用 Qwen2 的对话模板
        if for_training and experience_and_reflection:
            # 训练时：包含问题和答案
            messages = [
                {
                    "role": "system",
                    "content": "你是一位专业的心理咨询师。请根据提供的心理咨询案例，生成深入、专业的经验与反思总结。"
                },
                {
                    "role": "user",
                    "content": f"请对以下心理咨询案例进行深入分析和反思：\n\n{input_text}"
                },
                {
                    "role": "assistant",
                    "content": experience_and_reflection
                }
            ]
        else:
            # 推理时：只有问题
            messages = [
                {
                    "role": "system",
                    "content": "你是一位专业的心理咨询师。请根据提供的心理咨询案例，生成深入、专业的经验与反思总结。"
                },
                {
                    "role": "user",
                    "content": f"请对以下心理咨询案例进行深入分析和反思：\n\n{input_text}"
                }
            ]
        
        # 使用 tokenizer 的 chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not for_training
        )
        
        return prompt
    
    def prepare_inputs(
        self,
        case_description: List[str],
        consultation_process: List[str],
        experience_and_reflection: str = None,
        for_training: bool = False
    ) -> Dict:
        """
        准备模型输入
        
        Returns:
            tokenized 输入字典
        """
        # 构建提示词
        prompt = self.build_prompt(
            case_description,
            consultation_process,
            for_training=for_training,
            experience_and_reflection=experience_and_reflection
        )
        
        # Tokenize
        if for_training:
            # 训练时：整个序列都需要tokenize
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        else:
            # 推理时：只tokenize输入部分
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt"
            )
        
        return inputs
    
    def generate_summary(
        self,
        case_description: List[str],
        consultation_process: List[str],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """
        生成情绪总结
        
        Args:
            case_description: 案例描述
            consultation_process: 咨询过程
            temperature: 温度参数
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚
        
        Returns:
            生成的经验与反思文本
        """
        # 准备输入
        inputs = self.prepare_inputs(
            case_description,
            consultation_process,
            for_training=False
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # 解码（只解码新生成的部分）
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return summary.strip()
    
    def save_model(self, save_path: Path):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        print(f"\n保存模型到: {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 LoRA 权重
        if self.use_lora:
            self.model.save_pretrained(save_path)
            print("✓ LoRA 权重保存完成")
        else:
            self.model.save_pretrained(save_path)
            print("✓ 完整模型保存完成")
        
        # 保存 tokenizer
        self.tokenizer.save_pretrained(save_path)
        print("✓ Tokenizer 保存完成")
    
    def load_lora_weights(self, lora_path: Path):
        """
        加载 LoRA 权重
        
        Args:
            lora_path: LoRA权重路径
        """
        print(f"\n从 {lora_path} 加载 LoRA 权重...")
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path,
            device_map="auto"
        )
        print("✓ LoRA 权重加载完成")


def test_model():
    """
    测试 Qwen2 模型
    """
    print("\n" + "=" * 60)
    print("测试 Qwen2 Emotion Summary 模型")
    print("=" * 60 + "\n")
    
    # 创建模型实例（使用较小的参数用于测试）
    model = Qwen2EmotionSummaryModel(
        model_name="Qwen/Qwen2-7B-Instruct",
        use_4bit=True,
        use_lora=True,
        lora_r=32,  # 测试用较小的rank
    )
    
    # 测试数据
    test_case = [
        "来访者，女性，27岁，本科学历，公司职员。主诉：近3个月来情绪低落，经常感到焦虑。"
    ]
    
    test_process = [
        "来访者主诉: 近3个月情绪低落，对工作和生活失去兴趣。",
        "详细陈述: 工作压力大，与同事关系紧张，晚上经常失眠。",
        "咨询师回复: 通过认知行为疗法帮助来访者识别负面思维模式。"
    ]
    
    print("测试输入:")
    print(f"案例描述: {test_case[0]}")
    print(f"咨询过程段落数: {len(test_process)}")
    
    print("\n生成摘要...")
    summary = model.generate_summary(test_case, test_process)
    
    print("\n生成的经验与反思:")
    print("-" * 60)
    print(summary)
    print("-" * 60)
    
    print("\n✓ 测试完成！")


if __name__ == "__main__":
    test_model()

