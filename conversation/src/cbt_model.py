"""
CBT 心理健康模型加载器
基于 Qwen3-4B-Instruct-2507_mental_health_cbt 模型
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from pathlib import Path
from src.llm import LLM


class CBTModel(LLM):
    """CBT 心理健康模型实现（基于 Qwen3-4B-Instruct）"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False
    ):
        """
        初始化 CBT 模型
        
        Args:
            model_path: 本地模型路径（如果为None，使用默认路径 conversation/model）
            base_model: 基础模型名称（用于加载适配器）
            device: 设备（cuda 或 cpu）
            load_in_8bit: 是否使用8bit量化
        """
        # 确定模型路径
        if model_path is None:
            # 默认路径：conversation/model（相对于src目录）
            src_dir = Path(__file__).parent
            conversation_dir = src_dir.parent
            model_path = conversation_dir / "model"
        
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型路径不存在: {self.model_path}\n"
                f"请先运行: python conversation/scripts/download_model.py"
            )
        
        print(f"正在加载 CBT 模型: {self.model_path}")
        
        try:
            # 尝试加载 LoRA 适配器（如果存在）
            from peft import PeftModel, PeftConfig
            
            # 检查是否有 adapter_model.safetensors 文件
            adapter_file = self.model_path / "adapter_model.safetensors"
            adapter_config = self.model_path / "adapter_config.json"
            
            if adapter_file.exists() and adapter_config.exists():
                # 这是 LoRA 适配器，需要加载基础模型
                # 从适配器配置中读取基础模型名称
                import json
                with open(adapter_config, 'r', encoding='utf-8') as f:
                    adapter_config_data = json.load(f)
                    adapter_base_model = adapter_config_data.get('base_model_name_or_path', base_model or "Qwen/Qwen2.5-4B-Instruct")
                
                # 如果配置中的基础模型名称不完整，尝试修正
                if not adapter_base_model.startswith("Qwen/") and "Qwen" in adapter_base_model:
                    adapter_base_model = f"Qwen/{adapter_base_model}"
                elif not adapter_base_model.startswith("Qwen/"):
                    # 如果配置中没有指定，使用默认值
                    adapter_base_model = base_model or "Qwen/Qwen2.5-4B-Instruct"
                
                # 处理模型名称变体
                if "Qwen3" in adapter_base_model:
                    # Qwen3 系列模型
                    pass
                elif "Qwen2.5" not in adapter_base_model and "Qwen2" not in adapter_base_model:
                    # 如果只是版本号，补充完整名称
                    if adapter_base_model == "Qwen/Qwen3-4B-Instruct-2507":
                        adapter_base_model = "Qwen/Qwen2.5-4B-Instruct"  # 使用可用的基础模型
                
                print(f"检测到 LoRA 适配器，加载基础模型: {adapter_base_model}")
                
                # 加载分词器
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True
                )
                
                # 加载基础模型
                base_model_instance = AutoModelForCausalLM.from_pretrained(
                    adapter_base_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    load_in_8bit=load_in_8bit
                )
                
                # 加载 LoRA 适配器
                print(f"加载 LoRA 适配器: {self.model_path}")
                self.model = PeftModel.from_pretrained(
                    base_model_instance,
                    str(self.model_path),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                
                if not load_in_8bit and device != "cuda":
                    self.model = self.model.to(device)
            else:
                # 完整模型，直接加载
                print("加载完整模型...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    load_in_8bit=load_in_8bit
                )
                
                if not load_in_8bit and device != "cuda":
                    self.model = self.model.to(device)
        
        except ImportError:
            # 如果没有 peft，尝试直接加载
            print("未安装 peft，尝试直接加载模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                load_in_8bit=load_in_8bit
            )
            
            if not load_in_8bit and device != "cuda":
                self.model = self.model.to(device)
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"CBT 模型加载完成")
    
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
            max_length=2048
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

