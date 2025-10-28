#!/usr/bin/env python3
"""
QA 模型类和工具函数
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QADataset(Dataset):
    """问答数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        model_type: str = "extractive"
    ):
        """
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            model_type: 模型类型（extractive 或 generative）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
        # 加载数据
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        logger.info(f"加载了 {len(self.samples)} 条样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        problem = sample.get("problem", "")
        context = sample.get("context", "")
        answer = sample.get("answer", "")
        
        if self.model_type == "extractive":
            # 抽取式 QA：输入是问题+上下文，输出是答案在上下文中的位置
            inputs = self.tokenizer(
                problem,
                context,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # 找到答案在上下文中的位置
            # 使用更精确的答案定位方法
            answer_encoding = self.tokenizer(
                answer,
                add_special_tokens=False,
                return_tensors='pt'
            )
            
            # 在上下文中搜索答案
            answer_tokens = answer_encoding['input_ids'][0]
            context_tokens = inputs['input_ids'][0]
            
            # 寻找答案在上下文中的位置
            start_position = torch.tensor([0])
            end_position = torch.tensor([0])
            
            # 在上下文中搜索答案token序列
            for i in range(len(context_tokens) - len(answer_tokens) + 1):
                if torch.equal(context_tokens[i:i+len(answer_tokens)], answer_tokens):
                    start_position = torch.tensor([i])
                    end_position = torch.tensor([i + len(answer_tokens) - 1])
                    break
            
            # 如果没找到，使用默认位置
            if start_position.item() == 0 and end_position.item() == 0:
                start_position = torch.tensor([1])
                end_position = torch.tensor([min(len(answer_tokens), self.max_length - 1)])
            
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'start_positions': start_position.squeeze(0),
                'end_positions': end_position.squeeze(0)
            }
        
        else:  # generative
            # 生成式 QA：输入是问题+上下文，输出是答案文本
            input_text = f"Question: {problem}\nContext: {context}\nAnswer:"
            
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            labels = self.tokenizer(
                answer,
                max_length=256,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )['input_ids']
            
            # 将 padding token 设置为 -100，使其在计算损失时被忽略
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0)
            }


class QAModel:
    """问答模型封装类"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        model_type: str = "extractive",
        max_length: int = 512,
        device: str = None
    ):
        """
        Args:
            model_name: 预训练模型名称
            model_type: 模型类型（extractive 或 generative）
            max_length: 最大序列长度
            device: 设备（cuda 或 cpu）
        """
        self.model_name = model_name
        self.model_type = model_type
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"初始化 QA 模型：{model_name}")
        logger.info(f"模型类型：{model_type}")
        logger.info(f"设备：{self.device}")
        
        # 设置离线模式
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)

        
        # 确保有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型 - 强制使用safetensors格式和离线模式
        model_kwargs = {"use_safetensors": True, "local_files_only": True}
        if model_type == "extractive":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, **model_kwargs)
        elif model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        else:  # generative (causal LM)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                **model_kwargs
            )
        
        if self.device != 'auto':
            self.model.to(self.device)
        
        logger.info(f"模型参数量：{self.count_parameters():,}")
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model(self):
        """获取模型"""
        return self.model
    
    def get_tokenizer(self):
        """获取分词器"""
        return self.tokenizer
    
    def save(self, output_dir: str):
        """保存模型"""
        logger.info(f"保存模型到：{output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load(self, model_dir: str):
        """加载模型"""
        logger.info(f"从 {model_dir} 加载模型")
        
        if self.model_type == "extractive":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        elif self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None
            )
        
        if self.device != 'auto':
            self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    def predict(
        self,
        problem: str,
        context: str,
        max_answer_length: int = 256
    ) -> str:
        """
        预测答案
        
        Args:
            problem: 问题
            context: 上下文
            max_answer_length: 最大答案长度
        
        Returns:
            预测的答案文本
        """
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == "extractive":
                # 抽取式 QA
                inputs = self.tokenizer(
                    problem,
                    context,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # 获取答案的起始和结束位置
                start_idx = torch.argmax(outputs.start_logits)
                end_idx = torch.argmax(outputs.end_logits)
                
                # 确保 end >= start
                if end_idx < start_idx:
                    end_idx = start_idx
                
                # 解码答案
                answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                return answer.strip()
            
            else:  # generative or seq2seq
                # 生成式 QA
                input_text = f"Question: {problem}\nContext: {context}\nAnswer:"
                
                inputs = self.tokenizer(
                    input_text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_answer_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=False
                )
                
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 对于 causal LM，需要去掉输入部分
                if self.model_type == "generative":
                    answer = answer[len(input_text):].strip()
                
                return answer


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    model_type: str = "extractive",
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = QADataset(
        train_path,
        tokenizer,
        max_length=max_length,
        model_type=model_type
    )
    
    val_dataset = QADataset(
        val_path,
        tokenizer,
        max_length=max_length,
        model_type=model_type
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def calculate_f1(prediction: str, ground_truth: str) -> float:
    """
    计算 F1 分数（基于词级别）
    
    使用与 SQuAD 相同的评估方法
    """
    def normalize_answer(s):
        """标准化答案文本"""
        import string
        import re
        
        # 转小写
        s = s.lower()
        
        # 移除标点
        s = ''.join(ch if ch not in string.punctuation else ' ' for ch in s)
        
        # 移除冠词
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        
        # 标准化空白字符
        s = ' '.join(s.split())
        
        return s
    
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    # 计算交集
    common = set(pred_tokens) & set(truth_tokens)
    num_common = sum(min(pred_tokens.count(w), truth_tokens.count(w)) for w in common)
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


if __name__ == "__main__":
    # 测试代码
    print("QA 模型工具函数测试")
    
    # 测试 F1 计算
    pred = "The capital of France is Paris"
    truth = "Paris is the capital of France"
    f1 = calculate_f1(pred, truth)
    print(f"F1 Score: {f1:.4f}")

