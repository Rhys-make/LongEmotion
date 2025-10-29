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
    """问答数据集（支持长上下文滑窗分片）"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        model_type: str = "extractive",
        doc_stride: int = 128
    ):
        """
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度（模型输入长度）
            model_type: 模型类型（extractive 或 generative）
            doc_stride: 长上下文切片的重叠步长
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        self.doc_stride = doc_stride
        
        # 加载原始样本
        raw_samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_samples.append(json.loads(line))
        
        logger.info(f"加载了 {len(raw_samples)} 条样本（原始）")
        
        # 预处理为特征（对于抽取式，进行滑窗拆分；生成式保持一条）
        self.samples = []
        for sample in raw_samples:
            problem = sample.get("problem", "")
            context = sample.get("context", "")
            answer = sample.get("answer", "")
            
            if self.model_type == "extractive":
                # 在原始 context 中找到答案的字符级位置（首个匹配）
                answer_text = answer if isinstance(answer, str) else str(answer)
                start_char = -1
                if answer_text:
                    start_char = context.find(answer_text)
                end_char = start_char + len(answer_text) if start_char >= 0 else -1
                
                # 基于滑窗的编码
                encoded = self.tokenizer(
                    problem,
                    context,
                    max_length=self.max_length,
                    truncation='only_second',
                    padding='max_length',
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    stride=self.doc_stride,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                offset_mapping = encoded['offset_mapping']  # 每个token对应原始文本中的字符范围
                overflow_to_sample = encoded.get('overflow_to_sample_mapping', None)
                
                # sequence_ids 用于区分 question(0) 与 context(1)
                # transformers 的 BatchEncoding 支持 sequence_ids(i)
                for i in range(input_ids.size(0)):
                    # 映射到第 i 个特征的 sequence ids
                    try:
                        seq_ids = encoded.sequence_ids(i)
                    except Exception:
                        seq_ids = None
                    
                    # 仅在 context token 范围内标注答案
                    start_pos = 0
                    end_pos = 0
                    if start_char >= 0 and end_char >= 0 and seq_ids is not None:
                        # 找到当前特征的 context token 起止范围
                        token_start_index = 0
                        while token_start_index < len(seq_ids) and seq_ids[token_start_index] != 1:
                            token_start_index += 1
                        token_end_index = len(seq_ids) - 1
                        while token_end_index >= 0 and seq_ids[token_end_index] != 1:
                            token_end_index -= 1
                        
                        # 将字符级答案映射为 token 级别
                        if token_start_index <= token_end_index:
                            # 跳过不含字符映射的 tokens
                            while (
                                token_start_index <= token_end_index and
                                (offset_mapping[i][token_start_index][0].item() == 0 and \
                                 offset_mapping[i][token_start_index][1].item() == 0)
                            ):
                                token_start_index += 1
                            while (
                                token_end_index >= token_start_index and
                                (offset_mapping[i][token_end_index][0].item() == 0 and \
                                 offset_mapping[i][token_end_index][1].item() == 0)
                            ):
                                token_end_index -= 1
                            
                            if token_start_index <= token_end_index:
                                # 若答案不在该切片范围，则保持 (0,0)
                                if not (start_char >= offset_mapping[i][token_start_index][0].item() and \
                                        end_char <= offset_mapping[i][token_end_index][1].item()):
                                    pass
                                else:
                                    # 在切片范围内，二分定位 token 位置
                                    s = start_char
                                    e = end_char
                                    # 找到 start token
                                    while token_start_index < len(seq_ids) and \
                                        offset_mapping[i][token_start_index][0].item() <= s and \
                                        offset_mapping[i][token_start_index][1].item() <= s:
                                        token_start_index += 1
                                    while token_start_index > 0 and offset_mapping[i][token_start_index-1][0].item() > s:
                                        token_start_index -= 1
                                    # 回退一位确保覆盖
                                    while token_start_index > 0 and offset_mapping[i][token_start_index][0].item() > s:
                                        token_start_index -= 1
                                    
                                    # 找到 end token
                                    j = token_start_index
                                    while j < len(seq_ids) and offset_mapping[i][j][1].item() < e:
                                        j += 1
                                    if j >= len(seq_ids):
                                        j = len(seq_ids) - 1
                                    
                                    start_pos = max(0, token_start_index)
                                    end_pos = max(start_pos, j)
                    
                    self.samples.append({
                        'input_ids': input_ids[i],
                        'attention_mask': attention_mask[i],
                        'start_positions': torch.tensor(start_pos, dtype=torch.long),
                        'end_positions': torch.tensor(end_pos, dtype=torch.long)
                    })
            else:
                # 生成式不切片，直接构造输入与labels
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
                labels[labels == self.tokenizer.pad_token_id] = -100
                self.samples.append({
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': labels.squeeze(0)
                })
        
        logger.info(f"构建了 {len(self.samples)} 条特征（含滑窗分片）")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class QAModel:
    """问答模型封装类"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        model_type: str = "extractive",
        max_length: int = 512,
        device: str = None,
        doc_stride: int = 128,
        offline: bool = False
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
        self.doc_stride = doc_stride
        self.offline = offline
        
        logger.info(f"初始化 QA 模型：{model_name}")
        logger.info(f"模型类型：{model_type}")
        logger.info(f"设备：{self.device}")
        
        # 设置离线/在线模式
        import os
        if self.offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
        
        # 加载分词器（根据 offline 决定是否仅本地）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=self.offline)

        
        # 确保有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型 - 允许在线下载且不强制 safetensors
        model_kwargs = {"use_safetensors": False, "local_files_only": self.offline}
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
        from pathlib import Path
        local_dir = str(Path(model_dir).resolve())
        
        if self.model_type == "extractive":
            self.model = AutoModelForQuestionAnswering.from_pretrained(local_dir, local_files_only=self.offline)
        elif self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=self.offline)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                local_files_only=self.offline
            )
        
        if self.device != 'auto':
            self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=self.offline)
    
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
                # 抽取式 QA：对长上下文使用滑窗分片并聚合最佳答案
                encoded = self.tokenizer(
                    problem,
                    context,
                    max_length=self.max_length,
                    truncation='only_second',
                    padding='max_length',
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                    stride=self.doc_stride,
                    return_tensors='pt'
                )

                best_score = -1e18
                best_answer = ""

                for i in range(encoded['input_ids'].size(0)):
                    # 将该切片移到设备
                    inputs = {
                        'input_ids': encoded['input_ids'][i:i+1].to(self.device),
                        'attention_mask': encoded['attention_mask'][i:i+1].to(self.device)
                    }
                    if 'token_type_ids' in encoded:
                        inputs['token_type_ids'] = encoded['token_type_ids'][i:i+1].to(self.device)

                    outputs = self.model(**inputs)

                    # 仅在 context 区域选取
                    try:
                        seq_ids = encoded.sequence_ids(i)
                    except Exception:
                        seq_ids = None

                    if seq_ids is None:
                        context_mask = torch.ones(inputs['input_ids'].shape[1], dtype=torch.float32, device=self.device)
                    else:
                        context_mask = torch.tensor([1 if sid == 1 else 0 for sid in seq_ids], dtype=torch.float32, device=self.device)

                    very_neg = torch.tensor(-1e9, device=self.device)
                    mask_logits = torch.where(context_mask > 0, torch.tensor(0.0, device=self.device), very_neg)

                    start_logits = outputs.start_logits[0] + mask_logits
                    end_logits = outputs.end_logits[0] + mask_logits

                    # 选取分数最高的 span（使用和SQuAD类似的启发：start+end 最大）
                    start_idx = torch.argmax(start_logits)
                    end_idx = torch.argmax(end_logits)
                    if end_idx < start_idx:
                        end_idx = start_idx

                    score = (start_logits[start_idx] + end_logits[end_idx]).item()

                    if score > best_score:
                        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                        answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                        best_score = score
                        best_answer = answer_text

                return best_answer
            
            else:  # generative or seq2seq
                # 生成式 QA
                input_text = f"Question: {problem}\nContext: {context}\nAnswer:"
                
                inputs = self.tokenizer(
                    input_text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)

                # 对 LED 增设全局注意力（将第一个 token 设为全局）
                model_name_lower = (self.model_name or "").lower()
                if "led" in model_name_lower and "global_attention_mask" not in inputs:
                    input_ids = inputs["input_ids"]
                    global_attention_mask = torch.zeros_like(input_ids)
                    global_attention_mask[:, 0] = 1
                    inputs["global_attention_mask"] = global_attention_mask
                
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
    num_workers: int = 0,
    doc_stride: int = 128
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
        model_type=model_type,
        doc_stride=doc_stride
    )
    
    val_dataset = QADataset(
        val_path,
        tokenizer,
        max_length=max_length,
        model_type=model_type,
        doc_stride=doc_stride
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

