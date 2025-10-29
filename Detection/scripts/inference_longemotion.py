"""
LongEmotion 测试集推理脚本
专门处理 LongEmotion 格式的长文本多段落情感检测
"""
import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from collections import Counter

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import BertTokenizer, BertModel


class LongEmotionInference:
    """LongEmotion 推理器"""
    
    # 情感标签映射 (dair数据集的6类情感)
    EMOTION_LABELS = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        batch_size: int = 16
    ):
        """
        初始化推理器
        
        Args:
            model_path: 模型权重文件路径 (best_model.pt)
            device: 设备
            max_length: 最大序列长度
            batch_size: 批次大小
        """
        # 检查设备可用性
        if device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDA不可用，切换到CPU")
            device = "cpu"
        
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        print(f"正在加载模型从 {model_path}...")
        print(f"使用设备: {device}")
        
        # 加载分词器
        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        except Exception as e:
            print(f"加载分词器失败，尝试从本地加载: {e}")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=False)
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print("模型加载完成！")
    
    def _load_model(self, model_path: str):
        """加载训练好的模型 - 使用与训练时相同的简单结构"""
        try:
            from transformers import AutoModel
            import torch.nn as nn
            
            # 定义简单的分类器（与simple_train.py中的结构完全一致）
            class SimpleEmotionClassifier(nn.Module):
                """简单情感分类器 - 单层Linear"""
                def __init__(self, model_name, num_labels=6):
                    super().__init__()
                    self.bert = AutoModel.from_pretrained(model_name)
                    self.dropout = nn.Dropout(0.1)
                    self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
                
                def forward(self, input_ids, attention_mask):
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = outputs.pooler_output
                    pooled_output = self.dropout(pooled_output)
                    logits = self.classifier(pooled_output)
                    return logits
            
            # 创建模型
            print("创建模型结构（简单单层分类器）...")
            model = SimpleEmotionClassifier(
                model_name="bert-base-chinese",
                num_labels=6  # dair数据集的6类
            )
            
            # 加载权重
            print(f"加载模型权重...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的保存格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            # 移动到设备
            print(f"移动模型到设备: {self.device}")
            model = model.to(self.device)
            
            print("[OK] 模型加载成功！")
            return model
            
        except Exception as e:
            print(f"[ERROR] 加载模型时出错: {e}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_segment(self, text: str) -> Dict[str, Any]:
        """
        预测单个段落的情感
        
        Args:
            text: 段落文本
            
        Returns:
            预测结果: {emotion_id, emotion_name, probabilities, confidence}
        """
        # 文本预处理
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            # 对于分类任务，使用 softmax
            probabilities = torch.softmax(logits, dim=-1)
            
            # 获取最高概率的情感
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_id].item()
        
        return {
            'emotion_id': predicted_id,
            'emotion_name': self.EMOTION_LABELS[predicted_id],
            'probabilities': {
                self.EMOTION_LABELS[i]: float(probabilities[0, i])
                for i in range(len(self.EMOTION_LABELS))
            },
            'confidence': confidence
        }
    
    def predict_segments_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测多个段落的情感
        
        Args:
            texts: 段落文本列表
            
        Returns:
            预测结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # 文本预处理
            encoding = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 推理
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 对于分类任务，使用 softmax
                probabilities = torch.softmax(logits, dim=-1)
                
                # 获取最高概率的情感
                predicted_ids = torch.argmax(probabilities, dim=-1)
            
            # 处理批次结果
            for j in range(len(batch_texts)):
                predicted_id = predicted_ids[j].item()
                confidence = probabilities[j, predicted_id].item()
                
                result = {
                    'emotion_id': predicted_id,
                    'emotion_name': self.EMOTION_LABELS[predicted_id],
                    'probabilities': {
                        self.EMOTION_LABELS[k]: float(probabilities[j, k])
                        for k in range(len(self.EMOTION_LABELS))
                    },
                    'confidence': confidence
                }
                results.append(result)
        
        return results
    
    def find_unique_emotion_segment(
        self,
        segment_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        找出表达独特情感的段落
        
        在 n 个段落中，n-1 个段落表达相同情感，1 个段落表达独特情感
        
        Args:
            segment_predictions: 每个段落的预测结果
            
        Returns:
            独特情感段落信息
        """
        # 统计每种情感出现的次数
        emotion_counts = Counter([pred['emotion_name'] for pred in segment_predictions])
        
        # 找出只出现1次的情感 (独特情感)
        unique_emotions = [emotion for emotion, count in emotion_counts.items() if count == 1]
        
        if len(unique_emotions) == 1:
            # 找到独特情感
            unique_emotion = unique_emotions[0]
            
            # 找到该情感对应的段落索引
            for idx, pred in enumerate(segment_predictions):
                if pred['emotion_name'] == unique_emotion:
                    return {
                        'unique_segment_index': idx,
                        'unique_emotion': unique_emotion,
                        'confidence': pred['confidence'],
                        'emotion_distribution': dict(emotion_counts),
                        'total_segments': len(segment_predictions),
                        'status': 'success'
                    }
        
        # 如果没有找到唯一的独特情感，使用启发式方法
        # 方法1: 找出现次数最少且置信度最高的情感
        min_count = min(emotion_counts.values())
        rare_emotions = [emotion for emotion, count in emotion_counts.items() if count == min_count]
        
        # 在这些稀有情感中，找置信度最高的
        best_idx = None
        best_confidence = 0
        best_emotion = None
        
        for idx, pred in enumerate(segment_predictions):
            if pred['emotion_name'] in rare_emotions and pred['confidence'] > best_confidence:
                best_idx = idx
                best_confidence = pred['confidence']
                best_emotion = pred['emotion_name']
        
        return {
            'unique_segment_index': best_idx,
            'unique_emotion': best_emotion,
            'confidence': best_confidence,
            'emotion_distribution': dict(emotion_counts),
            'total_segments': len(segment_predictions),
            'status': 'heuristic',
            'note': f'No single unique emotion found. Used heuristic: rarest emotion ({min_count} occurrences) with highest confidence.'
        }
    
    def inference_longemotion_test(
        self,
        test_file: str,
        output_file: str,
        output_detailed: Optional[str] = None
    ):
        """
        对 LongEmotion 格式的测试集进行推理
        
        Args:
            test_file: 测试集文件路径 (JSONL格式)
            output_file: 输出文件路径 (提交格式)
            output_detailed: 详细结果输出路径 (可选)
        """
        print(f"读取测试集: {test_file}")
        
        # 读取测试集
        test_samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_samples.append(json.loads(line))
        
        print(f"测试样本数: {len(test_samples)}")
        
        # 推理结果
        results = []
        detailed_results = []
        
        # 处理每个样本
        for sample_idx, sample in enumerate(tqdm(test_samples, desc="推理进度")):
            # 提取段落
            segments = sample['text']
            
            # 提取每个段落的文本
            segment_texts = [seg['context'] for seg in segments]
            
            # 批量预测所有段落
            segment_predictions = self.predict_segments_batch(segment_texts)
            
            # 找出独特情感段落
            unique_result = self.find_unique_emotion_segment(segment_predictions)
            
            # 构建输出结果 (提交格式)
            result = {
                'sample_id': sample_idx,
                'unique_segment_index': unique_result['unique_segment_index'],
                'unique_emotion': unique_result['unique_emotion'],
                'confidence': unique_result['confidence']
            }
            results.append(result)
            
            # 构建详细结果
            if output_detailed:
                detailed_result = {
                    'sample_id': sample_idx,
                    'total_length': sample.get('length', 0),
                    'total_segments': len(segments),
                    'unique_segment_index': unique_result['unique_segment_index'],
                    'unique_emotion': unique_result['unique_emotion'],
                    'confidence': unique_result['confidence'],
                    'emotion_distribution': unique_result['emotion_distribution'],
                    'status': unique_result['status'],
                    'segment_predictions': [
                        {
                            'index': seg['index'],
                            'text_preview': seg['context'][:100] + '...' if len(seg['context']) > 100 else seg['context'],
                            'predicted_emotion': pred['emotion_name'],
                            'confidence': pred['confidence']
                        }
                        for seg, pred in zip(segments, segment_predictions)
                    ]
                }
                
                if 'note' in unique_result:
                    detailed_result['note'] = unique_result['note']
                
                detailed_results.append(detailed_result)
        
        # 保存结果
        print(f"\n保存结果到: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        # 保存详细结果
        if output_detailed:
            print(f"保存详细结果到: {output_detailed}")
            with open(output_detailed, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 统计信息
        print("\n=== 推理统计 ===")
        print(f"总样本数: {len(results)}")
        
        emotion_counts = Counter([r['unique_emotion'] for r in results])
        print(f"\n独特情感分布:")
        for emotion, count in emotion_counts.most_common():
            print(f"  {emotion}: {count} ({count/len(results)*100:.1f}%)")
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"\n平均置信度: {avg_confidence:.4f}")
        
        # 成功率统计
        if detailed_results:
            success_count = sum(1 for r in detailed_results if r['status'] == 'success')
            print(f"找到唯一独特情感的样本: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
        print("\n推理完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LongEmotion 测试集推理")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../model/best_model.pt",
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../test_data/test.jsonl",
        help="测试集文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../submission/predictions.jsonl",
        help="预测结果输出路径"
    )
    parser.add_argument(
        "--output_detailed",
        type=str,
        default="../submission/predictions_detailed.json",
        help="详细结果输出路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批次大小"
    )
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = LongEmotionInference(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # 执行推理
    inference.inference_longemotion_test(
        test_file=args.test_file,
        output_file=args.output_file,
        output_detailed=args.output_detailed
    )


if __name__ == "__main__":
    main()

