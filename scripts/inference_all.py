"""
批量推理所有任务
生成提交文件
"""
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
import torch

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.classification_model import EmotionClassificationModel
from models.detection_model import EmotionDetectionModelWrapper
from models.conversation_model import EmotionConversationModel
from models.summary_model import EmotionSummaryModel
from models.qa_model import EmotionQAModel


class InferenceEngine:
    """推理引擎"""
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        data_dir: str = "./data",
        output_dir: str = "./results",
        device: str = "cuda"
    ):
        """
        初始化推理引擎
        
        Args:
            checkpoint_dir: 检查点目录
            data_dir: 数据目录
            output_dir: 输出目录
            device: 设备
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def infer_classification(self, batch_size: int = 32):
        """推理分类任务"""
        print("\n" + "="*60)
        print("推理任务: 情感分类 (Emotion Classification)")
        print("="*60)
        
        # 加载模型
        model_path = self.checkpoint_dir / "classification" / "best_model"
        model = EmotionClassificationModel.load(str(model_path), device=self.device)
        
        # 加载测试数据
        test_dataset = load_from_disk(str(self.data_dir / "classification" / "test"))
        
        print(f"数据集字段: {test_dataset.column_names}")
        
        # 推理 - 使用 context 字段，截取前512字符
        results = []
        texts = [item['context'][:512] for item in test_dataset]
        
        for i in tqdm(range(0, len(texts), batch_size), desc="推理中"):
            batch_texts = texts[i:i + batch_size]
            predictions = model.predict(batch_texts)
            # 添加原始数据信息
            for j, pred in enumerate(predictions):
                idx = i + j
                pred['id'] = test_dataset[idx]['id']
                pred['Subject'] = test_dataset[idx]['Subject']
                pred['choices'] = test_dataset[idx]['choices']
            results.extend(predictions)
        
        # 保存结果
        output_file = self.output_dir / "classification_test.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ 结果已保存到 {output_file}")
        return results
    
    def infer_detection(self, batch_size: int = 32):
        """推理检测任务"""
        print("\n" + "="*60)
        print("推理任务: 情感检测 (Emotion Detection)")
        print("="*60)
        
        # 加载模型
        model_path = self.checkpoint_dir / "detection" / "best_model"
        model = EmotionDetectionModelWrapper.load(str(model_path), device=self.device)
        
        # 加载测试数据
        test_dataset = load_from_disk(str(self.data_dir / "detection" / "test"))
        
        print(f"数据集字段: {test_dataset.column_names}")
        
        # 推理 - text 是列表，提取第一个 context
        results = []
        texts = []
        for item in test_dataset:
            if item['text'] and isinstance(item['text'], list):
                text = item['text'][0].get('context', '')[:512]
            else:
                text = ''
            texts.append(text)
        
        for i in tqdm(range(0, len(texts), batch_size), desc="推理中"):
            batch_texts = texts[i:i + batch_size]
            predictions = model.predict(batch_texts)
            results.extend(predictions)
        
        # 保存结果
        output_file = self.output_dir / "detection_test.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ 结果已保存到 {output_file}")
        return results
    
    def infer_conversation(self):
        """推理对话任务"""
        print("\n" + "="*60)
        print("推理任务: 情感对话 (Emotion Conversation)")
        print("="*60)
        
        # 加载模型
        model_path = self.checkpoint_dir / "conversation" / "best_model"
        
        # 如果没有微调模型，使用基础模型
        if not model_path.exists():
            print("⚠ 未找到微调模型，使用基础模型")
            model = EmotionConversationModel(device=self.device)
        else:
            model = EmotionConversationModel.load(str(model_path), device=self.device)
        
        # 加载测试数据
        test_dataset = load_from_disk(str(self.data_dir / "conversation" / "test"))
        
        print(f"数据集字段: {test_dataset.column_names}")
        
        # 推理 - 使用 conversation_history 字段
        results = []
        
        for item in tqdm(test_dataset, desc="推理中"):
            # 使用完整对话历史的前1024字符作为上下文
            context = item['conversation_history'][:1024]
            emotion = None  # 数据集中没有指定情感
            
            response = model.generate_response(context, emotion)
            
            results.append({
                'id': item['id'],
                'context': context,
                'response': response
            })
        
        # 保存结果
        output_file = self.output_dir / "conversation_test.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ 结果已保存到 {output_file}")
        return results
    
    def infer_summary(self, batch_size: int = 8):
        """推理摘要任务"""
        print("\n" + "="*60)
        print("推理任务: 情感摘要 (Emotion Summary)")
        print("="*60)
        
        # 加载模型
        model_path = self.checkpoint_dir / "summary" / "best_model"
        model = EmotionSummaryModel.load(str(model_path), device=self.device)
        
        # 加载测试数据
        test_dataset = load_from_disk(str(self.data_dir / "summary" / "test"))
        
        print(f"数据集字段: {test_dataset.column_names}")
        
        # 推理 - 合并 consultation_process 作为输入
        results = []
        texts = []
        for item in test_dataset:
            if isinstance(item['consultation_process'], list):
                text = ' '.join(item['consultation_process'])[:1024]
            else:
                text = str(item['consultation_process'])[:1024]
            texts.append(text)
        
        for i in tqdm(range(0, len(texts), batch_size), desc="推理中"):
            batch_texts = texts[i:i + batch_size]
            predictions = model.batch_generate(batch_texts, batch_size=batch_size)
            # 添加原始数据信息
            for j, pred in enumerate(predictions):
                idx = i + j
                pred['id'] = test_dataset[idx]['id']
            results.extend(predictions)
        
        # 保存结果
        output_file = self.output_dir / "summary_test.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ 结果已保存到 {output_file}")
        return results
    
    def infer_qa(self):
        """推理问答任务"""
        print("\n" + "="*60)
        print("推理任务: 情感问答 (Emotion QA)")
        print("="*60)
        
        # 加载模型
        model_path = self.checkpoint_dir / "qa" / "best_model"
        model = EmotionQAModel.load(str(model_path), device=self.device)
        
        # 加载测试数据
        test_dataset = load_from_disk(str(self.data_dir / "qa" / "test"))
        
        print(f"数据集字段: {test_dataset.column_names}")
        
        # 推理 - 使用 problem 和 context 字段
        results = []
        
        for item in tqdm(test_dataset, desc="推理中"):
            question = item['problem']
            context = item['context'][:512]  # 截取前512字符
            
            answer_info = model.extract_answer(question, context)
            
            results.append({
                'problem': question,
                'source': item.get('source', ''),
                'context': context,
                'answer': answer_info['answer'],
                'confidence': answer_info['confidence']
            })
        
        # 保存结果
        output_file = self.output_dir / "qa_test.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ 结果已保存到 {output_file}")
        return results
    
    def infer_all(self, tasks: list = None):
        """推理所有任务"""
        all_tasks = {
            "classification": self.infer_classification,
            "detection": self.infer_detection,
            "conversation": self.infer_conversation,
            "summary": self.infer_summary,
            "qa": self.infer_qa
        }
        
        if tasks is None:
            tasks = list(all_tasks.keys())
        
        results = {}
        
        print("\n" + "="*60)
        print(f"开始推理 {len(tasks)} 个任务")
        print("="*60)
        
        for task in tasks:
            if task in all_tasks:
                try:
                    task_results = all_tasks[task]()
                    results[task] = {"status": "success", "count": len(task_results)}
                except Exception as e:
                    print(f"\n✗ {task} 推理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    results[task] = {"status": "failed", "error": str(e)}
            else:
                print(f"\n✗ 未知任务: {task}")
        
        print("\n" + "="*60)
        print("推理总结")
        print("="*60)
        
        for task, result in results.items():
            status = "✓" if result['status'] == 'success' else "✗"
            if result['status'] == 'success':
                print(f"{status} {task}: {result['count']} 条结果")
            else:
                print(f"{status} {task}: {result['status']}")
        
        print(f"\n所有结果已保存到 {self.output_dir}")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="推理 LongEmotion 任务")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="检查点目录"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="数据目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="输出目录"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="要推理的任务 (classification detection conversation summary qa)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )
    
    args = parser.parse_args()
    
    # 创建推理引擎
    engine = InferenceEngine(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # 推理
    results = engine.infer_all(tasks=args.tasks)

