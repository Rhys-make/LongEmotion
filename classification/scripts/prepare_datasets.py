#!/usr/bin/env python3
"""
数据准备脚本：下载并格式化 GoEmotions 和 LongEmotion 数据集
用于情绪分类任务的训练集、验证集和测试集准备
"""

import json
import os
import random
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import spacy
import numpy as np

# 全局变量：spacy 模型
nlp = None


def load_spacy_model():
    """加载 spacy 英文模型"""
    global nlp
    if nlp is None:
        try:
            print("📦 正在加载 spacy 英文模型...")
            nlp = spacy.load("en_core_web_sm")
            print("✅ spacy 模型加载成功")
        except OSError:
            print("⚠️  spacy 英文模型未安装，正在下载...")
            print("   请稍等，这可能需要几分钟...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            print("✅ spacy 模型下载并加载成功")
    return nlp


def extract_subject(text):
    """
    使用 spacy 提取句子的情感主体
    
    策略：
    1. 优先查找主语（包括代词）
    2. 如果没有主语，查找直接宾语
    3. 查找命名实体（人名、地名、组织等）
    4. 提取第一个名词块
    5. 以上都失败则返回 "unknown"
    
    Args:
        text: 英文文本
    
    Returns:
        主体字符串
    """
    try:
        nlp_model = load_spacy_model()
        doc = nlp_model(text)
        
        # 策略1: 查找主语（nsubj, nsubjpass）- 包括代词
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                # 获取主语及其修饰词
                subject_tokens = [token]
                for child in token.children:
                    if child.dep_ in ("compound", "amod", "det", "poss"):
                        subject_tokens.append(child)
                
                subject_tokens.sort(key=lambda x: x.i)
                subject = " ".join([t.text for t in subject_tokens])
                
                # 对于代词，保留常见的人称代词
                if token.pos_ == "PRON":
                    # 保留 I, you, he, she, we, they 等
                    if token.text.lower() in ["i", "you", "he", "she", "we", "they", "it"]:
                        return token.text
                    else:
                        continue  # 跳过其他代词，继续查找
                
                if len(subject) <= 50:
                    return subject
        
        # 策略2: 查找直接宾语（dobj）
        for token in doc:
            if token.dep_ == "dobj" and token.pos_ in ("NOUN", "PROPN"):
                dobj_tokens = [token]
                for child in token.children:
                    if child.dep_ in ("compound", "amod", "det"):
                        dobj_tokens.append(child)
                
                dobj_tokens.sort(key=lambda x: x.i)
                dobj = " ".join([t.text for t in dobj_tokens])
                if len(dobj) <= 50:
                    return dobj
        
        # 策略3: 查找命名实体（人名、组织等优先）
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT"):
                if len(ent.text) <= 50:
                    return ent.text
        
        # 策略4: 提取第一个有意义的名词块
        for chunk in doc.noun_chunks:
            # 排除太短或太长的块
            if 1 <= len(chunk.text.split()) <= 5 and len(chunk.text) <= 50:
                # 排除纯代词块
                if chunk.root.pos_ != "PRON" or chunk.root.text.lower() in ["i", "you", "he", "she", "we", "they"]:
                    return chunk.text
        
        # 策略5: 查找任意名词
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and len(token.text) <= 50:
                return token.text
        
        # 所有策略都失败，返回 unknown
        return "unknown"
    
    except Exception as e:
        return "unknown"


def ensure_dir(file_path):
    """确保目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")


def convert_go_emotions_to_format(dataset, split_name, choices):
    """
    将 GoEmotions 数据集转换为指定格式
    
    Args:
        dataset: GoEmotions 数据集的某个切分
        split_name: 切分名称（用于日志）
        choices: 情绪类别列表
    
    Returns:
        转换后的数据列表
    """
    converted_data = []
    
    print(f"\n🔄 正在转换 {split_name} 数据集...")
    
    # 确保 spacy 模型已加载
    load_spacy_model()
    
    for idx, example in enumerate(tqdm(dataset, desc=f"处理 {split_name}")):
        # 获取文本
        text = example['text']
        
        # 获取标签（GoEmotions 是多标签，我们取第一个标签）
        labels = example['labels']
        
        # 如果有多个标签，取第一个；如果没有标签，跳过
        if len(labels) == 0:
            continue
        
        # 获取第一个标签的名称
        label_idx = labels[0]
        answer = choices[label_idx]
        
        # 使用 spacy 提取主语
        subject = extract_subject(text)
        
        # 构建转换后的格式
        converted_example = {
            "id": idx,
            "Context": text,
            "Subject": subject,
            "Choices": choices,
            "Answer": answer
        }
        
        converted_data.append(converted_example)
    
    print(f"✅ {split_name} 转换完成，共 {len(converted_data)} 条数据")
    return converted_data


def save_to_jsonl(data, file_path):
    """保存数据为 JSONL 格式"""
    ensure_dir(file_path)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"💾 已保存到: {file_path}")


def download_and_process_go_emotions():
    """下载并处理 GoEmotions 数据集"""
    print("\n" + "="*60)
    print("📥 开始下载 GoEmotions 数据集...")
    print("="*60)
    
    # 加载数据集
    ds = load_dataset("go_emotions")
    
    # 获取所有情绪类别
    choices = ds["train"].features["labels"].feature.names
    print(f"\n📋 情绪类别共 {len(choices)} 个:")
    print(f"   {', '.join(choices[:10])}...")
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    # 处理训练集
    train_data = convert_go_emotions_to_format(ds["train"], "train", choices)
    train_path = project_root / "data" / "classification" / "train.jsonl"
    save_to_jsonl(train_data, str(train_path))
    
    # 处理验证集
    val_data = convert_go_emotions_to_format(ds["validation"], "validation", choices)
    val_path = project_root / "data" / "classification" / "validation.jsonl"
    save_to_jsonl(val_data, str(val_path))
    
    return choices


def download_and_process_long_emotion():
    """下载并处理 LongEmotion 数据集的测试集"""
    print("\n" + "="*60)
    print("📥 开始下载 LongEmotion 数据集（classification 部分）...")
    print("="*60)
    
    try:
        # 方法1: 尝试直接加载 classification 数据文件
        from huggingface_hub import hf_hub_download
        
        # 下载 classification 数据文件
        print("正在从 Hugging Face 下载 Emotion_Classification.jsonl...")
        file_path = hf_hub_download(
            repo_id="LongEmotion/LongEmotion",
            filename="Emotion Classification/Emotion_Classification.jsonl",
            repo_type="dataset"
        )
        
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        
        # 处理测试集
        print(f"\n🔄 正在转换 test 数据集...")
        test_data = []
        
        # 读取 JSONL 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="处理 test"):
                if line.strip():
                    example = json.loads(line)
                    test_data.append(example)
        
        test_path = project_root / "data" / "classification" / "test.jsonl"
        save_to_jsonl(test_data, str(test_path))
        
        print(f"✅ test 转换完成，共 {len(test_data)} 条数据")
        
    except Exception as e:
        print(f"⚠️  下载 LongEmotion 数据集时出错: {e}")
        print("   尝试使用备用方法...")
        
        # 方法2: 尝试使用 load_dataset 加载单个数据文件
        try:
            print("   使用备用方法加载数据...")
            ds = load_dataset(
                "json",
                data_files={
                    "test": "hf://datasets/LongEmotion/LongEmotion/Emotion Classification/Emotion_Classification.jsonl"
                }
            )
            
            project_root = Path(__file__).parent.parent.parent
            test_data = list(ds["test"])
            
            test_path = project_root / "data" / "classification" / "test.jsonl"
            save_to_jsonl(test_data, str(test_path))
            
            print(f"✅ test 转换完成，共 {len(test_data)} 条数据")
            
        except Exception as e2:
            print(f"⚠️  备用方法也失败了: {e2}")
            print("   请手动下载数据集或检查网络连接")


def create_combined_emotion_sample(sample1, sample2, all_choices):
    """
    合并两个样本，创建组合情绪样本
    使训练集更接近测试集（包含组合情绪）
    """
    # 合并文本（模拟长文本）
    combined_context = f"{sample1['Context']} {sample2['Context']}"
    
    # 组合情绪标签
    emotion1 = sample1['Answer']
    emotion2 = sample2['Answer']
    
    # 避免重复
    if emotion1 == emotion2:
        combined_answer = emotion1
    else:
        # 按字母顺序排列（保持一致性）
        emotions = sorted([emotion1, emotion2])
        combined_answer = f"{emotions[0]} & {emotions[1]}"
    
    # 创建新的Choices（包含原始情绪和组合情绪）
    # 随机选择4-6个其他情绪作为干扰项
    num_distractors = random.randint(3, 6)
    distractors = random.sample([c for c in all_choices if c not in [emotion1, emotion2]], 
                                 min(num_distractors, len(all_choices) - 2))
    
    choices = [emotion1, emotion2]
    if emotion1 != emotion2:
        choices.append(combined_answer)
    choices.extend(distractors)
    random.shuffle(choices)
    
    return {
        'id': sample1['id'],
        'Context': combined_context,
        'Subject': sample1['Subject'],
        'Choices': choices,
        'Answer': combined_answer
    }


def augment_training_data(train_data, augment_ratio=0.3, seed=42):
    """
    增强训练数据，生成组合情绪样本
    
    Args:
        train_data: 原始训练数据
        augment_ratio: 增强比例（生成原始数据量的x倍组合样本）
        seed: 随机种子
    
    Returns:
        增强后的训练数据
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 获取所有唯一的情绪标签
    all_emotions = set()
    for item in train_data:
        if 'Choices' in item:
            all_emotions.update(item['Choices'])
        if 'Answer' in item:
            all_emotions.add(item['Answer'])
    all_emotions = sorted(list(all_emotions))
    
    print(f"\n📊 数据增强:")
    print(f"   原始数据: {len(train_data)} 条")
    print(f"   情绪类别: {len(all_emotions)} 个")
    
    # 生成组合情绪样本
    num_augmented = int(len(train_data) * augment_ratio)
    augmented_samples = []
    
    print(f"   生成组合情绪样本: {num_augmented} 条...")
    
    for i in tqdm(range(num_augmented), desc="增强数据"):
        # 随机选择两个样本
        sample1, sample2 = random.sample(train_data, 2)
        
        # 创建组合样本
        combined_sample = create_combined_emotion_sample(sample1, sample2, all_emotions)
        combined_sample['id'] = len(train_data) + i
        
        augmented_samples.append(combined_sample)
    
    # 合并原始数据和增强数据
    final_data = train_data + augmented_samples
    
    # 打乱顺序
    random.shuffle(final_data)
    
    # 重新分配ID
    for i, item in enumerate(final_data):
        item['id'] = i
    
    # 统计组合情绪比例
    combined_count = sum(1 for item in final_data if ' & ' in item['Answer'])
    
    print(f"\n✅ 数据增强完成:")
    print(f"   原始样本: {len(train_data)}")
    print(f"   增强样本: {len(augmented_samples)}")
    print(f"   总样本数: {len(final_data)}")
    print(f"   组合情绪样本: {combined_count} ({combined_count/len(final_data)*100:.1f}%)")
    
    return final_data


def main():
    """主函数"""
    # 命令行参数
    parser = argparse.ArgumentParser(description='情绪分类数据集准备脚本')
    parser.add_argument('--augment', action='store_true',
                        help='是否进行数据增强（生成组合情绪样本）')
    parser.add_argument('--augment_ratio', type=float, default=0.3,
                        help='数据增强比例（默认0.3，即生成30%%的组合样本）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认42）')
    args = parser.parse_args()
    
    print("\n" + "🎯" + "="*58)
    print("       情绪分类数据集准备脚本")
    print("="*59 + "🎯\n")
    
    if args.augment:
        print("📌 数据增强模式: 开启")
        print(f"   增强比例: {args.augment_ratio * 100:.0f}%")
    else:
        print("📌 数据增强模式: 关闭")
        print("   提示: 使用 --augment 参数启用数据增强，提升组合情绪预测准确度")
    
    # 1. 处理 GoEmotions 数据集（训练集和验证集）
    try:
        choices = download_and_process_go_emotions()
        print("\n✅ GoEmotions 数据集处理完成！")
        
        # 数据增强（如果启用）
        if args.augment:
            # 读取训练数据
            project_root = Path(__file__).parent.parent.parent
            train_path = project_root / "data" / "classification" / "train.jsonl"
            
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        train_data.append(json.loads(line))
            
            # 增强数据
            augmented_data = augment_training_data(
                train_data, 
                augment_ratio=args.augment_ratio,
                seed=args.seed
            )
            
            # 保存增强后的训练数据
            save_to_jsonl(augmented_data, str(train_path))
        
    except Exception as e:
        print(f"\n❌ GoEmotions 数据集处理失败: {e}")
        return
    
    # 2. 处理 LongEmotion 数据集（测试集）
    try:
        download_and_process_long_emotion()
        print("\n✅ LongEmotion 数据集处理完成！")
    except Exception as e:
        print(f"\n⚠️  LongEmotion 数据集处理出现问题: {e}")
    
    # 完成
    print("\n" + "🎉" + "="*58)
    print("       所有数据集准备完成！")
    print("="*59 + "🎉\n")
    
    # 显示生成的文件
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "classification"
    
    print("📁 生成的文件:")
    for file in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
        file_path = data_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / 1024 / 1024  # MB
            print(f"   ✓ {file} ({size:.2f} MB)")
        else:
            print(f"   ✗ {file} (未生成)")
    
    print("\n💡 提示: 数据集已保存到 data/classification/ 目录")


if __name__ == "__main__":
    main()

