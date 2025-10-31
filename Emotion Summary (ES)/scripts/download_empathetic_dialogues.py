# -*- coding: utf-8 -*-
"""
下载并转换 Empathetic Dialogues 数据集为 ES 格式
数据源: https://huggingface.co/datasets/facebook/empathetic_dialogues
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def download_and_convert():
    """下载 Empathetic Dialogues 并转换为 ES 格式"""
    
    print("="*70)
    print("开始下载 Empathetic Dialogues 数据集")
    print("="*70)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 需要安装 datasets 库")
        print("请运行: pip install datasets")
        return
    
    # 下载数据集
    print("\n📥 正在从 Hugging Face 下载数据集...")
    try:
        # 新版本需要 trust_remote_code=True
        dataset = load_dataset("facebook/empathetic_dialogues", trust_remote_code=True)
    except Exception as e:
        print(f"⚠️  使用 trust_remote_code 参数重试...")
        dataset = load_dataset("facebook/empathetic_dialogues", trust_remote_code=True)
    
    print(f"✅ 下载完成！")
    print(f"   训练集: {len(dataset['train'])} 条对话片段")
    print(f"   验证集: {len(dataset['validation'])} 条对话片段")
    
    # 按对话ID组织数据
    print("\n🔄 正在组织对话数据...")
    train_conversations = organize_by_conversation(dataset['train'])
    val_conversations = organize_by_conversation(dataset['validation'])
    
    # 合并所有对话
    all_conversations = list(train_conversations.values()) + list(val_conversations.values())
    print(f"✅ 总共 {len(all_conversations)} 个完整对话")
    
    # 随机打乱
    random.shuffle(all_conversations)
    
    # 分割为 80% 训练 + 20% 验证
    split_idx = int(len(all_conversations) * 0.8)
    train_convs = all_conversations[:split_idx]
    val_convs = all_conversations[split_idx:]
    
    print(f"\n📊 数据分割:")
    print(f"   训练集: {len(train_convs)} 个对话 (80%)")
    print(f"   验证集: {len(val_convs)} 个对话 (20%)")
    
    # 转换为 ES 格式
    print("\n🔄 正在转换为 ES 格式...")
    train_es_data = [convert_to_es_format(conv, idx) for idx, conv in enumerate(train_convs)]
    val_es_data = [convert_to_es_format(conv, idx) for idx, conv in enumerate(val_convs)]
    
    # 保存数据
    print("\n💾 正在保存数据...")
    
    # 创建目录
    train_dir = Path("data/train")
    val_dir = Path("data/validation")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练集
    train_file = train_dir / "Emotion_Summary.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_es_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存验证集
    val_file = val_dir / "Emotion_Summary.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_es_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 数据保存成功！")
    print(f"   训练集: {train_file} ({len(train_es_data)} 样本)")
    print(f"   验证集: {val_file} ({len(val_es_data)} 样本)")
    
    # 显示示例
    print("\n" + "="*70)
    print("📝 数据示例 (训练集第一条):")
    print("="*70)
    print(json.dumps(train_es_data[0], ensure_ascii=False, indent=2)[:500] + "...")
    
    print("\n" + "="*70)
    print("✅ 所有操作完成！")
    print("="*70)
    print("\n📝 下一步:")
    print("   1. cd \"Emotion Summary (ES)\"")
    print("   2. python scripts/simple_train.py")
    print("="*70)

def organize_by_conversation(dataset):
    """按对话ID组织数据"""
    conversations = defaultdict(list)
    
    for item in dataset:
        conv_id = item['conv_id']
        conversations[conv_id].append(item)
    
    # 按 utterance_idx 排序每个对话
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x['utterance_idx'])
    
    return conversations

def convert_to_es_format(conversation, item_id):
    """
    将 Empathetic Dialogues 对话转换为 ES 格式
    
    格式映射:
    - case_description: 初始情绪上下文和提示
    - consultation_process: 对话内容
    - experience_and_reflection: 生成的反思总结
    """
    
    if not conversation:
        return {
            "id": item_id,
            "case_description": ["No conversation data available."],
            "consultation_process": [],
            "experience_and_reflection": "No data to reflect upon."
        }
    
    # 获取情绪上下文和初始提示
    first_item = conversation[0]
    emotion_context = first_item.get('context', 'unknown')
    initial_prompt = first_item.get('prompt', '')
    
    # 构建 case_description
    case_description = [
        f"Emotional Context: {emotion_context.capitalize()}",
        f"Initial Situation: {initial_prompt}" if initial_prompt else f"A conversation about {emotion_context} feelings."
    ]
    
    # 构建 consultation_process (对话内容)
    consultation_process = []
    for item in conversation:
        speaker = "Person A" if item['speaker_idx'] == 0 else "Person B"
        utterance = item['utterance'].replace('_comma_', ',')
        consultation_process.append(f"{speaker}: {utterance}")
    
    # 生成 experience_and_reflection
    reflection = generate_reflection(emotion_context, conversation)
    
    return {
        "id": item_id,
        "case_description": case_description,
        "consultation_process": consultation_process,
        "experience_and_reflection": reflection
    }

def generate_reflection(emotion_context, conversation):
    """
    基于对话内容生成经验与反思
    """
    
    num_turns = len(conversation)
    
    reflection_parts = []
    
    # 引言
    reflection_parts.append(
        f"This case presents a conversation centered around {emotion_context} emotions. "
        f"Through {num_turns} exchanges, we observe the development of an empathetic dialogue "
        f"that demonstrates the importance of active listening and emotional validation."
    )
    
    # 对话分析
    reflection_parts.append(
        "\n\nIn analyzing this conversation, several key therapeutic elements emerge. "
        "The dialogue showcases how empathetic responses can create a safe space for "
        "emotional expression. Each participant's contribution reflects an attempt to "
        "understand and validate the other's emotional experience, which is fundamental "
        "to building trust and rapport in any supportive relationship."
    )
    
    # 情绪维度分析
    if emotion_context in ['sad', 'grief', 'lonely', 'devastated', 'disappointed']:
        reflection_parts.append(
            "\n\nThe emotional tone of this exchange reflects themes of loss and vulnerability. "
            "In such conversations, it is crucial to provide space for the expression of difficult "
            "emotions without rushing to fix or minimize them. The progression of the dialogue "
            "demonstrates how acknowledging pain can be more healing than offering premature reassurance."
        )
    elif emotion_context in ['anxious', 'afraid', 'terrified', 'nervous']:
        reflection_parts.append(
            "\n\nThis conversation touches on themes of fear and uncertainty. When addressing anxiety, "
            "it is important to validate the person's concerns while gently exploring their roots. "
            "The dialogue illustrates how naming and discussing fears can reduce their power and "
            "help individuals feel less alone in their struggles."
        )
    elif emotion_context in ['joyful', 'excited', 'proud', 'grateful', 'hopeful']:
        reflection_parts.append(
            "\n\nThe positive emotional quality of this exchange highlights the importance of "
            "celebrating joy and achievement. Sharing positive experiences strengthens relationships "
            "and builds resilience. This conversation demonstrates how mutual celebration can deepen "
            "connections and create lasting positive memories."
        )
    elif emotion_context in ['angry', 'annoyed', 'furious', 'jealous']:
        reflection_parts.append(
            "\n\nThis dialogue addresses challenging emotions that often require careful navigation. "
            "When dealing with anger or frustration, it is essential to validate the emotion while "
            "helping the person explore its underlying causes. The conversation shows how creating "
            "space for these feelings can lead to greater self-understanding and eventual resolution."
        )
    else:
        reflection_parts.append(
            "\n\nThe emotional landscape of this conversation is complex and multifaceted. "
            "Each exchange reveals layers of human experience that require careful attention "
            "and thoughtful response. The progression of the dialogue demonstrates the value "
            "of staying present with whatever emotions arise."
        )
    
    # 专业反思
    reflection_parts.append(
        "\n\nFrom a therapeutic perspective, this interaction underscores several important principles. "
        "First, the power of presence and attentive listening cannot be overstated. Second, "
        "validating emotions without judgment creates the foundation for genuine connection. "
        "Third, allowing conversations to unfold naturally, without forcing solutions, often "
        "leads to organic insights and healing."
    )
    
    # 结论
    reflection_parts.append(
        "\n\nReflecting on this case reinforces the understanding that effective emotional support "
        "is less about having perfect answers and more about creating a space where people feel "
        "heard, understood, and accepted. These principles apply across various contexts, from "
        "professional counseling to everyday interactions with friends and family. Each conversation "
        "is an opportunity to practice empathy and deepen our understanding of the human experience."
    )
    
    return "".join(reflection_parts)

if __name__ == "__main__":
    random.seed(42)  # 设置随机种子以保证可复现性
    download_and_convert()


