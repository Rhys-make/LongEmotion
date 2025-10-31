# -*- coding: utf-8 -*-
"""
下载并转换 Empathetic Dialogues 数据集为 ES 格式 (使用 Parquet 格式)
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
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ 需要安装 datasets 和 huggingface_hub 库")
        print("请运行: pip install datasets huggingface_hub")
        return
    
    # 尝试使用旧版本的方法加载数据
    print("\n📥 正在从 Hugging Face 下载数据集...")
    print("⚠️  如果失败，将创建示例数据...")
    
    try:
        # 尝试使用 datasets 库
        import datasets
        datasets.logging.set_verbosity_error()
        
        # 降级方法：直接从 git 历史版本加载
        dataset = load_dataset(
            "facebook/empathetic_dialogues",
            download_mode="force_redownload",
            verification_mode="no_checks"
        )
        
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
        
    except Exception as e:
        print(f"⚠️  下载失败: {e}")
        print("📝 将创建示例英文数据集...")
        
        # 创建示例数据
        all_conversations = create_sample_english_data()
        print(f"✅ 创建了 {len(all_conversations)} 个示例对话")
    
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
    example = json.dumps(train_es_data[0], ensure_ascii=False, indent=2)
    if len(example) > 1000:
        print(example[:1000] + "\n  ... (truncated) ...")
    else:
        print(example)
    
    print("\n" + "="*70)
    print("✅ 所有操作完成！")
    print("="*70)
    print("\n📝 下一步:")
    print("   1. 确认数据正确")
    print("   2. python scripts/simple_train.py")
    print("="*70)

def create_sample_english_data():
    """创建示例英文心理咨询数据"""
    
    sample_conversations = []
    
    # 示例对话模板
    templates = [
        {
            "emotion": "anxious",
            "situation": "I have a big presentation at work tomorrow and I can't stop worrying about it.",
            "responses": [
                "I understand how nerve-wracking presentations can be. What specifically are you worried about?",
                "I'm mostly worried about forgetting what to say or making mistakes in front of everyone.",
                "Those are common concerns. Have you prepared well for the presentation?",
                "Yes, I've practiced several times, but I still feel anxious.",
                "It sounds like you've done your preparation. Sometimes anxiety persists even when we're well-prepared. Have you tried any relaxation techniques?",
                "I haven't really. Maybe I should try some breathing exercises.",
                "That's a great idea. Deep breathing can help calm your nervous system. Remember, it's okay to be nervous - it shows you care about doing well."
            ]
        },
        {
            "emotion": "sad",
            "situation": "My best friend moved to another country last month, and I've been feeling really lonely.",
            "responses": [
                "Moving away from close friends is never easy. How have you been coping with this change?",
                "Not very well, honestly. We used to see each other almost every day.",
                "That's a significant change in your daily life. Have you been able to keep in touch?",
                "We video call once a week, but it's not the same as being together in person.",
                "You're right, it's definitely different. The good news is that you're maintaining contact. How are you filling the time you used to spend together?",
                "I've been trying to join some local groups to meet new people, but it's hard.",
                "That shows real strength - actively working to build new connections while grieving the loss of daily contact with your friend. These things take time."
            ]
        },
        {
            "emotion": "stressed",
            "situation": "I'm juggling work deadlines, family responsibilities, and trying to maintain my health. I feel overwhelmed.",
            "responses": [
                "It sounds like you have a lot on your plate right now. What feels most overwhelming to you?",
                "Everything, honestly. I feel like I'm failing at all of it.",
                "When we're stressed, it can feel that way. Let's break it down - which area is causing you the most stress right now?",
                "Probably work. I have three major projects due next week and I haven't made enough progress.",
                "That does sound stressful. Have you been able to prioritize which project needs attention first?",
                "Not really. I've been jumping between them all and not making real progress on any.",
                "That scattered approach often leads to more stress. What if we tried to identify just one project to focus on first? Sometimes focusing on one thing at a time can reduce that overwhelming feeling."
            ]
        },
        {
            "emotion": "grief",
            "situation": "My grandmother passed away six months ago, and I still cry almost every day.",
            "responses": [
                "I'm so sorry for your loss. Grief doesn't follow a timeline, and six months is still very recent. Tell me about your grandmother.",
                "She raised me when my parents were working. She was everything to me.",
                "It sounds like you had a very special bond. Losing someone who played such a central role in your life is profound.",
                "Everyone tells me I should be getting over it by now, but I'm not.",
                "There's no 'should' when it comes to grief. The depth of your grief reflects the depth of your love. How are you honoring her memory?",
                "I haven't really thought about that. I've just been trying to get through each day.",
                "Getting through each day is an accomplishment when you're grieving. When you're ready, finding ways to honor her memory might bring some comfort. But there's no rush - grief takes as long as it takes."
            ]
        },
        {
            "emotion": "hopeful",
            "situation": "After years of struggling with depression, I finally feel like I'm starting to see improvements.",
            "responses": [
                "That's wonderful to hear! What changes have you noticed?",
                "I'm waking up with more energy, and I've started doing things I used to enjoy again.",
                "Those are significant improvements. What do you think has contributed to this positive shift?",
                "I think it's a combination of therapy, medication, and making myself stick to a routine.",
                "It sounds like you've been working hard on your recovery. How does it feel to notice these changes?",
                "It feels amazing, but also a little scary. I'm afraid I'll slip back into how I was before.",
                "That fear is understandable - you've been through a difficult time. But the fact that you've developed these coping strategies and you're aware of what helps suggests you're building resilience. Recovery isn't always linear, but you've shown you have the tools to keep moving forward."
            ]
        }
    ]
    
    # 扩展为更多样本
    for i in range(100):  # 创建100个示例对话
        template = templates[i % len(templates)]
        conv = []
        
        # 第一条：情境描述
        conv.append({
            'context': template['emotion'],
            'prompt': template['situation'],
            'utterance': template['situation'],
            'speaker_idx': 0,
            'utterance_idx': 0
        })
        
        # 添加对话内容
        for idx, response in enumerate(template['responses']):
            conv.append({
                'context': template['emotion'],
                'prompt': '',
                'utterance': response,
                'speaker_idx': idx % 2,
                'utterance_idx': idx + 1
            })
        
        sample_conversations.append(conv)
    
    return sample_conversations

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
    将对话转换为 ES 格式
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
        speaker = "Client" if item['speaker_idx'] == 0 else "Counselor"
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
    """基于对话内容生成经验与反思"""
    
    num_turns = len(conversation)
    
    reflection_parts = []
    
    # 引言
    reflection_parts.append(
        f"This case presents a counseling conversation centered around {emotion_context} emotions. "
        f"Through {num_turns} exchanges, we observe the development of a therapeutic dialogue "
        f"that demonstrates the importance of active listening, emotional validation, and empathetic response."
    )
    
    # 核心分析
    reflection_parts.append(
        "\n\nIn analyzing this therapeutic interaction, several key counseling principles emerge. "
        "The counselor's approach showcases how creating a safe, non-judgmental space allows clients "
        "to explore their emotions more deeply. Each response demonstrates careful attention to the "
        "client's emotional state while gently guiding them toward insight and coping strategies."
    )
    
    # 情绪特定分析
    emotion_reflections = {
        'anxious': "\n\nAnxiety-focused counseling requires particular attention to validation while also "
                  "introducing practical coping mechanisms. This dialogue illustrates how acknowledging "
                  "the legitimacy of anxious feelings, while simultaneously exploring their roots and "
                  "offering concrete strategies, can help clients feel both understood and empowered.",
        
        'sad': "\n\nWorking with sadness and loneliness demands patience and presence. This case demonstrates "
              "how sitting with difficult emotions, rather than rushing to fix them, creates space for genuine "
              "processing. The progression shows how validation of loss can gradually open pathways to adaptation.",
        
        'stressed': "\n\nStress management in counseling often involves helping clients regain a sense of control. "
                   "This dialogue exemplifies how breaking down overwhelming situations into manageable components "
                   "can reduce the intensity of stress while building the client's confidence in their coping abilities.",
        
        'grief': "\n\nGrief counseling requires special sensitivity to the unique timeline of each individual's mourning "
                "process. This case illustrates the importance of normalizing prolonged grief while gently introducing "
                "ways to honor memory and find meaning, all at the client's own pace.",
        
        'hopeful': "\n\nSupporting clients in recovery involves both celebrating progress and addressing fears of relapse. "
                  "This dialogue demonstrates how acknowledging both the achievements and the understandable anxieties "
                  "helps build sustainable resilience and realistic optimism."
    }
    
    reflection_parts.append(emotion_reflections.get(emotion_context, 
        "\n\nThis emotional experience, while complex, is addressed through fundamental counseling principles: "
        "empathy, validation, and collaborative exploration of solutions."))
    
    # 专业反思
    reflection_parts.append(
        "\n\nFrom a clinical perspective, this interaction demonstrates several evidence-based practices. "
        "The use of open-ended questions encourages deeper exploration. Reflective listening ensures the "
        "client feels heard. The balance between emotional support and practical guidance helps move the "
        "client toward both insight and action."
    )
    
    # 结论
    reflection_parts.append(
        "\n\nReflecting on this case reinforces core therapeutic truths: effective counseling is less about "
        "providing answers and more about facilitating the client's own discovery process. By maintaining "
        "a stance of empathetic curiosity and avoiding premature problem-solving, counselors create the "
        "conditions for genuine therapeutic change. Each session is an opportunity to witness resilience, "
        "honor struggle, and affirm the human capacity for growth even amid difficulty."
    )
    
    return "".join(reflection_parts)

if __name__ == "__main__":
    random.seed(42)  # 设置随机种子以保证可复现性
    download_and_convert()

