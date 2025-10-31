"""
转换 PsyQA 数据集为 ES 任务格式
将数据分为 85% 训练集和 15% 验证集

注意: PsyQA 完整数据集需要签署用户协议获取
本脚本使用 PsyQA_example.json 作为示例数据
"""

import json
import os
import random
from pathlib import Path
import shutil

# 设置随机种子以确保可复现性
random.seed(42)

def load_psyqa_data(file_path):
    """
    加载 PsyQA 数据文件
    PsyQA 格式:
    - question: 问题标题
    - description: 详细描述
    - keywords: 关键词
    - answers: 回答列表
    - questionID: 问题ID
    """
    print(f"正在加载数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载 {len(data)} 条数据")
    return data

def convert_to_es_format(psyqa_item, item_id):
    """
    将 PsyQA 格式转换为 ES 格式
    
    PsyQA 格式:
    - question: 问题标题
    - description: 详细描述  
    - keywords: 关键词
    - answers: 回答列表（每个回答包含 answer_text）
    - questionID: 问题ID
    
    ES 格式:
    - id: 编号
    - case_description: 案例描述（列表）
    - consultation_process: 咨询过程（列表）
    - experience_and_reflection: 经验与反思（字符串）
    """
    
    # 提取问题信息
    question = psyqa_item.get('question', '')
    description = psyqa_item.get('description', '')
    keywords = psyqa_item.get('keywords', '')
    answers = psyqa_item.get('answers', [])
    
    # 构建案例描述 - 包含问题和详细描述
    case_parts = []
    if question:
        case_parts.append(f"来访者问题: {question}")
    if description:
        case_parts.append(f"详细描述: {description}")
    if keywords:
        case_parts.append(f"关键词: {keywords}")
    
    case_description = "\n\n".join(case_parts) if case_parts else "案例信息缺失"
    
    # 构建咨询过程 - 从回答中提取
    consultation_parts = []
    
    # 添加问题描述作为初始咨询过程
    if question:
        consultation_parts.append(f"来访者主诉: {question}")
    
    if description and description != question:
        consultation_parts.append(f"来访者详细陈述: {description}")
    
    # 处理每个回答
    for idx, answer_obj in enumerate(answers):
        answer_text = answer_obj.get('answer_text', '')
        if answer_text:
            # 将回答分段
            answer_paragraphs = answer_text.split('。')
            # 过滤掉空段落和很短的段落
            answer_paragraphs = [p.strip() + '。' for p in answer_paragraphs if len(p.strip()) > 10]
            
            if len(answer_paragraphs) > 0:
                consultation_parts.append(f"咨询师回复 {idx+1}:")
                
                # 如果回答很长，分成多个部分
                if len(answer_paragraphs) > 3:
                    # 开头部分 - 通常是共情和理解
                    consultation_parts.append("初步理解与共情: " + " ".join(answer_paragraphs[:2]))
                    
                    # 中间部分 - 通常是分析
                    if len(answer_paragraphs) > 4:
                        consultation_parts.append("深入分析: " + " ".join(answer_paragraphs[2:-2]))
                    
                    # 结尾部分 - 通常是建议
                    consultation_parts.append("咨询建议: " + " ".join(answer_paragraphs[-2:]))
                else:
                    consultation_parts.append(answer_text)
    
    # 如果没有回答，添加占位符
    if not answers:
        consultation_parts.append("咨询师正在评估来访者的情况...")
    
    # 生成经验与反思
    reflection = generate_reflection(question, description, answers, keywords)
    
    # 构建 ES 格式
    es_item = {
        "id": item_id,
        "case_description": [case_description],
        "consultation_process": consultation_parts,
        "experience_and_reflection": reflection
    }
    
    return es_item

def generate_reflection(question, description, answers, keywords):
    """
    基于问题、描述、回答和关键词生成经验与反思
    模拟真实心理咨询师的反思过程
    """
    
    # 根据关键词调整反思内容
    keywords_str = keywords if keywords else "心理健康"
    
    # 计算回答的平均长度来判断咨询的深度
    answer_lengths = [len(ans.get('answer_text', '')) for ans in answers]
    avg_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
    
    # 生成个性化的反思
    reflection_parts = []
    
    # 开篇 - 案例概述
    reflection_parts.append(
        f"本案例涉及{keywords_str}相关的心理困扰。通过对来访者问题的深入分析，"
        f"我们可以看到这类问题往往反映了个体在成长、人际关系、情绪调节等多个维度上的挑战。"
    )
    
    # 咨询过程反思
    if avg_length > 200:
        reflection_parts.append(
            "\n\n在本次咨询中，我采用了较为深入的分析和引导策略。通过倾听来访者的叙述，"
            "识别其核心关切，并运用专业的心理学知识帮助来访者理解问题的本质。"
            "咨询过程中注重建立安全的咨询关系，让来访者能够充分表达自己的感受和想法。"
        )
    else:
        reflection_parts.append(
            "\n\n在本次咨询中，我提供了针对性的建议和支持。虽然时间有限，"
            "但力求抓住问题的核心，为来访者提供实用的应对策略和心理支持。"
        )
    
    # 关键洞察
    reflection_parts.append(
        "\n\n每个人的心理困扰都有其独特的成因和表现形式。作为咨询师，"
        "我们需要保持开放和好奇的态度，避免过早下结论或套用固定模式。"
        "真正有效的咨询来自于对来访者个体经验的深入理解和尊重。"
    )
    
    # 治疗效果与展望
    reflection_parts.append(
        "\n\n通过系统的心理咨询和持续的支持，来访者有机会更好地认识自己，"
        "理解问题的根源，并逐步找到适合自己的应对方法。咨询师的角色是陪伴者、"
        "引导者和支持者，帮助来访者激发内在的力量和资源，走向更健康、"
        "更平衡的心理状态。"
    )
    
    # 专业反思
    reflection_parts.append(
        "\n\n这个案例再次提醒我们，心理咨询工作需要扎实的专业知识、"
        "敏锐的觉察能力、真诚的同理心，以及对人性的深刻理解。每一次咨询"
        "都是一次共同成长的机会——来访者在咨询中获得成长和改变，"
        "咨询师也在实践中不断深化对人类心理的理解，提升专业能力。"
    )
    
    return "".join(reflection_parts)

def split_and_save_data(all_data, train_ratio=0.85):
    """
    分割数据并保存
    train_ratio: 训练集比例（默认85%）
    """
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 计算分割点
    total_count = len(all_data)
    train_count = int(total_count * train_ratio)
    
    train_data = all_data[:train_count]
    valid_data = all_data[train_count:]
    
    print(f"\n数据分割完成:")
    print(f"总数据量: {total_count}")
    print(f"训练集: {len(train_data)} ({len(train_data)/total_count*100:.1f}%)")
    print(f"验证集: {len(valid_data)} ({len(valid_data)/total_count*100:.1f}%)")
    
    return train_data, valid_data

def save_jsonl(data, file_path):
    """保存为 JSONL 格式"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存到: {file_path}")

def main():
    print("=" * 60)
    print("PsyQA 数据集转换工具")
    print("=" * 60)
    print("\n注意: PsyQA 完整数据集需要签署用户协议获取")
    print("本脚本使用 temp_psyqa_repo/PsyQA_example.json 作为示例数据")
    print("如需完整数据集，请访问: https://github.com/thu-coai/PsyQA")
    print("=" * 60 + "\n")
    
    # 检查数据文件是否存在
    script_dir = Path(__file__).parent.parent
    psyqa_file = script_dir / "temp_psyqa_repo" / "PsyQA_example.json"
    
    if not psyqa_file.exists():
        print(f"错误: 找不到数据文件 {psyqa_file}")
        print("\n请确保已经克隆了 PsyQA 仓库:")
        print("git clone https://github.com/thu-coai/PsyQA.git temp_psyqa_repo")
        return
    
    # 加载数据集
    all_psyqa_data = load_psyqa_data(psyqa_file)
    
    if not all_psyqa_data:
        print("\n错误: 数据文件为空！")
        return
    
    print(f"\n总共加载了 {len(all_psyqa_data)} 条原始数据")
    
    # 转换格式
    print("\n开始转换数据格式...")
    converted_data = []
    
    for idx, item in enumerate(all_psyqa_data, 1):
        try:
            es_item = convert_to_es_format(item, idx)
            converted_data.append(es_item)
            
            if idx % 100 == 0:
                print(f"已转换 {idx}/{len(all_psyqa_data)} 条数据")
        except Exception as e:
            print(f"转换第 {idx} 条数据时出错: {e}")
            continue
    
    print(f"成功转换 {len(converted_data)} 条数据")
    
    # 分割数据
    train_data, valid_data = split_and_save_data(converted_data, train_ratio=0.85)
    
    # 保存数据
    base_dir = Path(__file__).parent.parent / "data"
    
    train_path = base_dir / "train" / "Emotion_Summary.jsonl"
    valid_path = base_dir / "validation" / "Emotion_Summary.jsonl"
    
    print("\n保存数据...")
    save_jsonl(train_data, train_path)
    save_jsonl(valid_data, valid_path)
    
    print("\n" + "=" * 60)
    print("数据处理完成！")
    print("=" * 60)
    print(f"训练集位置: {train_path}")
    print(f"验证集位置: {valid_path}")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(valid_data)}")
    
    # 显示示例
    print("\n" + "=" * 60)
    print("训练集示例 (第1条):")
    print("=" * 60)
    if train_data:
        example = train_data[0]
        print(f"ID: {example['id']}")
        print(f"\n案例描述:")
        print(example['case_description'][0][:300] + "..." if len(example['case_description'][0]) > 300 else example['case_description'][0])
        print(f"\n咨询过程段落数: {len(example['consultation_process'])}")
        print(f"前3段咨询过程:")
        for i, proc in enumerate(example['consultation_process'][:3], 1):
            print(f"  {i}. {proc[:100]}..." if len(proc) > 100 else f"  {i}. {proc}")
        print(f"\n经验反思长度: {len(example['experience_and_reflection'])} 字符")
        print(f"经验反思片段: {example['experience_and_reflection'][:200]}...")
    
    print("\n" + "=" * 60)
    print("提示: 这是基于 PsyQA 示例数据的转换结果")
    print("如需完整训练数据，请联系 PsyQA 作者获取完整数据集")
    print("=" * 60)

if __name__ == "__main__":
    main()

