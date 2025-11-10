"""
推理脚本：运行咨询对话
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import sys

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.factory import CounselorFactory


def load_intake_form(input_file: str) -> Dict:
    """
    加载客户信息表
    
    Args:
        input_file: 输入文件路径
        
    Returns:
        客户信息字典或列表
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def convert_testset_format(testset_item: Dict) -> Dict:
    """
    将测试集格式转换为推理脚本需要的格式
    
    Args:
        testset_item: 测试集项目，包含 id 和 conversation_history
        
    Returns:
        转换后的格式，包含 id, client_information, reason_counseling, cbt_plan
    """
    item_id = testset_item.get('id', 0)
    conversation_history = testset_item.get('conversation_history', '')
    
    # 尝试从对话历史中提取信息
    # 如果 conversation_history 是字符串，尝试解析或提取前部分
    if isinstance(conversation_history, str):
        # 提取前200字符作为咨询原因
        reason_counseling = conversation_history[:200] if conversation_history else "需要心理咨询和支持"
        # 如果字符串很长，可能包含完整的对话
        if len(conversation_history) > 200:
            # 尝试找到第一个完整的句子
            sentences = conversation_history.split('。')[:3]
            reason_counseling = '。'.join(sentences) if sentences else reason_counseling
    else:
        reason_counseling = str(conversation_history)[:200]
    
    return {
        "id": item_id,
        "client_information": f"测试集样本 ID: {item_id}",
        "reason_counseling": reason_counseling,
        "cbt_plan": "基于认知行为理论，帮助客户识别和改变负面思维模式，建立积极的应对策略",
        "original_conversation_history": conversation_history
    }


def run_conversation(
    client_data: Dict,
    counselor_type: str,
    llm_type: str,
    max_turns: int,
    output_dir: str
):
    """
    运行咨询对话
    
    Args:
        client_data: 客户数据
        counselor_type: 咨询师类型
        llm_type: LLM类型
        max_turns: 最大轮次
        output_dir: 输出目录
    """
    # 创建咨询师代理
    counselor = CounselorFactory.get_counselor(counselor_type, llm_type)
    
    # 提取客户信息
    client_information = client_data.get('client_information', '')
    reason_counseling = client_data.get('reason_counseling', '')
    cbt_plan = client_data.get('cbt_plan', '')
    
    # 初始化对话历史
    history: List[Dict] = []
    
    # 开始对话
    print(f"\n开始咨询对话 (最大轮次: {max_turns})")
    print("=" * 50)
    
    for turn in range(max_turns):
        # 获取客户输入（如果是第一次，使用初始信息）
        if turn == 0 and reason_counseling:
            client_message = reason_counseling
        else:
            # 这里可以改为从输入获取，或使用默认提示
            client_message = input(f"\n客户 (轮次 {turn + 1}): ") if sys.stdin.isatty() else "继续咨询"
        
        # 添加到历史
        history.append({
            "role": "client",
            "message": client_message
        })
        
        # 生成咨询师回复
        print("\n咨询师正在思考...")
        counselor_response = counselor.generate(
            history=history,
            client_information=client_information,
            reason_counseling=reason_counseling,
            cbt_plan=cbt_plan
        )
        
        # 添加到历史
        history.append({
            "role": "counselor",
            "message": counselor_response
        })
        
        # 显示回复
        print(f"\n咨询师: {counselor_response}")
        
        # 检查是否结束对话（可以添加结束条件）
        # 例如：如果客户说"谢谢"、"结束"等
    
    # 保存对话历史
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"conversation_{client_data.get('id', 'default')}.json"
    
    output_data = {
        "client_data": client_data,
        "conversation_history": history,
        "counselor_type": counselor_type,
        "llm_type": llm_type,
        "max_turns": max_turns
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n对话已保存到: {output_file}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="LongEmotion咨询对话推理脚本")
    parser.add_argument("--input_file", type=str, required=True, help="输入文件路径（JSON格式）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--counselor_type", type=str, default="cactus", help="咨询师类型")
    parser.add_argument("--llm_type", type=str, default="chatgpt", help="LLM类型")
    parser.add_argument("--max_turns", type=int, default=20, help="最大对话轮次")
    
    args = parser.parse_args()
    
    # 加载客户数据
    if isinstance(args.input_file, str) and args.input_file.startswith('['):
        # 如果输入是JSON数组字符串
        client_data_list = json.loads(args.input_file)
    else:
        # 从文件加载
        client_data = load_intake_form(args.input_file)
        client_data_list = [client_data] if not isinstance(client_data, list) else client_data
    
    # 检查是否是测试集格式（包含 conversation_history 字段）
    if isinstance(client_data_list, list) and len(client_data_list) > 0:
        first_item = client_data_list[0]
        if isinstance(first_item, dict) and 'conversation_history' in first_item:
            # 这是测试集格式，需要转换
            print("检测到测试集格式，正在转换...")
            client_data_list = [convert_testset_format(item) for item in client_data_list]
    
    # 处理每个客户
    for idx, client_data in enumerate(client_data_list):
        if isinstance(client_data, dict) and 'id' not in client_data:
            client_data['id'] = idx
        
        print(f"\n处理客户 {client_data.get('id', idx)}")
        run_conversation(
            client_data=client_data,
            counselor_type=args.counselor_type,
            llm_type=args.llm_type,
            max_turns=args.max_turns,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

