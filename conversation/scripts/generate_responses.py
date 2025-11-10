"""
为测试集生成 Counselor 回复
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

from conversation.src.factory import CounselorFactory


def parse_conversation_history(conversation_history: str) -> List[Dict]:
    """
    解析对话历史字符串
    
    Args:
        conversation_history: 对话历史字符串
        
    Returns:
        解析后的对话列表
    """
    if not conversation_history or not isinstance(conversation_history, str):
        return []
    
    # 尝试解析 JSON 格式
    try:
        parsed = json.loads(conversation_history)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            messages = []
            for key, value in parsed.items():
                if isinstance(value, str):
                    messages.append({"role": "client", "message": value})
            return messages
    except:
        pass
    
    # 如果不是 JSON，尝试按行分割
    lines = conversation_history.split('\n')
    messages = []
    current_role = "client"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 尝试识别角色
        if line.startswith("客户:") or line.startswith("Client:") or line.startswith("用户:"):
            current_role = "client"
            message = line.split(":", 1)[1].strip() if ":" in line else line
        elif line.startswith("咨询师:") or line.startswith("Counselor:") or line.startswith("助手:"):
            current_role = "counselor"
            message = line.split(":", 1)[1].strip() if ":" in line else line
        else:
            message = line
        
        if message:
            messages.append({
                "role": current_role,
                "message": message
            })
    
    return messages


def extract_context_from_history(conversation_history: List[Dict]) -> Dict:
    """
    从对话历史中提取上下文
    
    Args:
        conversation_history: 对话历史列表
        
    Returns:
        包含 client_information, reason_counseling, cbt_plan 的字典
    """
    if not conversation_history:
        return {
            "client_information": "",
            "reason_counseling": "",
            "cbt_plan": "基于认知行为理论，帮助客户识别和改变负面思维模式，建立积极的应对策略"
        }
    
    # 提取第一条客户消息作为咨询原因
    first_client_message = ""
    for msg in conversation_history:
        if msg.get('role') == 'client':
            first_client_message = msg.get('message', '')
            break
    
    client_information = f"对话包含 {len(conversation_history)} 条消息"
    
    return {
        "client_information": client_information,
        "reason_counseling": first_client_message[:500] if first_client_message else "需要情感支持和咨询",
        "cbt_plan": "基于认知行为理论，帮助客户识别和改变负面思维模式，建立积极的应对策略"
    }


def generate_response_for_item(
    item: Dict,
    counselor,
    item_id: int
) -> Dict:
    """
    为单个测试项生成 Counselor 回复
    
    Args:
        item: 测试项
        counselor: 咨询师代理
        item_id: 项目ID
        
    Returns:
        包含 id 和 predicted_response 的字典
    """
    conversation_history_str = item.get('conversation_history', '')
    
    # 解析对话历史
    conversation_history = parse_conversation_history(conversation_history_str)
    
    # 提取上下文
    context = extract_context_from_history(conversation_history)
    
    # 找到最后一个客户消息（需要回复的）
    last_client_idx = -1
    for i in range(len(conversation_history) - 1, -1, -1):
        if conversation_history[i].get('role') == 'client':
            last_client_idx = i
            break
    
    # 构建用于生成的历史（到最后一个客户消息为止）
    if last_client_idx >= 0:
        history_for_generation = conversation_history[:last_client_idx + 1]
    else:
        # 如果没有客户消息，使用全部历史
        history_for_generation = conversation_history
    
    # 生成 Counselor 回复
    try:
        counselor_response = counselor.generate(
            history=history_for_generation,
            client_information=context['client_information'],
            reason_counseling=context['reason_counseling'],
            cbt_plan=context['cbt_plan']
        )
    except Exception as e:
        print(f"生成回复时出错 (ID: {item_id}): {e}")
        counselor_response = "[生成失败]"
    
    return {
        "id": item_id,
        "predicted_response": counselor_response
    }


def main():
    parser = argparse.ArgumentParser(description="为测试集生成 Counselor 回复")
    parser.add_argument(
        "--input_file",
        type=str,
        default="conversation/data/longemotion_emotion_conversation.json",
        help="输入测试集文件"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="conversation/output/predicted_responses.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--counselor_type",
        type=str,
        default="cactus",
        help="咨询师类型"
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        default="cbt",
        help="LLM类型（chatgpt, llama2, llama3, longemotion, cbt）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（None表示全部）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("生成 Counselor 回复")
    print("=" * 60)
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    print(f"咨询师类型: {args.counselor_type}")
    print(f"LLM类型: {args.llm_type}")
    print("=" * 60)
    
    # 加载测试集
    print("\n正在加载测试集...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        testset = json.load(f)
    
    print(f"✅ 测试集加载成功，共 {len(testset)} 条记录")
    
    # 限制样本数
    if args.max_samples:
        testset = testset[:args.max_samples]
        print(f"限制为前 {len(testset)} 条记录")
    
    # 创建咨询师代理
    print(f"\n正在初始化咨询师代理...")
    try:
        counselor = CounselorFactory.get_counselor(args.counselor_type, args.llm_type)
        print(f"✅ 咨询师代理初始化成功")
    except Exception as e:
        print(f"❌ 咨询师代理初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 生成回复
    print(f"\n开始生成 Counselor 回复...")
    results = []
    
    for item in tqdm(testset, desc="处理中"):
        item_id = item.get('id', len(results))
        result = generate_response_for_item(item, counselor, item_id)
        results.append(result)
    
    # 保存结果
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 同时输出为指定格式（每行一个JSON对象）
    output_txt = output_path.with_suffix('.txt')
    with open(output_txt, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n✅ 完成！共处理 {len(results)} 条记录")
    print(f"📁 JSON格式: {output_path}")
    print(f"📁 文本格式: {output_txt}")
    
    # 显示前3条结果
    print("\n前3条结果预览:")
    for i, result in enumerate(results[:3]):
        print(f"\n{json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

