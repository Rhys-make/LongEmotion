"""
转换预测结果为比赛提交格式
从: {"sample_id": 0, "unique_segment_index": 24, ...}
到: {"id": 0, "predicted_index": 24}
"""
import json

def convert_format(input_file, output_file):
    """转换格式"""
    print(f"读取: {input_file}")
    
    # 读取原始预测结果
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    print(f"共 {len(results)} 条预测结果")
    
    # 转换格式
    converted = []
    for result in results:
        converted.append({
            "id": result["sample_id"],
            "predicted_index": result["unique_segment_index"]
        })
    
    # 保存新格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"保存到: {output_file}")
    print("\n前3条结果:")
    for item in converted[:3]:
        print(f"  {item}")
    
    print(f"\n✅ 格式转换完成！")
    print(f"提交文件: {output_file}")

if __name__ == "__main__":
    # 使用Detection文件夹内的相对路径
    input_file = "../submission/predictions.jsonl"
    output_file = "../submission/Emotion_Detection_Result.jsonl"
    
    convert_format(input_file, output_file)

