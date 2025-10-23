"""
示例脚本：展示如何使用项目的各个功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))


def example_classification():
    """情感分类示例"""
    print("\n" + "="*60)
    print("示例 1: 情感分类")
    print("="*60)
    
    from models.classification_model import EmotionClassificationModel
    
    # 初始化模型（使用预训练模型）
    model = EmotionClassificationModel()
    
    # 测试文本
    texts = [
        "今天天气真好，心情特别开心！",
        "这部电影太让人难过了，我都哭了",
        "这个服务态度太差了，我很生气"
    ]
    
    # 预测
    results = model.predict(texts)
    
    for result in results:
        print(f"\n文本: {result['text']}")
        print(f"情感: {result['emotion']} (置信度: {result['confidence']:.2f})")


def example_preprocessing():
    """文本预处理示例"""
    print("\n" + "="*60)
    print("示例 2: 文本预处理")
    print("="*60)
    
    from utils.preprocess import get_preprocessor
    
    preprocessor = get_preprocessor()
    
    # 测试文本
    text = "  这是一段    有很多   空格的文本！！！  "
    
    print(f"原始文本: '{text}'")
    
    cleaned = preprocessor.clean_text(text)
    print(f"清洗后: '{cleaned}'")


def example_evaluator():
    """评估指标示例"""
    print("\n" + "="*60)
    print("示例 3: 评估指标计算")
    print("="*60)
    
    from utils.evaluator import get_evaluator
    
    evaluator = get_evaluator()
    
    # 模拟预测和标签
    predictions = [0, 1, 2, 0, 1]
    labels = [0, 1, 2, 1, 1]
    
    metrics = evaluator.compute_classification_metrics(predictions, labels)
    
    print("分类指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def example_api_client():
    """API 调用示例"""
    print("\n" + "="*60)
    print("示例 4: API 调用")
    print("="*60)
    print("注意: 需要先启动 API 服务")
    print("运行: cd api && python main.py")
    print("\n示例代码:")
    
    code = '''
import requests

# 情感分类
response = requests.post(
    "http://localhost:8000/classify",
    json={"text": "今天真是太棒了！"}
)
print(response.json())

# 情感检测
response = requests.post(
    "http://localhost:8000/detect",
    json={"text": "这部电影让我又高兴又难过"}
)
print(response.json())

# 情感对话
response = requests.post(
    "http://localhost:8000/conversation",
    json={"context": "我今天心情不好", "emotion": "happiness"}
)
print(response.json())
'''
    print(code)


def example_config():
    """配置使用示例"""
    print("\n" + "="*60)
    print("示例 5: 配置使用")
    print("="*60)
    
    from config import MODEL_CONFIGS, EMOTION_LABELS
    
    print("情感标签:")
    for i, label in enumerate(EMOTION_LABELS):
        print(f"  {i}: {label}")
    
    print("\n模型配置:")
    for task, config in MODEL_CONFIGS.items():
        print(f"\n{task}:")
        for key, value in config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LongEmotion 项目使用示例")
    print("="*60)
    
    # 运行所有示例
    try:
        example_preprocessing()
        example_evaluator()
        example_config()
        
        # 分类示例需要下载模型，可能较慢
        # example_classification()
        
        example_api_client()
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("某些示例可能需要先下载模型或数据")
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 下载数据集: python scripts/download_dataset.py")
    print("2. 训练模型: python scripts/train_all.py")
    print("3. 推理: python scripts/inference_all.py")
    print("4. 启动API: cd api && python main.py")

