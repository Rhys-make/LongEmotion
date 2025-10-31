"""
情感检测推理示例
使用detection_hug模型进行情感分类
"""
import torch
from transformers import BertTokenizer
from detection_model import EmotionDetectionModel

def load_model(model_path="model.pt", device="cpu"):
    """加载模型"""
    print(f"加载模型: {model_path}")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(".")
    
    # 创建模型
    model = EmotionDetectionModel(
        model_name="bert-base-chinese",
        num_emotions=6,
        dropout=0.1
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("✅ 模型加载成功")
    return model, tokenizer

def predict(text, model, tokenizer, device="cpu"):
    """预测单个文本的情感"""
    # 情感标签
    EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    # 编码
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_id].item()
    
    return {
        'emotion': EMOTIONS[predicted_id],
        'confidence': confidence,
        'all_probabilities': {
            EMOTIONS[i]: float(probabilities[0, i])
            for i in range(len(EMOTIONS))
        }
    }

if __name__ == "__main__":
    # 示例
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device=device)
    
    # 测试文本
    test_texts = [
        "我今天很开心！",
        "这让我感到非常难过。",
        "我爱你。"
    ]
    
    print("\n" + "="*60)
    print("情感检测结果")
    print("="*60)
    
    for text in test_texts:
        result = predict(text, model, tokenizer, device)
        print(f"\n文本: {text}")
        print(f"情感: {result['emotion']}")
        print(f"置信度: {result['confidence']:.4f}")

