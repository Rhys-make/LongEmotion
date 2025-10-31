# -*- coding: utf-8 -*-
"""
上传Emotion Summary模型到Hugging Face
"""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def upload_model():
    """上传模型到Hugging Face Hub"""
    
    print("\n" + "="*80)
    print("🚀 上传Emotion Summary模型到Hugging Face")
    print("="*80 + "\n")
    
    # 检查huggingface_hub是否安装
    try:
        from huggingface_hub import HfApi, create_repo
        print("✓ huggingface_hub 已安装")
    except ImportError:
        print("⚠️ huggingface_hub 未安装")
        print("请运行: pip install huggingface_hub")
        return
    
    # 配置
    model_path = "model/emotion_summary"
    repo_name = "emotion-summary-mt5-small"  # 你的仓库名称
    
    print(f"📁 本地模型路径: {model_path}")
    print(f"📦 Hugging Face仓库: {repo_name}")
    
    # 检查模型目录是否存在
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 模型目录不存在: {model_path}")
        print("请先运行: python prepare_model_for_submission.py")
        return
    
    # 列出要上传的文件
    files = os.listdir(model_path)
    print(f"\n📄 将上传以下文件:")
    for file in sorted(files):
        file_path = os.path.join(model_path, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file:30s} ({size_mb:.2f} MB)")
    
    print("\n" + "="*80)
    print("⚠️ 上传前准备:")
    print("="*80)
    print("\n1. 确保已登录Hugging Face:")
    print("   huggingface-cli login")
    print("\n2. 或者设置环境变量:")
    print("   export HUGGING_FACE_HUB_TOKEN=your_token")
    print("\n3. 确认仓库名称:")
    print(f"   - 当前设置: {repo_name}")
    print("   - 可以修改此脚本中的repo_name变量")
    
    print("\n" + "="*80)
    response = input("\n是否继续上传? (yes/no): ")
    
    if response.lower() not in ['yes', 'y', '是']:
        print("\n❌ 取消上传")
        return
    
    try:
        print("\n📤 开始上传...")
        
        # 初始化API
        api = HfApi()
        
        # 创建仓库（如果不存在）
        print(f"\n1️⃣ 创建/检查仓库: {repo_name}")
        try:
            create_repo(
                repo_id=repo_name,
                repo_type="model",
                exist_ok=True
            )
            print("   ✓ 仓库已准备")
        except Exception as e:
            print(f"   ⚠️ 仓库创建失败（可能已存在）: {e}")
        
        # 上传整个文件夹
        print(f"\n2️⃣ 上传模型文件...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
        )
        
        print("\n" + "="*80)
        print("✅ 上传成功！")
        print("="*80)
        print(f"\n🌐 模型链接: https://huggingface.co/{repo_name}")
        print(f"\n📖 使用方法:")
        print(f"""
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained("{repo_name}")
tokenizer = MT5Tokenizer.from_pretrained("{repo_name}")
""")
        
    except Exception as e:
        print(f"\n❌ 上传失败: {e}")
        print("\n请检查:")
        print("  1. 是否已登录: huggingface-cli login")
        print("  2. Token是否有写入权限")
        print("  3. 网络连接是否正常")

def show_usage():
    """显示使用说明"""
    print("\n" + "="*80)
    print("📚 Hugging Face 上传指南")
    print("="*80)
    
    print("\n1️⃣ 安装依赖:")
    print("   pip install huggingface_hub")
    
    print("\n2️⃣ 登录Hugging Face:")
    print("   方式1: huggingface-cli login")
    print("   方式2: 在 https://huggingface.co/settings/tokens 创建token，然后设置环境变量")
    
    print("\n3️⃣ 准备模型:")
    print("   python prepare_model_for_submission.py")
    
    print("\n4️⃣ 上传模型:")
    print("   python upload_to_huggingface.py")
    
    print("\n5️⃣ 使用Git LFS (可选，用于大文件):")
    print("   git lfs install")
    print("   git clone https://huggingface.co/your-username/emotion-summary-mt5-small")
    print("   cd emotion-summary-mt5-small")
    print("   cp -r ../model/emotion_summary/* .")
    print("   git add .")
    print('   git commit -m "Upload model"')
    print("   git push")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage()
    else:
        upload_model()

