"""
上传Detection模型到Hugging Face Hub
"""
from huggingface_hub import HfApi, create_repo
import os

# 配置
REPO_NAME = "your-username/longemotion-detection"  # 修改为你的用户名
MODEL_PATH = "model/best_model.pt"
LOCAL_DIR = "."

def upload_model():
    """上传模型到Hugging Face"""
    
    print("="*60)
    print("上传LongEmotion Detection模型到Hugging Face")
    print("="*60)
    
    # 初始化API
    api = HfApi()
    
    # 步骤1: 创建仓库（如果不存在）
    print("\n[1/3] 创建仓库...")
    try:
        create_repo(
            repo_id=REPO_NAME,
            repo_type="model",
            exist_ok=True,
            private=False  # 设置为True则为私有仓库
        )
        print(f"✅ 仓库创建/确认: {REPO_NAME}")
    except Exception as e:
        print(f"❌ 创建仓库失败: {e}")
        return
    
    # 步骤2: 上传模型文件
    print("\n[2/3] 上传模型文件...")
    try:
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="best_model.pt",
            repo_id=REPO_NAME,
            repo_type="model",
        )
        print("✅ 模型文件上传成功")
    except Exception as e:
        print(f"❌ 上传模型失败: {e}")
        return
    
    # 步骤3: 上传README和其他文件
    print("\n[3/3] 上传README和文档...")
    files_to_upload = [
        ("README.md", "README.md"),
        ("快速使用指南.md", "快速使用指南.md"),
        ("scripts/inference_longemotion.py", "inference_longemotion.py"),
        ("scripts/detection_model.py", "detection_model.py"),
    ]
    
    for local_path, remote_path in files_to_upload:
        if os.path.exists(local_path):
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=REPO_NAME,
                    repo_type="model",
                )
                print(f"✅ 上传: {remote_path}")
            except Exception as e:
                print(f"⚠️ 上传 {remote_path} 失败: {e}")
    
    print("\n" + "="*60)
    print(f"🎉 上传完成！")
    print(f"模型地址: https://huggingface.co/{REPO_NAME}")
    print("="*60)

if __name__ == "__main__":
    # 检查是否已登录
    print("请确保已执行: huggingface-cli login")
    input("按回车继续...")
    
    upload_model()

