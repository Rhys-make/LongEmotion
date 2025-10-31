# Qwen2-7B-Instruct 完整部署指南
## 从零开始部署 Emotion Summary 模型

---

## ✅ 已完成的工作

### 1. 数据准备 ✓
- ✅ 成功克隆 PsyQA 仓库
- ✅ 转换 100 条 PsyQA 示例数据为 ES 格式
- ✅ 数据已按 85%-15% 分割为训练集和验证集
- ✅ 数据保存在:
  - `data/train/Emotion_Summary.jsonl` (85 samples)
  - `data/validation/Emotion_Summary.jsonl` (15 samples)

### 2. 模型代码 ✓
- ✅ 创建 `model_qwen2.py` - Qwen2 模型封装
- ✅ 创建 `train_qwen2.py` - LoRA 微调训练脚本
- ✅ 配置文件已准备
- ✅ 启动脚本已创建（Windows + Linux）

### 3. 技术方案 ✓
- ✅ 选定 **Qwen2-7B-Instruct** 模型
- ✅ 使用 **4-bit量化** + **LoRA** 高效微调
- ✅ 支持最长 8K tokens 输入（可扩展到 32K）
- ✅ 优化的中文心理咨询任务处理

---

## 📦 步骤1: 安装依赖

### 方法1: 使用国内镜像源（推荐）

```bash
# 使用清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers>=4.37.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple peft>=0.7.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple accelerate>=0.25.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple bitsandbytes>=0.41.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets
```

### 方法2: 使用阿里云镜像

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ transformers peft accelerate bitsandbytes datasets
```

### 方法3: 一键安装所有依赖

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_qwen2.txt
```

### 核心依赖列表

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| `torch` | >=2.0.0 | 深度学习框架 |
| `transformers` | >=4.37.0 | Hugging Face 模型库 |
| `peft` | >=0.7.0 | LoRA 微调 |
| `accelerate` | >=0.25.0 | 分布式训练 |
| `bitsandbytes` | >=0.41.0 | 4-bit 量化 |
| `datasets` | latest | 数据集处理 |

---

## 🔧 步骤2: 环境验证

运行以下命令验证环境：

```python
# 检查 CUDA
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA版本: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

# 检查依赖
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print('BitsAndBytes: OK')"
```

### 预期输出
```
CUDA可用: True
CUDA版本: 11.8
GPU: NVIDIA GeForce RTX 3090
Transformers: 4.37.2
PEFT: 0.8.0
BitsAndBytes: OK
```

---

## 🚀 步骤3: 开始训练

### 选项A: 使用启动脚本（最简单）

#### Windows:
```cmd
cd scripts
start_training_qwen2.bat
```

#### Linux/Mac:
```bash
cd scripts
chmod +x start_training_qwen2.sh
./start_training_qwen2.sh
```

### 选项B: Python 直接运行

```bash
cd scripts
python train_qwen2.py
```

### 选项C: 自定义参数运行

```bash
cd scripts

# 显存较小的配置 (12GB)
python train_qwen2.py \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 4096 \
    --lora_r 32

# 显存较大的配置 (24GB+)
python train_qwen2.py \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 8192 \
    --lora_r 64

# 更多训练轮次
python train_qwen2.py \
    --num_epochs 5 \
    --learning_rate 1e-4
```

---

## ⏱️ 步骤4: 等待训练完成

### 训练时间估算

| GPU 型号 | 单 Epoch 时间 | 3 Epochs 总时间 |
|----------|---------------|-----------------|
| RTX 4090 | ~1-2 小时 | 3-6 小时 |
| RTX 3090 | ~2-3 小时 | 6-9 小时 |
| RTX 3080 | ~3-4 小时 | 9-12 小时 |
| V100 | ~2-3 小时 | 6-9 小时 |
| A100 | ~1-1.5 小时 | 3-4.5 小时 |

### 监控训练进度

#### 方法1: 查看终端输出
训练过程会实时显示:
- Loss 值
- 训练步数
- 预估剩余时间

#### 方法2: 使用 TensorBoard
```bash
# 新开一个终端
cd "Emotion Summary (ES)"
tensorboard --logdir=model/qwen2_emotion_summary
# 访问 http://localhost:6006
```

#### 方法3: 监控 GPU 使用
```bash
# Linux/Mac
watch -n 1 nvidia-smi

# Windows (PowerShell)
while($true) { nvidia-smi; sleep 1; cls }
```

---

## 💾 步骤5: 模型保存位置

训练完成后，模型会保存在:

```
model/
└── qwen2_emotion_summary/
    ├── checkpoint-100/      # 训练过程检查点
    ├── checkpoint-200/
    ├── ...
    └── final/               # 最终模型 ⭐
        ├── adapter_config.json
        ├── adapter_model.bin  # LoRA 权重
        ├── tokenizer_config.json
        └── ...
```

**重点**: `final/` 目录包含了微调后的 LoRA 权重

---

## 🔮 步骤6: 使用模型推理

### 简单测试

```python
import sys
sys.path.append('scripts')
from model_qwen2 import Qwen2EmotionSummaryModel

# 加载基础模型
model = Qwen2EmotionSummaryModel(
    model_name="Qwen/Qwen2-7B-Instruct",
    use_4bit=True,
    use_lora=True
)

# 加载微调的 LoRA 权重
model.load_lora_weights("../model/qwen2_emotion_summary/final")

# 测试数据
case_desc = [
    "来访者，女性，27岁，公司职员。主诉：近3个月情绪低落，焦虑。"
]

consult_process = [
    "来访者主诉: 工作压力大，失眠。",
    "咨询师分析: 识别负面思维模式。",
    "治疗方案: 认知行为疗法。"
]

# 生成摘要
summary = model.generate_summary(
    case_description=case_desc,
    consultation_process=consult_process
)

print("生成的经验与反思:")
print(summary)
```

### 批量推理

```python
import json
from pathlib import Path

# 加载测试数据
test_file = Path("../data/test/Emotion_Summary.jsonl")
results = []

with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        
        # 生成摘要
        summary = model.generate_summary(
            case_description=sample['case_description'],
            consultation_process=sample['consultation_process']
        )
        
        # 保存结果
        result = {
            'id': sample['id'],
            'predicted_summary': summary,
            'reference_summary': sample.get('experience_and_reflection', '')
        }
        results.append(result)

# 保存预测结果
with open('../submission/predictions.jsonl', 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"完成！生成了 {len(results)} 个预测")
```

---

## 📊 步骤7: 评估模型

### 使用 ROUGE 评估

```bash
cd scripts
python evaluate.py --model_path ../model/qwen2_emotion_summary/final
```

### 手动评估

对比生成的摘要和参考摘要，评估:
- ✅ 事实一致性
- ✅ 完整性
- ✅ 流畅性
- ✅ 专业性

---

## 🐛 常见问题排查

### 问题1: 下载模型慢或失败

**解决方案**: 使用镜像站

```bash
# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

或手动下载:
1. 访问 https://hf-mirror.com/Qwen/Qwen2-7B-Instruct
2. 下载所有文件到本地目录
3. 修改 `model_name` 为本地路径

### 问题2: CUDA Out of Memory

**解决方案1**: 减小 batch_size
```bash
python train_qwen2.py --batch_size 1
```

**解决方案2**: 增加梯度累积
```bash
python train_qwen2.py --gradient_accumulation_steps 16
```

**解决方案3**: 减小序列长度
```bash
python train_qwen2.py --max_length 4096
```

**解决方案4**: 减小 LoRA rank
```bash
python train_qwen2.py --lora_r 32
```

### 问题3: BitsAndBytes 安装失败 (Windows)

**原因**: Windows 上 bitsandbytes 支持有限

**解决方案**:
```bash
# 使用 CPU 版本或不使用量化
pip install bitsandbytes-windows

# 或训练时不使用 4-bit
python train_qwen2.py --use_4bit False  # 需要更多显存
```

### 问题4: 训练速度慢

**检查清单**:
- ✅ 确认使用 GPU (`nvidia-smi`)
- ✅ 确认 CUDA 版本匹配 PyTorch
- ✅ 使用 4-bit 量化
- ✅ 启用梯度检查点（默认开启）
- ✅ 使用较小的 max_length

---

## 📈 性能优化建议

### 1. 数据优化
- 使用完整 PsyQA 数据集（需要申请）
- 当前只有 100 个示例，完整数据集有 2W+ 样本

### 2. 训练优化
```bash
# 更长的训练
python train_qwen2.py --num_epochs 5

# 更小的学习率（更稳定）
python train_qwen2.py --learning_rate 1e-4

# 更大的 LoRA rank（更高精度）
python train_qwen2.py --lora_r 128
```

### 3. 推理优化
```python
# 调整生成参数
summary = model.generate_summary(
    case_description=case_desc,
    consultation_process=consult_process,
    temperature=0.7,      # 降低随机性
    top_p=0.9,           # nucleus sampling
    repetition_penalty=1.1  # 避免重复
)
```

---

## 📚 下一步建议

1. **获取完整数据集**
   - 联系 PsyQA 作者获取完整数据集
   - Email: thu-sunhao@foxmail.com

2. **扩展训练数据**
   - 添加更多心理咨询案例
   - 数据增强技术

3. **模型调优**
   - 尝试不同的 LoRA 参数
   - 调整训练超参数
   - 使用学习率调度器

4. **评估改进**
   - 使用多个评估指标
   - 人工评估质量
   - A/B 测试

---

## ✅ 完整流程总结

```bash
# 1. 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_qwen2.txt

# 2. 验证环境
python -c "import torch; print(torch.cuda.is_available())"

# 3. 开始训练
cd scripts
python train_qwen2.py

# 4. 等待完成（6-12小时）

# 5. 使用模型
python inference.py --model_path ../model/qwen2_emotion_summary/final

# 6. 生成提交文件
python generate_submission.py
```

---

## 🎉 恭喜！

你现在已经有了一个完整的 Qwen2-7B-Instruct 情绪总结模型！

如有任何问题，请参考:
- `QWEN2_TRAINING_GUIDE.md` - 详细训练指南
- `scripts/model_qwen2.py` - 模型代码
- `scripts/train_qwen2.py` - 训练代码

**祝训练顺利！** 🚀

