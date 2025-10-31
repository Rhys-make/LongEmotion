# 📦 Emotion Summary 模型提交指南

## ✅ 模型准备完成

### 📊 模型信息

- **模型类型**: mT5-small (微调)
- **任务**: 情感信息提取与总结
- **框架**: PyTorch + Transformers
- **模型大小**: 1.16 GB
- **提交包位置**: `model/emotion_summary/`

---

## 📁 提交包内容

```
model/emotion_summary/
├── config.json                 # 模型配置
├── generation_config.json      # 生成配置
├── model.safetensors          # 模型权重 (1.14 GB)
├── tokenizer_config.json       # Tokenizer配置
├── tokenizer.json              # Tokenizer词表 (15.6 MB)
├── spiece.model                # SentencePiece模型 (4.1 MB)
├── special_tokens_map.json     # 特殊token映射
├── README.md                   # 模型卡片
├── inference_example.py        # 推理示例代码
├── model_info.json            # 模型元信息
└── .gitattributes             # Git LFS配置
```

---

## 🚀 提交方式（3种选择）

### 方式1: 使用提供的脚本上传（推荐）

**优点**: 简单快捷，一键上传

```bash
# 1. 安装依赖
pip install huggingface_hub

# 2. 登录Hugging Face
huggingface-cli login
# 输入你的token (从 https://huggingface.co/settings/tokens 获取)

# 3. 修改仓库名称（可选）
# 编辑 upload_to_huggingface.py，修改 repo_name 变量

# 4. 运行上传脚本
python upload_to_huggingface.py
```

---

### 方式2: 使用Git上传

**优点**: 更灵活，支持版本控制

```bash
# 1. 安装Git LFS
git lfs install

# 2. 在Hugging Face创建新仓库
# 访问 https://huggingface.co/new
# 创建名为 "emotion-summary-mt5-small" 的模型仓库

# 3. 克隆仓库
git clone https://huggingface.co/你的用户名/emotion-summary-mt5-small
cd emotion-summary-mt5-small

# 4. 复制模型文件
cp -r ../model/emotion_summary/* .

# 5. 提交并上传
git add .
git commit -m "Upload Emotion Summary Model (mT5-small)"
git push
```

---

### 方式3: 手动上传（如果网络受限）

**优点**: 可以选择性上传文件

1. 访问 https://huggingface.co/new
2. 创建新模型仓库
3. 点击 "Files and versions" → "Add file" → "Upload files"
4. 上传 `model/emotion_summary/` 目录下的所有文件

⚠️ **注意**: `model.safetensors` 文件较大(1.14GB)，建议使用Git LFS

---

## 📝 提交到比赛

### 文件清单

提交给比赛组委会需要包含：

1. ✅ **模型文件**: `model/emotion_summary/` 完整目录
2. ✅ **推理结果**: `results/Emotion_Summary_Result.jsonl`
3. ✅ **推理代码**: `model/emotion_summary/inference_example.py`
4. ✅ **模型说明**: `model/emotion_summary/README.md`

### 压缩打包（如果需要）

```bash
# 压缩模型目录（Windows PowerShell）
Compress-Archive -Path "model\emotion_summary\*" -DestinationPath "emotion_summary_model.zip"

# 或使用7-Zip（更高压缩率）
7z a -tzip emotion_summary_model.zip "model\emotion_summary\*"
```

---

## 🧪 测试模型

在提交前测试模型是否可用：

```bash
cd model/emotion_summary
python inference_example.py
```

### 快速测试代码

```python
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import json

# 加载模型
model = MT5ForConditionalGeneration.from_pretrained("./model/emotion_summary")
tokenizer = MT5Tokenizer.from_pretrained("./model/emotion_summary")

# 读取一个测试样本
with open("data/test/Emotion_Summary.jsonl", "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

# 准备输入
case_text = " ".join(sample["case_description"])
input_text = f"Summarize: {case_text}"

# 编码
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# 生成
output_ids = model.generate(input_ids, max_length=256, num_beams=4)

# 解码
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Generated: {output}")
```

---

## 📋 提交检查清单

- [ ] 模型文件完整（11个文件）
- [ ] 模型大小正确（约1.16GB）
- [ ] README.md 包含使用说明
- [ ] inference_example.py 可以运行
- [ ] 结果文件 Emotion_Summary_Result.jsonl 完整（150条）
- [ ] 已测试模型加载和推理
- [ ] （可选）已上传到Hugging Face

---

## 🌐 Hugging Face链接格式

上传成功后，你的模型链接格式：

```
https://huggingface.co/你的用户名/emotion-summary-mt5-small
```

### 使用你的模型

其他人可以这样使用：

```python
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# 直接从Hugging Face加载
model = MT5ForConditionalGeneration.from_pretrained("你的用户名/emotion-summary-mt5-small")
tokenizer = MT5Tokenizer.from_pretrained("你的用户名/emotion-summary-mt5-small")
```

---

## ⚠️ 常见问题

### Q1: 上传失败怎么办？

**A**: 检查以下几点：
1. 确认已登录: `huggingface-cli whoami`
2. Token权限: 需要有 `write` 权限
3. 网络连接: 尝试使用VPN
4. 文件大小: 大文件需要Git LFS

### Q2: 比赛要求什么格式？

**A**: 参考Detection任务的格式：
- 模型权重文件
- config.json
- tokenizer文件
- README.md
- inference示例代码

### Q3: 模型太大无法上传？

**A**: 三种解决方案：
1. 使用Git LFS上传大文件
2. 分片上传（Hugging Face支持）
3. 只提交必要文件给比赛（本地保存完整版）

### Q4: 可以提交训练代码吗？

**A**: 可以！添加训练代码会更完整：
- `scripts/train.py` - 训练脚本
- `scripts/fast_train.py` - 快速训练版本
- `requirements_es.txt` - 依赖列表

---

## 📞 技术支持

如果遇到问题：

1. **Hugging Face文档**: https://huggingface.co/docs/hub/models-uploading
2. **Git LFS教程**: https://git-lfs.github.com/
3. **Transformers文档**: https://huggingface.co/docs/transformers

---

## 📌 重要提示

1. ⚠️ **模型权重是1.14GB**，上传需要时间和稳定网络
2. ✅ **推理结果已准备好**: `results/Emotion_Summary_Result.jsonl` (150条)
3. 🎯 **模型和结果都符合比赛要求**
4. 💡 **建议先测试模型，再上传**

---

## ✅ 提交完成后

- [ ] 在比赛平台提交模型链接或文件
- [ ] 提交推理结果文件
- [ ] 填写模型说明文档
- [ ] 等待评审结果

---

**祝提交顺利！** 🎉

