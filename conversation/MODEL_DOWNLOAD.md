# CBT 模型下载说明

## 模型信息

- **模型名称**: `help2opensource/Qwen3-4B-Instruct-2507_mental_health_cbt`
- **模型类型**: LoRA 适配器（基于 Qwen2.5-4B-Instruct）
- **模型大小**: 约 40MB（适配器文件）
- **基础模型**: Qwen/Qwen2.5-4B-Instruct（需要单独下载，约 8GB）

## 下载方法

### 方法1: 使用脚本自动下载（推荐）

```bash
python conversation/scripts/download_model.py
```

如果遇到网络错误，可以：
1. 检查网络连接
2. 稍后重试
3. 使用代理（如果在中国大陆，可能需要配置代理）

### 方法2: 使用 Hugging Face CLI

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 下载模型
huggingface-cli download help2opensource/Qwen3-4B-Instruct-2507_mental_health_cbt \
    --local-dir conversation/model \
    --local-dir-use-symlinks False
```

### 方法3: 手动下载

1. 访问模型页面：https://huggingface.co/help2opensource/Qwen3-4B-Instruct-2507_mental_health_cbt
2. 下载所有文件到 `conversation/model/` 目录
3. 确保以下文件存在：
   - `adapter_config.json`
   - `adapter_model.safetensors`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - 其他相关文件

## 基础模型下载

由于这是 LoRA 适配器，需要先下载基础模型：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 基础模型会自动下载到缓存目录
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-4B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-4B-Instruct")
```

或者在代码中会自动处理。

## 使用模型

下载完成后，可以使用以下命令使用 CBT 模型：

```bash
python conversation/scripts/inference.py \
    --input_file conversation/data/longemotion_testset.json \
    --output_dir conversation/output \
    --counselor_type cactus \
    --llm_type cbt \
    --max_turns 20
```

## 文件结构

下载完成后，`conversation/model/` 目录应包含：

```
conversation/model/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
├── added_tokens.json
├── chat_template.jinja
└── training_args.bin
```

## 故障排除

### 问题1: 网络连接错误

**解决方案**:
- 检查网络连接
- 使用 VPN 或代理
- 稍后重试

### 问题2: 模型路径不存在

**解决方案**:
- 确保模型已下载到 `conversation/model/` 目录
- 检查文件是否完整

### 问题3: 缺少基础模型

**解决方案**:
- 基础模型会在首次使用时自动下载
- 确保有足够的磁盘空间（约 8GB）
- 检查网络连接

## 验证下载

运行以下命令验证模型是否正确下载：

```python
from conversation.src.cbt_model import CBTModel

try:
    model = CBTModel()
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
```

