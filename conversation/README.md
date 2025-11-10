# Cactus: 基于认知行为理论的心理学咨询对话系统

这是 Cactus 项目的实现，基于认知行为理论（Cognitive Behavioral Theory, CBT）的心理学咨询对话系统。

## 📌 数据集和模型

本项目集成了 Hugging Face 上的数据集和模型：

- **数据集**: [LangAGI-Lab/cactus](https://huggingface.co/datasets/LangAGI-Lab/cactus)
- **模型**: [LangAGI-Lab/camel](https://huggingface.co/LangAGI-Lab/camel)
- **CBT 模型**: [help2opensource/Qwen3-4B-Instruct-2507_mental_health_cbt](https://huggingface.co/help2opensource/Qwen3-4B-Instruct-2507_mental_health_cbt)
- **Collection**: [Cactus Collection](https://huggingface.co/collections/LangAGI-Lab/cactus-towards-psychological-counseling-conversations)

### 下载 CBT 模型

首先需要下载 CBT 心理健康模型到本地：

```bash
python scripts/download_model.py \
    --model_name help2opensource/Qwen3-4B-Instruct-2507_mental_health_cbt \
    --output_dir conversation/model
```

模型将下载到 `conversation/model/` 目录。

### 加载数据集

```bash
python scripts/load_longemotion_dataset.py \
    --output_file data/longemotion_test.json \
    --max_samples 100 \
    --split test
```

### 使用模型

#### 使用 CBT 模型（推荐）

```bash
python scripts/inference.py \
    --input_file data/longemotion_testset.json \
    --output_dir output \
    --counselor_type cactus \
    --llm_type cbt \
    --max_turns 20
```

#### 使用 LongEmotion 模型

```bash
python scripts/inference.py \
    --input_file data/longemotion_test.json \
    --output_dir output \
    --counselor_type cactus \
    --llm_type longemotion \
    --max_turns 20
```

## 项目结构

```
conversation/
├── prompts/                  # 提示模板文件
│   ├── agent_cactus_chatgpt.txt
│   ├── agent_cactus_llama2.txt
│   ├── agent_cactus_llama3.txt
│   └── agent_cactus_longemotion.txt
├── model/                    # 本地模型目录（下载的 CBT 模型）
├── src/                      # 源代码
│   ├── __init__.py
│   ├── config.py            # 配置管理
│   ├── llm.py               # LLM实现
│   ├── agent.py             # 咨询师代理
│   ├── factory.py           # 工厂类
│   ├── longemotion_dataset.py    # LongEmotion数据集加载器
│   ├── longemotion_model.py      # LongEmotion模型加载器
│   └── cbt_model.py         # CBT模型加载器
├── scripts/                  # 脚本文件
│   ├── inference.py         # 推理脚本
│   ├── inference.sh         # 推理脚本（Shell）
│   ├── download_model.py    # 下载 CBT 模型
│   ├── download_testset.py  # 下载测试集
│   ├── load_longemotion_dataset.py # 加载数据集
│   └── run_vllm.sh          # vLLM服务器启动脚本
├── conf.d/                   # 配置文件目录
│   └── config.yaml.example   # 配置示例
├── requirements.txt          # 依赖包
└── README.md                 # 本文件
```

## 安装

### 1. 创建虚拟环境

推荐使用 `conda` 或 `virtualenv`：

#### 使用 Conda

```bash
conda create -n cactus python=3.8
conda activate cactus
```

#### 使用 Virtualenv

```bash
# 如果未安装virtualenv
pip install virtualenv

# 创建虚拟环境
virtualenv .venv
source .venv/bin/activate  # Linux & macOS
.venv\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置文件

复制配置文件示例并填写：

```bash
cp conf.d/config.yaml.example conf.d/config.yaml
```

编辑 `conf.d/config.yaml`：

```yaml
openai:
  key: <<Your openai API key>>

llama2:
  host: http://<<Server IP or URL>>/v1

llama3:
  host: http://<<Server IP or URL>>/v1
```

## 使用方法

### 1. 准备输入数据

创建JSON格式的客户信息表（intake form）：

```json
{
  "id": 1,
  "client_information": "25岁女性，工作压力大",
  "reason_counseling": "最近感到焦虑，难以入睡",
  "cbt_plan": "帮助客户识别焦虑的触发因素，建立应对策略"
}
```

或数组格式：

```json
[
  {
    "id": 1,
    "client_information": "...",
    "reason_counseling": "...",
    "cbt_plan": "..."
  }
]
```

### 2. 下载测试集（可选）

如果要使用 LongEmotion 测试集：

```bash
# 从 Hugging Face 下载 emotion_conversation 测试集
python scripts/download_testset.py \
    --output_file data/longemotion_testset.json \
    --split default \
    --max_samples 100
```

这会下载 [LongEmotion/LongEmotion](https://huggingface.co/datasets/LongEmotion/LongEmotion/viewer/default/emotion_conversation) 数据集的 `emotion_conversation` 子集作为测试集。

### 3. 运行推理

#### 使用Python脚本

```bash
# 使用自定义数据 + ChatGPT
python scripts/inference.py \
    --input_file ./data/intake_forms.json \
    --output_dir ./output \
    --counselor_type cactus \
    --llm_type chatgpt \
    --max_turns 20

# 使用 LongEmotion 测试集 + LongEmotion 模型
python scripts/inference.py \
    --input_file ./data/longemotion_testset.json \
    --output_dir ./output \
    --counselor_type cactus \
    --llm_type longemotion \
    --max_turns 20
```

#### 使用Shell脚本

```bash
sh scripts/inference.sh \
    --input_file ./data/intake_forms.json \
    --output_dir ./output \
    --counselor_type cactus \
    --llm_type chatgpt \
    --max_turns 20
```

### 3. 运行vLLM服务器（可选）

如果需要使用Llama2或Llama3模型，需要先启动vLLM服务器：

```bash
sh scripts/run_vllm.sh \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000
```

然后在 `config.yaml` 中配置对应的host。

## 添加新的咨询师代理

### 1. 创建提示文件

在 `prompts` 目录下创建文件，命名格式：`agent_{counselor_type}_{llm_type}.txt`

例如：`agent_new_counselor_chatgpt.txt`

提示文件应包含以下变量：
- `{client_information}` - 客户信息
- `{reason_counseling}` - 咨询原因
- `{cbt_plan}` - CBT计划
- `{history}` - 对话历史

### 2. 创建咨询师代理类

在 `src/agent.py` 中添加新类：

```python
class NewCounselorAgent(CounselorAgent):
    def __init__(self, llm_type):
        super().__init__(llm_type)
        self.language = "english"  # 或 "chinese"
        prompt_text = self.load_prompt(f"agent_new_{llm_type}.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["client_information", "reason_counseling", "cbt_plan", "history"],
            template=prompt_text
        )
    
    def generate(self, history, client_information="", reason_counseling="", cbt_plan=""):
        formatted_history = self.format_history(history)
        prompt = self.prompt_template.format(
            client_information=client_information,
            reason_counseling=reason_counseling,
            cbt_plan=cbt_plan,
            history=formatted_history
        )
        return self.llm.generate(prompt)
```

### 3. 添加到工厂类

在 `src/factory.py` 的 `CounselorFactory` 中添加：

```python
if counselor_type == "new":
    return NewCounselorAgent(llm_type)
```

## 添加新的LLM

### 1. 创建LLM类

在 `src/llm.py` 中添加：

```python
class NewLLM(LLM):
    def __init__(self):
        config = get_config()
        # 从配置中读取参数
        api_key = config.get('new', {}).get('key', '')
        self.llm = ChatOpenAI(
            model_name="new-model",
            temperature=0.7,
            openai_api_key=api_key
        )
    
    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content
```

### 2. 添加到工厂类

在 `src/factory.py` 的 `LLMFactory` 中添加：

```python
elif llm_type == "new":
    return NewLLM()
```

## 参数说明

- `--input_file`: 输入文件路径（JSON格式）
- `--output_dir`: 输出目录
- `--counselor_type`: 咨询师类型（默认：cactus）
- `--llm_type`: LLM类型（chatgpt, llama2, llama3, longemotion, cbt）
- `--max_turns`: 最大对话轮次（默认：20）

## 输出格式

输出为JSON文件，包含：
- 客户数据
- 对话历史
- 咨询师类型和LLM类型
- 最大轮次

## 引用

```
@misc{lee2024cactus,
      title={Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory}, 
      author={Suyeon Lee and Sunghwan Kim and Minju Kim and Dongjin Kang and Dongil Yang and Harim Kim and Minseok Kang and Dayi Jung and Min Hee Kim and Seungbeen Lee and Kyoung-Mee Chung and Youngjae Yu and Dongha Lee and Jinyoung Yeo},
      year={2024},
      eprint={2407.03103},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.03103}, 
}
```

## 许可证

GPL-2.0

## 链接

原始项目和数据集：https://github.com/coding-groot/cactus

Hugging Face: https://huggingface.co/collections/DLI-Lab/cactus-towards-psychological-counseling-conversations-6672312f6f64b0d7be75dd0b

