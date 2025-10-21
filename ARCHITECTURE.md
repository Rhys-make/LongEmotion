# LongEmotion 架构设计文档

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                       LongEmotion 系统                           │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  数据层      │    │  模型层      │    │  服务层      │
│  (Data)      │    │  (Models)    │    │  (API)       │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ - 数据下载   │    │ - 5个任务    │    │ - FastAPI    │
│ - 预处理     │    │ - 模型训练   │    │ - REST API   │
│ - 数据加载   │    │ - 模型推理   │    │ - 文档生成   │
└──────────────┘    └──────────────┘    └──────────────┘
```

## 模块依赖关系

```
api/main.py
    │
    ├─→ models/classification_model.py
    │       └─→ transformers.BertForSequenceClassification
    │
    ├─→ models/detection_model.py
    │       └─→ transformers.BertModel + Custom Head
    │
    ├─→ models/conversation_model.py
    │       └─→ transformers.AutoModelForCausalLM
    │
    ├─→ models/summary_model.py
    │       └─→ transformers.AutoModelForSeq2SeqLM
    │
    └─→ models/qa_model.py
            └─→ transformers.BertForQuestionAnswering

scripts/train_all.py
    │
    ├─→ utils/trainer.py
    │       ├─→ torch.optim.AdamW
    │       └─→ transformers.get_linear_schedule_with_warmup
    │
    ├─→ utils/evaluator.py
    │       ├─→ sklearn.metrics
    │       └─→ rouge_score
    │
    └─→ utils/preprocess.py
            └─→ re, typing
```

## 数据流图

### 训练流程

```
HuggingFace Dataset
        │
        ▼
load_from_disk()
        │
        ▼
TextPreprocessor
        │
        ├─→ clean_text()
        ├─→ process_classification_data()
        ├─→ process_detection_data()
        ├─→ process_conversation_data()
        ├─→ process_summary_data()
        └─→ process_qa_data()
        │
        ▼
DataLoader (batch)
        │
        ▼
Model Training
        │
        ├─→ forward()
        ├─→ loss.backward()
        ├─→ optimizer.step()
        └─→ scheduler.step()
        │
        ▼
Evaluation
        │
        ├─→ compute_metrics()
        └─→ early_stopping_check()
        │
        ▼
Save Best Model
```

### 推理流程

```
Test Dataset
        │
        ▼
Load Model
        │
        ▼
Batch Inference
        │
        ├─→ preprocess()
        ├─→ model.predict()
        └─→ postprocess()
        │
        ▼
Save Results (.jsonl)
```

### API 请求流程

```
HTTP Request
        │
        ▼
FastAPI Router
        │
        ▼
Pydantic Validation
        │
        ▼
ModelManager
        │
        ├─→ lazy_load_model()
        │
        ▼
Model Inference
        │
        ▼
Response Model
        │
        ▼
HTTP Response (JSON)
```

## 类图

### 模型层

```
┌─────────────────────────────────┐
│ EmotionClassificationModel      │
├─────────────────────────────────┤
│ + model: BertForSequenceClass   │
│ + tokenizer: BertTokenizer      │
├─────────────────────────────────┤
│ + preprocess(texts)             │
│ + predict(texts)                │
│ + save(path)                    │
│ + load(path): cls               │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ EmotionDetectionModelWrapper    │
├─────────────────────────────────┤
│ + model: EmotionDetectionModel  │
│ + tokenizer: BertTokenizer      │
│ + threshold: float              │
├─────────────────────────────────┤
│ + preprocess(texts)             │
│ + predict(texts)                │
│ + save(path)                    │
│ + load(path): cls               │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ EmotionConversationModel        │
├─────────────────────────────────┤
│ + model: AutoModelForCausalLM   │
│ + tokenizer: AutoTokenizer      │
│ + generation_config             │
├─────────────────────────────────┤
│ + format_prompt(context, emot)  │
│ + generate_response(context)    │
│ + batch_generate(contexts)      │
│ + save(path)                    │
│ + load(path): cls               │
└─────────────────────────────────┘
```

### 训练层

```
┌─────────────────────────────────┐
│ UnifiedTrainer                  │
├─────────────────────────────────┤
│ + model: nn.Module              │
│ + train_dataloader: DataLoader  │
│ + eval_dataloader: DataLoader   │
│ + optimizer: AdamW              │
│ + scheduler: LRScheduler        │
│ + evaluator: Evaluator          │
├─────────────────────────────────┤
│ + train()                       │
│ + _train_epoch()                │
│ + _eval_epoch()                 │
│ + save_model(name)              │
│ + load_model(path): cls         │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ Evaluator                       │
├─────────────────────────────────┤
│ + compute_classification_metrics│
│ + compute_detection_metrics     │
│ + compute_generation_metrics    │
│ + compute_qa_metrics            │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ TextPreprocessor                │
├─────────────────────────────────┤
│ + emotion_labels: dict          │
├─────────────────────────────────┤
│ + clean_text(text)              │
│ + process_classification_data() │
│ + process_detection_data()      │
│ + process_conversation_data()   │
│ + process_summary_data()        │
│ + process_qa_data()             │
└─────────────────────────────────┘
```

### API 层

```
┌─────────────────────────────────┐
│ ModelManager                    │
├─────────────────────────────────┤
│ - _classification_model         │
│ - _detection_model              │
│ - _conversation_model           │
│ - _summary_model                │
│ - _qa_model                     │
├─────────────────────────────────┤
│ + classification_model @property│
│ + detection_model @property     │
│ + conversation_model @property  │
│ + summary_model @property       │
│ + qa_model @property            │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ FastAPI App                     │
├─────────────────────────────────┤
│ + /                             │
│ + /health                       │
│ + /classify (POST)              │
│ + /classify/batch (POST)        │
│ + /detect (POST)                │
│ + /detect/batch (POST)          │
│ + /conversation (POST)          │
│ + /summary (POST)               │
│ + /summary/batch (POST)         │
│ + /qa (POST)                    │
└─────────────────────────────────┘
```

## 时序图

### 训练时序

```
User → TaskTrainer → DataLoader → Model → Optimizer → Evaluator → Disk
  │         │            │          │         │          │         │
  │   train_all()        │          │         │          │         │
  │         ├───load_dataset()      │         │          │         │
  │         │            │          │         │          │         │
  │         ├────init_model()       │         │          │         │
  │         │            │          │         │          │         │
  │         ├─────────batch()       │         │          │         │
  │         │            └─────forward()      │          │         │
  │         │                       └──loss.backward()   │         │
  │         │                                  └─step()   │         │
  │         │                                             │         │
  │         ├─────────────────────────────compute_metrics()        │
  │         │                                             │         │
  │         └─────────────────────────────────────────save_model() │
  │                                                       │         │
  ◄─────────────────────────────────────────────────────────────────┘
```

### API 请求时序

```
Client → FastAPI → ModelManager → Model → Response
  │         │           │           │         │
  │    POST /classify   │           │         │
  │         │           │           │         │
  │         ├──validate_request()  │         │
  │         │           │           │         │
  │         └───get_model()        │         │
  │                     └──load_if_needed()  │
  │                                 │         │
  │                         model.predict()  │
  │                                 │         │
  │                                 └──result │
  │                                           │
  │         ◄─────────────────────────────────┘
  │         │                                  
  │    format_response()                      
  │         │                                  
  ◄─────────┘                                  
```

## 配置管理

### 配置层次

```
config.py (全局配置)
    │
    ├─→ MODEL_CONFIGS
    │       ├─→ classification
    │       ├─→ detection
    │       ├─→ conversation
    │       ├─→ summary
    │       └─→ qa
    │
    ├─→ TRAINING_CONFIG
    │       ├─→ early_stopping_patience
    │       ├─→ warmup_ratio
    │       └─→ weight_decay
    │
    ├─→ API_CONFIG
    │       ├─→ host
    │       ├─→ port
    │       └─→ version
    │
    └─→ HF_CONFIG
            ├─→ dataset_name
            ├─→ cache_dir
            └─→ use_auth_token
```

## 错误处理机制

### 训练错误处理

```
try:
    TaskTrainer.train_all()
        │
        ├─→ try: train_classification()
        │   └─→ except: log_error() + continue
        │
        ├─→ try: train_detection()
        │   └─→ except: log_error() + continue
        │
        └─→ ... (其他任务)
finally:
    save_training_summary()
```

### API 错误处理

```
@app.post("/classify")
async def classify(request):
    try:
        validate(request)
        result = model.predict()
        return response
    except ValidationError:
        raise HTTPException(422)
    except ModelError:
        raise HTTPException(500)
    except Exception as e:
        log_error(e)
        raise HTTPException(500)
```

## 扩展点

### 1. 添加新模型

```python
# models/new_model.py
class NewTaskModel:
    def __init__(self): ...
    def predict(self): ...
    def save(self): ...
    @classmethod
    def load(cls): ...
```

### 2. 添加新预处理

```python
# utils/preprocess.py
class TextPreprocessor:
    def process_new_task_data(self, examples):
        # 新任务的数据处理
        ...
```

### 3. 添加新评估指标

```python
# utils/evaluator.py
class Evaluator:
    def compute_new_task_metrics(self, preds, labels):
        # 新任务的指标计算
        ...
```

### 4. 添加新 API 端点

```python
# api/main.py
@app.post("/new_task")
async def new_task(request: NewTaskRequest):
    result = model_manager.new_task_model.predict()
    return NewTaskResponse(**result)
```

## 性能优化策略

### 模型加载优化

1. **延迟加载**: 仅在需要时加载模型
2. **模型缓存**: 保持模型在内存中
3. **模型量化**: 使用 8-bit/4-bit 量化

### 推理优化

1. **批量处理**: 将多个请求合并处理
2. **异步处理**: 使用 FastAPI 异步特性
3. **GPU 加速**: 使用 CUDA 加速推理

### 内存优化

1. **梯度检查点**: 减少训练显存
2. **混合精度**: 使用 FP16 训练
3. **动态批量**: 根据输入长度调整批量大小

## 安全考虑

### API 安全

1. **输入验证**: Pydantic 模型验证
2. **速率限制**: 限制请求频率
3. **认证授权**: JWT/OAuth2

### 模型安全

1. **对抗样本防御**: 输入过滤
2. **模型水印**: 保护模型版权
3. **隐私保护**: 差分隐私

## 监控与日志

### 日志系统

```
logs/
├── training/
│   ├── classification.log
│   ├── detection.log
│   └── ...
│
├── inference/
│   └── predictions.log
│
└── api/
    ├── access.log
    └── error.log
```

### 监控指标

- 请求延迟
- 错误率
- GPU/CPU 使用率
- 内存使用
- 模型准确率

---

**文档版本**: 1.0  
**最后更新**: 2024  
**维护者**: LongEmotion Team

