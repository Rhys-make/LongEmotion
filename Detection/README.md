# LongEmotion 比赛核心文件

## 📁 文件结构

### model/
- `best_model.pt` - 训练好的BERT模型（验证准确率91.47%）

### test_data/
- `test.jsonl` - 比赛测试集（136个样本）

### scripts/
- `run_inference_final.py` - 推理运行脚本
- `inference_longemotion.py` - 推理核心逻辑
- `convert_submission_format.py` - 格式转换脚本
- `detection_model.py` - 模型定义

### submission/
- `submission.jsonl` - 提交文件（格式: {"id": 0, "predicted_index": 24}）
- `Emotion_Detection_Result.jsonl` - 备份提交文件

### reports/
- 项目进度报告和自查报告

## 🚀 使用方法

### 运行推理
```bash
# 从Detection文件夹内运行
cd Detection
python scripts/run_inference_final.py
```

### 转换格式（可选）
```bash
# 从Detection/scripts文件夹内运行
cd Detection/scripts
python convert_submission_format.py
```

## 📊 模型性能
- 验证准确率: 91.47%
- 平均预测置信度: 89.27%
- 推理时间: ~5-10分钟/136样本

## 📝 提交
提交文件: `submission/submission.jsonl`
