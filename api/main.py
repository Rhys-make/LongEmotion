"""
FastAPI 服务接口
提供五个情感任务的 REST API
"""
import sys
from pathlib import Path
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.classification_model import EmotionClassificationModel
from models.detection_model import EmotionDetectionModelWrapper
from models.conversation_model import EmotionConversationModel
from models.summary_model import EmotionSummaryModel
from models.qa_model import EmotionQAModel


# ==================== 请求/响应模型 ====================

class TextRequest(BaseModel):
    """文本输入请求"""
    text: str


class BatchTextRequest(BaseModel):
    """批量文本输入请求"""
    texts: List[str]


class ClassificationResponse(BaseModel):
    """分类响应"""
    label: int
    emotion: str
    confidence: float
    probabilities: dict


class DetectionResponse(BaseModel):
    """检测响应"""
    emotions: List[dict]
    all_scores: dict


class ConversationRequest(BaseModel):
    """对话请求"""
    context: str
    emotion: Optional[str] = None


class ConversationResponse(BaseModel):
    """对话响应"""
    response: str


class SummaryResponse(BaseModel):
    """摘要响应"""
    summary: str


class QARequest(BaseModel):
    """问答请求"""
    question: str
    context: str


class QAResponse(BaseModel):
    """问答响应"""
    answer: str
    confidence: float


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="LongEmotion API",
    description="情感分析与生成的统一 API 接口",
    version="1.0.0"
)


# ==================== 全局模型加载 ====================

class ModelManager:
    """模型管理器"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", device: str = "cuda"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        
        # 延迟加载模型
        self._classification_model = None
        self._detection_model = None
        self._conversation_model = None
        self._summary_model = None
        self._qa_model = None
    
    @property
    def classification_model(self):
        if self._classification_model is None:
            print("加载分类模型...")
            model_path = self.checkpoint_dir / "classification" / "best_model"
            self._classification_model = EmotionClassificationModel.load(
                str(model_path), device=self.device
            )
        return self._classification_model
    
    @property
    def detection_model(self):
        if self._detection_model is None:
            print("加载检测模型...")
            model_path = self.checkpoint_dir / "detection" / "best_model"
            self._detection_model = EmotionDetectionModelWrapper.load(
                str(model_path), device=self.device
            )
        return self._detection_model
    
    @property
    def conversation_model(self):
        if self._conversation_model is None:
            print("加载对话模型...")
            model_path = self.checkpoint_dir / "conversation" / "best_model"
            
            if model_path.exists():
                self._conversation_model = EmotionConversationModel.load(
                    str(model_path), device=self.device
                )
            else:
                print("⚠ 未找到微调模型，使用基础模型")
                self._conversation_model = EmotionConversationModel(device=self.device)
        return self._conversation_model
    
    @property
    def summary_model(self):
        if self._summary_model is None:
            print("加载摘要模型...")
            model_path = self.checkpoint_dir / "summary" / "best_model"
            self._summary_model = EmotionSummaryModel.load(
                str(model_path), device=self.device
            )
        return self._summary_model
    
    @property
    def qa_model(self):
        if self._qa_model is None:
            print("加载问答模型...")
            model_path = self.checkpoint_dir / "qa" / "best_model"
            self._qa_model = EmotionQAModel.load(
                str(model_path), device=self.device
            )
        return self._qa_model


# 创建全局模型管理器
model_manager = ModelManager()


# ==================== API 路由 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用 LongEmotion API",
        "version": "1.0.0",
        "endpoints": {
            "classification": "/classify",
            "detection": "/detect",
            "conversation": "/conversation",
            "summary": "/summary",
            "qa": "/qa"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: TextRequest):
    """
    情感分类接口
    
    输入文本，返回情感分类结果
    """
    try:
        results = model_manager.classification_model.predict([request.text])
        result = results[0]
        
        return ClassificationResponse(
            label=result['label'],
            emotion=result['emotion'],
            confidence=result['confidence'],
            probabilities=result['probabilities']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")


@app.post("/classify/batch")
async def classify_batch(request: BatchTextRequest):
    """批量情感分类"""
    try:
        results = model_manager.classification_model.predict(request.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量分类失败: {str(e)}")


@app.post("/detect", response_model=DetectionResponse)
async def detect(request: TextRequest):
    """
    情感检测接口
    
    输入文本，返回检测到的多个情感
    """
    try:
        results = model_manager.detection_model.predict([request.text])
        result = results[0]
        
        return DetectionResponse(
            emotions=result['emotions'],
            all_scores=result['all_scores']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/detect/batch")
async def detect_batch(request: BatchTextRequest):
    """批量情感检测"""
    try:
        results = model_manager.detection_model.predict(request.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量检测失败: {str(e)}")


@app.post("/conversation", response_model=ConversationResponse)
async def conversation(request: ConversationRequest):
    """
    情感对话接口
    
    输入对话上下文和目标情感，生成回复
    """
    try:
        response = model_manager.conversation_model.generate_response(
            context=request.context,
            emotion=request.emotion
        )
        
        return ConversationResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")


@app.post("/summary", response_model=SummaryResponse)
async def summary(request: TextRequest):
    """
    情感摘要接口
    
    输入长文本，返回摘要
    """
    try:
        summary_text = model_manager.summary_model.generate_summary(request.text)
        
        return SummaryResponse(summary=summary_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"摘要生成失败: {str(e)}")


@app.post("/summary/batch")
async def summary_batch(request: BatchTextRequest):
    """批量情感摘要"""
    try:
        results = model_manager.summary_model.batch_generate(request.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量摘要失败: {str(e)}")


@app.post("/qa", response_model=QAResponse)
async def qa(request: QARequest):
    """
    情感问答接口
    
    输入问题和上下文，返回答案
    """
    try:
        result = model_manager.qa_model.extract_answer(
            question=request.question,
            context=request.context
        )
        
        return QAResponse(
            answer=result['answer'],
            confidence=result['confidence']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


# ==================== 启动服务 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动 LongEmotion API 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="模型目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--reload", action="store_true", help="自动重载")
    
    args = parser.parse_args()
    
    # 更新模型管理器配置
    model_manager.checkpoint_dir = Path(args.checkpoint_dir)
    model_manager.device = args.device
    
    # 启动服务
    print(f"启动 LongEmotion API 服务...")
    print(f"访问地址: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

