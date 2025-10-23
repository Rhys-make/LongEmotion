"""
模型模块
包含五个任务的模型实现
"""
from .classification_model import EmotionClassificationModel
from .detection_model import EmotionDetectionModel, EmotionDetectionModelWrapper
from .conversation_model import EmotionConversationModel
from .summary_model import EmotionSummaryModel
from .qa_model import EmotionQAModel, GenerativeQAModel

__all__ = [
    'EmotionClassificationModel',
    'EmotionDetectionModel',
    'EmotionDetectionModelWrapper',
    'EmotionConversationModel',
    'EmotionSummaryModel',
    'EmotionQAModel',
    'GenerativeQAModel'
]

