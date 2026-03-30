"""
特征工程模块 - 医疗领域特征提取与实体识别

该模块提供医疗文本的特征提取、实体识别和向量化功能，
支持专科领域（心血管、神经内科等）的特征工程需求。
"""

from .entity_recognizer import MedicalEntityRecognizer, EntityType
from .feature_extractor import FeatureExtractor, FeatureConfig
from .vectorizer import MedicalVectorizer

__all__ = [
    "MedicalEntityRecognizer",
    "EntityType",
    "FeatureExtractor",
    "FeatureConfig",
    "MedicalVectorizer",
]
