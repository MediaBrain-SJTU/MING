"""
特征工程模块初始化文件

本模块提供医疗文本特征提取、实体识别和特征工程的核心功能。
"""

from ming.feature_engineering.entity_recognizer import MedicalEntityRecognizer
from ming.feature_engineering.feature_extractor import FeatureExtractor
from ming.feature_engineering.feature_selector import FeatureSelector
from ming.feature_engineering.feature_utils import (
    preprocess_text,
    extract_medical_features,
    create_feature_embedding,
    FeatureProcessingError,
)

__all__ = [
    "MedicalEntityRecognizer",
    "FeatureExtractor",
    "FeatureSelector",
    "preprocess_text",
    "extract_medical_features",
    "create_feature_embedding",
    "FeatureProcessingError",
]
