"""
特征提取模块

提供医疗文本的特征提取功能，包括文本特征、实体特征、
语义特征等多维度特征工程。
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from collections import Counter

from .entity_recognizer import MedicalEntityRecognizer, EntityType, MedicalEntity


@dataclass
class FeatureConfig:
    """特征提取配置"""
    # 文本特征
    extract_text_features: bool = True
    extract_statistical_features: bool = True
    
    # 实体特征
    extract_entity_features: bool = True
    entity_recognizer_config: Optional[Dict] = None
    
    # 语义特征
    extract_semantic_features: bool = True
    max_sequence_length: int = 512
    
    # 专科特征
    extract_specialty_features: bool = True
    target_specialties: List[str] = field(default_factory=lambda: [
        "心血管", "神经内科", "呼吸", "消化", "内分泌"
    ])
    
    # 性能优化
    use_cache: bool = True
    cache_size: int = 10000


@dataclass
class TextFeatures:
    """文本特征数据结构"""
    # 基础统计
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    
    # 长度特征
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    
    # 复杂度特征
    punctuation_count: int = 0
    digit_count: int = 0
    chinese_char_ratio: float = 0.0
    
    # 医疗特征
    medical_term_density: float = 0.0
    question_mark_count: int = 0


@dataclass
class EntityFeatures:
    """实体特征数据结构"""
    # 实体统计
    total_entity_count: int = 0
    entity_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 实体密度
    entity_density: float = 0.0  # 实体数/文本长度
    unique_entity_ratio: float = 0.0  # 唯一实体数/总实体数
    
    # 主要实体
    dominant_entity_type: Optional[str] = None
    top_entities: List[str] = field(default_factory=list)


@dataclass
class SpecialtyFeatures:
    """专科特征数据结构"""
    # 专科相关性分数
    specialty_scores: Dict[str, float] = field(default_factory=dict)
    
    # 主要专科
    primary_specialty: Optional[str] = None
    secondary_specialty: Optional[str] = None
    
    # 专科置信度
    specialty_confidence: float = 0.0


@dataclass
class ExtractedFeatures:
    """提取的完整特征数据结构"""
    text_id: str = ""
    raw_text: str = ""
    
    # 各维度特征
    text_features: TextFeatures = field(default_factory=TextFeatures)
    entity_features: EntityFeatures = field(default_factory=EntityFeatures)
    specialty_features: SpecialtyFeatures = field(default_factory=SpecialtyFeatures)
    
    # 原始实体列表
    entities: List[MedicalEntity] = field(default_factory=list)
    
    # 元数据
    extraction_time_ms: float = 0.0
    feature_version: str = "1.0"
    
    def to_vector(self) -> np.ndarray:
        """将特征转换为向量表示"""
        vector = []
        
        # 文本特征向量
        tf = self.text_features
        vector.extend([
            tf.char_count,
            tf.word_count,
            tf.sentence_count,
            tf.avg_word_length,
            tf.avg_sentence_length,
            tf.punctuation_count,
            tf.digit_count,
            tf.chinese_char_ratio,
            tf.medical_term_density,
            tf.question_mark_count,
        ])
        
        # 实体特征向量
        ef = self.entity_features
        vector.extend([
            ef.total_entity_count,
            ef.entity_density,
            ef.unique_entity_ratio,
        ])
        
        # 实体类型分布 (归一化)
        entity_types = [et.name for et in EntityType]
        for et_name in entity_types:
            vector.append(ef.entity_type_distribution.get(et_name, 0))
        
        # 专科特征向量
        sf = self.specialty_features
        for specialty in ["心血管", "神经内科", "呼吸", "消化", "内分泌"]:
            vector.append(sf.specialty_scores.get(specialty, 0.0))
        
        return np.array(vector, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "text_id": self.text_id,
            "raw_text": self.raw_text,
            "text_features": {
                "char_count": self.text_features.char_count,
                "word_count": self.text_features.word_count,
                "sentence_count": self.text_features.sentence_count,
                "avg_word_length": self.text_features.avg_word_length,
                "avg_sentence_length": self.text_features.avg_sentence_length,
                "punctuation_count": self.text_features.punctuation_count,
                "digit_count": self.text_features.digit_count,
                "chinese_char_ratio": self.text_features.chinese_char_ratio,
                "medical_term_density": self.text_features.medical_term_density,
                "question_mark_count": self.text_features.question_mark_count,
            },
            "entity_features": {
                "total_entity_count": self.entity_features.total_entity_count,
                "entity_type_distribution": self.entity_features.entity_type_distribution,
                "entity_density": self.entity_features.entity_density,
                "unique_entity_ratio": self.entity_features.unique_entity_ratio,
                "dominant_entity_type": self.entity_features.dominant_entity_type,
                "top_entities": self.entity_features.top_entities,
            },
            "specialty_features": {
                "specialty_scores": self.specialty_features.specialty_scores,
                "primary_specialty": self.specialty_features.primary_specialty,
                "secondary_specialty": self.specialty_features.secondary_specialty,
                "specialty_confidence": self.specialty_features.specialty_confidence,
            },
            "entities": [e.to_dict() for e in self.entities],
            "extraction_time_ms": self.extraction_time_ms,
            "feature_version": self.feature_version,
        }


class FeatureExtractor:
    """
    医疗文本特征提取器
    
    提供多维度特征提取功能，包括文本统计特征、
    医疗实体特征、专科领域特征等。
    
    Attributes:
        config: 特征提取配置
        entity_recognizer: 实体识别器
        _cache: 特征缓存
    
    Example:
        >>> config = FeatureConfig(extract_text_features=True)
        >>> extractor = FeatureExtractor(config)
        >>> features = extractor.extract("患者有高血压病史")
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征提取器
        
        Args:
            config: 特征提取配置，使用默认配置如果为None
        """
        self.config = config or FeatureConfig()
        self.entity_recognizer = MedicalEntityRecognizer(
            self.config.entity_recognizer_config.get("custom_dict_path") 
            if self.config.entity_recognizer_config else None
        )
        self._cache: Dict[str, ExtractedFeatures] = {}
        self._cache_order: List[str] = []
    
    def extract(self, text: str, text_id: str = "") -> ExtractedFeatures:
        """
        提取单个文本的特征
        
        Args:
            text: 输入文本
            text_id: 文本唯一标识，用于缓存
            
        Returns:
            提取的特征对象
            
        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract("患者有高血压病史", "doc_001")
            >>> print(features.text_features.char_count)
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = text_id or text
        if self.config.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        features = ExtractedFeatures(
            text_id=text_id,
            raw_text=text
        )
        
        # 提取文本特征
        if self.config.extract_text_features:
            features.text_features = self._extract_text_features(text)
        
        # 提取实体特征
        if self.config.extract_entity_features:
            entities = self.entity_recognizer.recognize(text)
            features.entities = entities
            features.entity_features = self._extract_entity_features(text, entities)
        
        # 提取专科特征
        if self.config.extract_specialty_features:
            features.specialty_features = self._extract_specialty_features(
                text, features.entities
            )
        
        # 记录提取时间
        features.extraction_time_ms = (time.time() - start_time) * 1000
        
        # 更新缓存
        if self.config.use_cache:
            self._update_cache(cache_key, features)
        
        return features
    
    def batch_extract(
        self, 
        texts: List[str], 
        text_ids: Optional[List[str]] = None
    ) -> List[ExtractedFeatures]:
        """
        批量提取文本特征
        
        Args:
            texts: 文本列表
            text_ids: 文本ID列表，可选
            
        Returns:
            特征对象列表
        """
        if text_ids is None:
            text_ids = [f"doc_{i}" for i in range(len(texts))]
        
        return [
            self.extract(text, text_id) 
            for text, text_id in zip(texts, text_ids)
        ]
    
    def _extract_text_features(self, text: str) -> TextFeatures:
        """提取文本统计特征"""
        features = TextFeatures()
        
        # 基础统计
        features.char_count = len(text)
        features.word_count = len(text.split())
        
        # 句子统计 (按标点符号分割)
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features.sentence_count = len(sentences)
        
        # 平均长度
        if features.word_count > 0:
            features.avg_word_length = features.char_count / features.word_count
        if features.sentence_count > 0:
            features.avg_sentence_length = features.word_count / features.sentence_count
        
        # 标点符号
        features.punctuation_count = len(re.findall(r'[，。！？、；：""''（）【】]', text))
        
        # 数字
        features.digit_count = len(re.findall(r'\d', text))
        
        # 中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if features.char_count > 0:
            features.chinese_char_ratio = chinese_chars / features.char_count
        
        # 医疗术语密度 (简单估计)
        medical_patterns = [
            r'[病患]者', r'[诊治]疗', r'[检检]查', r'[术手]',
            r'[病症]状', r'[药用]药', r'[医医院]'
        ]
        medical_term_count = sum(
            len(re.findall(pattern, text)) 
            for pattern in medical_patterns
        )
        if features.char_count > 0:
            features.medical_term_density = medical_term_count / features.char_count
        
        # 问号数量 (用于判断问题类型)
        features.question_mark_count = text.count('?') + text.count('？')
        
        return features
    
    def _extract_entity_features(
        self, 
        text: str, 
        entities: List[MedicalEntity]
    ) -> EntityFeatures:
        """提取实体相关特征"""
        features = EntityFeatures()
        
        if not entities:
            return features
        
        # 实体统计
        features.total_entity_count = len(entities)
        
        # 实体类型分布
        type_counts = Counter(e.entity_type.name for e in entities)
        features.entity_type_distribution = dict(type_counts)
        
        # 实体密度
        if len(text) > 0:
            features.entity_density = len(entities) / len(text)
        
        # 唯一实体比例
        unique_entities = set((e.text, e.entity_type) for e in entities)
        if len(entities) > 0:
            features.unique_entity_ratio = len(unique_entities) / len(entities)
        
        # 主导实体类型
        if type_counts:
            features.dominant_entity_type = type_counts.most_common(1)[0][0]
        
        # Top实体 (按出现频率)
        entity_texts = [e.text for e in entities]
        entity_freq = Counter(entity_texts)
        features.top_entities = [e for e, _ in entity_freq.most_common(5)]
        
        return features
    
    def _extract_specialty_features(
        self, 
        text: str, 
        entities: List[MedicalEntity]
    ) -> SpecialtyFeatures:
        """提取专科领域特征"""
        features = SpecialtyFeatures()
        
        specialty_keywords = self.entity_recognizer.specialty_keywords
        
        # 计算各专科相关性分数
        scores = {}
        for specialty, keywords in specialty_keywords.items():
            score = 0.0
            
            # 基于关键词匹配
            for keyword in keywords:
                if keyword in text:
                    score += 1.0
            
            # 基于实体类型加权
            for entity in entities:
                if entity.text in keywords:
                    score += entity.confidence * 2.0
            
            # 归一化
            if len(keywords) > 0:
                score = min(1.0, score / len(keywords))
            
            scores[specialty] = score
        
        features.specialty_scores = scores
        
        # 确定主要和次要专科
        sorted_specialties = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if sorted_specialties:
            features.primary_specialty = sorted_specialties[0][0]
            features.specialty_confidence = sorted_specialties[0][1]
            
            if len(sorted_specialties) > 1:
                features.secondary_specialty = sorted_specialties[1][0]
        
        return features
    
    def _update_cache(self, key: str, features: ExtractedFeatures) -> None:
        """更新特征缓存"""
        if len(self._cache) >= self.config.cache_size:
            # LRU淘汰
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = features
        self._cache_order.append(key)
    
    def clear_cache(self) -> None:
        """清空特征缓存"""
        self._cache.clear()
        self._cache_order.clear()
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = []
        
        # 文本特征
        names.extend([
            "char_count", "word_count", "sentence_count",
            "avg_word_length", "avg_sentence_length",
            "punctuation_count", "digit_count",
            "chinese_char_ratio", "medical_term_density",
            "question_mark_count"
        ])
        
        # 实体特征
        names.extend([
            "total_entity_count", "entity_density", "unique_entity_ratio"
        ])
        
        # 实体类型分布
        names.extend([f"entity_type_{et.name}" for et in EntityType])
        
        # 专科特征
        names.extend([
            f"specialty_{s}" 
            for s in ["心血管", "神经内科", "呼吸", "消化", "内分泌"]
        ])
        
        return names
