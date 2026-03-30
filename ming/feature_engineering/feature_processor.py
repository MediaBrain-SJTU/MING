"""
特征处理器
对提取的特征进行处理、转换和增强
"""
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

from ming.feature_engineering.feature_extractor import MedicalFeatures


@dataclass
class ProcessedFeatures:
    """处理后的特征"""
    question_id: str
    feature_vector: Optional[np.ndarray] = None
    feature_dict: Dict[str, Any] = field(default_factory=dict)
    normalized_features: Dict[str, float] = field(default_factory=dict)
    specialty_onehot: Dict[str, int] = field(default_factory=dict)
    difficulty_onehot: Dict[str, int] = field(default_factory=dict)
    entity_features: Dict[str, float] = field(default_factory=dict)
    metadata_features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "question_id": self.question_id,
            "feature_dict": self.feature_dict,
            "normalized_features": self.normalized_features,
            "specialty_onehot": self.specialty_onehot,
            "difficulty_onehot": self.difficulty_onehot,
            "entity_features": self.entity_features,
            "metadata_features": self.metadata_features
        }
        if self.feature_vector is not None:
            result["feature_vector"] = self.feature_vector.tolist()
        return result


class FeatureProcessor:
    """
    特征处理器
    
    功能：
    - 特征标准化
    - 特征编码
    - 特征选择
    - 特征增强
    """
    
    SPECIALTY_LIST = [
        "cardiovascular", "neurology", "hematology", "endocrinology",
        "gastroenterology", "pediatrics", "obstetrics_gynecology",
        "psychiatry", "immunology", "pathology", "general"
    ]
    
    DIFFICULTY_LIST = ["easy", "medium", "hard"]
    
    ANSWER_TYPE_LIST = [
        "diagnosis", "treatment", "examination", "mechanism",
        "classification", "complication", "prognosis", "knowledge"
    ]
    
    REASONING_COMPLEXITY_LIST = ["simple", "medium", "complex"]
    
    ENTITY_TYPES = [
        "DISEASE", "SYMPTOM", "MEDICINE", "BODY_PART",
        "EXAMINATION", "TREATMENT", "LAB_VALUE", "CLINICAL_DEPT"
    ]
    
    def __init__(
        self,
        normalize: bool = True,
        max_entity_count: int = 50,
        feature_dim: int = 128
    ):
        """
        初始化特征处理器
        
        Args:
            normalize: 是否进行特征标准化
            max_entity_count: 实体数量上限
            feature_dim: 特征向量维度
        """
        self.normalize = normalize
        self.max_entity_count = max_entity_count
        self.feature_dim = feature_dim
        
        self._normalization_params = {}
        self._feature_stats = {}
    
    def fit(
        self,
        features_list: List[MedicalFeatures]
    ) -> "FeatureProcessor":
        """
        拟合特征处理器，计算标准化参数
        
        Args:
            features_list: 特征列表
            
        Returns:
            self
        """
        entity_counts = [f.entity_count for f in features_list]
        question_lengths = [f.question_length for f in features_list]
        
        self._normalization_params = {
            "entity_count": {
                "mean": np.mean(entity_counts) if entity_counts else 0,
                "std": np.std(entity_counts) if entity_counts else 1
            },
            "question_length": {
                "mean": np.mean(question_lengths) if question_lengths else 0,
                "std": np.std(question_lengths) if question_lengths else 1
            }
        }
        
        all_entity_types = Counter()
        for f in features_list:
            for entity_type, count in f.entity_type_distribution.items():
                all_entity_types[entity_type] += count
        
        self._feature_stats["entity_type_frequencies"] = dict(all_entity_types)
        self._feature_stats["total_samples"] = len(features_list)
        
        return self
    
    def process(
        self,
        features: MedicalFeatures
    ) -> ProcessedFeatures:
        """
        处理单个特征
        
        Args:
            features: 原始特征
            
        Returns:
            ProcessedFeatures: 处理后的特征
        """
        processed = ProcessedFeatures(question_id=features.question_id or "")
        
        processed.feature_dict = self._extract_feature_dict(features)
        
        processed.specialty_onehot = self._one_hot_encode(
            features.specialty, self.SPECIALTY_LIST
        )
        
        processed.difficulty_onehot = self._one_hot_encode(
            features.difficulty_level, self.DIFFICULTY_LIST
        )
        
        processed.entity_features = self._process_entity_features(features)
        
        processed.normalized_features = self._normalize_features(features)
        
        processed.metadata_features = {
            "has_age_info": features.has_age_info,
            "has_gender_info": features.has_gender_info,
            "has_numeric_data": features.has_numeric_data,
            "has_lab_values": features.has_lab_values,
            "age_value": features.age_value,
            "gender": features.gender
        }
        
        processed.feature_vector = self._build_feature_vector(processed)
        
        return processed
    
    def _extract_feature_dict(self, features: MedicalFeatures) -> Dict[str, Any]:
        """提取特征字典"""
        return {
            "entity_count": features.entity_count,
            "question_length": features.question_length,
            "option_count": features.option_count,
            "specialty": features.specialty,
            "difficulty_level": features.difficulty_level,
            "answer_type": features.answer_type,
            "reasoning_complexity": features.reasoning_complexity,
            "disease_count": len(features.disease_entities),
            "symptom_count": len(features.symptom_entities),
            "medicine_count": len(features.medicine_entities),
            "body_part_count": len(features.body_part_entities),
            "examination_count": len(features.examination_entities),
            "treatment_count": len(features.treatment_entities),
            "lab_value_count": len(features.lab_value_entities),
            "clinical_dept_count": len(features.clinical_dept_entities)
        }
    
    def _one_hot_encode(
        self,
        value: str,
        categories: List[str]
    ) -> Dict[str, int]:
        """独热编码"""
        one_hot = {cat: 0 for cat in categories}
        if value in one_hot:
            one_hot[value] = 1
        return one_hot
    
    def _process_entity_features(
        self,
        features: MedicalFeatures
    ) -> Dict[str, float]:
        """处理实体特征"""
        entity_features = {}
        
        total_entities = max(features.entity_count, 1)
        
        for entity_type in self.ENTITY_TYPES:
            count = features.entity_type_distribution.get(entity_type, 0)
            entity_features[f"{entity_type.lower()}_ratio"] = count / total_entities
            entity_features[f"{entity_type.lower()}_count"] = count
        
        entity_features["total_entity_count"] = features.entity_count
        entity_features["entity_diversity"] = len(features.entity_type_distribution) / len(self.ENTITY_TYPES)
        
        return entity_features
    
    def _normalize_features(
        self,
        features: MedicalFeatures
    ) -> Dict[str, float]:
        """标准化特征"""
        normalized = {}
        
        if self.normalize and self._normalization_params:
            for key, params in self._normalization_params.items():
                value = getattr(features, key, 0)
                if params["std"] > 0:
                    normalized[key] = (value - params["mean"]) / params["std"]
                else:
                    normalized[key] = 0
        else:
            normalized["entity_count"] = features.entity_count
            normalized["question_length"] = features.question_length
        
        return normalized
    
    def _build_feature_vector(
        self,
        processed: ProcessedFeatures
    ) -> np.ndarray:
        """构建特征向量"""
        feature_list = []
        
        feature_list.extend(processed.specialty_onehot.values())
        feature_list.extend(processed.difficulty_onehot.values())
        
        feature_list.extend(processed.entity_features.values())
        
        feature_list.extend(processed.normalized_features.values())
        
        answer_type_onehot = self._one_hot_encode(
            processed.feature_dict.get("answer_type", "knowledge"),
            self.ANSWER_TYPE_LIST
        )
        feature_list.extend(answer_type_onehot.values())
        
        reasoning_onehot = self._one_hot_encode(
            processed.feature_dict.get("reasoning_complexity", "medium"),
            self.REASONING_COMPLEXITY_LIST
        )
        feature_list.extend(reasoning_onehot.values())
        
        metadata_values = [
            float(processed.metadata_features.get("has_age_info", 0)),
            float(processed.metadata_features.get("has_gender_info", 0)),
            float(processed.metadata_features.get("has_numeric_data", 0)),
            float(processed.metadata_features.get("has_lab_values", 0))
        ]
        feature_list.extend(metadata_values)
        
        feature_vector = np.array(feature_list, dtype=np.float32)
        
        if len(feature_vector) < self.feature_dim:
            padding = np.zeros(self.feature_dim - len(feature_vector), dtype=np.float32)
            feature_vector = np.concatenate([feature_vector, padding])
        elif len(feature_vector) > self.feature_dim:
            feature_vector = feature_vector[:self.feature_dim]
        
        return feature_vector
    
    def batch_process(
        self,
        features_list: List[MedicalFeatures],
        fit_first: bool = False
    ) -> List[ProcessedFeatures]:
        """
        批量处理特征
        
        Args:
            features_list: 特征列表
            fit_first: 是否先拟合
            
        Returns:
            处理后的特征列表
        """
        if fit_first:
            self.fit(features_list)
        
        return [self.process(f) for f in features_list]
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = []
        
        names.extend([f"specialty_{s}" for s in self.SPECIALTY_LIST])
        names.extend([f"difficulty_{d}" for d in self.DIFFICULTY_LIST])
        
        for entity_type in self.ENTITY_TYPES:
            names.append(f"{entity_type.lower()}_ratio")
            names.append(f"{entity_type.lower()}_count")
        names.extend(["total_entity_count", "entity_diversity"])
        
        names.extend(["normalized_entity_count", "normalized_question_length"])
        
        names.extend([f"answer_type_{a}" for a in self.ANSWER_TYPE_LIST])
        names.extend([f"reasoning_{r}" for r in self.REASONING_COMPLEXITY_LIST])
        
        names.extend(["has_age", "has_gender", "has_numeric", "has_lab"])
        
        return names
    
    def export_normalization_params(self, output_path: str) -> None:
        """导出标准化参数"""
        params = {
            "normalization_params": self._normalization_params,
            "feature_stats": self._feature_stats,
            "config": {
                "normalize": self.normalize,
                "max_entity_count": self.max_entity_count,
                "feature_dim": self.feature_dim
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    
    def import_normalization_params(self, input_path: str) -> None:
        """导入标准化参数"""
        with open(input_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        self._normalization_params = params.get("normalization_params", {})
        self._feature_stats = params.get("feature_stats", {})
    
    def augment_features(
        self,
        features: MedicalFeatures,
        augmentation_ratio: float = 0.1
    ) -> MedicalFeatures:
        """
        特征增强
        
        Args:
            features: 原始特征
            augmentation_ratio: 增强比例
            
        Returns:
            增强后的特征
        """
        import copy
        augmented = copy.deepcopy(features)
        
        if augmented.entity_count > 0:
            noise = int(augmented.entity_count * augmentation_ratio)
            augmented.entity_count = max(1, augmented.entity_count + noise)
        
        return augmented
    
    def select_features(
        self,
        processed_features: ProcessedFeatures,
        selected_names: List[str]
    ) -> Dict[str, Any]:
        """
        特征选择
        
        Args:
            processed_features: 处理后的特征
            selected_names: 选择的特征名称
            
        Returns:
            选择的特征字典
        """
        all_features = {
            **processed_features.feature_dict,
            **processed_features.specialty_onehot,
            **processed_features.difficulty_onehot,
            **processed_features.entity_features,
            **processed_features.normalized_features,
            **processed_features.metadata_features
        }
        
        return {name: all_features.get(name) for name in selected_names if name in all_features}
