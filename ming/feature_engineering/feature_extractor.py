"""
医疗特征提取器
从医疗文本中提取结构化特征
目标：特征提取耗时 <= 50ms/条, 特征覆盖率 >= 95%
"""
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
import json

from ming.feature_engineering.entity_recognition import (
    MedicalEntityRecognizer,
    MedicalEntity,
    EntityRecognitionResult
)


@dataclass
class MedicalFeatures:
    """医疗特征数据结构"""
    question_id: Optional[str] = None
    specialty: str = ""
    difficulty_level: str = ""
    
    disease_entities: List[str] = field(default_factory=list)
    symptom_entities: List[str] = field(default_factory=list)
    medicine_entities: List[str] = field(default_factory=list)
    body_part_entities: List[str] = field(default_factory=list)
    examination_entities: List[str] = field(default_factory=list)
    treatment_entities: List[str] = field(default_factory=list)
    lab_value_entities: List[str] = field(default_factory=list)
    clinical_dept_entities: List[str] = field(default_factory=list)
    
    entity_count: int = 0
    entity_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    question_length: int = 0
    option_count: int = 0
    
    has_numeric_data: bool = False
    has_lab_values: bool = False
    has_age_info: bool = False
    has_gender_info: bool = False
    
    age_value: Optional[int] = None
    gender: Optional[str] = None
    
    key_phrases: List[str] = field(default_factory=list)
    
    answer_type: str = ""
    reasoning_complexity: str = ""
    
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "specialty": self.specialty,
            "difficulty_level": self.difficulty_level,
            "disease_entities": self.disease_entities,
            "symptom_entities": self.symptom_entities,
            "medicine_entities": self.medicine_entities,
            "body_part_entities": self.body_part_entities,
            "examination_entities": self.examination_entities,
            "treatment_entities": self.treatment_entities,
            "lab_value_entities": self.lab_value_entities,
            "clinical_dept_entities": self.clinical_dept_entities,
            "entity_count": self.entity_count,
            "entity_type_distribution": self.entity_type_distribution,
            "question_length": self.question_length,
            "option_count": self.option_count,
            "has_numeric_data": self.has_numeric_data,
            "has_lab_values": self.has_lab_values,
            "has_age_info": self.has_age_info,
            "has_gender_info": self.has_gender_info,
            "age_value": self.age_value,
            "gender": self.gender,
            "key_phrases": self.key_phrases,
            "answer_type": self.answer_type,
            "reasoning_complexity": self.reasoning_complexity,
            "processing_time_ms": self.processing_time_ms
        }


class MedicalFeatureExtractor:
    """
    医疗特征提取器
    
    从医疗问答文本中提取多维特征，包括：
    - 实体特征
    - 语义特征
    - 结构特征
    - 元数据特征
    """
    
    SPECIALTY_KEYWORDS = {
        "cardiovascular": [
            "心", "心脏", "心肌", "冠状动脉", "心律", "血压", "心绞痛",
            "心肌梗死", "心力衰竭", "房颤", "心内膜炎"
        ],
        "neurology": [
            "脑", "神经", "脊髓", "癫痫", "帕金森", "头痛", "眩晕",
            "脑梗死", "脑出血", "意识障碍"
        ],
        "hematology": [
            "血液", "贫血", "血小板", "白血病", "淋巴瘤", "出血",
            "骨髓", "凝血", "ITP"
        ],
        "endocrinology": [
            "甲状腺", "糖尿病", "血糖", "胰岛素", "激素", "内分泌",
            "甲亢", "甲减", "肾上腺"
        ],
        "gastroenterology": [
            "胃", "肠", "肝", "胆", "胰腺", "消化", "腹泻", "便秘",
            "肝硬化", "胃炎"
        ],
        "pediatrics": [
            "患儿", "儿童", "小儿", "新生儿", "婴幼儿", "麻疹", "风疹",
            "幼儿急疹", "猩红热"
        ],
        "obstetrics_gynecology": [
            "妊娠", "孕妇", "子宫", "卵巢", "输卵管", "胎", "产",
            "月经", "流产"
        ],
        "psychiatry": [
            "精神", "幻觉", "妄想", "抑郁", "焦虑", "精神分裂",
            "情感", "心理"
        ],
        "immunology": [
            "免疫", "自身免疫", "红斑狼疮", "类风湿", "抗体",
            "补体", "免疫球蛋白"
        ],
        "pathology": [
            "病理", "肿瘤", "癌", "恶性", "良性", "组织", "细胞"
        ]
    }
    
    DIFFICULTY_INDICATORS = {
        "easy": ["下列哪项", "以下哪项", "正确的是", "错误的是"],
        "medium": ["首先考虑", "最可能", "最合理", "应选择"],
        "hard": ["最合理的治疗", "最佳方案", "首选检查", "应首先"]
    }
    
    AGE_PATTERNS = [
        r'(\d+)\s*岁',
        r'年龄\s*(\d+)',
        r'(\d+)\s*岁\s*(男|女)',
        r'(男|女)\s*[，,]?\s*(\d+)\s*岁',
        r'患儿\s*(\d+)\s*岁'
    ]
    
    GENDER_PATTERNS = [
        (r'(男[性]?[，,]?\s*\d+岁?|男性患者)', "男"),
        (r'(女[性]?[，,]?\s*\d+岁?|女性患者)', "女"),
        (r'患儿', "儿童")
    ]
    
    KEY_PHRASE_PATTERNS = [
        r'主要致病菌',
        r'诊断.*?是',
        r'治疗.*?为',
        r'首选',
        r'禁忌',
        r'最可能',
        r'应首先',
        r'错误的是',
        r'正确的是'
    ]
    
    ANSWER_TYPE_PATTERNS = {
        "diagnosis": [r'诊断', r'最可能.*?诊断', r'首先考虑.*?诊断'],
        "treatment": [r'治疗', r'用药', r'方案'],
        "examination": [r'检查', r'首选.*?检查', r'最有价值.*?检查'],
        "mechanism": [r'机制', r'原理', r'原因'],
        "classification": [r'分类', r'类型', r'分型'],
        "complication": [r'并发症', r'合并症'],
        "prognosis": [r'预后', r'生存期']
    }
    
    def __init__(
        self,
        entity_recognizer: Optional[MedicalEntityRecognizer] = None,
        specialty: str = "cardiovascular",
        extract_metadata: bool = True
    ):
        """
        初始化特征提取器
        
        Args:
            entity_recognizer: 实体识别器实例
            specialty: 默认专科领域
            extract_metadata: 是否提取元数据
        """
        if entity_recognizer is None:
            self.entity_recognizer = MedicalEntityRecognizer(specialty=specialty)
        else:
            self.entity_recognizer = entity_recognizer
        
        self.specialty = specialty
        self.extract_metadata = extract_metadata
    
    def extract(
        self,
        text: str,
        question_id: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        meta_info: Optional[str] = None
    ) -> MedicalFeatures:
        """
        从文本中提取特征
        
        Args:
            text: 输入文本（问题）
            question_id: 问题ID
            options: 选项字典
            meta_info: 元信息
            
        Returns:
            MedicalFeatures: 提取的特征
        """
        start_time = time.time()
        
        features = MedicalFeatures(question_id=question_id)
        
        entity_result = self.entity_recognizer.recognize(text)
        self._populate_entity_features(features, entity_result)
        
        features.question_length = len(text)
        features.option_count = len(options) if options else 0
        
        if self.extract_metadata:
            self._extract_age_gender(features, text)
            self._detect_numeric_data(features, text)
        
        features.specialty = self._detect_specialty(text, entity_result)
        features.difficulty_level = self._detect_difficulty(text)
        
        features.key_phrases = self._extract_key_phrases(text)
        features.answer_type = self._detect_answer_type(text)
        features.reasoning_complexity = self._estimate_reasoning_complexity(
            text, entity_result, options
        )
        
        if meta_info:
            features.difficulty_level = self._parse_meta_info(
                features, meta_info
            )
        
        features.processing_time_ms = (time.time() - start_time) * 1000
        
        return features
    
    def _populate_entity_features(
        self,
        features: MedicalFeatures,
        entity_result: EntityRecognitionResult
    ) -> None:
        """填充实体特征"""
        entity_mapping = {
            "DISEASE": "disease_entities",
            "SYMPTOM": "symptom_entities",
            "MEDICINE": "medicine_entities",
            "BODY_PART": "body_part_entities",
            "EXAMINATION": "examination_entities",
            "TREATMENT": "treatment_entities",
            "LAB_VALUE": "lab_value_entities",
            "CLINICAL_DEPT": "clinical_dept_entities"
        }
        
        for entity in entity_result.entities:
            attr_name = entity_mapping.get(entity.entity_type)
            if attr_name:
                entity_list = getattr(features, attr_name)
                if entity.text not in entity_list:
                    entity_list.append(entity.text)
        
        features.entity_count = len(entity_result.entities)
        features.entity_type_distribution = self.entity_recognizer.get_entity_count_by_type(
            entity_result.entities
        )
        
        features.has_lab_values = len(features.lab_value_entities) > 0
    
    def _extract_age_gender(
        self,
        features: MedicalFeatures,
        text: str
    ) -> None:
        """提取年龄和性别信息"""
        for pattern in self.AGE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                features.has_age_info = True
                try:
                    features.age_value = int(match.group(1))
                except (IndexError, ValueError):
                    pass
                break
        
        for pattern, gender in self.GENDER_PATTERNS:
            if re.search(pattern, text):
                features.has_gender_info = True
                features.gender = gender
                break
    
    def _detect_numeric_data(
        self,
        features: MedicalFeatures,
        text: str
    ) -> None:
        """检测数值数据"""
        numeric_pattern = r'\d+\.?\d*\s*(mmol/L|g/L|mg/dL|U/L|pg/mL|ng/mL|%|mmHg|kPa|次/分|℃)'
        if re.search(numeric_pattern, text):
            features.has_numeric_data = True
    
    def _detect_specialty(
        self,
        text: str,
        entity_result: EntityRecognitionResult
    ) -> str:
        """检测专科领域"""
        specialty_scores = {}
        
        for specialty, keywords in self.SPECIALTY_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            specialty_scores[specialty] = score
        
        for entity in entity_result.entities:
            for specialty, entities in self.entity_recognizer.SPECIALTY_ENTITIES.items():
                for entity_type, entity_list in entities.items():
                    if entity.text in entity_list:
                        specialty_scores[specialty] = specialty_scores.get(specialty, 0) + 2
        
        if specialty_scores:
            max_score = max(specialty_scores.values())
            if max_score > 0:
                for specialty, score in specialty_scores.items():
                    if score == max_score:
                        return specialty
        
        return self.specialty
    
    def _detect_difficulty(self, text: str) -> str:
        """检测问题难度"""
        for difficulty, indicators in self.DIFFICULTY_INDICATORS.items():
            for indicator in indicators:
                if indicator in text:
                    return difficulty
        return "medium"
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """提取关键短语"""
        key_phrases = []
        for pattern in self.KEY_PHRASE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                key_phrases.append(match.group(0))
        return key_phrases
    
    def _detect_answer_type(self, text: str) -> str:
        """检测答案类型"""
        for answer_type, patterns in self.ANSWER_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return answer_type
        return "knowledge"
    
    def _estimate_reasoning_complexity(
        self,
        text: str,
        entity_result: EntityRecognitionResult,
        options: Optional[Dict[str, str]]
    ) -> str:
        """估计推理复杂度"""
        score = 0
        
        score += min(entity_result.entities.count / 10, 3)
        
        if options:
            option_text = " ".join(options.values())
            score += len(options) * 0.5
            
            unique_entities = set()
            for opt_text in options.values():
                opt_result = self.entity_recognizer.recognize(opt_text)
                for e in opt_result.entities:
                    unique_entities.add(e.text)
            score += len(unique_entities) * 0.2
        
        reasoning_keywords = ["因为", "由于", "导致", "引起", "机制", "原理", "首先", "其次"]
        for keyword in reasoning_keywords:
            if keyword in text:
                score += 0.5
        
        if score < 2:
            return "simple"
        elif score < 4:
            return "medium"
        else:
            return "complex"
    
    def _parse_meta_info(
        self,
        features: MedicalFeatures,
        meta_info: str
    ) -> str:
        """解析元信息"""
        if "第一部分" in meta_info or "历年真题" in meta_info:
            features.difficulty_level = "hard"
        elif "模拟试题" in meta_info:
            features.difficulty_level = "medium"
        
        for specialty in self.SPECIALTY_KEYWORDS.keys():
            if specialty in meta_info.lower():
                features.specialty = specialty
                break
        
        return features.difficulty_level
    
    def extract_from_jsonl(
        self,
        jsonl_path: str,
        show_progress: bool = False
    ) -> List[MedicalFeatures]:
        """
        从JSONL文件提取特征
        
        Args:
            jsonl_path: JSONL文件路径
            show_progress: 是否显示进度
            
        Returns:
            特征列表
        """
        features_list = []
        
        try:
            import jsonlines
            with jsonlines.open(jsonl_path) as reader:
                items = list(reader)
        except ImportError:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f]
        
        items_iter = items
        if show_progress:
            try:
                from tqdm import tqdm
                items_iter = tqdm(items, desc="Extracting features")
            except ImportError:
                pass
        
        for idx, item in enumerate(items_iter):
            features = self.extract(
                text=item.get("question", ""),
                question_id=str(idx),
                options=item.get("options"),
                meta_info=item.get("meta_info")
            )
            features_list.append(features)
        
        return features_list
    
    def get_feature_statistics(
        self,
        features_list: List[MedicalFeatures]
    ) -> Dict[str, Any]:
        """
        获取特征统计信息
        
        Args:
            features_list: 特征列表
            
        Returns:
            统计信息字典
        """
        if not features_list:
            return {}
        
        stats = {
            "total_samples": len(features_list),
            "avg_entity_count": sum(f.entity_count for f in features_list) / len(features_list),
            "avg_processing_time_ms": sum(f.processing_time_ms for f in features_list) / len(features_list),
            "specialty_distribution": Counter(f.specialty for f in features_list),
            "difficulty_distribution": Counter(f.difficulty_level for f in features_list),
            "answer_type_distribution": Counter(f.answer_type for f in features_list),
            "reasoning_complexity_distribution": Counter(f.reasoning_complexity for f in features_list),
            "samples_with_age": sum(1 for f in features_list if f.has_age_info),
            "samples_with_gender": sum(1 for f in features_list if f.has_gender_info),
            "samples_with_lab_values": sum(1 for f in features_list if f.has_lab_values)
        }
        
        entity_type_counts = Counter()
        for f in features_list:
            for entity_type, count in f.entity_type_distribution.items():
                entity_type_counts[entity_type] += count
        stats["entity_type_distribution"] = dict(entity_type_counts)
        
        return stats
    
    def export_features(
        self,
        features_list: List[MedicalFeatures],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        导出特征到文件
        
        Args:
            features_list: 特征列表
            output_path: 输出路径
            format: 输出格式 (json, jsonl, csv)
        """
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([f.to_dict() for f in features_list], f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for features in features_list:
                    f.write(json.dumps(features.to_dict(), ensure_ascii=False) + '\n')
        elif format == "csv":
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                header = list(MedicalFeatures.__dataclass_fields__.keys())
                writer.writerow(header)
                for features in features_list:
                    row = [getattr(features, field) for field in header]
                    writer.writerow(row)
