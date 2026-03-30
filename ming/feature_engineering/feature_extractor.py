"""
特征提取器模块

本模块提供从医疗文本中提取各种特征的功能，包括统计特征、语义特征、
实体特征和专科特征等。
"""

import re
import time
import hashlib
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ming.feature_engineering.entity_recognizer import (
    MedicalEntityRecognizer,
    Entity,
    EntityType,
)


@dataclass
class FeatureSet:
    """特征集合数据类"""

    statistical_features: Dict[str, float] = field(default_factory=dict)
    entity_features: Dict[str, Any] = field(default_factory=dict)
    semantic_features: Dict[str, float] = field(default_factory=dict)
    specialty_features: Dict[str, float] = field(default_factory=dict)
    context_features: Dict[str, Any] = field(default_factory=dict)
    feature_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "statistical_features": self.statistical_features,
            "entity_features": self.entity_features,
            "semantic_features": self.semantic_features,
            "specialty_features": self.specialty_features,
            "context_features": self.context_features,
            "metadata": self.metadata,
        }
        if self.feature_embedding is not None:
            result["feature_embedding"] = self.feature_embedding.tolist()
        return result

    def get_feature_vector(self) -> np.ndarray:
        """获取扁平化的特征向量"""
        all_features = []

        # 统计特征
        all_features.extend(list(self.statistical_features.values()))

        # 实体特征（转换为数值类型）
        entity_counts = self.entity_features.get("entity_counts", {})
        all_features.extend(list(entity_counts.values()))

        # 语义特征
        all_features.extend(list(self.semantic_features.values()))

        # 专科特征
        all_features.extend(list(self.specialty_features.values()))

        return np.array(all_features, dtype=np.float32)


class FeatureExtractor:
    """
    特征提取器

    从医疗文本中提取多维度特征，包括：
    1. 统计特征：文本长度、句子数、词汇丰富度等
    2. 实体特征：医疗实体计数、分布、密度等
    3. 语义特征：关键词、TF-IDF、主题特征等
    4. 专科特征：心血管、神经内科等专科相关特征
    5. 上下文特征：对话历史、上下文关联特征

    Attributes:
        entity_recognizer: 医疗实体识别器实例
        tfidf_vectorizer: TF-IDF向量化器
        scaler: 特征标准化器
        stop_words: 停用词集合
    """

    # 专科关键词词典
    CARDIOVASCULAR_KEYWORDS: Set[str] = {
        "心脏", "心肌", "冠脉", "冠状动脉", "血压", "心率", "心律",
        "心电图", "ST段", "T波", "早搏", "房颤", "心衰", "心绞痛",
        "心肌梗死", "支架", "搭桥", "他汀", "阿司匹林", "氯吡格雷",
        "高血压", "低血压", "血脂", "胆固醇", "甘油三酯", "动脉硬化"
    }

    NEUROLOGY_KEYWORDS: Set[str] = {
        "神经", "大脑", "脑", "头颅", "头痛", "头晕", "眩晕", "晕厥",
        "肌力", "肌张力", "震颤", "抽搐", "癫痫", "中风", "脑梗死",
        "脑出血", "偏瘫", "截瘫", "麻木", "感觉障碍", "意识障碍",
        "巴氏征", "脑电图", "肌电图", "头颅CT", "头颅MRI", "脑脊液"
    }

    RESPIRATORY_KEYWORDS: Set[str] = {
        "肺", "呼吸", "气管", "支气管", "咳嗽", "咳痰", "咯血", "气喘",
        "呼吸困难", "胸闷", "胸痛", "肺炎", "肺结核", "肺癌", "哮喘",
        "肺气肿", "肺心病", "血氧", "氧饱和度", "呼吸机", "胸部CT"
    }

    GASTROENTEROLOGY_KEYWORDS: Set[str] = {
        "胃", "肠", "肝", "胆", "脾", "胰", "食管", "腹痛", "腹胀",
        "腹泻", "便秘", "恶心", "呕吐", "呕血", "黑便", "黄疸",
        "胃炎", "溃疡", "肝硬化", "肝炎", "肠镜", "胃镜", "B超"
    }

    ENDOCRINOLOGY_KEYWORDS: Set[str] = {
        "血糖", "胰岛素", "甲状腺", "甲亢", "甲减", "糖尿病", "激素",
        "垂体", "肾上腺", "性腺", "钙", "磷", "电解质", "代谢综合征",
        "糖耐量", "糖化血红蛋白", "尿酸", "痛风", "骨质疏松"
    }

    def __init__(
        self,
        entity_recognizer: Optional[MedicalEntityRecognizer] = None,
        stop_words: Optional[Set[str]] = None,
        embedding_dim: int = 128
    ):
        """
        初始化特征提取器

        Args:
            entity_recognizer: 医疗实体识别器实例，None则创建默认实例
            stop_words: 停用词集合，None则使用默认停用词
            embedding_dim: 特征嵌入维度
        """
        self.entity_recognizer = entity_recognizer or MedicalEntityRecognizer()
        self.embedding_dim = embedding_dim

        # 停用词配置
        self.stop_words = stop_words or self._load_default_stopwords()

        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(self.stop_words) if self.stop_words else None,
            ngram_range=(1, 2)
        )

        # 标准化器
        self.scaler = StandardScaler()

        # 专科关键词映射
        self.specialty_keywords: Dict[str, Set[str]] = {
            "cardiovascular": self.CARDIOVASCULAR_KEYWORDS,
            "neurology": self.NEUROLOGY_KEYWORDS,
            "respiratory": self.RESPIRATORY_KEYWORDS,
            "gastroenterology": self.GASTROENTEROLOGY_KEYWORDS,
            "endocrinology": self.ENDOCRINOLOGY_KEYWORDS
        }

        # 专科名称映射
        self.specialty_name_map: Dict[str, str] = {
            "cardiovascular": "心血管内科",
            "neurology": "神经内科",
            "respiratory": "呼吸内科",
            "gastroenterology": "消化内科",
            "endocrinology": "内分泌科"
        }

    def _load_default_stopwords(self) -> Set[str]:
        """加载默认停用词"""
        return {
            "的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
            "着", "没有", "看", "好", "自己", "这", "那", "他", "她", "它",
            "们", "这个", "那个", "什么", "怎么", "为什么", "哪", "哪里", "谁",
            "多少", "几", "啊", "吧", "呢", "吗", "啦", "呀", "了", "着", "过",
            "患者", "检查", "治疗", "诊断", "病情", "症状", "情况", "建议",
            "考虑", "可能", "应该", "可以", "需要", "必须", "注意", "随访"
        }

    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        """
        提取统计特征

        Args:
            text: 输入文本

        Returns:
            Dict[str, float]: 统计特征字典
        """
        # 基本长度特征
        char_length = len(text)
        word_list = list(jieba.cut(text))
        word_length = len(word_list)

        # 句子分割
        sentences = re.split(r'[。！？；.!?;]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # 词汇丰富度
        unique_words = set(word_list)
        unique_word_count = len(unique_words)
        type_token_ratio = unique_word_count / word_length if word_length > 0 else 0

        # 停用词比例
        stop_word_count = sum(1 for word in word_list if word in self.stop_words)
        stop_word_ratio = stop_word_count / word_length if word_length > 0 else 0

        # 数字和符号特征
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / char_length if char_length > 0 else 0

        return {
            "char_length": float(char_length),
            "word_length": float(word_length),
            "sentence_count": float(sentence_count),
            "unique_word_count": float(unique_word_count),
            "type_token_ratio": type_token_ratio,
            "stop_word_ratio": stop_word_ratio,
            "digit_ratio": digit_ratio,
            "avg_sentence_length": float(word_length / sentence_count) if sentence_count > 0 else 0
        }

    def extract_entity_features(
        self,
        text: str,
        entities: Optional[List[Entity]] = None
    ) -> Dict[str, Any]:
        """
        提取实体特征

        Args:
            text: 输入文本
            entities: 预识别的实体列表，None则重新识别

        Returns:
            Dict[str, Any]: 实体特征字典
        """
        if entities is None:
            entities, _ = self.entity_recognizer.recognize(text)

        word_length = len(list(jieba.cut(text)))

        # 实体计数
        entity_counts: Dict[str, int] = defaultdict(int)
        for entity in entities:
            entity_counts[entity.entity_type.value] += 1

        # 实体密度（每100词的实体数）
        entity_density = (len(entities) / word_length * 100) if word_length > 0 else 0

        # 实体类型丰富度
        entity_type_count = len(entity_counts)

        # 专科实体匹配
        specialty_entity_counts: Dict[str, int] = defaultdict(int)
        for entity in entities:
            if entity.entity_type == EntityType.MEDICAL_SPECIALTY:
                specialty_entity_counts[entity.text] += 1

        # 实体共现
        entity_texts = [e.text for e in entities]
        entity_cooccurrence = self._analyze_entity_cooccurrence(entity_texts)

        return {
            "entity_counts": dict(entity_counts),
            "total_entities": len(entities),
            "entity_density": entity_density,
            "entity_type_count": entity_type_count,
            "specialty_entity_counts": dict(specialty_entity_counts),
            "entity_cooccurrence": entity_cooccurrence,
            "entity_list": [e.to_dict() for e in entities]
        }

    def _analyze_entity_cooccurrence(
        self,
        entity_texts: List[str]
    ) -> List[Tuple[str, str, int]]:
        """分析实体共现关系"""
        cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        for i in range(len(entity_texts)):
            for j in range(i + 1, len(entity_texts)):
                pair = tuple(sorted([entity_texts[i], entity_texts[j]]))
                cooccurrence[pair] += 1

        # 转换为列表并排序
        cooccurrence_list = [
            (pair[0], pair[1], count)
            for pair, count in cooccurrence.items()
        ]
        cooccurrence_list.sort(key=lambda x: x[2], reverse=True)
        return cooccurrence_list[:5]  # 返回前5个最频繁的共现

    def extract_semantic_features(self, text: str) -> Dict[str, float]:
        """
        提取语义特征

        Args:
            text: 输入文本

        Returns:
            Dict[str, float]: 语义特征字典
        """
        # TF-IDF关键词提取
        keywords = jieba.analyse.extract_tags(
            text,
            topK=10,
            withWeight=True,
            allowPOS=('n', 'vn', 'v')
        )

        # 关键词权重统计
        keyword_weights = [weight for _, weight in keywords]
        avg_keyword_weight = sum(keyword_weights) / len(keyword_weights) if keyword_weights else 0
        max_keyword_weight = max(keyword_weights) if keyword_weights else 0

        # 主题相关度（基于关键词类别）
        word_list = list(jieba.cut(text))

        return {
            "keyword_count": len(keywords),
            "avg_keyword_weight": avg_keyword_weight,
            "max_keyword_weight": max_keyword_weight,
            "top_keywords": [word for word, _ in keywords[:5]]
        }

    def extract_specialty_features(self, text: str) -> Dict[str, float]:
        """
        提取专科相关特征

        Args:
            text: 输入文本

        Returns:
            Dict[str, float]: 专科特征字典
        """
        word_list = set(jieba.cut(text))
        specialty_scores: Dict[str, float] = {}

        for specialty, keywords in self.specialty_keywords.items():
            # 计算匹配的关键词数量
            matched_words = word_list.intersection(keywords)
            match_count = len(matched_words)
            total_keywords = len(keywords)

            # 计算专科相关度得分
            specialty_score = match_count / total_keywords if total_keywords > 0 else 0
            specialty_scores[f"{specialty}_score"] = specialty_score
            specialty_scores[f"{specialty}_match_count"] = float(match_count)

        # 判断主要专科方向
        max_score = max(specialty_scores.values()) if specialty_scores else 0
        primary_specialty = max(
            self.specialty_keywords.keys(),
            key=lambda x: specialty_scores.get(f"{x}_score", 0)
        ) if max_score > 0 else "general"

        specialty_scores["primary_specialty"] = primary_specialty
        specialty_scores["max_specialty_score"] = max_score

        return specialty_scores

    def extract_context_features(
        self,
        current_text: str,
        history_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        提取上下文特征

        Args:
            current_text: 当前文本
            history_texts: 历史文本列表

        Returns:
            Dict[str, Any]: 上下文特征字典
        """
        context_features: Dict[str, Any] = {
            "has_history": history_texts is not None and len(history_texts) > 0,
            "history_length": len(history_texts) if history_texts else 0,
        }

        if history_texts:
            # 当前文本与历史文本的相似度
            current_words = set(jieba.cut(current_text))
            history_words = set()
            for hist_text in history_texts:
                history_words.update(jieba.cut(hist_text))

            # 词汇重叠度
            overlap = current_words.intersection(history_words)
            context_features["vocabulary_overlap"] = len(overlap) / len(current_words) if current_words else 0

            # 历史文本长度统计
            history_lengths = [len(text) for text in history_texts]
            context_features["avg_history_length"] = sum(history_lengths) / len(history_lengths)

        return context_features

    def create_feature_embedding(
        self,
        features: FeatureSet,
        output_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        创建特征嵌入

        Args:
            features: 特征集合
            output_dim: 输出维度，None则使用默认维度

        Returns:
            np.ndarray: 特征嵌入向量
        """
        output_dim = output_dim or self.embedding_dim

        # 收集所有特征值
        feature_values = []

        # 统计特征
        feature_values.extend(list(features.statistical_features.values()))

        # 实体特征
        entity_counts = features.entity_features.get("entity_counts", {})
        feature_values.extend(list(entity_counts.values()))

        # 语义特征
        semantic_values = [
            v for v in features.semantic_features.values()
            if isinstance(v, (int, float))
        ]
        feature_values.extend(semantic_values)

        # 专科特征
        specialty_values = [
            v for v in features.specialty_features.values()
            if isinstance(v, (int, float))
        ]
        feature_values.extend(specialty_values)

        # 转换为numpy数组
        feature_array = np.array(feature_values, dtype=np.float32)

        # 如果特征维度不足，用0填充；如果超过，截断
        if len(feature_array) < output_dim:
            padding = np.zeros(output_dim - len(feature_array), dtype=np.float32)
            feature_array = np.concatenate([feature_array, padding])
        elif len(feature_array) > output_dim:
            feature_array = feature_array[:output_dim]

        return feature_array

    def extract(
        self,
        text: str,
        history_texts: Optional[List[str]] = None,
        entities: Optional[List[Entity]] = None
    ) -> Tuple[FeatureSet, float]:
        """
        完整提取所有特征

        Args:
            text: 输入文本
            history_texts: 历史文本列表
            entities: 预识别的实体列表

        Returns:
            Tuple[FeatureSet, float]: 特征集合和处理时间（毫秒）
        """
        start_time = time.time()

        # 提取各维度特征
        statistical_features = self.extract_statistical_features(text)
        entity_features = self.extract_entity_features(text, entities)
        semantic_features = self.extract_semantic_features(text)
        specialty_features = self.extract_specialty_features(text)
        context_features = self.extract_context_features(text, history_texts)

        # 创建特征集
        feature_set = FeatureSet(
            statistical_features=statistical_features,
            entity_features=entity_features,
            semantic_features=semantic_features,
            specialty_features=specialty_features,
            context_features=context_features,
            metadata={
                "text_hash": hashlib.md5(text.encode()).hexdigest(),
                "extraction_timestamp": time.time()
            }
        )

        # 创建特征嵌入
        feature_set.feature_embedding = self.create_feature_embedding(feature_set)

        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒

        if processing_time > 50:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("特征提取耗时超过阈值: %.2fms", processing_time)

        return feature_set, processing_time

    def batch_extract(
        self,
        texts: List[str],
        history_texts_list: Optional[List[Optional[List[str]]]] = None
    ) -> Tuple[List[FeatureSet], Dict[str, float]]:
        """
        批量提取特征

        Args:
            texts: 输入文本列表
            history_texts_list: 历史文本列表的列表

        Returns:
            Tuple[List[FeatureSet], Dict[str, float]]: 特征集合列表和统计信息
        """
        if history_texts_list is None:
            history_texts_list = [None] * len(texts)

        all_features = []
        total_time = 0.0
        max_time = 0.0
        min_time = float('inf')

        for text, history_texts in zip(texts, history_texts_list):
            features, proc_time = self.extract(text, history_texts)
            all_features.append(features)
            total_time += proc_time
            max_time = max(max_time, proc_time)
            min_time = min(min_time, proc_time)

        stats = {
            "total_count": len(texts),
            "total_time_ms": total_time,
            "avg_time_ms": total_time / len(texts) if texts else 0,
            "max_time_ms": max_time,
            "min_time_ms": min_time,
            "meets_time_constraint": all(t <= 50 for t in [total_time / len(texts), max_time])
        }

        return all_features, stats


def main():
    """测试函数"""
    extractor = FeatureExtractor()

    # 测试文本
    test_texts = [
        "患者因高血压病史10年，近日出现头痛、头晕，血压180/110mmHg，心电图示ST段压低，诊断为高血压危象",
        "神经内科会诊：患者左侧肢体肌力3级，肌张力增高，巴氏征阳性，头颅CT示右侧基底节区脑梗死",
        "心血管内科：患者有冠心病史，近日胸痛发作，含服硝酸甘油可缓解，冠状动脉造影示左前降支狭窄75%"
    ]

    # 批量提取特征
    all_features, stats = extractor.batch_extract(test_texts)

    print("=== 特征提取结果 ===")
    for i, (text, features) in enumerate(zip(test_texts, all_features)):
        print(f"\n文本{i+1}: {text[:50]}...")
        print(f"  统计特征数: {len(features.statistical_features)}")
        print(f"  实体数量: {features.entity_features.get('total_entities', 0)}")
        print(f"  专科得分: {features.specialty_features.get('max_specialty_score', 0):.3f}")
        print(f"  主要专科: {features.specialty_features.get('primary_specialty', 'N/A')}")
        print(f"  特征嵌入维度: {features.feature_embedding.shape if features.feature_embedding is not None else 'None'}")

    print("\n=== 统计信息 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # 检查实体覆盖率
    coverage = extractor.entity_recognizer.get_entity_coverage()
    print("\n=== 实体覆盖率 ===")
    print(f"总体覆盖率: {coverage['overall_coverage']:.1f}%")


if __name__ == "__main__":
    main()
