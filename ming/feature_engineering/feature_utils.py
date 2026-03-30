"""
特征工程工具函数模块

本模块提供特征工程的辅助工具函数，包括：
1. 文本预处理
2. 特征验证
3. 特征增强
4. 批处理工具
"""

import re
import hashlib
import time
import logging
from typing import List, Dict, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from functools import wraps
import numpy as np
import jieba

from ming.feature_engineering.entity_recognizer import (
    MedicalEntityRecognizer,
    Entity,
    EntityType,
)
from ming.feature_engineering.feature_extractor import FeatureSet, FeatureExtractor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureProcessingError(Exception):
    """特征处理异常类"""
    pass


class ValidationError(FeatureProcessingError):
    """特征验证异常类"""
    pass


def timing_decorator(func: Callable) -> Callable:
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        logger.debug(f"{func.__name__} 执行时间: {elapsed_time:.2f}ms")
        return result
    return wrapper


def preprocess_text(
    text: str,
    remove_punctuation: bool = True,
    normalize_numbers: bool = True,
    lowercase: bool = False,
    remove_whitespace: bool = True
) -> str:
    """
    文本预处理函数

    Args:
        text: 输入文本
        remove_punctuation: 是否移除标点符号
        normalize_numbers: 是否标准化数字表示
        lowercase: 是否转换为小写
        remove_whitespace: 是否移除多余空白

    Returns:
        str: 预处理后的文本
    """
    if not isinstance(text, str):
        raise ValueError("输入必须为字符串类型")

    result = text

    # 移除空白字符
    if remove_whitespace:
        result = re.sub(r'\s+', ' ', result).strip()

    # 移除标点符号
    if remove_punctuation:
        punctuation_pattern = r'[，。！？；：""''（）【】《》、,.;:\'\"()\[\]<>]'
        result = re.sub(punctuation_pattern, '', result)

    # 标准化数字表示
    if normalize_numbers:
        # 将中文数字转换为阿拉伯数字（简化版）
        cn_num_map = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10', '百': '100', '千': '1000', '万': '10000'
        }
        for cn_num, arab_num in cn_num_map.items():
            result = result.replace(cn_num, arab_num)

    # 转换为小写（中文不需要，但保留对英文内容的处理）
    if lowercase:
        result = result.lower()

    return result


def extract_medical_features(
    text: str,
    recognizer: Optional[MedicalEntityRecognizer] = None,
    extractor: Optional[FeatureExtractor] = None,
    history_texts: Optional[List[str]] = None
) -> Tuple[FeatureSet, Dict[str, Any]]:
    """
    一站式医疗特征提取

    Args:
        text: 输入文本
        recognizer: 实体识别器实例
        extractor: 特征提取器实例
        history_texts: 历史文本列表

    Returns:
        Tuple[FeatureSet, Dict[str, Any]]: 特征集合和元数据信息
    """
    if recognizer is None:
        recognizer = MedicalEntityRecognizer()

    if extractor is None:
        extractor = FeatureExtractor(entity_recognizer=recognizer)

    start_time = time.time()

    # 实体识别
    entities, entity_time = recognizer.recognize(text)

    # 特征提取
    features, feature_time = extractor.extract(text, history_texts, entities)

    total_time = (time.time() - start_time) * 1000

    metadata = {
        "entity_count": len(entities),
        "entity_time_ms": entity_time,
        "feature_time_ms": feature_time,
        "total_time_ms": total_time,
        "text_hash": hashlib.md5(text.encode()).hexdigest(),
        "meets_time_constraint": total_time <= 50
    }

    return features, metadata


def validate_feature_set(
    features: FeatureSet,
    raise_errors: bool = False
) -> Tuple[bool, List[str]]:
    """
    验证特征集合的完整性和有效性

    Args:
        features: 特征集合
        raise_errors: 是否抛出异常

    Returns:
        Tuple[bool, List[str]]: 验证是否通过和错误信息列表
    """
    errors: List[str] = []

    # 检查统计特征
    if not features.statistical_features:
        errors.append("统计特征为空")

    # 检查实体特征
    entity_counts = features.entity_features.get("entity_counts", {})
    if not entity_counts:
        errors.append("实体计数特征为空")

    # 检查专科特征覆盖率
    specialty_scores = [
        v for k, v in features.specialty_features.items()
        if k.endswith("_score") and isinstance(v, (int, float))
    ]
    if not specialty_scores:
        errors.append("专科特征得分为空")

    # 检查特征嵌入
    if features.feature_embedding is None:
        errors.append("特征嵌入为空")
    elif not isinstance(features.feature_embedding, np.ndarray):
        errors.append("特征嵌入类型错误，应为numpy.ndarray")
    elif len(features.feature_embedding.shape) != 1:
        errors.append(f"特征嵌入形状错误: {features.feature_embedding.shape}，应为1维数组")

    # 检查是否有数值异常
    all_numeric_features: List[float] = []
    all_numeric_features.extend(list(features.statistical_features.values()))
    all_numeric_features.extend([
        v for v in features.semantic_features.values()
        if isinstance(v, (int, float))
    ])
    all_numeric_features.extend([
        v for v in features.specialty_features.values()
        if isinstance(v, (int, float))
    ])

    for value in all_numeric_features:
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                errors.append(f"特征值包含异常值: {value}")

    is_valid = len(errors) == 0

    if not is_valid and raise_errors:
        raise ValidationError("特征验证失败: " + "; ".join(errors))

    return is_valid, errors


def create_feature_embedding(
    features: FeatureSet,
    output_dim: int = 128,
    normalize: bool = True
) -> np.ndarray:
    """
    创建特征嵌入向量

    Args:
        features: 特征集合
        output_dim: 输出维度
        normalize: 是否归一化

    Returns:
        np.ndarray: 特征嵌入向量
    """
    # 收集所有数值特征
    feature_values: List[float] = []

    # 统计特征
    feature_values.extend([
        float(v) for v in features.statistical_features.values()
        if isinstance(v, (int, float))
    ])

    # 实体特征
    entity_counts = features.entity_features.get("entity_counts", {})
    feature_values.extend([
        float(v) for v in entity_counts.values()
        if isinstance(v, (int, float))
    ])

    # 添加实体密度和类型数
    feature_values.append(float(features.entity_features.get("entity_density", 0)))
    feature_values.append(float(features.entity_features.get("entity_type_count", 0)))

    # 语义特征
    feature_values.extend([
        float(v) for v in features.semantic_features.values()
        if isinstance(v, (int, float))
    ])

    # 专科特征
    feature_values.extend([
        float(v) for v in features.specialty_features.values()
        if isinstance(v, (int, float))
    ])

    # 转换为numpy数组
    feature_array = np.array(feature_values, dtype=np.float32)

    # 归一化
    if normalize and len(feature_array) > 0:
        mean = np.mean(feature_array)
        std = np.std(feature_array) + 1e-8
        feature_array = (feature_array - mean) / std

    # 调整维度
    if len(feature_array) < output_dim:
        padding = np.zeros(output_dim - len(feature_array), dtype=np.float32)
        feature_array = np.concatenate([feature_array, padding])
    elif len(feature_array) > output_dim:
        feature_array = feature_array[:output_dim]

    return feature_array


def batch_feature_extraction(
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True,
    history_texts_list: Optional[List[Optional[List[str]]]] = None
) -> Tuple[List[FeatureSet], List[Dict[str, Any]]]:
    """
    批量特征提取

    Args:
        texts: 输入文本列表
        batch_size: 批处理大小
        show_progress: 是否显示进度
        history_texts_list: 历史文本列表的列表

    Returns:
        Tuple[List[FeatureSet], List[Dict[str, Any]]]: 特征集合列表和元数据列表
    """
    recognizer = MedicalEntityRecognizer()
    extractor = FeatureExtractor(entity_recognizer=recognizer)

    if history_texts_list is None:
        history_texts_list = [None] * len(texts)

    all_features: List[FeatureSet] = []
    all_metadata: List[Dict[str, Any]] = []

    n_batches = (len(texts) + batch_size - 1) // batch_size

    iterator = range(n_batches)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="特征提取")
        except ImportError:
            pass

    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))

        batch_texts = texts[start_idx:end_idx]
        batch_histories = history_texts_list[start_idx:end_idx]

        for text, history in zip(batch_texts, batch_histories):
            try:
                features, metadata = extract_medical_features(
                    text,
                    recognizer,
                    extractor,
                    history
                )
                all_features.append(features)
                all_metadata.append(metadata)
            except Exception as e:
                logger.error(f"处理文本时出错: {str(e)[:50]}...")
                # 创建空的特征集合
                all_features.append(FeatureSet())
                all_metadata.append({
                    "error": str(e),
                    "text_hash": hashlib.md5(text.encode()).hexdigest()
                })

    return all_features, all_metadata


@dataclass
class FeatureAugmentor:
    """
    特征增强器

    提供多种特征增强策略，用于提升模型鲁棒性：
    1. 同义词替换
    2. 实体替换
    3. 文本重组
    4. 噪声注入
    """

    synonym_dict: Dict[str, List[str]] = field(default_factory=dict)
    entity_types_for_augment: Set[EntityType] = field(
        default_factory=lambda: {EntityType.DISEASE, EntityType.SYMPTOM, EntityType.DRUG}
    )
    max_replacements: int = 3
    recognizer: Optional[MedicalEntityRecognizer] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.recognizer is None:
            self.recognizer = MedicalEntityRecognizer()

        # 初始化默认同义词词典
        self._init_default_synonyms()

    def _init_default_synonyms(self):
        """初始化默认同义词词典"""
        default_synonyms: Dict[str, List[str]] = {
            "高血压": ["血压高", "原发性高血压", "高血压病"],
            "糖尿病": ["消渴", "DM", "2型糖尿病"],
            "冠心病": ["冠状动脉性心脏病", "缺血性心脏病"],
            "头痛": ["头疼", "头痛不适"],
            "头晕": ["头昏", "眩晕", "头晕目眩"],
            "心电图": ["ECG", "心电"],
            "CT": ["计算机断层扫描", "CT检查"],
            "阿司匹林": ["ASA", "乙酰水杨酸"],
            "患者": ["病人", "就诊者", "受试者"],
            "诊断": ["确诊", "诊断为", "考虑为"],
            "治疗": ["医治", "诊疗", "诊治"]
        }

        for word, synonyms in default_synonyms.items():
            if word not in self.synonym_dict:
                self.synonym_dict[word] = synonyms

    def synonym_replacement(
        self,
        text: str,
        n_replacements: Optional[int] = None
    ) -> str:
        """
        同义词替换增强

        Args:
            text: 输入文本
            n_replacements: 最大替换次数

        Returns:
            str: 增强后的文本
        """
        n_replacements = n_replacements or self.max_replacements
        words = list(jieba.cut(text))
        replacements_made = 0

        for i, word in enumerate(words):
            if replacements_made >= n_replacements:
                break

            synonyms = self.synonym_dict.get(word, [])
            if synonyms and len(synonyms) > 0:
                import random
                synonym = random.choice(synonyms)
                words[i] = synonym
                replacements_made += 1

        return ''.join(words)

    def entity_replacement(
        self,
        text: str,
        n_replacements: Optional[int] = None
    ) -> str:
        """
        实体替换增强

        Args:
            text: 输入文本
            n_replacements: 最大替换次数

        Returns:
            str: 增强后的文本
        """
        n_replacements = n_replacements or self.max_replacements

        # 识别实体
        entities, _ = self.recognizer.recognize(text)
        entities = [
            e for e in entities
            if e.entity_type in self.entity_types_for_augment
        ]

        if not entities:
            return text

        # 按实体长度降序排序（优先替换长实体）
        entities.sort(key=lambda x: len(x.text), reverse=True)

        result = text
        replacements_made = 0

        # 收集同类型的实体列表用于替换
        same_type_entities: Dict[EntityType, List[str]] = {}
        for entity_type in self.entity_types_for_augment:
            entity_words = self.recognizer.entity_dicts.get(entity_type, set())
            same_type_entities[entity_type] = list(entity_words)

        for entity in entities[:n_replacements]:
            if replacements_made >= n_replacements:
                break

            candidates = same_type_entities.get(entity.entity_type, [])
            candidates = [c for c in candidates if c != entity.text]

            if candidates:
                import random
                replacement = random.choice(candidates)
                result = result.replace(entity.text, replacement, 1)
                replacements_made += 1

        return result

    def text_reordering(self, text: str) -> str:
        """
        文本重排序增强

        Args:
            text: 输入文本

        Returns:
            str: 重排序后的文本
        """
        # 按句子分割
        sentences = re.split(r'([。！？；.!?;])', text)

        # 合并分割的句子和标点
        merged_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                merged_sentences.append(sentences[i] + sentences[i + 1])
            else:
                merged_sentences.append(sentences[i])

        if len(merged_sentences) <= 2:
            return text

        import random
        # 保持第一句和最后一句不变，打乱中间
        first = merged_sentences[0]
        last = merged_sentences[-1]
        middle = merged_sentences[1:-1]
        random.shuffle(middle)

        reordered = [first] + middle + [last]
        return ''.join(reordered)

    def augment(
        self,
        text: str,
        methods: Optional[List[str]] = None,
        n_variations: int = 1
    ) -> List[str]:
        """
        生成增强文本

        Args:
            text: 输入文本
            methods: 增强方法列表，可选值: ['synonym', 'entity', 'reorder']
            n_variations: 生成的变体数量

        Returns:
            List[str]: 增强后的文本列表
        """
        methods = methods or ['synonym', 'entity']
        variations: List[str] = []

        for _ in range(n_variations):
            result = text

            if 'synonym' in methods:
                result = self.synonym_replacement(result)
            if 'entity' in methods:
                result = self.entity_replacement(result)
            if 'reorder' in methods:
                result = self.text_reordering(result)

            variations.append(result)

        return variations


def calculate_feature_statistics(
    features_list: List[FeatureSet]
) -> Dict[str, Dict[str, float]]:
    """
    计算特征统计信息

    Args:
        features_list: 特征集合列表

    Returns:
        Dict[str, Dict[str, float]]: 各特征类别的统计信息
    """
    statistics: Dict[str, Dict[str, float]] = {}

    if not features_list:
        return statistics

    # 收集所有特征值
    all_stat_values: Dict[str, List[float]] = defaultdict(list)
    all_entity_counts: Dict[str, List[int]] = defaultdict(list)
    all_specialty_scores: Dict[str, List[float]] = defaultdict(list)

    for features in features_list:
        # 统计特征
        for name, value in features.statistical_features.items():
            if isinstance(value, (int, float)):
                all_stat_values[name].append(float(value))

        # 实体计数
        entity_counts = features.entity_features.get("entity_counts", {})
        for entity_type, count in entity_counts.items():
            all_entity_counts[entity_type].append(count)

        # 专科得分
        for name, value in features.specialty_features.items():
            if isinstance(value, (int, float)) and name.endswith("_score"):
                all_specialty_scores[name].append(float(value))

    # 计算统计量
    def calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr))
        }

    statistics["statistical"] = {
        name: calc_stats(values)
        for name, values in all_stat_values.items()
    }

    statistics["entity_counts"] = {
        entity_type: calc_stats([float(v) for v in values])
        for entity_type, values in all_entity_counts.items()
    }

    statistics["specialty_scores"] = {
        specialty: calc_stats(values)
        for specialty, values in all_specialty_scores.items()
    }

    return statistics


def save_features_to_json(
    features_list: List[FeatureSet],
    output_path: str,
    include_embeddings: bool = True
) -> None:
    """
    保存特征到JSON文件

    Args:
        features_list: 特征集合列表
        output_path: 输出路径
        include_embeddings: 是否包含特征嵌入
    """
    import json

    output_data: List[Dict[str, Any]] = []

    for features in features_list:
        feature_dict = features.to_dict()
        if not include_embeddings:
            feature_dict.pop("feature_embedding", None)
        output_data.append(feature_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"特征已保存到: {output_path}")


def load_features_from_json(
    input_path: str
) -> List[FeatureSet]:
    """
    从JSON文件加载特征

    Args:
        input_path: 输入路径

    Returns:
        List[FeatureSet]: 特征集合列表
    """
    import json

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features_list: List[FeatureSet] = []

    for item in data:
        features = FeatureSet(
            statistical_features=item.get("statistical_features", {}),
            entity_features=item.get("entity_features", {}),
            semantic_features=item.get("semantic_features", {}),
            specialty_features=item.get("specialty_features", {}),
            context_features=item.get("context_features", {}),
            metadata=item.get("metadata", {})
        )

        if "feature_embedding" in item and item["feature_embedding"]:
            features.feature_embedding = np.array(item["feature_embedding"], dtype=np.float32)

        features_list.append(features)

    return features_list


def main():
    """测试函数"""
    # 测试文本预处理
    test_text = "患者，男，65岁，因“反复头痛、头晕10年，加重1周”入院。"
    processed = preprocess_text(test_text)
    print(f"原始文本: {test_text}")
    print(f"预处理后: {processed}")

    # 测试一站式特征提取
    features, metadata = extract_medical_features(test_text)
    print(f"\n特征提取元数据:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # 测试特征验证
    is_valid, errors = validate_feature_set(features, raise_errors=False)
    print(f"\n特征验证结果: {'通过' if is_valid else '失败'}")
    if errors:
        print(f"错误信息: {errors}")

    # 测试特征增强
    augmentor = FeatureAugmentor()
    augmented = augmentor.augment(test_text, n_variations=2)
    print(f"\n原始文本: {test_text}")
    for i, var in enumerate(augmented):
        print(f"增强变体{i+1}: {var}")

    # 测试批量处理
    test_texts = [
        "患者因高血压病史10年，近日出现头痛、头晕，血压180/110mmHg",
        "神经内科会诊：患者左侧肢体肌力3级，肌张力增高，巴氏征阳性",
        "心血管内科：患者有冠心病史，近日胸痛发作，含服硝酸甘油可缓解"
    ]

    all_features, all_metadata = batch_feature_extraction(test_texts, show_progress=False)
    print(f"\n批量处理结果:")
    for i, (feat, meta) in enumerate(zip(all_features, all_metadata)):
        entity_count = feat.entity_features.get("total_entities", 0)
        print(f"  文本{i+1}: 实体数={entity_count}, 耗时={meta.get('total_time_ms', 0):.2f}ms")

    # 计算统计信息
    stats = calculate_feature_statistics(all_features)
    print(f"\n特征统计信息:")
    if "statistical" in stats:
        for name, stat in stats["statistical"].items():
            if stat:
                print(f"  {name}: mean={stat.get('mean', 0):.2f}, std={stat.get('std', 0):.2f}")


if __name__ == "__main__":
    main()
