"""
特征选择器模块

本模块提供特征选择和优化功能，包括：
1. 特征重要性评估
2. 冗余特征移除
3. 专科特征加权
4. 特征子集优化
"""

import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    VarianceThreshold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

from ming.feature_engineering.feature_extractor import FeatureSet

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """特征重要性数据类"""
    feature_name: str
    importance_score: float
    feature_type: str
    specialty_weight: float = 1.0
    selected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "feature_name": self.feature_name,
            "importance_score": float(self.importance_score),
            "feature_type": self.feature_type,
            "specialty_weight": float(self.specialty_weight),
            "selected": self.selected
        }


class FeatureSelector:
    """
    特征选择器

    提供多种特征选择策略，针对医疗专科任务进行优化：
    1. 基于方差的特征过滤
    2. 基于统计检验的特征选择
    3. 基于模型的特征重要性排序
    4. 专科领域知识引导的特征加权

    Attributes:
        variance_threshold: 方差阈值
        specialty_weights: 专科特征权重字典
        selected_features: 选中的特征名称列表
        scaler: 特征标准化器
    """

    # 预定义的专科特征重要性权重
    SPECIALTY_FEATURE_WEIGHTS: Dict[str, Dict[str, float]] = {
        "cardiovascular": {
            "entity_counts_疾病": 2.0,
            "entity_counts_症状": 1.8,
            "entity_counts_药物": 1.5,
            "entity_counts_检查": 1.5,
            "cardiovascular_score": 2.5,
            "digit_ratio": 1.2,
            "avg_keyword_weight": 1.3
        },
        "neurology": {
            "entity_counts_疾病": 2.0,
            "entity_counts_症状": 2.0,
            "entity_counts_检查": 1.5,
            "neurology_score": 2.5,
            "type_token_ratio": 1.2,
            "entity_density": 1.3
        },
        "respiratory": {
            "entity_counts_疾病": 2.0,
            "entity_counts_症状": 1.8,
            "entity_counts_检查": 1.5,
            "respiratory_score": 2.5
        },
        "gastroenterology": {
            "entity_counts_疾病": 2.0,
            "entity_counts_症状": 1.8,
            "entity_counts_检查": 1.5,
            "gastroenterology_score": 2.5
        },
        "endocrinology": {
            "entity_counts_疾病": 2.0,
            "entity_counts_药物": 1.8,
            "entity_counts_检查": 1.5,
            "endocrinology_score": 2.5,
            "digit_ratio": 1.3
        }
    }

    def __init__(
        self,
        variance_threshold: float = 0.01,
        specialty: str = "general",
        custom_weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化特征选择器

        Args:
            variance_threshold: 方差阈值，低于该阈值的特征将被移除
            specialty: 专科类型，用于加载预定义权重
            custom_weights: 自定义特征权重
        """
        self.variance_threshold = variance_threshold
        self.specialty = specialty
        self.selected_features: List[str] = []
        self.feature_importances_: List[FeatureImportance] = []
        self.scaler = StandardScaler()

        # 初始化方差阈值过滤器
        self.variance_filter = VarianceThreshold(threshold=variance_threshold)

        # 设置专科特征权重
        self.specialty_weights = self._initialize_weights(custom_weights)

        # 特征选择状态
        self._is_fitted = False

    def _initialize_weights(
        self,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        初始化特征权重

        Args:
            custom_weights: 自定义权重字典

        Returns:
            Dict[str, float]: 合并后的权重字典
        """
        base_weights: Dict[str, float] = defaultdict(lambda: 1.0)

        # 加载预定义的专科权重
        if self.specialty in self.SPECIALTY_FEATURE_WEIGHTS:
            base_weights.update(self.SPECIALTY_FEATURE_WEIGHTS[self.specialty])

        # 合并自定义权重
        if custom_weights:
            base_weights.update(custom_weights)

        return dict(base_weights)

    def _flatten_features(
        self,
        features_list: List[FeatureSet]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        将FeatureSet对象列表转换为特征矩阵

        Args:
            features_list: 特征集合列表

        Returns:
            Tuple[np.ndarray, List[str]]: 特征矩阵和特征名称列表
        """
        all_feature_values: List[List[float]] = []
        all_feature_names: List[str] = []

        if not features_list:
            return np.array([]), []

        # 使用第一个样本确定特征名称
        sample_features = features_list[0]

        # 1. 统计特征
        for name, value in sample_features.statistical_features.items():
            if isinstance(value, (int, float)):
                all_feature_names.append(f"stat_{name}")

        # 2. 实体特征（计数）
        entity_counts = sample_features.entity_features.get("entity_counts", {})
        for entity_type in ["疾病", "症状", "药物", "检查", "治疗", "身体部位", "专科"]:
            all_feature_names.append(f"entity_{entity_type}_count")

        # 3. 实体密度和类型数
        all_feature_names.append("entity_density")
        all_feature_names.append("entity_type_count")

        # 4. 语义特征
        for name, value in sample_features.semantic_features.items():
            if isinstance(value, (int, float)):
                all_feature_names.append(f"semantic_{name}")

        # 5. 专科特征
        for name, value in sample_features.specialty_features.items():
            if isinstance(value, (int, float)):
                all_feature_names.append(f"specialty_{name}")

        # 构建特征矩阵
        for features in features_list:
            feature_row: List[float] = []

            # 统计特征
            for name in sample_features.statistical_features.keys():
                value = features.statistical_features.get(name, 0.0)
                feature_row.append(float(value) if isinstance(value, (int, float)) else 0.0)

            # 实体计数
            entity_counts = features.entity_features.get("entity_counts", {})
            for entity_type in ["疾病", "症状", "药物", "检查", "治疗", "身体部位", "专科"]:
                count = entity_counts.get(entity_type, 0)
                feature_row.append(float(count))

            # 实体密度和类型数
            feature_row.append(float(features.entity_features.get("entity_density", 0.0)))
            feature_row.append(float(features.entity_features.get("entity_type_count", 0)))

            # 语义特征
            for name in sample_features.semantic_features.keys():
                value = features.semantic_features.get(name, 0.0)
                feature_row.append(float(value) if isinstance(value, (int, float)) else 0.0)

            # 专科特征
            for name in sample_features.specialty_features.keys():
                value = features.specialty_features.get(name, 0.0)
                feature_row.append(float(value) if isinstance(value, (int, float)) else 0.0)

            all_feature_values.append(feature_row)

        return np.array(all_feature_values, dtype=np.float32), all_feature_names

    def _apply_specialty_weights(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        应用专科特征权重

        Args:
            X: 特征矩阵
            feature_names: 特征名称列表

        Returns:
            np.ndarray: 加权后的特征矩阵
        """
        X_weighted = X.copy()

        for i, feature_name in enumerate(feature_names):
            weight = self.specialty_weights.get(feature_name, 1.0)
            if weight != 1.0:
                X_weighted[:, i] *= weight

        return X_weighted

    def fit(
        self,
        features_list: List[FeatureSet],
        labels: Optional[np.ndarray] = None,
        n_features_to_select: int = 50
    ) -> Tuple[List[FeatureImportance], Dict[str, Any]]:
        """
        拟合特征选择器

        Args:
            features_list: 特征集合列表
            labels: 标签数组（用于监督式特征选择）
            n_features_to_select: 要选择的特征数量

        Returns:
            Tuple[List[FeatureImportance], Dict[str, Any]]: 特征重要性列表和统计信息
        """
        logger.info("开始特征选择拟合过程...")

        # 1. 构建特征矩阵
        X, feature_names = self._flatten_features(features_list)
        logger.info(f"构建特征矩阵完成: {X.shape}")

        if X.shape[1] == 0:
            logger.warning("没有有效特征可用于选择")
            return [], {"selected_count": 0, "total_count": 0}

        # 2. 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 3. 应用专科特征权重
        X_weighted = self._apply_specialty_weights(X_scaled, feature_names)

        # 4. 基于方差过滤
        X_variance_filtered = self.variance_filter.fit_transform(X_weighted)
        variance_mask = self.variance_filter.get_support()
        variance_keep_names = [name for name, keep in zip(feature_names, variance_mask) if keep]
        logger.info(f"方差过滤后保留特征数: {len(variance_keep_names)}")

        # 5. 计算特征重要性
        importances = self._calculate_feature_importance(
            X_weighted,
            feature_names,
            labels
        )

        # 6. 选择Top-N特征
        # 按重要性排序
        sorted_importances = sorted(
            importances,
            key=lambda x: x.importance_score * x.specialty_weight,
            reverse=True
        )

        # 选择Top-N特征
        n_select = min(n_features_to_select, len(sorted_importances))
        selected = sorted_importances[:n_select]
        self.selected_features = [f.feature_name for f in selected]

        # 更新selected状态
        for f in sorted_importances:
            f.selected = f.feature_name in self.selected_features

        self.feature_importances_ = sorted_importances
        self._is_fitted = True

        stats = {
            "total_count": len(feature_names),
            "variance_filtered_count": len(variance_keep_names),
            "selected_count": n_select,
            "variance_threshold": self.variance_threshold,
            "specialty": self.specialty
        }

        logger.info(f"特征选择完成: 选中{n_select}个特征")

        return sorted_importances, stats

    def _calculate_feature_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        labels: Optional[np.ndarray] = None
    ) -> List[FeatureImportance]:
        """
        计算特征重要性

        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
            labels: 标签数组

        Returns:
            List[FeatureImportance]: 特征重要性列表
        """
        n_features = X.shape[1]
        importance_scores = np.zeros(n_features)

        # 1. 无监督：基于方差和特征值范围
        variance_scores = np.var(X, axis=0)
        variance_scores = variance_scores / (np.max(variance_scores) + 1e-8)  # 归一化
        importance_scores += variance_scores * 0.3

        # 2. 有监督：基于标签计算
        if labels is not None and len(np.unique(labels)) >= 2:
            try:
                # F值分数
                f_scores, _ = f_classif(X, labels)
                f_scores = f_scores / (np.max(f_scores) + 1e-8)
                importance_scores += f_scores * 0.4

                # 互信息
                mi_scores = mutual_info_classif(X, labels, random_state=42)
                mi_scores = mi_scores / (np.max(mi_scores) + 1e-8)
                importance_scores += mi_scores * 0.3
            except Exception as e:
                logger.warning(f"计算有监督特征重要性时出错: {e}")
                # 退化为随机森林特征重要性
                try:
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf.fit(X, labels)
                    rf_scores = rf.feature_importances_
                    importance_scores += rf_scores * 0.7
                except Exception as e2:
                    logger.warning(f"使用随机森林计算特征重要性也失败: {e2}")

        # 3. 归一化
        importance_scores = importance_scores / (np.max(importance_scores) + 1e-8)

        # 4. 创建特征重要性对象
        feature_importances: List[FeatureImportance] = []
        for i, (name, score) in enumerate(zip(feature_names, importance_scores)):
            # 确定特征类型
            feature_type = "unknown"
            if name.startswith("stat_"):
                feature_type = "statistical"
            elif name.startswith("entity_"):
                feature_type = "entity"
            elif name.startswith("semantic_"):
                feature_type = "semantic"
            elif name.startswith("specialty_"):
                feature_type = "specialty"

            # 获取专科权重
            specialty_weight = self.specialty_weights.get(name, 1.0)

            feature_importances.append(FeatureImportance(
                feature_name=name,
                importance_score=float(score),
                feature_type=feature_type,
                specialty_weight=specialty_weight
            ))

        return feature_importances

    def transform(
        self,
        features_list: List[FeatureSet]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        应用特征选择

        Args:
            features_list: 特征集合列表

        Returns:
            Tuple[np.ndarray, List[str]]: 选择后的特征矩阵和特征名称列表
        """
        if not self._is_fitted:
            raise RuntimeError("特征选择器尚未拟合，请先调用fit()方法")

        # 构建特征矩阵
        X, feature_names = self._flatten_features(features_list)

        if X.shape[1] == 0:
            return np.array([]), []

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 应用专科权重
        X_weighted = self._apply_specialty_weights(X_scaled, feature_names)

        # 获取选中特征的索引
        feature_indices = [
            feature_names.index(name)
            for name in self.selected_features
            if name in feature_names
        ]

        # 选择特征
        X_selected = X_weighted[:, feature_indices]

        return X_selected, self.selected_features

    def fit_transform(
        self,
        features_list: List[FeatureSet],
        labels: Optional[np.ndarray] = None,
        n_features_to_select: int = 50
    ) -> Tuple[np.ndarray, List[str], List[FeatureImportance], Dict[str, Any]]:
        """
        拟合并应用特征选择

        Args:
            features_list: 特征集合列表
            labels: 标签数组
            n_features_to_select: 要选择的特征数量

        Returns:
            选择后的特征矩阵、特征名称列表、特征重要性列表和统计信息
        """
        importances, stats = self.fit(features_list, labels, n_features_to_select)
        X_selected, selected_names = self.transform(features_list)
        return X_selected, selected_names, importances, stats

    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """获取特征重要性摘要"""
        if not self.feature_importances_:
            return {"error": "特征选择器尚未拟合"}

        # 按特征类型分组
        type_groups: Dict[str, List[FeatureImportance]] = defaultdict(list)
        for fi in self.feature_importances_:
            type_groups[fi.feature_type].append(fi)

        summary: Dict[str, Any] = {
            "total_features": len(self.feature_importances_),
            "selected_features": len(self.selected_features),
            "specialty": self.specialty,
            "by_type": {},
            "top_features": []
        }

        # 各类型统计
        for feat_type, items in type_groups.items():
            selected_count = sum(1 for item in items if item.selected)
            avg_importance = sum(item.importance_score for item in items) / len(items)
            summary["by_type"][feat_type] = {
                "count": len(items),
                "selected_count": selected_count,
                "avg_importance": float(avg_importance),
                "avg_weight": float(sum(item.specialty_weight for item in items) / len(items))
            }

        # 前10个重要特征
        top_selected = [
            fi for fi in self.feature_importances_ if fi.selected
        ][:10]
        summary["top_features"] = [fi.to_dict() for fi in top_selected]

        return summary

    def export_weights(self, filepath: str) -> None:
        """导出特征权重到文件"""
        import json

        weight_data = {
            "specialty": self.specialty,
            "variance_threshold": self.variance_threshold,
            "selected_features": self.selected_features,
            "feature_importances": [fi.to_dict() for fi in self.feature_importances_],
            "specialty_weights": self.specialty_weights
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weight_data, f, ensure_ascii=False, indent=2)

        logger.info(f"特征权重已导出到: {filepath}")


def main():
    """测试函数"""
    from ming.feature_engineering.feature_extractor import FeatureExtractor

    # 创建特征提取器
    extractor = FeatureExtractor()

    # 测试文本
    test_texts = [
        "患者因高血压病史10年，近日出现头痛、头晕，血压180/110mmHg，心电图示ST段压低，诊断为高血压危象",
        "神经内科会诊：患者左侧肢体肌力3级，肌张力增高，巴氏征阳性，头颅CT示右侧基底节区脑梗死",
        "心血管内科：患者有冠心病史，近日胸痛发作，含服硝酸甘油可缓解，冠状动脉造影示左前降支狭窄75%",
        "患者有糖尿病史5年，血糖控制不佳，空腹血糖12.3mmol/L，HbA1c 8.5%，伴有四肢麻木",
        "患者因咳嗽、咳痰1周入院，胸部CT示双肺纹理增多，右下肺可见斑片状阴影，诊断为社区获得性肺炎"
    ]

    # 提取特征
    all_features, _ = extractor.batch_extract(test_texts)

    # 创建特征选择器（心血管专科）
    selector = FeatureSelector(
        variance_threshold=0.01,
        specialty="cardiovascular"
    )

    # 模拟标签（0=心血管，1=神经，2=内分泌，3=呼吸）
    labels = np.array([0, 1, 0, 2, 3])

    # 拟合特征选择器
    importances, stats = selector.fit(all_features, labels, n_features_to_select=15)

    print("=== 特征选择统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== 前10个重要特征 ===")
    for fi in importances[:10]:
        print(f"{fi.feature_name}: 得分={fi.importance_score:.3f}, 权重={fi.specialty_weight:.1f}, 类型={fi.feature_type}")

    # 获取摘要
    summary = selector.get_feature_importance_summary()
    print("\n=== 特征类型统计 ===")
    for feat_type, type_stats in summary["by_type"].items():
        print(f"{feat_type}: 总数={type_stats['count']}, 选中={type_stats['selected_count']}, 平均重要性={type_stats['avg_importance']:.3f}")

    # 应用特征选择
    X_selected, selected_names = selector.transform(all_features)
    print(f"\n选择后的特征矩阵形状: {X_selected.shape}")
    print(f"选中的特征名称: {selected_names[:5]}...")

    # 导出权重
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    selector.export_weights(temp_path)
    print(f"\n特征权重已导出到临时文件: {temp_path}")
    os.unlink(temp_path)


if __name__ == "__main__":
    main()
