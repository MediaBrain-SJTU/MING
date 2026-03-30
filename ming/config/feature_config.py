"""
特征工程配置类

提供结构化的特征工程配置，支持YAML序列化
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """特征工程配置类"""

    # 基本设置
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4

    # 实体识别配置
    entity_recognition: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "max_entity_length": 50,
            "confidence_threshold": 0.5,
            "merge_overlapping": True,
        }
    )

    # 实体类型配置
    entity_types: List[str] = field(
        default_factory=lambda: [
            "疾病",
            "症状",
            "药物",
            "检查",
            "治疗",
            "身体部位",
            "医疗器械",
            "细菌",
            "病毒",
            "专科",
        ]
    )

    # 特征提取配置
    feature_extraction: Dict[str, Any] = field(
        default_factory=lambda: {
            "statistical": True,
            "entity": True,
            "semantic": True,
            "specialty": True,
            "embedding_dim": 768,
        }
    )

    # 特征选择配置
    feature_selection: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "method": "mutual_info",
            "n_features": 100,
            "threshold": 0.01,
        }
    )

    # 专科特征权重
    specialty_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "cardiovascular": 1.0,
            "neurology": 1.0,
            "respiratory": 1.0,
            "gastroenterology": 1.0,
            "endocrinology": 1.0,
        }
    )

    # 性能约束
    performance: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_extraction_time_ms": 50,
            "min_coverage": 0.95,
            "target_f1": 0.92,
        }
    )

    # 输出配置
    output: Dict[str, Any] = field(
        default_factory=lambda: {
            "save_features": True,
            "save_entities": True,
            "output_dir": "./feature_output",
            "format": "json",
        }
    )

    @classmethod
    def from_yaml(cls, config_path: str) -> "FeatureConfig":
        """从YAML文件加载配置"""
        from ming.config.config_loader import load_config

        config_dict = load_config(config_path)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FeatureConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "entity_recognition": self.entity_recognition,
            "entity_types": self.entity_types,
            "feature_extraction": self.feature_extraction,
            "feature_selection": self.feature_selection,
            "specialty_weights": self.specialty_weights,
            "performance": self.performance,
            "output": self.output,
        }

    def save_yaml(self, config_path: str) -> None:
        """保存为YAML文件"""
        from ming.config.config_loader import save_config

        save_config(self.to_dict(), config_path)

    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []

        if self.batch_size <= 0:
            errors.append(f"批量大小应大于0，当前值: {self.batch_size}")

        if self.num_workers < 0:
            errors.append(f"工作线程数应非负，当前值: {self.num_workers}")

        if len(self.entity_types) == 0:
            errors.append("实体类型列表不能为空")

        # 验证实体识别配置
        er_config = self.entity_recognition
        if er_config.get("confidence_threshold", 0) < 0 or er_config.get(
            "confidence_threshold", 1
        ) > 1:
            errors.append(
                f"置信度阈值应在[0, 1]范围内，当前值: {er_config.get('confidence_threshold')}"
            )

        # 验证性能约束
        perf_config = self.performance
        if perf_config.get("max_extraction_time_ms", 0) <= 0:
            errors.append(
                f"最大提取时间应大于0ms，当前值: {perf_config.get('max_extraction_time_ms')}"
            )

        if perf_config.get("min_coverage", 0) < 0 or perf_config.get("min_coverage", 0) > 1:
            errors.append(
                f"最小覆盖率应在[0, 1]范围内，当前值: {perf_config.get('min_coverage')}"
            )

        return errors

    def is_valid(self) -> bool:
        """检查配置是否有效"""
        return len(self.validate()) == 0
