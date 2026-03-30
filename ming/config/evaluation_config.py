"""
评估配置类

提供结构化的评估配置，支持YAML序列化
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class EvaluationConfig:
    """评估配置类"""

    # 基本配置
    output_dir: str = "./evaluation_results"
    eval_batch_size: int = 8
    max_length: int = 2048
    max_new_tokens: int = 512
    num_beams: int = 1
    temperature: float = 0.0
    top_p: float = 1.0

    # 评估选项
    do_sample: bool = False
    use_cache: bool = True
    compute_entity_metrics: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True

    # 性能监控
    measure_inference_time: bool = True
    measure_memory_usage: bool = True

    # 结果保存
    save_outputs: bool = True
    save_references: bool = True
    save_metrics: bool = True

    # 复现性
    seed: int = 42
    deterministic: bool = True

    # 专科评估配置
    specialty_eval: bool = True
    target_specialties: List[str] = field(
        default_factory=lambda: [
            "cardiovascular",
            "neurology",
            "respiratory",
            "gastroenterology",
            "endocrinology",
        ]
    )
    benchmark_name: str = "MING-7B 专科医疗基准测试"
    target_improvement: float = 0.15  # 15%目标提升

    # 数据集配置
    datasets: Dict[str, str] = field(default_factory=dict)

    # 报告配置
    generate_report: bool = True
    report_prefix: str = "ming_evaluation"

    @classmethod
    def from_yaml(cls, config_path: str) -> "EvaluationConfig":
        """从YAML文件加载配置"""
        from ming.config.config_loader import load_config

        config_dict = load_config(config_path)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvaluationConfig":
        """从字典创建配置"""
        # 处理嵌套配置
        if "specialty" in config_dict:
            specialty_config = config_dict.pop("specialty")
            for key, value in specialty_config.items():
                config_dict[f"specialty_{key}"] = value

        # 处理报告配置
        if "report" in config_dict:
            report_config = config_dict.pop("report")
            if "generate" in report_config:
                config_dict["generate_report"] = report_config["generate"]
            if "prefix" in report_config:
                config_dict["report_prefix"] = report_config["prefix"]

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "output_dir": self.output_dir,
            "eval_batch_size": self.eval_batch_size,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
            "compute_entity_metrics": self.compute_entity_metrics,
            "compute_bleu": self.compute_bleu,
            "compute_rouge": self.compute_rouge,
            "measure_inference_time": self.measure_inference_time,
            "measure_memory_usage": self.measure_memory_usage,
            "save_outputs": self.save_outputs,
            "save_references": self.save_references,
            "save_metrics": self.save_metrics,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "specialty": {
                "eval": self.specialty_eval,
                "target_specialties": self.target_specialties,
                "benchmark_name": self.benchmark_name,
                "target_improvement": self.target_improvement,
            },
            "datasets": self.datasets,
            "report": {
                "generate": self.generate_report,
                "prefix": self.report_prefix,
            },
        }
        return result

    def save_yaml(self, config_path: str) -> None:
        """保存为YAML文件"""
        from ming.config.config_loader import save_config

        save_config(self.to_dict(), config_path)

    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []

        if self.eval_batch_size <= 0:
            errors.append(f"评估批量大小应大于0，当前值: {self.eval_batch_size}")

        if self.max_new_tokens <= 0 or self.max_new_tokens > 4096:
            errors.append(f"最大生成token数应在(0, 4096]范围内，当前值: {self.max_new_tokens}")

        if self.temperature < 0 or self.temperature > 2:
            errors.append(f"温度应在[0, 2]范围内，当前值: {self.temperature}")

        if self.top_p < 0 or self.top_p > 1:
            errors.append(f"top_p应在[0, 1]范围内，当前值: {self.top_p}")

        if self.target_improvement <= 0 or self.target_improvement > 1:
            errors.append(f"目标提升比例应在(0, 1]范围内，当前值: {self.target_improvement}")

        return errors

    def is_valid(self) -> bool:
        """检查配置是否有效"""
        return len(self.validate()) == 0
