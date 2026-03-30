"""
专科领域评估器模块

本模块提供医疗专科领域特定的评估功能：
1. 多专科基准测试
2. 专科性能比较分析
3. 难度分层评估
4. EM值提升计算（对比基线）
"""

import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import numpy as np

from ming.evaluation.evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    EvaluationResult,
    EvaluationSample,
)
from ming.evaluation.metrics import (
    compute_em_score,
    compute_f1_score,
    compute_medical_entity_score,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 专科类型定义
MEDICAL_SPECIALTIES = {
    "cardiovascular": "心血管内科",
    "neurology": "神经内科",
    "respiratory": "呼吸内科",
    "gastroenterology": "消化内科",
    "endocrinology": "内分泌科",
    "nephrology": "肾内科",
    "hematology": "血液内科",
    "rheumatology": "风湿免疫科",
    "infectious": "感染科",
    "general_surgery": "普通外科",
    "neurosurgery": "神经外科",
    "cardiothoracic": "心胸外科",
    "obstetrics_gynecology": "妇产科",
    "pediatrics": "儿科",
    "ophthalmology": "眼科",
    "otolaryngology": "耳鼻喉科",
    "dermatology": "皮肤科",
    "emergency": "急诊科",
    "critical_care": "重症医学科",
    "general": "综合",
}

# 专科难度级别
DIFFICULTY_LEVELS = {
    "easy": "简单",
    "medium": "中等",
    "hard": "困难",
    "expert": "专家级",
}


@dataclass
class SpecialtyBenchmark:
    """专科基准测试配置"""

    name: str
    description: str
    specialties: List[str]
    data_paths: Dict[str, str]
    difficulty_weights: Dict[str, float] = field(default_factory=dict)
    baseline_results: Optional[Dict[str, Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "specialties": self.specialties,
            "data_paths": self.data_paths,
            "difficulty_weights": self.difficulty_weights,
            "baseline_results": self.baseline_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecialtyBenchmark":
        """从字典创建"""
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: str) -> "SpecialtyBenchmark":
        """从JSON文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, filepath: str) -> None:
        """保存为JSON"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


@dataclass
class SpecialtyResult:
    """单专科评估结果"""

    specialty: str
    sample_count: int
    em_score: float
    f1_score: float
    entity_f1: float
    difficulty_scores: Dict[str, float] = field(default_factory=dict)
    baseline_em: Optional[float] = None
    em_improvement: Optional[float] = None
    em_improvement_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "specialty": self.specialty,
            "specialty_name": MEDICAL_SPECIALTIES.get(self.specialty, self.specialty),
            "sample_count": self.sample_count,
            "em_score": self.em_score,
            "f1_score": self.f1_score,
            "entity_f1": self.entity_f1,
            "difficulty_scores": self.difficulty_scores,
        }
        if self.baseline_em is not None:
            result["baseline_em"] = self.baseline_em
        if self.em_improvement is not None:
            result["em_improvement"] = self.em_improvement
        if self.em_improvement_percent is not None:
            result["em_improvement_percent"] = self.em_improvement_percent
        return result


@dataclass
class SpecialtyEvaluationSummary:
    """专科评估汇总"""

    benchmark_name: str
    overall_em: float
    overall_f1: float
    overall_entity_f1: float
    weighted_em: float
    total_samples: int
    specialty_results: List[SpecialtyResult]
    average_improvement: Optional[float] = None
    improvement_percent: Optional[float] = None
    target_improvement: float = 0.15  # 15%目标提升
    improvement_achieved: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "benchmark_name": self.benchmark_name,
            "overall": {
                "em_score": self.overall_em,
                "f1_score": self.overall_f1,
                "entity_f1": self.overall_entity_f1,
                "weighted_em": self.weighted_em,
                "total_samples": self.total_samples,
            },
            "improvement": {
                "average_improvement": self.average_improvement,
                "improvement_percent": self.improvement_percent,
                "target_improvement": self.target_improvement,
                "achieved": self.improvement_achieved,
            },
            "specialty_results": [r.to_dict() for r in self.specialty_results],
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """生成文本摘要"""
        lines = [
            "=" * 70,
            f"专科评估汇总: {self.benchmark_name}",
            "=" * 70,
            f"总样本数: {self.total_samples}",
            f"总体EM分数: {self.overall_em:.4f}",
            f"总体F1分数: {self.overall_f1:.4f}",
            f"总体实体F1: {self.overall_entity_f1:.4f}",
            f"加权EM分数: {self.weighted_em:.4f}",
        ]

        if self.improvement_percent is not None:
            lines.extend(
                [
                    "",
                    f"EM值提升百分比: {self.improvement_percent:.2%}",
                    f"目标提升: {self.target_improvement:.0%}",
                    f"是否达标: {'是' if self.improvement_achieved else '否'}",
                ]
            )

        lines.extend(
            [
                "",
                "各专科表现:",
                "-" * 70,
                f"{'专科':<20} {'样本数':<8} {'EM':<8} {'F1':<8} {'提升':<10}",
                "-" * 70,
            ]
        )

        for r in self.specialty_results:
            specialty_name = MEDICAL_SPECIALTIES.get(r.specialty, r.specialty)
            imp_str = (
                f"{r.em_improvement_percent:+.2%}"
                if r.em_improvement_percent is not None
                else "N/A"
            )
            lines.append(
                f"{specialty_name:<20} {r.sample_count:<8} "
                f"{r.em_score:<8.4f} {r.f1_score:<8.4f} {imp_str:<10}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


class SpecialtyEvaluator:
    """
    专科领域评估器

    专门针对医疗专科领域的评估，支持：
    1. 多专科数据集评估
    2. 难度分层评估
    3. 基线对比与提升计算
    4. 加权综合评分
    """

    def __init__(
        self,
        model_evaluator: ModelEvaluator,
        benchmark: SpecialtyBenchmark,
    ):
        """
        初始化专科评估器

        Args:
            model_evaluator: 基础模型评估器
            benchmark: 专科基准测试配置
        """
        self.model_evaluator = model_evaluator
        self.benchmark = benchmark
        self.config = model_evaluator.config

        logger.info(f"专科评估器初始化完成，基准测试: {benchmark.name}")

    def load_specialty_samples(
        self,
        specialty: str,
    ) -> List[EvaluationSample]:
        """
        加载专科样本

        Args:
            specialty: 专科类型

        Returns:
            List[EvaluationSample]: 样本列表
        """
        if specialty not in self.benchmark.data_paths:
            raise ValueError(
                f"专科 {specialty} 没有配置数据路径. "
                f"可用专科: {list(self.benchmark.data_paths.keys())}"
            )

        data_path = self.benchmark.data_paths[specialty]
        samples = self.model_evaluator.load_dataset(data_path)

        # 更新专科标记
        for sample in samples:
            sample.specialty = specialty

        logger.info(f"加载 {specialty} 专科样本: {len(samples)} 个")
        return samples

    def evaluate_specialty(
        self,
        specialty: str,
        samples: Optional[List[EvaluationSample]] = None,
    ) -> Tuple[SpecialtyResult, EvaluationResult]:
        """
        评估单个专科

        Args:
            specialty: 专科类型
            samples: 可选的预加载样本

        Returns:
            Tuple[SpecialtyResult, EvaluationResult]: 专科结果和完整评估结果
        """
        if samples is None:
            samples = self.load_specialty_samples(specialty)

        if not samples:
            logger.warning(f"专科 {specialty} 没有样本")
            empty_result = SpecialtyResult(
                specialty=specialty,
                sample_count=0,
                em_score=0.0,
                f1_score=0.0,
                entity_f1=0.0,
            )
            return empty_result, EvaluationResult(
                config=self.config, metrics=self.model_evaluator.evaluate([], specialty)
            )

        # 执行评估
        eval_result = self.model_evaluator.evaluate(samples, specialty)

        # 计算专科指标
        predictions = [s.prediction or "" for s in eval_result.samples]
        references = [s.reference or "" for s in eval_result.samples]
        pred_entities = [s.pred_entities or [] for s in eval_result.samples]
        ref_entities = [s.ref_entities or [] for s in eval_result.samples]

        em_score = compute_em_score(predictions, references)
        f1_score, _, _ = compute_f1_score(predictions, references)

        entity_metrics = compute_medical_entity_score(pred_entities, ref_entities)
        entity_f1 = entity_metrics.get("f1", 0.0)

        # 按难度分层计算
        difficulty_scores = self._compute_difficulty_scores(eval_result.samples)

        # 计算基线对比
        baseline_em = None
        em_improvement = None
        em_improvement_percent = None

        if self.benchmark.baseline_results:
            specialty_baseline = self.benchmark.baseline_results.get(specialty, {})
            baseline_em = specialty_baseline.get("em_score")

            if baseline_em is not None and baseline_em > 0:
                em_improvement = em_score - baseline_em
                em_improvement_percent = em_improvement / baseline_em

        specialty_result = SpecialtyResult(
            specialty=specialty,
            sample_count=len(samples),
            em_score=em_score,
            f1_score=f1_score,
            entity_f1=entity_f1,
            difficulty_scores=difficulty_scores,
            baseline_em=baseline_em,
            em_improvement=em_improvement,
            em_improvement_percent=em_improvement_percent,
        )

        return specialty_result, eval_result

    def _compute_difficulty_scores(
        self,
        samples: List[EvaluationSample],
    ) -> Dict[str, float]:
        """按难度级别计算分数"""
        difficulty_groups: Dict[str, List[Tuple[str, str]]] = {
            level: [] for level in DIFFICULTY_LEVELS
        }

        for sample in samples:
            difficulty = sample.metadata.get("difficulty", "medium")
            if difficulty not in difficulty_groups:
                difficulty = "medium"
            difficulty_groups[difficulty].append(
                (sample.prediction or "", sample.reference or "")
            )

        difficulty_scores = {}
        for level, pairs in difficulty_groups.items():
            if pairs:
                preds, refs = zip(*pairs)
                score = compute_em_score(list(preds), list(refs))
                difficulty_scores[level] = float(score)
            else:
                difficulty_scores[level] = 0.0

        return difficulty_scores

    def evaluate_all_specialties(
        self,
        specialties: Optional[List[str]] = None,
    ) -> Tuple[SpecialtyEvaluationSummary, Dict[str, EvaluationResult]]:
        """
        评估所有专科

        Args:
            specialties: 要评估的专科列表，默认为基准测试配置的所有专科

        Returns:
            Tuple[SpecialtyEvaluationSummary, Dict[str, EvaluationResult]]: 汇总结果和各专科的详细结果
        """
        if specialties is None:
            specialties = self.benchmark.specialties

        all_results: List[SpecialtyResult] = []
        eval_results: Dict[str, EvaluationResult] = {}

        total_samples = 0
        em_sum = 0.0
        f1_sum = 0.0
        entity_f1_sum = 0.0
        weighted_em_sum = 0.0
        total_weight = 0.0

        improvement_sum = 0.0
        improvement_count = 0

        for specialty in specialties:
            logger.info(f"\n开始评估专科: {MEDICAL_SPECIALTIES.get(specialty, specialty)}")
            result, eval_result = self.evaluate_specialty(specialty)

            all_results.append(result)
            eval_results[specialty] = eval_result

            # 更新统计
            sample_count = result.sample_count
            total_samples += sample_count
            em_sum += result.em_score * sample_count
            f1_sum += result.f1_score * sample_count
            entity_f1_sum += result.entity_f1 * sample_count

            # 加权计算
            difficulty_weights = self.benchmark.difficulty_weights
            if difficulty_weights:
                weight = sum(
                    difficulty_weights.get(d, 1.0) * s
                    for d, s in result.difficulty_scores.items()
                )
            else:
                weight = sample_count

            weighted_em_sum += result.em_score * weight
            total_weight += weight

            # 提升计算
            if result.em_improvement is not None:
                improvement_sum += result.em_improvement
                improvement_count += 1

        # 计算总体指标
        overall_em = em_sum / total_samples if total_samples > 0 else 0.0
        overall_f1 = f1_sum / total_samples if total_samples > 0 else 0.0
        overall_entity_f1 = entity_f1_sum / total_samples if total_samples > 0 else 0.0
        weighted_em = weighted_em_sum / total_weight if total_weight > 0 else 0.0

        # 计算平均提升
        average_improvement = None
        improvement_percent = None
        improvement_achieved = False

        if improvement_count > 0:
            average_improvement = improvement_sum / improvement_count
            baseline_em_total = sum(
                r.baseline_em or 0.0 for r in all_results if r.baseline_em
            )
            if baseline_em_total > 0:
                improvement_percent = average_improvement / (
                    baseline_em_total / improvement_count
                )
                improvement_achieved = (
                    improvement_percent >= self.benchmark.target_improvement
                )

        # 按EM分数排序
        all_results.sort(key=lambda x: x.em_score, reverse=True)

        summary = SpecialtyEvaluationSummary(
            benchmark_name=self.benchmark.name,
            overall_em=overall_em,
            overall_f1=overall_f1,
            overall_entity_f1=overall_entity_f1,
            weighted_em=weighted_em,
            total_samples=total_samples,
            specialty_results=all_results,
            average_improvement=average_improvement,
            improvement_percent=improvement_percent,
            target_improvement=self.benchmark.target_improvement,
            improvement_achieved=improvement_achieved,
        )

        return summary, eval_results

    def compare_with_baseline(
        self,
        baseline_results: Dict[str, Dict[str, float]],
        current_results: Dict[str, SpecialtyResult],
    ) -> Dict[str, Any]:
        """
        与基线结果进行比较

        Args:
            baseline_results: 基线结果字典 {specialty: {metric: value}}
            current_results: 当前结果字典 {specialty: SpecialtyResult}

        Returns:
            Dict[str, Any]: 比较结果
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "target_improvement": self.benchmark.target_improvement,
            "specialty_comparison": [],
            "overall": {},
        }

        total_improvement = 0.0
        total_baseline = 0.0
        count = 0

        for specialty, current in current_results.items():
            baseline = baseline_results.get(specialty, {})
            baseline_em = baseline.get("em_score", 0.0)

            if baseline_em > 0:
                improvement = current.em_score - baseline_em
                improvement_pct = improvement / baseline_em

                comp = {
                    "specialty": specialty,
                    "specialty_name": MEDICAL_SPECIALTIES.get(specialty, specialty),
                    "baseline_em": baseline_em,
                    "current_em": current.em_score,
                    "absolute_improvement": improvement,
                    "relative_improvement": improvement_pct,
                    "sample_count": current.sample_count,
                }

                total_improvement += improvement
                total_baseline += baseline_em
                count += 1
            else:
                comp = {
                    "specialty": specialty,
                    "specialty_name": MEDICAL_SPECIALTIES.get(specialty, specialty),
                    "baseline_em": None,
                    "current_em": current.em_score,
                    "absolute_improvement": None,
                    "relative_improvement": None,
                    "sample_count": current.sample_count,
                }

            comparison["specialty_comparison"].append(comp)

        if count > 0:
            comparison["overall"] = {
                "average_absolute_improvement": total_improvement / count,
                "average_relative_improvement": (total_improvement / count)
                / (total_baseline / count)
                if total_baseline > 0
                else 0.0,
                "total_specialties": count,
            }

        return comparison

    def save_evaluation_report(
        self,
        summary: SpecialtyEvaluationSummary,
        eval_results: Dict[str, EvaluationResult],
        output_dir: Optional[str] = None,
    ) -> str:
        """
        保存完整的评估报告

        Args:
            summary: 评估汇总
            eval_results: 各专科的评估结果
            output_dir: 输出目录

        Returns:
            str: 报告文件路径
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存汇总结果
        summary_path = Path(output_dir) / f"specialty_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)

        # 保存各专科详细结果
        details_dir = Path(output_dir) / f"specialty_details_{timestamp}"
        details_dir.mkdir(exist_ok=True)

        for specialty, result in eval_results.items():
            detail_path = details_dir / f"{specialty}.json"
            result.save(str(detail_path))

        # 保存文本摘要
        text_path = Path(output_dir) / f"specialty_summary_{timestamp}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(summary.summary())

        logger.info(f"专科评估报告已保存到: {output_dir}")
        logger.info(f"  汇总JSON: {summary_path}")
        logger.info(f"  文本摘要: {text_path}")

        return str(summary_path)


def create_default_benchmark(
    data_dir: str = "ming/eval/datasets",
    baseline_em: float = 0.65,
    target_improvement: float = 0.15,
) -> SpecialtyBenchmark:
    """
    创建默认的专科基准测试配置

    Args:
        data_dir: 数据目录
        baseline_em: 基线EM分数
        target_improvement: 目标提升百分比

    Returns:
        SpecialtyBenchmark: 基准测试配置
    """
    specialties = [
        "cardiovascular",
        "neurology",
        "respiratory",
        "gastroenterology",
        "endocrinology",
    ]

    data_paths = {}
    for s in specialties:
        data_paths[s] = f"{data_dir}/{s}.jsonl"

    difficulty_weights = {
        "easy": 0.2,
        "medium": 0.3,
        "hard": 0.3,
        "expert": 0.2,
    }

    baseline_results = {
        s: {"em_score": baseline_em} for s in specialties
    }

    return SpecialtyBenchmark(
        name="MING-7B 专科医疗基准测试",
        description="针对5个核心医疗专科的问答性能基准测试",
        specialties=specialties,
        data_paths=data_paths,
        difficulty_weights=difficulty_weights,
        baseline_results=baseline_results,
        target_improvement=target_improvement,
    )


def main():
    """测试函数"""
    print("专科领域评估器模块测试")
    print()

    # 创建默认基准配置
    benchmark = create_default_benchmark()
    print("默认基准配置:")
    print(f"  名称: {benchmark.name}")
    print(f"  描述: {benchmark.description}")
    print(f"  专科数量: {len(benchmark.specialties)}")
    print("  专科列表:")
    for s in benchmark.specialties:
        print(f"    - {MEDICAL_SPECIALTIES.get(s, s)}")
    print(f"  目标提升: {benchmark.target_improvement:.0%}")
    print()

    # 测试专科结果
    results = [
        SpecialtyResult(
            specialty="cardiovascular",
            sample_count=100,
            em_score=0.78,
            f1_score=0.85,
            entity_f1=0.92,
            baseline_em=0.65,
            em_improvement=0.13,
            em_improvement_percent=0.20,
        ),
        SpecialtyResult(
            specialty="neurology",
            sample_count=80,
            em_score=0.75,
            f1_score=0.82,
            entity_f1=0.90,
            baseline_em=0.64,
            em_improvement=0.11,
            em_improvement_percent=0.17,
        ),
    ]

    # 创建汇总
    summary = SpecialtyEvaluationSummary(
        benchmark_name=benchmark.name,
        overall_em=0.765,
        overall_f1=0.835,
        overall_entity_f1=0.91,
        weighted_em=0.77,
        total_samples=180,
        specialty_results=results,
        average_improvement=0.12,
        improvement_percent=0.185,
        target_improvement=0.15,
        improvement_achieved=True,
    )

    print("评估汇总示例:")
    print(summary.summary())


if __name__ == "__main__":
    main()
