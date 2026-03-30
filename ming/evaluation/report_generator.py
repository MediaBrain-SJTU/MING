"""
评估报告生成器模块

本模块提供评估报告的生成功能，支持：
1. 多维度评估结果汇总
2. JSON格式报告输出
3. 对比分析报告
4. 报告验证与复现性检查
"""

import json
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import numpy as np

from ming.evaluation.evaluator import EvaluationResult, EvaluationConfig
from ming.evaluation.specialty_evaluator import (
    SpecialtyEvaluationSummary,
    SpecialtyResult,
    MEDICAL_SPECIALTIES,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """报告章节基类"""

    title: str
    description: str = ""
    content: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "title": self.title,
            "description": self.description,
            "content": self.content,
        }


@dataclass
class EvaluationReport:
    """
    评估报告类

    包含完整的评估报告信息，确保可复现性和完整性
    """

    report_id: str
    report_version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""

    # 模型信息
    model_info: Dict[str, Any] = field(default_factory=dict)

    # 评估配置
    evaluation_config: Dict[str, Any] = field(default_factory=dict)

    # 报告章节
    sections: List[ReportSection] = field(default_factory=list)

    # 汇总指标
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

    # 复现性验证
    reproducibility: Dict[str, Any] = field(default_factory=dict)

    # 文件哈希（用于验证完整性）
    file_hashes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "report_id": self.report_id,
            "report_version": self.report_version,
            "created_at": self.created_at,
            "description": self.description,
            "model_info": self.model_info,
            "evaluation_config": self.evaluation_config,
            "sections": [s.to_dict() for s in self.sections],
            "summary_metrics": self.summary_metrics,
            "reproducibility": self.reproducibility,
            "file_hashes": self.file_hashes,
        }

    def save(self, filepath: str, add_hash: bool = True) -> None:
        """
        保存报告为JSON格式

        Args:
            filepath: 输出文件路径
            add_hash: 是否添加文件哈希用于验证
        """
        data = self.to_dict()

        if add_hash:
            content_str = json.dumps(data, ensure_ascii=False, sort_keys=True)
            data["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"评估报告已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str, verify_hash: bool = True) -> "EvaluationReport":
        """
        从JSON文件加载报告

        Args:
            filepath: 文件路径
            verify_hash: 是否验证哈希

        Returns:
            EvaluationReport: 加载的报告
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if verify_hash and "content_hash" in data:
            stored_hash = data.pop("content_hash")
            content_str = json.dumps(data, ensure_ascii=False, sort_keys=True)
            computed_hash = hashlib.sha256(content_str.encode()).hexdigest()

            if stored_hash != computed_hash:
                logger.warning(f"报告完整性验证失败: {filepath}")
                logger.warning(f"  存储哈希: {stored_hash}")
                logger.warning(f"  计算哈希: {computed_hash}")
            else:
                logger.info(f"报告完整性验证通过: {filepath}")

        # 重建sections
        sections = []
        for s in data.get("sections", []):
            sections.append(
                ReportSection(
                    title=s.get("title", ""),
                    description=s.get("description", ""),
                    content=s.get("content", {}),
                )
            )
        data["sections"] = sections

        return cls(**data)

    def verify_reproducibility(
        self,
        other_report: "EvaluationReport",
        tolerance: float = 1e-6,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        验证两个报告的可复现性

        Args:
            other_report: 另一个报告进行比较
            tolerance: 数值容忍度

        Returns:
            Tuple[bool, Dict[str, Any]]: 是否通过和详细结果
        """
        result = {
            "verified": True,
            "differences": [],
            "tolerance": tolerance,
            "verified_at": datetime.now().isoformat(),
        }

        # 验证模型信息
        if self.model_info.get("model_path") != other_report.model_info.get("model_path"):
            result["differences"].append(
                {
                    "field": "model_path",
                    "this": self.model_info.get("model_path"),
                    "other": other_report.model_info.get("model_path"),
                }
            )
            result["verified"] = False

        # 验证评估配置
        config_fields = ["max_new_tokens", "temperature", "eval_batch_size"]
        for field in config_fields:
            this_val = self.evaluation_config.get(field)
            other_val = other_report.evaluation_config.get(field)
            if this_val != other_val:
                result["differences"].append(
                    {
                        "field": f"config.{field}",
                        "this": this_val,
                        "other": other_val,
                    }
                )
                result["verified"] = False

        # 验证指标（在容忍范围内）
        metric_fields = ["em", "f1", "precision", "recall"]
        for field in metric_fields:
            this_val = self.summary_metrics.get("overall", {}).get(field, 0.0)
            other_val = other_report.summary_metrics.get("overall", {}).get(field, 0.0)
            if abs(this_val - other_val) > tolerance:
                result["differences"].append(
                    {
                        "field": f"metric.{field}",
                        "this": this_val,
                        "other": other_val,
                        "diff": abs(this_val - other_val),
                    }
                )
                result["verified"] = False

        return result["verified"], result

    def summary(self) -> str:
        """生成文本摘要"""
        lines = [
            "=" * 70,
            f"评估报告: {self.report_id}",
            "=" * 70,
            f"版本: {self.report_version}",
            f"生成时间: {self.created_at}",
            f"描述: {self.description}",
            "",
            "模型信息:",
            f"  名称: {self.model_info.get('name', 'N/A')}",
            f"  路径: {self.model_info.get('path', 'N/A')}",
            "",
            "汇总指标:",
        ]

        overall = self.summary_metrics.get("overall", {})
        lines.extend(
            [
                f"  EM分数: {overall.get('em', 0):.4f}",
                f"  F1分数: {overall.get('f1', 0):.4f}",
                f"  精确率: {overall.get('precision', 0):.4f}",
                f"  召回率: {overall.get('recall', 0):.4f}",
            ]
        )

        improvement = self.summary_metrics.get("improvement", {})
        if improvement:
            lines.extend(
                [
                    "",
                    "提升情况:",
                    f"  绝对提升: {improvement.get('absolute', 0):.4f}",
                    f"  相对提升: {improvement.get('relative', 0):.2%}",
                    f"  目标提升: {improvement.get('target', 0):.0%}",
                    f"  是否达标: {'是' if improvement.get('achieved', False) else '否'}",
                ]
            )

        performance = self.summary_metrics.get("performance", {})
        if performance:
            lines.extend(
                [
                    "",
                    "性能指标:",
                    f"  单样本推理时间: {performance.get('inference_time_ms', 0):.2f}ms",
                    f"  显存使用: {performance.get('memory_usage_gb', 0):.2f}GB",
                ]
            )

        # 专科表现（前5）
        specialty_results = self.summary_metrics.get("specialty_results", [])
        if specialty_results:
            lines.extend(
                [
                    "",
                    "专科表现Top 5:",
                    "-" * 50,
                    f"{'专科':<20} {'样本数':<8} {'EM':<8} {'提升':<10}",
                    "-" * 50,
                ]
            )
            for r in specialty_results[:5]:
                name = MEDICAL_SPECIALTIES.get(r.get("specialty", ""), r.get("specialty", ""))
                imp = r.get("em_improvement_percent", 0)
                imp_str = f"{imp:+.2%}" if imp != 0 else "N/A"
                lines.append(
                    f"{name:<20} {r.get('sample_count', 0):<8} "
                    f"{r.get('em_score', 0):<8.4f} {imp_str:<10}"
                )

        lines.extend(
            [
                "",
                "复现性:",
                f"  可复现: {'是' if self.reproducibility.get('verified', False) else '否'}",
            ]
        )

        lines.append("=" * 70)
        return "\n".join(lines)


class ReportGenerator:
    """
    评估报告生成器

    生成符合要求的JSON格式评估报告，确保：
    1. 100%复现率
    2. 完整的指标计算
    3. 5分钟内生成
    """

    def __init__(
        self,
        output_dir: str = "./evaluation_reports",
        report_prefix: str = "ming_evaluation",
    ):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
            report_prefix: 报告文件名前缀
        """
        self.output_dir = Path(output_dir)
        self.report_prefix = report_prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        evaluation_result: EvaluationResult,
        specialty_summary: Optional[SpecialtyEvaluationSummary] = None,
        baseline_results: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> EvaluationReport:
        """
        生成完整的评估报告

        Args:
            evaluation_result: 评估结果
            specialty_summary: 专科评估汇总（可选）
            baseline_results: 基线结果（可选）
            description: 报告描述

        Returns:
            EvaluationReport: 生成的评估报告
        """
        start_time = datetime.now()

        # 生成报告ID
        report_id = self._generate_report_id()

        # 创建报告
        report = EvaluationReport(
            report_id=report_id,
            description=description or "MING-7B 医疗大模型评估报告",
        )

        # 设置模型信息
        report.model_info = {
            "name": evaluation_result.model_name,
            "path": evaluation_result.model_path,
            "type": "MING-7B",
        }

        # 设置评估配置
        report.evaluation_config = evaluation_result.config.to_dict()

        # 1. 总体指标章节
        overall_section = self._create_overall_section(evaluation_result)
        report.sections.append(overall_section)

        # 2. 详细指标章节
        metrics_section = self._create_detailed_metrics_section(evaluation_result)
        report.sections.append(metrics_section)

        # 3. 性能指标章节
        performance_section = self._create_performance_section(evaluation_result)
        report.sections.append(performance_section)

        # 4. 实体识别章节
        entity_section = self._create_entity_section(evaluation_result)
        report.sections.append(entity_section)

        # 5. 专科评估章节（如果有）
        if specialty_summary:
            specialty_section = self._create_specialty_section(specialty_summary)
            report.sections.append(specialty_section)

            # 设置专科汇总指标
            report.summary_metrics["specialty_results"] = [
                r.to_dict() for r in specialty_summary.specialty_results
            ]

        # 6. 基线对比章节（如果有）
        if baseline_results:
            comparison_section = self._create_comparison_section(
                evaluation_result, baseline_results
            )
            report.sections.append(comparison_section)

        # 7. 错误分析章节
        error_section = self._create_error_analysis_section(evaluation_result)
        report.sections.append(error_section)

        # 设置汇总指标
        report.summary_metrics["overall"] = {
            "em": evaluation_result.metrics.em,
            "f1": evaluation_result.metrics.f1,
            "precision": evaluation_result.metrics.precision,
            "recall": evaluation_result.metrics.recall,
            "sample_count": evaluation_result.metrics.sample_count,
        }

        if specialty_summary:
            report.summary_metrics["improvement"] = {
                "absolute": specialty_summary.average_improvement or 0.0,
                "relative": specialty_summary.improvement_percent or 0.0,
                "target": specialty_summary.target_improvement,
                "achieved": specialty_summary.improvement_achieved,
            }

        report.summary_metrics["performance"] = {
            "inference_time_ms": evaluation_result.metrics.inference_time_per_sample,
            "memory_usage_gb": evaluation_result.metrics.memory_usage_gb,
            "total_time_seconds": evaluation_result.total_time_seconds,
        }

        # 设置复现性信息
        report.reproducibility = {
            "config_hash": self._compute_hash(report.evaluation_config),
            "metrics_hash": self._compute_hash(report.summary_metrics),
            "sample_count": evaluation_result.metrics.sample_count,
            "generation_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
        }

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{self.report_prefix}_{report_id}_{timestamp}.json"
        report.save(str(report_path))

        # 保存文本摘要
        text_path = self.output_dir / f"{self.report_prefix}_{report_id}_{timestamp}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(report.summary())

        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"报告生成完成，耗时: {generation_time:.2f}秒")
        logger.info(f"报告ID: {report_id}")
        logger.info(f"报告路径: {report_path}")

        return report

    def _generate_report_id(self) -> str:
        """生成唯一报告ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"{timestamp}_{random_str}"

    def _compute_hash(self, obj: Any) -> str:
        """计算对象的哈希值"""
        content_str = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]

    def _create_overall_section(
        self,
        result: EvaluationResult,
    ) -> ReportSection:
        """创建总体指标章节"""
        content = {
            "em_score": result.metrics.em,
            "f1_score": result.metrics.f1,
            "precision": result.metrics.precision,
            "recall": result.metrics.recall,
            "sample_count": result.metrics.sample_count,
            "evaluation_time_seconds": result.total_time_seconds,
        }

        return ReportSection(
            title="总体评估结果",
            description="模型在评估数据集上的整体表现",
            content=content,
        )

    def _create_detailed_metrics_section(
        self,
        result: EvaluationResult,
    ) -> ReportSection:
        """创建详细指标章节"""
        content = {}

        # BLEU分数
        if result.metrics.bleu:
            content["bleu_scores"] = result.metrics.bleu

        # ROUGE分数
        if result.metrics.rouge:
            content["rouge_scores"] = result.metrics.rouge

        return ReportSection(
            title="详细指标",
            description="BLEU、ROUGE等详细评估指标",
            content=content,
        )

    def _create_performance_section(
        self,
        result: EvaluationResult,
    ) -> ReportSection:
        """创建性能指标章节"""
        content = {
            "inference_time_per_sample_ms": result.metrics.inference_time_per_sample,
            "memory_usage_gb": result.metrics.memory_usage_gb,
            "total_evaluation_time_seconds": result.total_time_seconds,
            "samples_per_second": (
                result.metrics.sample_count / result.total_time_seconds
                if result.total_time_seconds > 0
                else 0.0
            ),
        }

        return ReportSection(
            title="性能指标",
            description="推理速度和显存使用等性能指标",
            content=content,
        )

    def _create_entity_section(
        self,
        result: EvaluationResult,
    ) -> ReportSection:
        """创建实体识别章节"""
        content = {}

        if result.metrics.entity_metrics:
            content["overall"] = {
                "precision": result.metrics.entity_metrics.get("precision", 0.0),
                "recall": result.metrics.entity_metrics.get("recall", 0.0),
                "f1": result.metrics.entity_metrics.get("f1", 0.0),
            }

            # 按实体类型分类
            type_f1 = {}
            for key, value in result.metrics.entity_metrics.items():
                if key.endswith("_f1") and key not in ["f1", "overall_f1"]:
                    entity_type = key[:-3]
                    type_f1[entity_type] = value

            if type_f1:
                content["entity_type_f1"] = type_f1

        return ReportSection(
            title="医疗实体识别",
            description="医疗实体识别性能指标",
            content=content,
        )

    def _create_specialty_section(
        self,
        summary: SpecialtyEvaluationSummary,
    ) -> ReportSection:
        """创建专科评估章节"""
        specialty_list = []
        for result in summary.specialty_results:
            specialty_list.append(result.to_dict())

        content = {
            "overall_em": summary.overall_em,
            "overall_f1": summary.overall_f1,
            "overall_entity_f1": summary.overall_entity_f1,
            "weighted_em": summary.weighted_em,
            "total_samples": summary.total_samples,
            "specialty_results": specialty_list,
        }

        return ReportSection(
            title="专科领域评估",
            description="各医疗专科领域的详细评估结果",
            content=content,
        )

    def _create_comparison_section(
        self,
        result: EvaluationResult,
        baseline: Dict[str, Any],
    ) -> ReportSection:
        """创建基线对比章节"""
        baseline_em = baseline.get("em_score", 0.0)
        current_em = result.metrics.em

        improvement = current_em - baseline_em
        improvement_pct = improvement / baseline_em if baseline_em > 0 else 0.0

        content = {
            "baseline": baseline,
            "current": {
                "em_score": current_em,
                "f1_score": result.metrics.f1,
            },
            "improvement": {
                "absolute": improvement,
                "relative": improvement_pct,
                "percentage": f"{improvement_pct:.2%}",
            },
        }

        return ReportSection(
            title="基线对比",
            description="与基线模型的性能对比",
            content=content,
        )

    def _create_error_analysis_section(
        self,
        result: EvaluationResult,
        max_samples: int = 10,
    ) -> ReportSection:
        """创建错误分析章节"""
        error_samples = []
        correct_samples = []

        for sample in result.samples:
            pred = sample.prediction or ""
            ref = sample.reference or ""

            # 标准化比较
            from ming.evaluation.metrics import normalize_answer

            is_correct = normalize_answer(pred) == normalize_answer(ref)

            sample_data = {
                "id": sample.id,
                "question": sample.question,
                "prediction": sample.prediction,
                "reference": sample.reference,
                "specialty": sample.specialty,
                "is_correct": is_correct,
            }

            if not is_correct and len(error_samples) < max_samples:
                error_samples.append(sample_data)
            elif is_correct and len(correct_samples) < max_samples // 2:
                correct_samples.append(sample_data)

        content = {
            "total_errors": sum(
                1
                for s in result.samples
                if normalize_answer(s.prediction or "")
                != normalize_answer(s.reference or "")
            ),
            "total_correct": sum(
                1
                for s in result.samples
                if normalize_answer(s.prediction or "")
                == normalize_answer(s.reference or "")
            ),
            "error_rate": (
                1 - result.metrics.em if result.metrics.sample_count > 0 else 0.0
            ),
            "error_examples": error_samples,
            "correct_examples": correct_samples,
        }

        return ReportSection(
            title="错误分析",
            description="错误案例和正确案例分析",
            content=content,
        )

    def generate_comparison_report(
        self,
        reports: Dict[str, EvaluationReport],
        description: str = "模型对比报告",
    ) -> EvaluationReport:
        """
        生成多模型对比报告

        Args:
            reports: 模型名称到报告的映射
            description: 报告描述

        Returns:
            EvaluationReport: 对比报告
        """
        report_id = self._generate_report_id()
        report = EvaluationReport(
            report_id=report_id,
            description=description,
        )

        report.model_info = {
            "comparison_models": list(reports.keys()),
            "report_count": len(reports),
        }

        # 指标对比章节
        comparison_content: Dict[str, Dict[str, Any]] = {}

        for model_name, r in reports.items():
            overall = r.summary_metrics.get("overall", {})
            performance = r.summary_metrics.get("performance", {})

            for metric_name, value in overall.items():
                if metric_name not in comparison_content:
                    comparison_content[metric_name] = {}
                comparison_content[metric_name][model_name] = value

            for metric_name, value in performance.items():
                if metric_name not in comparison_content:
                    comparison_content[metric_name] = {}
                comparison_content[metric_name][model_name] = value

        comparison_section = ReportSection(
            title="指标对比",
            description="各模型的指标对比",
            content=comparison_content,
        )
        report.sections.append(comparison_section)

        # 专科对比（如果有）
        specialty_comparison: Dict[str, Dict[str, float]] = {}
        for model_name, r in reports.items():
            specialty_results = r.summary_metrics.get("specialty_results", [])
            for result in specialty_results:
                specialty = result.get("specialty", "")
                if specialty not in specialty_comparison:
                    specialty_comparison[specialty] = {}
                specialty_comparison[specialty][model_name] = result.get(
                    "em_score", 0.0
                )

        if specialty_comparison:
            specialty_section = ReportSection(
                title="专科表现对比",
                description="各专科领域的EM分数对比",
                content=specialty_comparison,
            )
            report.sections.append(specialty_section)

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"comparison_{report_id}_{timestamp}.json"
        report.save(str(report_path))

        return report


def main():
    """测试函数"""
    print("评估报告生成器模块测试")

    # 创建报告生成器
    generator = ReportGenerator(output_dir="./test_reports")

    # 创建模拟评估结果
    from ming.evaluation.evaluator import EvaluationResult, EvaluationConfig
    from ming.evaluation.metrics import EvaluationMetrics

    config = EvaluationConfig()
    metrics = EvaluationMetrics(
        em=0.78,
        f1=0.85,
        precision=0.87,
        recall=0.83,
        sample_count=1000,
        inference_time_per_sample=45.5,
        memory_usage_gb=14.2,
    )

    result = EvaluationResult(
        config=config,
        metrics=metrics,
        model_name="MING-7B",
        model_path="./models/ming-7b",
        total_time_seconds=45.5,
    )

    # 生成报告
    report = generator.generate_report(
        result,
        description="测试报告 - MING-7B 医疗大模型评估",
    )

    # 打印摘要
    print("\n报告摘要:")
    print(report.summary())


if __name__ == "__main__":
    main()
