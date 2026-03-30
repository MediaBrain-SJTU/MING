"""
多维度评估指标体系模块

本模块提供医疗大模型的多维度评估能力，包括：
1. 基础NLP指标（EM, F1, BLEU, ROUGE）
2. 医疗专科指标（实体识别准确率、专科问题EM值）
3. 性能指标（推理速度、显存使用）
4. 评估报告生成（JSON格式）
"""

from ming.evaluation.metrics import (
    EvaluationMetrics,
    compute_em_score,
    compute_f1_score,
    compute_bleu_score,
    compute_rouge_score,
    compute_medical_entity_score,
    normalize_answer,
)
from ming.evaluation.evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    EvaluationResult,
    EvaluationSample,
    EvaluationDataset,
)
from ming.evaluation.report_generator import (
    ReportGenerator,
    EvaluationReport,
    ReportSection,
)
from ming.evaluation.specialty_evaluator import (
    SpecialtyEvaluator,
    SpecialtyBenchmark,
    SpecialtyResult,
    SpecialtyEvaluationSummary,
    MEDICAL_SPECIALTIES,
    DIFFICULTY_LEVELS,
    create_default_benchmark,
)

__all__ = [
    "EvaluationMetrics",
    "compute_em_score",
    "compute_f1_score",
    "compute_bleu_score",
    "compute_rouge_score",
    "compute_medical_entity_score",
    "normalize_answer",
    "ModelEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationSample",
    "EvaluationDataset",
    "ReportGenerator",
    "EvaluationReport",
    "ReportSection",
    "SpecialtyEvaluator",
    "SpecialtyBenchmark",
    "SpecialtyResult",
    "SpecialtyEvaluationSummary",
    "MEDICAL_SPECIALTIES",
    "DIFFICULTY_LEVELS",
    "create_default_benchmark",
]
