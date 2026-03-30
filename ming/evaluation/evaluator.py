"""
医疗问答评估器
提供完整的评估流程
"""
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

from ming.evaluation.metrics import (
    BaseMetric,
    ExactMatchMetric,
    F1Metric,
    ROUGEMetric,
    BLEUMetric,
    MedicalAccuracyMetric,
    MetricResult
)


@dataclass
class EvaluationSample:
    """评估样本"""
    question_id: str
    question: str
    options: Optional[Dict[str, str]]
    reference: str
    prediction: str
    specialty: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "options": self.options,
            "reference": self.reference,
            "prediction": self.prediction,
            "specialty": self.specialty,
            "difficulty": self.difficulty,
            "metadata": self.metadata
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    total_samples: int
    metrics: Dict[str, MetricResult]
    specialty_results: Dict[str, Dict[str, float]]
    difficulty_results: Dict[str, Dict[str, float]]
    processing_time_seconds: float
    reproducibility_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "specialty_results": self.specialty_results,
            "difficulty_results": self.difficulty_results,
            "processing_time_seconds": self.processing_time_seconds,
            "reproducibility_hash": self.reproducibility_hash
        }


class MedicalQAEvaluator:
    """
    医疗问答评估器
    
    功能：
    - 多维度指标计算
    - 按专科/难度分组评估
    - 结果可复现
    - 支持增量评估
    """
    
    DEFAULT_METRICS = ["exact_match", "f1", "rouge", "medical_accuracy"]
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        entity_recognizer: Optional[Any] = None,
        cache_results: bool = True
    ):
        """
        初始化评估器
        
        Args:
            metrics: 要计算的指标列表
            entity_recognizer: 实体识别器（用于医疗准确性评估）
            cache_results: 是否缓存结果
        """
        self.metrics_names = metrics or self.DEFAULT_METRICS
        self.entity_recognizer = entity_recognizer
        self.cache_results = cache_results
        
        self._metric_instances = self._init_metrics()
        self._cache: Dict[str, Any] = {}
    
    def _init_metrics(self) -> Dict[str, BaseMetric]:
        """初始化指标实例"""
        metric_instances = {}
        
        for name in self.metrics_names:
            if name == "exact_match":
                metric_instances[name] = ExactMatchMetric()
            elif name == "f1":
                metric_instances[name] = F1Metric()
            elif name == "rouge":
                metric_instances[name] = ROUGEMetric()
            elif name == "bleu":
                metric_instances[name] = BLEUMetric()
            elif name == "medical_accuracy":
                metric_instances[name] = MedicalAccuracyMetric(
                    entity_recognizer=self.entity_recognizer
                )
        
        return metric_instances
    
    def evaluate(
        self,
        samples: List[EvaluationSample],
        compute_per_sample: bool = True
    ) -> EvaluationResult:
        """
        执行评估
        
        Args:
            samples: 评估样本列表
            compute_per_sample: 是否计算每个样本的详细结果
            
        Returns:
            EvaluationResult: 评估结果
        """
        start_time = time.time()
        
        predictions = [s.prediction for s in samples]
        references = [s.reference for s in samples]
        
        metrics_results = {}
        for name, metric in self._metric_instances.items():
            result = metric.compute(predictions, references)
            metrics_results[name] = result
        
        specialty_results = self._evaluate_by_group(
            samples, "specialty"
        )
        
        difficulty_results = self._evaluate_by_group(
            samples, "difficulty"
        )
        
        processing_time = time.time() - start_time
        
        reproducibility_hash = self._compute_hash(samples, predictions, references)
        
        return EvaluationResult(
            total_samples=len(samples),
            metrics=metrics_results,
            specialty_results=specialty_results,
            difficulty_results=difficulty_results,
            processing_time_seconds=processing_time,
            reproducibility_hash=reproducibility_hash
        )
    
    def _evaluate_by_group(
        self,
        samples: List[EvaluationSample],
        group_by: str
    ) -> Dict[str, Dict[str, float]]:
        """按分组评估"""
        groups: Dict[str, List[EvaluationSample]] = {}
        
        for sample in samples:
            group_key = getattr(sample, group_by, "unknown") or "unknown"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(sample)
        
        results = {}
        for group_key, group_samples in groups.items():
            predictions = [s.prediction for s in group_samples]
            references = [s.reference for s in group_samples]
            
            group_metrics = {}
            for name, metric in self._metric_instances.items():
                result = metric.compute(predictions, references)
                group_metrics[name] = result.value
            
            results[group_key] = group_metrics
        
        return results
    
    def _compute_hash(
        self,
        samples: List[EvaluationSample],
        predictions: List[str],
        references: List[str]
    ) -> str:
        """计算可复现性哈希"""
        content = json.dumps({
            "samples": [s.to_dict() for s in samples],
            "predictions": predictions,
            "references": references
        }, ensure_ascii=False, sort_keys=True)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def evaluate_from_file(
        self,
        file_path: str,
        prediction_key: str = "prediction",
        reference_key: str = "answer"
    ) -> EvaluationResult:
        """
        从文件加载并评估
        
        Args:
            file_path: 文件路径
            prediction_key: 预测结果字段名
            reference_key: 参考答案字段名
            
        Returns:
            EvaluationResult: 评估结果
        """
        samples = []
        
        try:
            import jsonlines
            with jsonlines.open(file_path) as reader:
                items = list(reader)
        except ImportError:
            with open(file_path, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f if line.strip()]
        
        for i, item in enumerate(items):
            sample = EvaluationSample(
                question_id=str(i),
                question=item.get("question", ""),
                options=item.get("options"),
                reference=item.get(reference_key, ""),
                prediction=item.get(prediction_key, ""),
                specialty=item.get("specialty"),
                difficulty=item.get("difficulty"),
                metadata=item
            )
            samples.append(sample)
        
        return self.evaluate(samples)
    
    def compare_models(
        self,
        model_results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """
        比较多个模型的评估结果
        
        Args:
            model_results: 模型名称到评估结果的映射
            
        Returns:
            比较结果字典
        """
        comparison = {
            "models": list(model_results.keys()),
            "metrics_comparison": {},
            "best_model_per_metric": {}
        }
        
        all_metrics = set()
        for result in model_results.values():
            all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            metric_values = {}
            for model_name, result in model_results.items():
                if metric in result.metrics:
                    metric_values[model_name] = result.metrics[metric].value
            
            comparison["metrics_comparison"][metric] = metric_values
            
            if metric_values:
                best_model = max(metric_values.items(), key=lambda x: x[1])
                comparison["best_model_per_metric"][metric] = {
                    "model": best_model[0],
                    "value": best_model[1]
                }
        
        return comparison
    
    def get_error_analysis(
        self,
        samples: List[EvaluationSample],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        错误分析
        
        Args:
            samples: 评估样本
            top_k: 返回top k个错误样本
            
        Returns:
            错误分析结果
        """
        errors = []
        
        for sample in samples:
            em_metric = ExactMatchMetric()
            result = em_metric.compute([sample.prediction], [sample.reference])
            
            if result.value == 0:
                f1_metric = F1Metric()
                f1_result = f1_metric.compute([sample.prediction], [sample.reference])
                
                errors.append({
                    "sample": sample.to_dict(),
                    "f1_score": f1_result.value,
                    "error_type": self._classify_error(sample)
                })
        
        errors.sort(key=lambda x: x["f1_score"])
        
        error_types = {}
        for error in errors:
            error_type = error["error_type"]
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(samples) if samples else 0,
            "error_type_distribution": error_types,
            "top_errors": errors[:top_k]
        }
    
    def _classify_error(self, sample: EvaluationSample) -> str:
        """分类错误类型"""
        pred = sample.prediction.lower()
        ref = sample.reference.lower()
        
        if not pred or pred == "无" or pred == "未知":
            return "no_answer"
        
        if any(c in pred for c in "ABCD") and any(c in ref for c in "ABCD"):
            return "wrong_option"
        
        if len(pred) < len(ref) * 0.5:
            return "incomplete_answer"
        
        if len(pred) > len(ref) * 2:
            return "overly_verbose"
        
        return "semantic_error"
    
    def export_results(
        self,
        result: EvaluationResult,
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        导出评估结果
        
        Args:
            result: 评估结果
            output_path: 输出路径
            format: 输出格式
        """
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for name, metric_result in result.metrics.items():
                    writer.writerow([name, metric_result.value])
    
    def verify_reproducibility(
        self,
        result1: EvaluationResult,
        result2: EvaluationResult
    ) -> bool:
        """
        验证结果可复现性
        
        Args:
            result1: 第一次评估结果
            result2: 第二次评估结果
            
        Returns:
            是否可复现
        """
        return result1.reproducibility_hash == result2.reproducibility_hash
