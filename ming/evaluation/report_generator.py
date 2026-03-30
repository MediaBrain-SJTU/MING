"""
评估报告生成器
生成多维度评估报告
目标：报告生成时间 <= 5分钟
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: Dict[str, Any]
    subsections: List["ReportSection"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections]
        }


class EvaluationReportGenerator:
    """
    评估报告生成器
    
    功能：
    - 生成JSON格式评估报告
    - 多维度结果展示
    - 对比分析
    - 可视化数据准备
    """
    
    REPORT_VERSION = "1.0.0"
    
    def __init__(
        self,
        output_dir: str = "reports",
        include_visualization_data: bool = True
    ):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
            include_visualization_data: 是否包含可视化数据
        """
        self.output_dir = output_dir
        self.include_visualization_data = include_visualization_data
        
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(
        self,
        evaluation_result: Any,
        model_info: Optional[Dict[str, Any]] = None,
        training_info: Optional[Dict[str, Any]] = None,
        baseline_result: Optional[Any] = None,
        report_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            evaluation_result: 评估结果
            model_info: 模型信息
            training_info: 训练信息
            baseline_result: 基线评估结果
            report_name: 报告名称
            
        Returns:
            完整报告字典
        """
        start_time = time.time()
        
        report_name = report_name or f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report = {
            "report_info": self._generate_report_info(report_name),
            "model_info": model_info or {},
            "training_info": training_info or {},
            "evaluation_summary": self._generate_summary(evaluation_result),
            "detailed_metrics": self._generate_detailed_metrics(evaluation_result),
            "specialty_analysis": self._generate_specialty_analysis(evaluation_result),
            "difficulty_analysis": self._generate_difficulty_analysis(evaluation_result),
            "baseline_comparison": self._generate_baseline_comparison(
                evaluation_result, baseline_result
            ),
            "recommendations": self._generate_recommendations(evaluation_result),
            "visualization_data": self._generate_visualization_data(
                evaluation_result
            ) if self.include_visualization_data else None
        }
        
        generation_time = time.time() - start_time
        report["generation_info"] = {
            "generation_time_seconds": generation_time,
            "within_time_limit": generation_time <= 300
        }
        
        return report
    
    def _generate_report_info(self, report_name: str) -> Dict[str, Any]:
        """生成报告基本信息"""
        return {
            "report_name": report_name,
            "version": self.REPORT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "generator": "MING Evaluation Report Generator"
        }
    
    def _generate_summary(self, result: Any) -> Dict[str, Any]:
        """生成评估摘要"""
        summary = {
            "total_samples": result.total_samples,
            "processing_time_seconds": result.processing_time_seconds,
            "reproducibility_hash": result.reproducibility_hash,
            "key_metrics": {}
        }
        
        if hasattr(result, "metrics"):
            for name, metric_result in result.metrics.items():
                summary["key_metrics"][name] = {
                    "value": metric_result.value,
                    "description": self._get_metric_description(name)
                }
        
        return summary
    
    def _get_metric_description(self, metric_name: str) -> str:
        """获取指标描述"""
        descriptions = {
            "exact_match": "预测与参考答案完全匹配的比例",
            "f1": "词级别的F1分数，综合考虑精确率和召回率",
            "rouge": "ROUGE分数，衡量生成文本与参考文本的重叠度",
            "bleu": "BLEU分数，衡量翻译/生成质量",
            "medical_accuracy": "医疗准确性，综合考虑诊断、治疗、药物推荐准确性"
        }
        return descriptions.get(metric_name, f"{metric_name}指标")
    
    def _generate_detailed_metrics(self, result: Any) -> Dict[str, Any]:
        """生成详细指标"""
        detailed = {}
        
        if hasattr(result, "metrics"):
            for name, metric_result in result.metrics.items():
                detailed[name] = metric_result.to_dict()
        
        return detailed
    
    def _generate_specialty_analysis(self, result: Any) -> Dict[str, Any]:
        """生成专科分析"""
        analysis = {
            "overview": "按专科领域分组的评估结果",
            "specialties": {}
        }
        
        if hasattr(result, "specialty_results"):
            for specialty, metrics in result.specialty_results.items():
                analysis["specialties"][specialty] = {
                    "metrics": metrics,
                    "performance_level": self._get_performance_level(metrics)
                }
        
        return analysis
    
    def _generate_difficulty_analysis(self, result: Any) -> Dict[str, Any]:
        """生成难度分析"""
        analysis = {
            "overview": "按问题难度分组的评估结果",
            "difficulty_levels": {}
        }
        
        if hasattr(result, "difficulty_results"):
            for difficulty, metrics in result.difficulty_results.items():
                analysis["difficulty_levels"][difficulty] = {
                    "metrics": metrics,
                    "performance_level": self._get_performance_level(metrics)
                }
        
        return analysis
    
    def _get_performance_level(self, metrics: Dict[str, float]) -> str:
        """获取性能等级"""
        avg_score = sum(metrics.values()) / len(metrics) if metrics else 0
        
        if avg_score >= 0.9:
            return "excellent"
        elif avg_score >= 0.8:
            return "good"
        elif avg_score >= 0.7:
            return "fair"
        elif avg_score >= 0.6:
            return "poor"
        else:
            return "very_poor"
    
    def _generate_baseline_comparison(
        self,
        result: Any,
        baseline: Optional[Any]
    ) -> Dict[str, Any]:
        """生成基线对比"""
        comparison = {
            "has_baseline": baseline is not None,
            "improvements": {},
            "regressions": {},
            "summary": ""
        }
        
        if baseline is None:
            comparison["summary"] = "未提供基线结果进行对比"
            return comparison
        
        if hasattr(result, "metrics") and hasattr(baseline, "metrics"):
            for name, metric_result in result.metrics.items():
                if name in baseline.metrics:
                    current_value = metric_result.value
                    baseline_value = baseline.metrics[name].value
                    change = current_value - baseline_value
                    change_percent = (change / baseline_value * 100) if baseline_value > 0 else 0
                    
                    comparison_data = {
                        "current_value": current_value,
                        "baseline_value": baseline_value,
                        "absolute_change": change,
                        "percent_change": change_percent
                    }
                    
                    if change > 0:
                        comparison["improvements"][name] = comparison_data
                    elif change < 0:
                        comparison["regressions"][name] = comparison_data
        
        if comparison["improvements"]:
            comparison["summary"] = f"相比基线，在{len(comparison['improvements'])}个指标上有提升"
        elif comparison["regressions"]:
            comparison["summary"] = f"相比基线，在{len(comparison['regressions'])}个指标上有下降"
        else:
            comparison["summary"] = "与基线表现相当"
        
        return comparison
    
    def _generate_recommendations(self, result: Any) -> List[Dict[str, Any]]:
        """生成改进建议"""
        recommendations = []
        
        if hasattr(result, "metrics"):
            em_value = result.metrics.get("exact_match")
            if em_value and em_value.value < 0.7:
                recommendations.append({
                    "area": "exact_match",
                    "priority": "high",
                    "recommendation": "精确匹配率较低，建议检查答案格式统一性或增加训练数据多样性",
                    "expected_improvement": "提升5-10%"
                })
            
            f1_value = result.metrics.get("f1")
            if f1_value and f1_value.value < 0.8:
                recommendations.append({
                    "area": "f1_score",
                    "priority": "medium",
                    "recommendation": "F1分数有提升空间，建议优化模型对关键词的识别能力",
                    "expected_improvement": "提升3-5%"
                })
        
        if hasattr(result, "specialty_results"):
            weak_specialties = []
            for specialty, metrics in result.specialty_results.items():
                avg = sum(metrics.values()) / len(metrics) if metrics else 0
                if avg < 0.7:
                    weak_specialties.append(specialty)
            
            if weak_specialties:
                recommendations.append({
                    "area": "specialty_performance",
                    "priority": "high",
                    "recommendation": f"以下专科表现较弱：{', '.join(weak_specialties)}，建议增加专科训练数据",
                    "expected_improvement": "专科准确率提升10-15%"
                })
        
        if hasattr(result, "difficulty_results"):
            hard_metrics = result.difficulty_results.get("hard", {})
            if hard_metrics:
                avg_hard = sum(hard_metrics.values()) / len(hard_metrics)
                if avg_hard < 0.6:
                    recommendations.append({
                        "area": "difficulty_handling",
                        "priority": "medium",
                        "recommendation": "高难度问题表现不佳，建议增加复杂推理训练样本",
                        "expected_improvement": "高难度问题准确率提升15%"
                    })
        
        return recommendations
    
    def _generate_visualization_data(self, result: Any) -> Dict[str, Any]:
        """生成可视化数据"""
        viz_data = {
            "metrics_chart": {
                "type": "bar",
                "data": {}
            },
            "specialty_chart": {
                "type": "grouped_bar",
                "data": {}
            },
            "difficulty_chart": {
                "type": "bar",
                "data": {}
            }
        }
        
        if hasattr(result, "metrics"):
            viz_data["metrics_chart"]["data"] = {
                name: metric.value
                for name, metric in result.metrics.items()
            }
        
        if hasattr(result, "specialty_results"):
            viz_data["specialty_chart"]["data"] = result.specialty_results
        
        if hasattr(result, "difficulty_results"):
            viz_data["difficulty_chart"]["data"] = result.difficulty_results
        
        return viz_data
    
    def save_report(
        self,
        report: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        保存报告到文件
        
        Args:
            report: 报告内容
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        filename = filename or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def generate_comparison_report(
        self,
        results: Dict[str, Any],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        生成多模型对比报告
        
        Args:
            results: 模型名称到评估结果的映射
            model_names: 要对比的模型名称列表
            
        Returns:
            对比报告
        """
        model_names = model_names or list(results.keys())
        
        comparison = {
            "report_info": self._generate_report_info("model_comparison"),
            "models": model_names,
            "metrics_comparison": {},
            "best_model_per_metric": {},
            "ranking": []
        }
        
        all_metrics = set()
        for result in results.values():
            if hasattr(result, "metrics"):
                all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            metric_values = {}
            for model_name in model_names:
                result = results.get(model_name)
                if result and hasattr(result, "metrics") and metric in result.metrics:
                    metric_values[model_name] = result.metrics[metric].value
            
            comparison["metrics_comparison"][metric] = metric_values
            
            if metric_values:
                best_model = max(metric_values.items(), key=lambda x: x[1])
                comparison["best_model_per_metric"][metric] = {
                    "model": best_model[0],
                    "value": best_model[1]
                }
        
        model_scores = {model: 0 for model in model_names}
        for metric, best in comparison["best_model_per_metric"].items():
            model_scores[best["model"]] += 1
        
        ranking = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        comparison["ranking"] = [
            {"rank": i + 1, "model": model, "wins": score}
            for i, (model, score) in enumerate(ranking)
        ]
        
        return comparison
    
    def generate_training_report(
        self,
        training_result: Any,
        evaluation_result: Any,
        config: Any
    ) -> Dict[str, Any]:
        """
        生成训练报告
        
        Args:
            training_result: 训练结果
            evaluation_result: 评估结果
            config: 训练配置
            
        Returns:
            训练报告
        """
        report = {
            "report_info": self._generate_report_info("training_report"),
            "training_summary": {},
            "evaluation_summary": {},
            "resource_usage": {},
            "config": {}
        }
        
        if hasattr(training_result, "to_dict"):
            training_dict = training_result.to_dict()
            report["training_summary"] = {
                "total_steps": training_dict.get("total_steps", 0),
                "total_epochs": training_dict.get("total_epochs", 0),
                "total_time_hours": training_dict.get("total_time_hours", 0),
                "best_eval_em": training_dict.get("best_eval_em", 0),
                "em_improvement": training_dict.get("em_improvement", 0),
                "convergence_improvement": training_dict.get("convergence_improvement", 0)
            }
            report["resource_usage"] = {
                "peak_memory_gb": training_dict.get("peak_memory_gb", 0),
                "within_memory_constraint": training_dict.get("peak_memory_gb", 0) <= 22
            }
        
        if hasattr(evaluation_result, "to_dict"):
            eval_dict = evaluation_result.to_dict()
            report["evaluation_summary"] = eval_dict.get("metrics", {})
        
        if hasattr(config, "to_dict"):
            report["config"] = config.to_dict()
        
        return report
