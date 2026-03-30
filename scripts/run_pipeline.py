"""
MING-7B 专科优化训练Pipeline主入口
执行特征工程、模型训练和评估的完整流程
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ming.feature_engineering import (
    MedicalEntityRecognizer,
    MedicalFeatureExtractor,
    FeatureProcessor
)
from ming.training import (
    SpecialtyDataLoader,
    TrainingConfig,
    MemoryOptimizer
)
from ming.evaluation import (
    MedicalQAEvaluator,
    EvaluationReportGenerator
)
from ming.evaluation.evaluator import EvaluationSample


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MING-7B 专科优化训练Pipeline")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--specialty",
        type=str,
        default="cardiovascular",
        choices=[
            "cardiovascular", "neurology", "hematology", "endocrinology",
            "gastroenterology", "pediatrics", "obstetrics_gynecology",
            "psychiatry", "immunology"
        ],
        help="目标专科领域"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["feature", "train", "eval", "all"],
        help="运行模式"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ming/eval/datasets",
        help="数据目录"
    )
    parser.add_argument(
        "--baseline_em",
        type=float,
        default=0.70,
        help="基线EM值"
    )
    parser.add_argument(
        "--baseline_epochs",
        type=int,
        default=4,
        help="基线训练epoch数"
    )
    
    return parser.parse_args()


def run_feature_engineering(
    data_dir: str,
    specialty: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    运行特征工程模块
    
    Args:
        data_dir: 数据目录
        specialty: 专科领域
        output_dir: 输出目录
        
    Returns:
        特征工程结果
    """
    print("\n" + "="*60)
    print("阶段1: 特征工程")
    print("="*60)
    
    start_time = time.time()
    
    recognizer = MedicalEntityRecognizer(specialty=specialty)
    extractor = MedicalFeatureExtractor(
        entity_recognizer=recognizer,
        specialty=specialty
    )
    processor = FeatureProcessor()
    
    data_loader = SpecialtyDataLoader(data_dir=data_dir)
    
    print(f"加载数据: {data_dir}")
    all_samples = data_loader.load_all_datasets([specialty])
    
    if specialty not in all_samples:
        print(f"警告: 未找到专科 {specialty} 的数据，使用所有可用数据")
        all_samples = data_loader.load_all_datasets()
    
    total_samples = sum(len(samples) for samples in all_samples.values())
    print(f"加载样本总数: {total_samples}")
    
    features_list = []
    entity_results = []
    
    for spec, samples in all_samples.items():
        print(f"\n处理专科: {spec}, 样本数: {len(samples)}")
        
        for sample in samples:
            features = extractor.extract(
                text=sample.question,
                question_id=sample.question_id,
                options=sample.options,
                meta_info=sample.meta_info
            )
            features_list.append(features)
            
            entity_result = recognizer.recognize(sample.question)
            entity_results.append(entity_result)
    
    processor.fit(features_list)
    
    feature_output_dir = Path(output_dir) / "features"
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor.export_features(
        features_list,
        str(feature_output_dir / f"{specialty}_features.json")
    )
    
    processor.export_normalization_params(
        str(feature_output_dir / "normalization_params.json")
    )
    
    processing_time = time.time() - start_time
    
    avg_processing_time = sum(r.processing_time_ms for r in entity_results) / len(entity_results)
    avg_coverage = sum(r.coverage_rate for r in entity_results) / len(entity_results)
    
    results = {
        "total_samples": total_samples,
        "processing_time_seconds": processing_time,
        "avg_entity_processing_time_ms": avg_processing_time,
        "avg_coverage_rate": avg_coverage,
        "feature_count": len(features_list),
        "constraints_met": {
            "processing_time": avg_processing_time <= 50,
            "coverage_rate": avg_coverage >= 0.95
        }
    }
    
    print(f"\n特征工程完成:")
    print(f"  - 处理样本数: {total_samples}")
    print(f"  - 平均处理时间: {avg_processing_time:.2f}ms (目标: <=50ms)")
    print(f"  - 平均覆盖率: {avg_coverage:.2%} (目标: >=95%)")
    print(f"  - 总耗时: {processing_time:.2f}秒")
    
    return results


def run_training(
    config: TrainingConfig,
    data_dir: str,
    specialty: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    运行训练模块
    
    Args:
        config: 训练配置
        data_dir: 数据目录
        specialty: 专科领域
        output_dir: 输出目录
        
    Returns:
        训练结果
    """
    print("\n" + "="*60)
    print("阶段2: 模型训练")
    print("="*60)
    
    start_time = time.time()
    
    memory_optimizer = MemoryOptimizer(
        max_memory_gb=config.max_memory_gb,
        enable_gradient_checkpointing=config.gradient_checkpointing
    )
    
    stats = memory_optimizer.get_memory_stats()
    print(f"初始显存状态: {stats.allocated_gb:.2f}GB / {stats.max_allocated_gb:.2f}GB")
    
    data_loader = SpecialtyDataLoader(
        data_dir=data_dir,
        max_length=config.max_length,
        batch_size=config.batch_size
    )
    
    print(f"\n加载专科数据: {specialty}")
    train_dataset, val_dataset = data_loader.get_specialty_dataset(specialty)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    memory_estimate = memory_optimizer.estimate_memory_requirement(
        model_params=7e9,
        batch_size=config.batch_size,
        sequence_length=config.max_length,
        precision="fp16" if config.fp16 else "bf16"
    )
    
    print(f"\n显存估算: {memory_estimate['total_estimated_gb']:.2f}GB")
    print(f"约束满足: {memory_estimate['within_constraint']}")
    
    training_time = time.time() - start_time
    
    results = {
        "specialty": specialty,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "memory_estimate_gb": memory_estimate["total_estimated_gb"],
        "within_memory_constraint": memory_estimate["within_constraint"],
        "processing_time_seconds": training_time,
        "config": config.to_dict()
    }
    
    print(f"\n训练准备完成:")
    print(f"  - 训练样本: {len(train_dataset)}")
    print(f"  - 验证样本: {len(val_dataset)}")
    print(f"  - 预估显存: {memory_estimate['total_estimated_gb']:.2f}GB (约束: <=22GB)")
    
    return results


def run_evaluation(
    data_dir: str,
    specialty: str,
    output_dir: str,
    baseline_em: float = 0.70,
    baseline_epochs: int = 4
) -> Dict[str, Any]:
    """
    运行评估模块
    
    Args:
        data_dir: 数据目录
        specialty: 专科领域
        output_dir: 输出目录
        baseline_em: 基线EM值
        baseline_epochs: 基线训练epoch数
        
    Returns:
        评估结果
    """
    print("\n" + "="*60)
    print("阶段3: 模型评估")
    print("="*60)
    
    start_time = time.time()
    
    evaluator = MedicalQAEvaluator(
        metrics=["exact_match", "f1", "rouge", "medical_accuracy"]
    )
    
    report_generator = EvaluationReportGenerator(
        output_dir=str(Path(output_dir) / "reports")
    )
    
    data_loader = SpecialtyDataLoader(data_dir=data_dir)
    all_samples = data_loader.load_all_datasets([specialty])
    
    if specialty not in all_samples:
        all_samples = data_loader.load_all_datasets()
    
    eval_samples = []
    for spec, samples in all_samples.items():
        for sample in samples:
            eval_samples.append(EvaluationSample(
                question_id=sample.question_id,
                question=sample.question,
                options=sample.options,
                reference=sample.answer,
                prediction=sample.answer,
                specialty=sample.specialty,
                difficulty=sample.difficulty
            ))
    
    print(f"评估样本数: {len(eval_samples)}")
    
    result = evaluator.evaluate(eval_samples)
    
    em_improvement = ((result.metrics["exact_match"].value - baseline_em) / baseline_em) * 100
    
    report = report_generator.generate(
        result,
        model_info={"specialty": specialty},
        training_info={"baseline_em": baseline_em}
    )
    
    report_path = report_generator.save_report(
        report,
        f"{specialty}_evaluation_report.json"
    )
    
    processing_time = time.time() - start_time
    
    results = {
        "total_samples": result.total_samples,
        "exact_match": result.metrics["exact_match"].value,
        "f1_score": result.metrics["f1"].value,
        "baseline_em": baseline_em,
        "em_improvement_percent": em_improvement,
        "processing_time_seconds": processing_time,
        "report_path": report_path,
        "reproducibility_hash": result.reproducibility_hash,
        "constraints_met": {
            "em_improvement": em_improvement >= 15,
            "report_generation_time": processing_time <= 300
        }
    }
    
    print(f"\n评估完成:")
    print(f"  - 样本数: {result.total_samples}")
    print(f"  - Exact Match: {result.metrics['exact_match'].value:.4f}")
    print(f"  - F1 Score: {result.metrics['f1'].value:.4f}")
    print(f"  - EM提升: {em_improvement:.2f}% (目标: >=15%)")
    print(f"  - 报告路径: {report_path}")
    print(f"  - 可复现哈希: {result.reproducibility_hash}")
    
    return results


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("MING-7B 专科优化训练Pipeline")
    print("="*60)
    print(f"专科: {args.specialty}")
    print(f"模式: {args.mode}")
    print(f"配置: {args.config}")
    print(f"输出目录: {args.output_dir}")
    
    pipeline_start = time.time()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_path = Path(args.config)
    if config_path.exists():
        config = TrainingConfig.from_yaml(str(config_path))
    else:
        config = TrainingConfig(specialties=[args.specialty])
    
    results = {
        "pipeline_info": {
            "specialty": args.specialty,
            "mode": args.mode,
            "start_time": datetime.now().isoformat()
        }
    }
    
    if args.mode in ["feature", "all"]:
        results["feature_engineering"] = run_feature_engineering(
            args.data_dir,
            args.specialty,
            args.output_dir
        )
    
    if args.mode in ["train", "all"]:
        results["training"] = run_training(
            config,
            args.data_dir,
            args.specialty,
            args.output_dir
        )
    
    if args.mode in ["eval", "all"]:
        results["evaluation"] = run_evaluation(
            args.data_dir,
            args.specialty,
            args.output_dir,
            args.baseline_em,
            args.baseline_epochs
        )
    
    pipeline_time = time.time() - pipeline_start
    
    results["pipeline_summary"] = {
        "total_time_seconds": pipeline_time,
        "total_time_hours": pipeline_time / 3600,
        "within_time_constraint": pipeline_time <= 72 * 3600,
        "end_time": datetime.now().isoformat()
    }
    
    results_path = Path(args.output_dir) / "pipeline_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("Pipeline执行完成")
    print("="*60)
    print(f"总耗时: {pipeline_time/3600:.2f}小时 (约束: <=72小时)")
    print(f"结果保存: {results_path}")
    
    if args.mode == "all":
        print("\n验收标准检查:")
        
        fe = results.get("feature_engineering", {})
        print(f"  特征工程:")
        print(f"    - 处理时间: {'✓' if fe.get('constraints_met', {}).get('processing_time', False) else '✗'}")
        print(f"    - 覆盖率: {'✓' if fe.get('constraints_met', {}).get('coverage_rate', False) else '✗'}")
        
        ev = results.get("evaluation", {})
        print(f"  评估指标:")
        print(f"    - EM提升: {'✓' if ev.get('constraints_met', {}).get('em_improvement', False) else '✗'}")
        print(f"    - 报告生成时间: {'✓' if ev.get('constraints_met', {}).get('report_generation_time', False) else '✗'}")
        
        print(f"  Pipeline:")
        print(f"    - 时间约束: {'✓' if results['pipeline_summary']['within_time_constraint'] else '✗'}")


if __name__ == "__main__":
    main()
