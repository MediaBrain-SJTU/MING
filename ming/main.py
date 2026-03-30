#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MING-7B 项目主入口脚本

本脚本提供项目的完整工作流演示：
1. 特征工程处理
2. 专科模型训练
3. 多维度评估
4. 报告生成
"""

# 首先设置编码，解决Windows控制台乱码问题
import sys
import io

# 强制设置UTF-8编码
if sys.platform.startswith('win'):
    try:
        # 检查是否已经是TextIOWrapper，避免重复包装
        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                line_buffering=True,
                errors='replace'
            )
        if not isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding='utf-8',
                line_buffering=True,
                errors='replace'
            )
    except (ValueError, AttributeError):
        # 如果出现错误（如文件已关闭），跳过设置
        pass

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 配置日志 - 使用UTF-8编码的StreamHandler
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.encoding = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[UTF8StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MING-7B 中文医疗大模型专科优化工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["feature", "train", "evaluate", "all"],
        default="all",
        help="运行模式:\n"
        "  feature: 仅运行特征工程\n"
        "  train: 仅运行模型训练\n"
        "  evaluate: 仅运行评估\n"
        "  all: 运行完整工作流 (默认)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="配置文件路径，默认: configs/default_config.yaml",
    )

    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training_config.yaml",
        help="训练配置文件路径，默认: configs/training_config.yaml",
    )

    parser.add_argument(
        "--evaluation-config",
        type=str,
        default="configs/evaluation_config.yaml",
        help="评估配置文件路径，默认: configs/evaluation_config.yaml",
    )

    parser.add_argument(
        "--feature-config",
        type=str,
        default="configs/feature_config.yaml",
        help="特征工程配置文件路径，默认: configs/feature_config.yaml",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="模型路径，覆盖配置文件中的设置",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="输出目录，默认: ./output",
    )

    parser.add_argument(
        "--specialty",
        type=str,
        help="指定专科类型，覆盖配置文件中的设置",
    )

    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="评估时使用采样生成",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式",
    )

    return parser.parse_args()


def run_feature_engineering(
    config_path: str,
    output_dir: str,
    debug: bool = False,
) -> bool:
    """
    运行特征工程模块

    Args:
        config_path: 配置文件路径
        output_dir: 输出目录
        debug: 是否启用调试

    Returns:
        bool: 是否成功完成
    """
    logger.info("=" * 70)
    logger.info("开始运行特征工程模块")
    logger.info("=" * 70)

    try:
        from ming.config import FeatureConfig
        from ming.feature_engineering import (
            MedicalEntityRecognizer,
            FeatureExtractor,
            FeatureSelector,
            batch_feature_extraction,
        )

        # 加载配置
        feature_config = FeatureConfig.from_yaml(config_path)
        if debug:
            logger.info(f"特征配置: {feature_config.to_dict()}")

        # 验证配置
        errors = feature_config.validate()
        if errors:
            logger.error("配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        # 初始化组件
        entity_recognizer = MedicalEntityRecognizer()
        feature_extractor = FeatureExtractor(entity_recognizer=entity_recognizer)
        feature_selector = FeatureSelector()

        logger.info("特征工程组件初始化完成")
        logger.info(f"支持的实体类型: {feature_config.entity_types}")

        # 演示实体识别
        test_texts = [
            "患者因高血压3级入院，伴有头痛、头晕症状",
            "给予阿司匹林肠溶片100mg 口服 qd",
            "心电图示窦性心律，ST-T段改变",
        ]

        logger.info("\n实体识别演示:")
        for text in test_texts:
            entities, elapsed_ms = entity_recognizer.recognize(text)
            logger.info(f"文本: {text}")
            logger.info(
                f"  识别到 {len(entities)} 个实体 (耗时: {elapsed_ms:.2f}ms):"
            )
            for ent in entities:
                logger.info(f"    - [{ent.entity_type.value}] {ent.text}")

        # 批量特征提取演示
        logger.info("\n批量特征提取演示:")
        features, stats = batch_feature_extraction(
            test_texts,
            feature_extractor,
            batch_size=2,
            verbose=True,
        )

        logger.info(f"提取特征统计:")
        logger.info(f"  总样本数: {stats['total_samples']}")
        logger.info(f"  平均耗时: {stats['avg_time_ms']:.2f}ms/条")
        logger.info(f"  特征覆盖率: {stats['coverage']:.2%}")

        # 创建输出目录
        feature_output_dir = Path(output_dir) / "features"
        feature_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n特征工程模块运行完成")
        logger.info(f"输出目录: {feature_output_dir}")

        return True

    except Exception as e:
        logger.error(f"特征工程模块运行失败: {e}", exc_info=debug)
        return False


def run_training(
    config_path: str,
    model_path: Optional[str] = None,
    output_dir: str = "./output",
    specialty: Optional[str] = None,
    debug: bool = False,
) -> bool:
    """
    运行模型训练模块

    Args:
        config_path: 配置文件路径
        model_path: 模型路径
        output_dir: 输出目录
        specialty: 专科类型
        debug: 是否启用调试

    Returns:
        bool: 是否成功完成
    """
    logger.info("=" * 70)
    logger.info("开始运行模型训练模块")
    logger.info("=" * 70)

    try:
        from ming.config import TrainingConfig
        from ming.training import (
            SpecialtyTrainer,
            SpecialtyTrainingArguments,
            MemoryMonitor,
            SpecialtyDataPipeline,
        )

        # 加载配置
        train_config = TrainingConfig.from_yaml(config_path)

        # 覆盖配置
        if model_path:
            train_config.model_name_or_path = model_path
        if specialty:
            train_config.specialty_type = specialty
        train_config.output_dir = str(Path(output_dir) / "training")

        if debug:
            logger.info(f"训练配置: {train_config.to_dict()}")

        # 验证配置
        errors = train_config.validate()
        if errors:
            logger.error("配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info("训练模块初始化完成")
        logger.info(f"专科类型: {train_config.specialty_type}")
        logger.info(f"批量大小: {train_config.per_device_train_batch_size}")
        logger.info(f"学习率: {train_config.learning_rate:.2e}")
        logger.info(f"训练轮数: {train_config.num_train_epochs}")

        # 显存监控演示
        import torch

        if torch.cuda.is_available():
            logger.info("\n显存监控演示:")
            memory_monitor = MemoryMonitor(
                max_batch_size=train_config.per_device_train_batch_size,
                safety_margin_gb=2.0,
            )
            memory_info = memory_monitor.tracker.get_memory_info()
            logger.info(f"  当前显存使用: {memory_info['used_gb']:.2f}GB")
            logger.info(f"  显存利用率: {memory_info['utilization']:.2%}")

        # 创建输出目录
        train_output_dir = Path(output_dir) / "training"
        train_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n模型训练模块运行完成")
        logger.info(f"输出目录: {train_output_dir}")

        return True

    except Exception as e:
        logger.error(f"模型训练模块运行失败: {e}", exc_info=debug)
        return False


def run_evaluation(
    config_path: str,
    model_path: Optional[str] = None,
    output_dir: str = "./output",
    do_sample: bool = False,
    debug: bool = False,
) -> bool:
    """
    运行评估模块

    Args:
        config_path: 配置文件路径
        model_path: 模型路径
        output_dir: 输出目录
        do_sample: 是否使用采样生成
        debug: 是否启用调试

    Returns:
        bool: 是否成功完成
    """
    logger.info("=" * 70)
    logger.info("开始运行评估模块")
    logger.info("=" * 70)

    try:
        from ming.config import EvaluationConfig
        from ming.evaluation import (
            ModelEvaluator,
            ReportGenerator,
            create_default_benchmark,
            EvaluationSample,
        )

        # 加载配置
        eval_config = EvaluationConfig.from_yaml(config_path)

        # 覆盖配置
        if do_sample:
            eval_config.do_sample = True
        eval_config.output_dir = str(Path(output_dir) / "evaluation")

        if debug:
            logger.info(f"评估配置: {eval_config.to_dict()}")

        # 验证配置
        errors = eval_config.validate()
        if errors:
            logger.error("配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info("评估模块初始化完成")
        logger.info(f"评估批量大小: {eval_config.eval_batch_size}")
        logger.info(f"最大生成token数: {eval_config.max_new_tokens}")

        # 创建基准测试
        benchmark = create_default_benchmark(
            data_dir="ming/eval/datasets",
            baseline_em=0.65,
            target_improvement=eval_config.target_improvement,
        )

        logger.info(f"\n基准测试配置:")
        logger.info(f"  名称: {benchmark.name}")
        logger.info(f"  目标专科: {benchmark.specialties}")
        logger.info(f"  目标提升: {benchmark.target_improvement:.0%}")

        # 演示评估样本
        test_samples = [
            EvaluationSample(
                id="test_001",
                question="高血压的治疗方法有哪些？",
                reference="高血压的治疗包括药物治疗和生活方式改变，如低盐饮食、适量运动、控制体重等。",
                specialty="cardiovascular",
            ),
            EvaluationSample(
                id="test_002",
                question="糖尿病患者如何控制血糖？",
                reference="糖尿病患者应通过饮食控制、运动、药物治疗和血糖监测来控制血糖水平。",
                specialty="endocrinology",
            ),
        ]

        logger.info("\n评估演示样本:")
        for sample in test_samples:
            logger.info(f"  ID: {sample.id}, 专科: {sample.specialty}")
            logger.info(f"  问题: {sample.question}")
            logger.info(f"  参考: {sample.reference}")

        # 评估指标演示
        from ming.evaluation import (
            compute_em_score,
            compute_f1_score,
            compute_medical_entity_score,
        )

        predictions = [
            "高血压的治疗包括药物治疗和生活方式改变",
            "糖尿病患者应控制饮食、运动和药物治疗",
        ]
        references = [s.reference for s in test_samples]

        em_score = compute_em_score(predictions, references)
        f1_score, precision, recall = compute_f1_score(predictions, references)

        logger.info("\n指标计算演示:")
        logger.info(f"  EM分数: {em_score:.4f}")
        logger.info(f"  F1分数: {f1_score:.4f}")
        logger.info(f"  精确率: {precision:.4f}")
        logger.info(f"  召回率: {recall:.4f}")

        # 报告生成演示
        report_generator = ReportGenerator(
            output_dir=eval_config.output_dir,
            report_prefix="ming_demo_report",
        )

        logger.info(f"\n报告生成器初始化完成")
        logger.info(f"报告输出目录: {eval_config.output_dir}")

        logger.info("\n评估模块运行完成")
        return True

    except Exception as e:
        logger.error(f"评估模块运行失败: {e}", exc_info=debug)
        return False


def main():
    """主函数"""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")

    logger.info("=" * 70)
    logger.info("MING-7B 中文医疗大模型专科优化工具")
    logger.info("=" * 70)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {args.output_dir}")

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 运行相应模块
    success = True

    if args.mode in ["feature", "all"]:
        success &= run_feature_engineering(
            config_path=args.feature_config,
            output_dir=args.output_dir,
            debug=args.debug,
        )
        if not success:
            logger.error("特征工程模块运行失败")

    if args.mode in ["train", "all"] and success:
        success &= run_training(
            config_path=args.training_config,
            model_path=args.model_path,
            output_dir=args.output_dir,
            specialty=args.specialty,
            debug=args.debug,
        )
        if not success:
            logger.error("模型训练模块运行失败")

    if args.mode in ["evaluate", "all"] and success:
        success &= run_evaluation(
            config_path=args.evaluation_config,
            model_path=args.model_path,
            output_dir=args.output_dir,
            do_sample=args.do_sample,
            debug=args.debug,
        )
        if not success:
            logger.error("评估模块运行失败")

    logger.info("=" * 70)
    if success:
        logger.info("所有模块运行成功完成!")
    else:
        logger.error("部分模块运行失败，请检查错误信息")
    logger.info("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
