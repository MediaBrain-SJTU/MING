#!/usr/bin/env python3
"""
模型评估脚本

使用多维度评估指标体系对模型进行评估。

Usage:
    python scripts/evaluate.py --config configs/eval_config.yaml --model ./output/final
"""

import argparse
import yaml
import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ming.eval.evaluation_suite import (
    EvaluationSuite,
    EvaluationConfig
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_eval_config(config_dict: dict, args) -> EvaluationConfig:
    """从字典创建评估配置"""
    
    config = EvaluationConfig()
    
    # 命令行参数优先
    if args.model:
        config.model_path = args.model
    elif 'model' in config_dict and 'model_path' in config_dict['model']:
        config.model_path = config_dict['model']['model_path']
    
    if args.data:
        config.eval_data_path = args.data
    elif 'data' in config_dict and 'eval_data_path' in config_dict['data']:
        config.eval_data_path = config_dict['data']['eval_data_path']
    
    if args.max_samples:
        config.max_samples = args.max_samples
    elif 'data' in config_dict and 'max_samples' in config_dict['data']:
        config.max_samples = config_dict['data']['max_samples']
    
    # 生成配置
    if 'generation' in config_dict:
        gen_cfg = config_dict['generation']
        config.max_new_tokens = gen_cfg.get('max_new_tokens', config.max_new_tokens)
        config.temperature = gen_cfg.get('temperature', config.temperature)
        config.beam_size = gen_cfg.get('beam_size', config.beam_size)
    
    # 评估维度
    if 'evaluation' in config_dict:
        eval_cfg = config_dict['evaluation']
        config.evaluate_em = eval_cfg.get('evaluate_em', config.evaluate_em)
        config.evaluate_f1 = eval_cfg.get('evaluate_f1', config.evaluate_f1)
        config.evaluate_specialty = eval_cfg.get('evaluate_specialty', config.evaluate_specialty)
        config.evaluate_entity = eval_cfg.get('evaluate_entity', config.evaluate_entity)
        
        if 'target_specialties' in eval_cfg:
            config.target_specialties = eval_cfg['target_specialties']
    
    # 输出配置
    if 'output' in config_dict:
        out_cfg = config_dict['output']
        config.output_dir = out_cfg.get('output_dir', config.output_dir)
        config.save_predictions = out_cfg.get('save_predictions', config.save_predictions)
    
    # 可复现性
    if 'reproducibility' in config_dict:
        rep_cfg = config_dict['reproducibility']
        config.seed = rep_cfg.get('seed', config.seed)
    
    return config


def print_report_summary(report: dict):
    """打印报告摘要"""
    print("\n" + "=" * 60)
    print("评估报告摘要")
    print("=" * 60)
    
    print(f"\n报告ID: {report['report_id']}")
    print(f"评估时间: {report['timestamp']}")
    print(f"模型路径: {report['model_path']}")
    print(f"数据路径: {report['eval_data_path']}")
    
    print("\n总体指标:")
    for metric in report['overall_metrics']:
        print(f"  {metric['name']}: {metric['value']:.4f}")
    
    print("\n专科指标:")
    for spec in report['specialty_metrics']:
        print(f"  {spec['specialty']}: EM={spec['em_score']:.4f}, "
              f"F1={spec['f1_score']:.4f}, 样本数={spec['sample_count']}")
    
    print("\n实体识别指标:")
    entity_metrics = report['entity_metrics']
    print(f"  F1: {entity_metrics['f1']:.4f}")
    print(f"  Precision: {entity_metrics['precision']:.4f}")
    print(f"  Recall: {entity_metrics['recall']:.4f}")
    print(f"  Coverage: {entity_metrics['coverage']:.4%}")
    
    print("\n性能指标:")
    perf = report['performance']
    print(f"  评估时间: {perf['eval_time_seconds']:.2f}秒")
    print(f"  样本/秒: {perf['samples_per_second']:.2f}")
    print(f"  平均推理时间: {perf['avg_inference_time_ms']:.2f}ms")
    
    print("\n关键发现:")
    for finding in report['summary']['key_findings']:
        print(f"  - {finding}")
    
    print("\n" + "=" * 60)


def check_acceptance_criteria(report: dict, config_dict: dict) -> bool:
    """检查验收标准"""
    if 'acceptance_criteria' not in config_dict:
        return True
    
    criteria = config_dict['acceptance_criteria']
    passed = True
    
    print("\n" + "=" * 60)
    print("验收标准检查")
    print("=" * 60)
    
    # 实体识别F1
    entity_f1 = report['entity_metrics']['f1']
    threshold = criteria.get('entity_f1_threshold', 0.92)
    status = "✓ 通过" if entity_f1 >= threshold else "✗ 未通过"
    print(f"  实体识别F1: {entity_f1:.4f} >= {threshold} {status}")
    if entity_f1 < threshold:
        passed = False
    
    # 特征覆盖率
    coverage = report['entity_metrics']['coverage']
    threshold = criteria.get('entity_coverage_threshold', 0.95)
    status = "✓ 通过" if coverage >= threshold else "✗ 未通过"
    print(f"  实体覆盖率: {coverage:.2%} >= {threshold:.2%} {status}")
    if coverage < threshold:
        passed = False
    
    # 报告生成时间
    gen_time = report['performance']['eval_time_seconds'] / 60
    threshold = criteria.get('report_generation_time_min', 5)
    status = "✓ 通过" if gen_time <= threshold else "✗ 未通过"
    print(f"  报告生成时间: {gen_time:.2f}分钟 <= {threshold}分钟 {status}")
    if gen_time > threshold:
        passed = False
    
    print("\n" + "=" * 60)
    if passed:
        print("✓ 所有验收标准通过!")
    else:
        print("✗ 部分验收标准未通过")
    print("=" * 60)
    
    return passed


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='评估医疗大模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件评估
  python scripts/evaluate.py --config configs/eval_config.yaml
  
  # 指定模型和数据
  python scripts/evaluate.py --model ./output/final --data ./data/eval.jsonl
  
  # 限制评估样本数
  python scripts/evaluate.py --config configs/eval_config.yaml --max-samples 100
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/eval_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='模型路径'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='评估数据路径'
    )
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=None,
        help='最大评估样本数'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录'
    )
    parser.add_argument(
        '--check-criteria',
        action='store_true',
        help='检查验收标准'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config_dict = load_config(args.config)
    
    # 创建评估配置
    config = create_eval_config(config_dict, args)
    
    if args.output:
        config.output_dir = args.output
    
    # 验证配置
    if not config.model_path:
        logger.error("错误: 未指定模型路径")
        print("请使用 --model 参数指定模型路径，或在配置文件中设置")
        return 1
    
    if not os.path.exists(config.model_path):
        logger.error(f"错误: 模型路径不存在: {config.model_path}")
        return 1
    
    if not config.eval_data_path:
        logger.error("错误: 未指定评估数据路径")
        print("请使用 --data 参数指定数据路径，或在配置文件中设置")
        return 1
    
    if not os.path.exists(config.eval_data_path):
        logger.error(f"错误: 评估数据路径不存在: {config.eval_data_path}")
        return 1
    
    # 打印配置
    logger.info("=" * 50)
    logger.info("评估配置:")
    logger.info(f"  模型: {config.model_path}")
    logger.info(f"  数据: {config.eval_data_path}")
    logger.info(f"  最大样本数: {config.max_samples or '全部'}")
    logger.info(f"  输出目录: {config.output_dir}")
    logger.info(f"  随机种子: {config.seed}")
    logger.info("=" * 50)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 创建评估套件
    logger.info("初始化评估套件...")
    try:
        suite = EvaluationSuite(config)
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return 1
    
    # 执行评估
    logger.info("开始评估...")
    try:
        report = suite.evaluate()
        
        # 保存报告
        report_path = report.save()
        
        # 打印摘要
        report_dict = report.to_dict()
        print_report_summary(report_dict)
        
        logger.info(f"评估报告已保存: {report_path}")
        
        # 检查验收标准
        if args.check_criteria:
            passed = check_acceptance_criteria(report_dict, config_dict)
            return 0 if passed else 1
        
        return 0
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
