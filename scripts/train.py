#!/usr/bin/env python3
"""
模型训练脚本

使用优化后的训练Pipeline进行模型微调。

Usage:
    python scripts/train.py --config configs/training_config.yaml --specialty 心血管
"""

import argparse
import yaml
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ming.train.optimized_trainer import (
    OptimizedTrainer,
    OptimizedTrainingConfig
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


def merge_config(base_config: dict, overrides: dict) -> dict:
    """合并配置"""
    merged = base_config.copy()
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def create_training_config(config_dict: dict) -> OptimizedTrainingConfig:
    """从字典创建训练配置"""
    
    config = OptimizedTrainingConfig()
    
    # 模型配置
    if 'model' in config_dict:
        config.model_name_or_path = config_dict['model'].get(
            'model_name_or_path', config.model_name_or_path
        )
    
    # 数据配置
    if 'data' in config_dict:
        config.train_data_path = config_dict['data'].get(
            'train_data_path', config.train_data_path
        )
        config.eval_data_path = config_dict['data'].get(
            'eval_data_path', config.eval_data_path
        )
        config.max_seq_length = config_dict['data'].get(
            'max_seq_length', config.max_seq_length
        )
    
    # LoRA配置
    if 'lora' in config_dict:
        lora_cfg = config_dict['lora']
        config.use_lora = lora_cfg.get('use_lora', config.use_lora)
        config.lora_r = lora_cfg.get('r', config.lora_r)
        config.lora_alpha = lora_cfg.get('alpha', config.lora_alpha)
        config.lora_dropout = lora_cfg.get('dropout', config.lora_dropout)
        config.lora_target_modules = lora_cfg.get(
            'target_modules', config.lora_target_modules
        )
    
    # 量化配置
    if 'quantization' in config_dict:
        quant_cfg = config_dict['quantization']
        config.load_in_4bit = quant_cfg.get('load_in_4bit', config.load_in_4bit)
        config.load_in_8bit = quant_cfg.get('load_in_8bit', config.load_in_8bit)
    
    # 训练配置
    if 'training' in config_dict:
        train_cfg = config_dict['training']
        config.num_train_epochs = train_cfg.get(
            'num_train_epochs', config.num_train_epochs
        )
        config.per_device_train_batch_size = train_cfg.get(
            'per_device_train_batch_size', config.per_device_train_batch_size
        )
        config.gradient_accumulation_steps = train_cfg.get(
            'gradient_accumulation_steps', config.gradient_accumulation_steps
        )
        config.learning_rate = train_cfg.get(
            'learning_rate', config.learning_rate
        )
        config.weight_decay = train_cfg.get('weight_decay', config.weight_decay)
        config.warmup_ratio = train_cfg.get('warmup_ratio', config.warmup_ratio)
    
    # 内存优化配置
    if 'memory_optimization' in config_dict:
        mem_cfg = config_dict['memory_optimization']
        config.gradient_checkpointing = mem_cfg.get(
            'gradient_checkpointing', config.gradient_checkpointing
        )
        config.max_memory_mb = mem_cfg.get('max_memory_mb', config.max_memory_mb)
        config.optim = mem_cfg.get('optim', config.optim)
    
    # 保存配置
    if 'checkpoint' in config_dict:
        chk_cfg = config_dict['checkpoint']
        config.output_dir = chk_cfg.get('output_dir', config.output_dir)
        config.save_steps = chk_cfg.get('save_steps', config.save_steps)
        config.eval_steps = chk_cfg.get('eval_steps', config.eval_steps)
    
    # 早停配置
    if 'early_stopping' in config_dict:
        es_cfg = config_dict['early_stopping']
        config.early_stopping_patience = es_cfg.get(
            'patience', config.early_stopping_patience
        )
        config.early_stopping_threshold = es_cfg.get(
            'threshold', config.early_stopping_threshold
        )
    
    # 专科配置
    if 'specialty' in config_dict:
        spec_cfg = config_dict['specialty']
        config.feature_engineering = spec_cfg.get(
            'feature_engineering', config.feature_engineering
        )
    
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='训练医疗大模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件训练
  python scripts/train.py --config configs/training_config.yaml
  
  # 指定专科训练
  python scripts/train.py --config configs/training_config.yaml --specialty 心血管
  
  # 覆盖配置参数
  python scripts/train.py --config configs/training_config.yaml --epochs 5 --lr 1e-4
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/training_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--specialty', '-s',
        type=str,
        default=None,
        help='专科领域（如：心血管、神经内科）'
    )
    parser.add_argument(
        '--epochs',
        type=float,
        default=None,
        help='训练轮数'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学习率'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录'
    )
    parser.add_argument(
        '--train-data',
        type=str,
        default=None,
        help='训练数据路径'
    )
    parser.add_argument(
        '--eval-data',
        type=str,
        default=None,
        help='评估数据路径'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config_dict = load_config(args.config)
    
    # 命令行参数覆盖
    overrides = {
        'specialty': args.specialty,
        'num_train_epochs': args.epochs,
        'learning_rate': args.lr,
        'per_device_train_batch_size': args.batch_size,
        'output_dir': args.output_dir,
        'train_data_path': args.train_data,
        'eval_data_path': args.eval_data,
    }
    
    # 创建训练配置
    config = create_training_config(config_dict)
    
    # 应用命令行覆盖
    if args.specialty:
        config.specialty_focus = args.specialty
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.train_data:
        config.train_data_path = args.train_data
    if args.eval_data:
        config.eval_data_path = args.eval_data
    
    # 打印配置
    logger.info("=" * 50)
    logger.info("训练配置:")
    logger.info(f"  模型: {config.model_name_or_path}")
    logger.info(f"  训练数据: {config.train_data_path}")
    logger.info(f"  评估数据: {config.eval_data_path}")
    logger.info(f"  专科聚焦: {config.specialty_focus or '无'}")
    logger.info(f"  训练轮数: {config.num_train_epochs}")
    logger.info(f"  学习率: {config.learning_rate}")
    logger.info(f"  批次大小: {config.per_device_train_batch_size}")
    logger.info(f"  输出目录: {config.output_dir}")
    logger.info(f"  使用LoRA: {config.use_lora}")
    logger.info(f"  4-bit量化: {config.load_in_4bit}")
    logger.info(f"  梯度检查点: {config.gradient_checkpointing}")
    logger.info(f"  最大显存: {config.max_memory_mb}MB")
    logger.info("=" * 50)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 创建训练器
    logger.info("初始化训练器...")
    trainer = OptimizedTrainer(config)
    
    # 打印显存信息
    memory_info = trainer.get_memory_usage()
    if memory_info:
        logger.info("初始显存使用:")
        for key, value in memory_info.items():
            logger.info(f"  {key}: {value:.2f}MB")
    
    # 开始训练
    logger.info("开始训练...")
    try:
        results = trainer.train()
        
        logger.info("=" * 50)
        logger.info("训练完成!")
        logger.info(f"训练时间: {results['training_time']:.2f}秒")
        logger.info(f"最终损失: {results['final_loss']:.4f}")
        if results['best_eval_loss'] < float('inf'):
            logger.info(f"最佳评估损失: {results['best_eval_loss']:.4f}")
        logger.info("=" * 50)
        
        # 保存训练结果
        results_path = os.path.join(config.output_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"训练结果已保存: {results_path}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == '__main__':
    main()
