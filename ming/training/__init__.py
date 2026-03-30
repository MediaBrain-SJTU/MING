"""
训练Pipeline模块初始化文件

本模块提供优化的模型训练Pipeline，支持专科领域定向微调，包括：
1. 专科数据加载与预处理
2. 显存优化训练策略
3. 训练过程监控与显存控制
4. 快速收敛优化算法
"""

from ming.training.specialty_trainer import (
    SpecialtyTrainer,
    SpecialtyTrainingArguments,
    MemoryOptimizedTrainer,
    compute_basic_metrics,
    TrainerState,
)
from ming.training.convergence_optimizer import (
    ConvergenceOptimizer,
    ConvergenceConfig,
    EMA,
    SWA,
    EarlyStopping,
    GradientClipper,
)
from ming.training.data_pipeline import (
    SpecialtyDataPipeline,
    SpecialtyDataset,
    SpecialtySample,
    SpecialtySampler,
    DataCollatorForSpecialty,
)
from ming.training.memory_monitor import MemoryMonitor, GPUMemoryTracker

__all__ = [
    "SpecialtyTrainer",
    "SpecialtyTrainingArguments",
    "MemoryOptimizedTrainer",
    "compute_basic_metrics",
    "TrainerState",
    "ConvergenceOptimizer",
    "ConvergenceConfig",
    "EMA",
    "SWA",
    "EarlyStopping",
    "GradientClipper",
    "SpecialtyDataPipeline",
    "SpecialtyDataset",
    "SpecialtySample",
    "SpecialtySampler",
    "DataCollatorForSpecialty",
    "MemoryMonitor",
    "GPUMemoryTracker",
]
