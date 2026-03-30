"""
训练收敛优化器模块

本模块提供多种收敛优化策略，旨在提升训练收敛速度至少20%：
1. 动态学习率调度
2. 梯度裁剪与归一化
3. 早停机制
4. 模型权重平均（EMA/SWA）
5. 混合精度训练优化
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    """收敛配置类"""

    # 学习率调度
    lr_scheduler_type: str = "cosine_with_warmup"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    num_training_steps: int = 1000
    num_cycles: float = 0.5
    power: float = 1.0

    # 梯度控制
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    clip_gradients: bool = True

    # 早停
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    use_early_stopping: bool = True

    # 权重平均
    use_ema: bool = False
    ema_decay: float = 0.999
    use_swa: bool = False
    swa_start: float = 0.75
    swa_freq: int = 5

    # 优化器
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # 学习率
    learning_rate: float = 2e-5
    min_learning_rate: float = 1e-7


class EMA:
    """
    指数移动平均（Exponential Moving Average）

    对模型权重进行指数移动平均，提高模型的稳定性和泛化能力。
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        """
        初始化EMA

        Args:
            model: 目标模型
            decay: EMA衰减率
            device: 设备
        """
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}

        # 初始化影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().detach()
                if device:
                    self.shadow_params[name] = self.shadow_params[name].to(device)

    def step(self) -> None:
        """更新影子参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow_params
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow_params[name]
                self.shadow_params[name] = new_average

    def apply_shadow(self) -> None:
        """应用影子参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow_params
                self.backup_params[name] = param.data.clone()
                param.data = self.shadow_params[name]

    def restore(self) -> None:
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup_params
                param.data = self.backup_params[name]
        self.backup_params = {}


class SWA:
    """
    随机权重平均（Stochastic Weight Averaging）

    在训练的后期阶段对多个检查点的权重进行平均，提高泛化能力。
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start_step: int,
        swa_freq: int = 5,
        device: Optional[torch.device] = None,
    ):
        """
        初始化SWA

        Args:
            model: 目标模型
            swa_start_step: 开始SWA的步数
            swa_freq: SWA更新频率
            device: 设备
        """
        self.model = model
        self.swa_start_step = swa_start_step
        self.swa_freq = swa_freq
        self.device = device

        self.swa_params: Dict[str, torch.Tensor] = {}
        self.swa_count = 0
        self._initialized = False

    def _initialize(self) -> None:
        """初始化SWA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.swa_params[name] = param.data.clone().detach()
                if self.device:
                    self.swa_params[name] = self.swa_params[name].to(self.device)
        self._initialized = True

    def step(self, current_step: int) -> bool:
        """
        更新SWA参数

        Args:
            current_step: 当前训练步数

        Returns:
            bool: 是否执行了更新
        """
        if current_step < self.swa_start_step:
            return False

        if (current_step - self.swa_start_step) % self.swa_freq != 0:
            return False

        if not self._initialized:
            self._initialize()

        # 更新平均参数
        self.swa_count += 1
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.swa_params[name] *= (self.swa_count - 1) / self.swa_count
                self.swa_params[name] += param.data / self.swa_count

        logger.debug(f"SWA更新完成，当前平均模型数: {self.swa_count}")
        return True

    def apply_swa(self) -> None:
        """应用SWA参数到模型"""
        if not self._initialized:
            logger.warning("SWA尚未初始化，无法应用")
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.swa_params[name]

        logger.info(f"SWA参数已应用，共平均 {self.swa_count} 个模型")


class EarlyStopping:
    """
    早停机制

    监控验证集性能，当性能不再提升时提前停止训练，防止过拟合。
    """

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ):
        """
        初始化早停

        Args:
            patience: 耐心次数（连续多少次没有提升就停止）
            threshold: 提升阈值
            mode: 'max'表示指标越大越好，'min'表示越小越好
            verbose: 是否输出日志
        """
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.early_stop = False
        self.best_state: Optional[Dict[str, Any]] = None

    def __call__(
        self,
        score: float,
        model: Optional[nn.Module] = None,
        step: int = 0,
    ) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前分数
            model: 模型实例（用于保存最佳状态）
            step: 当前步数

        Returns:
            bool: 是否应该早停
        """
        improved = False

        if self.mode == "max":
            if score > self.best_score + self.threshold:
                improved = True
        else:
            if score < self.best_score - self.threshold:
                improved = True

        if improved:
            self.best_score = score
            self.counter = 0
            if model:
                self.best_state = {
                    k: v.clone().cpu() for k, v in model.state_dict().items()
                }
            if self.verbose:
                logger.info(f"性能提升，新最佳分数: {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"性能未提升，当前计数: {self.counter}/{self.patience}, "
                    f"最佳分数: {self.best_score:.6f}"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.warning(f"早停触发，已连续 {self.patience} 次未提升")

        return self.early_stop

    def restore_best_state(self, model: nn.Module) -> bool:
        """
        恢复最佳状态到模型

        Args:
            model: 模型实例

        Returns:
            bool: 是否成功恢复
        """
        if self.best_state is None:
            logger.warning("没有保存的最佳状态")
            return False

        model.load_state_dict(self.best_state)
        logger.info("已恢复最佳状态")
        return True


class GradientClipper:
    """
    梯度裁剪器

    对梯度进行裁剪，防止梯度爆炸。
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        verbose: bool = False,
    ):
        """
        初始化梯度裁剪器

        Args:
            max_norm: 最大梯度范数
            norm_type: 范数类型
            verbose: 是否输出日志
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.verbose = verbose

        self.clip_count = 0
        self.gradient_norm_history: List[float] = []

    def __call__(
        self,
        parameters: List[nn.Parameter],
        step: int = 0,
    ) -> Tuple[float, bool]:
        """
        执行梯度裁剪

        Args:
            parameters: 模型参数列表
            step: 当前步数

        Returns:
            Tuple[float, bool]: 梯度范数和是否被裁剪
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )

        clipped = False
        if total_norm > self.max_norm:
            clipped = True
            self.clip_count += 1
            if self.verbose:
                logger.warning(
                    f"梯度裁剪已执行，梯度范数: {total_norm:.4f} > {self.max_norm}"
                )

        self.gradient_norm_history.append(float(total_norm))
        return float(total_norm), clipped

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.gradient_norm_history:
            return {}

        import numpy as np

        norms = self.gradient_norm_history
        return {
            "avg_gradient_norm": sum(norms) / len(norms),
            "max_gradient_norm": max(norms),
            "min_gradient_norm": min(norms),
            "clip_ratio": self.clip_count / len(norms),
            "clip_count": self.clip_count,
            "total_steps": len(norms),
        }


class ConvergenceOptimizer:
    """
    收敛优化器

    集成多种收敛优化策略，统一管理：
    1. 学习率调度
    2. 梯度裁剪
    3. 早停机制
    4. EMA/SWA权重平均
    """

    def __init__(
        self,
        model: nn.Module,
        config: ConvergenceConfig,
        device: Optional[torch.device] = None,
    ):
        """
        初始化收敛优化器

        Args:
            model: 目标模型
            config: 收敛配置
            device: 设备
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 组件
        self.optimizer: Optional[optim.Optimizer] = None
        self.lr_scheduler: Optional[LRScheduler] = None
        self.grad_clipper: Optional[GradientClipper] = None
        self.ema: Optional[EMA] = None
        self.swa: Optional[SWA] = None
        self.early_stopping: Optional[EarlyStopping] = None

        # 状态
        self.current_step = 0
        self.current_epoch = 0
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.train_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}

        # 初始化组件
        self._initialize()

    def _initialize(self) -> None:
        """初始化所有组件"""
        # 1. 初始化优化器
        self._setup_optimizer()

        # 2. 初始化学习率调度器
        self._setup_lr_scheduler()

        # 3. 初始化梯度裁剪器
        if self.config.clip_gradients:
            self.grad_clipper = GradientClipper(
                max_norm=self.config.max_grad_norm,
            )

        # 4. 初始化EMA
        if self.config.use_ema:
            self.ema = EMA(
                self.model,
                decay=self.config.ema_decay,
                device=self.device,
            )

        # 5. 初始化SWA
        if self.config.use_swa:
            swa_start_step = int(self.config.num_training_steps * self.config.swa_start)
            self.swa = SWA(
                self.model,
                swa_start_step=swa_start_step,
                swa_freq=self.config.swa_freq,
                device=self.device,
            )

        # 6. 初始化早停
        if self.config.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                threshold=self.config.early_stopping_threshold,
            )

        logger.info("收敛优化器初始化完成")

    def _setup_optimizer(self) -> None:
        """设置优化器"""
        # 分离权重衰减参数和非权重衰减参数
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        elif self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.config.optimizer_type}")

        logger.info(f"优化器已设置: {self.config.optimizer_type}")

    def _setup_lr_scheduler(self) -> None:
        """设置学习率调度器"""
        if not self.optimizer:
            raise RuntimeError("优化器未初始化，无法设置学习率调度器")

        # 计算warmup步数
        warmup_steps = self.config.warmup_steps
        if self.config.warmup_ratio > 0:
            warmup_steps = int(
                self.config.num_training_steps * self.config.warmup_ratio
            )

        scheduler_type = self.config.lr_scheduler_type.lower()

        if scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.config.num_training_steps,
            )
        elif scheduler_type == "cosine_with_warmup":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.config.num_training_steps,
                num_cycles=self.config.num_cycles,
            )
        elif scheduler_type == "polynomial":
            self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.config.num_training_steps,
                power=self.config.power,
                lr_end=self.config.min_learning_rate,
            )
        elif scheduler_type == "cosine_annealing":
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_training_steps,
                eta_min=self.config.min_learning_rate,
            )
        elif scheduler_type == "reduce_on_plateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=self.config.min_learning_rate,
            )
        else:
            raise ValueError(f"不支持的学习率调度器类型: {scheduler_type}")

        logger.info(
            f"学习率调度器已设置: {scheduler_type}, warmup_steps={warmup_steps}"
        )

    def get_current_lr(self) -> float:
        """获取当前学习率"""
        if not self.optimizer:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def backward_step(
        self,
        loss: torch.Tensor,
        accumulate: bool = False,
    ) -> Tuple[Optional[float], Optional[bool]]:
        """
        反向传播步骤

        Args:
            loss: 损失张量
            accumulate: 是否累积梯度

        Returns:
            Tuple[Optional[float], Optional[bool]]: 梯度范数和是否被裁剪
        """
        if not self.optimizer:
            raise RuntimeError("优化器未初始化")

        # 缩放损失（用于梯度累积）
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        # 反向传播
        loss.backward()

        grad_norm: Optional[float] = None
        clipped: Optional[bool] = None

        # 如果不需要累积，执行优化步骤
        if not accumulate:
            # 梯度裁剪
            if self.grad_clipper:
                grad_norm, clipped = self.grad_clipper(
                    self.model.parameters(),
                    step=self.current_step,
                )

            # 优化器更新
            self.optimizer.step()

            # 学习率更新
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau需要metrics
                    pass
                else:
                    self.lr_scheduler.step()

            # 清空梯度
            self.optimizer.zero_grad()

            # 更新EMA
            if self.ema:
                self.ema.step()

            # 更新SWA
            if self.swa:
                self.swa.step(self.current_step)

            self.current_step += 1

        return grad_norm, clipped

    def validation_step(
        self,
        val_loss: float,
        val_metric: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        验证后处理

        Args:
            val_loss: 验证损失
            val_metric: 验证指标（如准确率）

        Returns:
            Dict[str, Any]: 验证结果和早停状态
        """
        self.val_loss_history.append(val_loss)

        # 更新ReduceLROnPlateau调度器
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(val_loss)

        result = {
            "early_stop": False,
            "best_score": None,
            "lr_reduced": False,
        }

        # 早停检查
        if self.early_stopping:
            early_stop_score = val_metric if val_metric is not None else -val_loss
            result["early_stop"] = self.early_stopping(
                early_stop_score,
                model=self.model,
                step=self.current_step,
            )
            result["best_score"] = self.early_stopping.best_score

        return result

    def apply_final_averaging(self) -> None:
        """应用最终的权重平均（EMA或SWA）"""
        if self.ema:
            self.ema.apply_shadow()
            logger.info("已应用EMA权重")

        if self.swa:
            self.swa.apply_swa()
            logger.info("已应用SWA权重")

    def restore_best_model(self) -> bool:
        """恢复最佳模型状态"""
        if self.early_stopping:
            return self.early_stopping.restore_best_state(self.model)
        return False

    def get_optimizer(self) -> optim.Optimizer:
        """获取优化器"""
        if not self.optimizer:
            raise RuntimeError("优化器未初始化")
        return self.optimizer

    def get_lr_scheduler(self) -> LRScheduler:
        """获取学习率调度器"""
        if not self.lr_scheduler:
            raise RuntimeError("学习率调度器未初始化")
        return self.lr_scheduler

    def get_stats(self) -> Dict[str, Any]:
        """获取收敛统计信息"""
        stats = {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "current_lr": self.get_current_lr(),
            "num_training_steps": self.config.num_training_steps,
        }

        if self.train_loss_history:
            stats["avg_train_loss"] = sum(self.train_loss_history) / len(
                self.train_loss_history
            )
            stats["min_train_loss"] = min(self.train_loss_history)

        if self.val_loss_history:
            stats["avg_val_loss"] = sum(self.val_loss_history) / len(
                self.val_loss_history
            )
            stats["min_val_loss"] = min(self.val_loss_history)

        if self.grad_clipper:
            stats["gradient_stats"] = self.grad_clipper.get_stats()

        if self.early_stopping:
            stats["early_stopping"] = {
                "counter": self.early_stopping.counter,
                "best_score": self.early_stopping.best_score,
                "early_stop": self.early_stopping.early_stop,
            }

        return stats


def main():
    """测试函数"""
    # 创建一个简单的模型
    model = nn.Linear(10, 2)

    # 创建收敛配置
    config = ConvergenceConfig(
        lr_scheduler_type="cosine_with_warmup",
        learning_rate=2e-5,
        num_training_steps=100,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        use_ema=True,
        use_early_stopping=True,
        early_stopping_patience=3,
    )

    # 创建收敛优化器
    optimizer = ConvergenceOptimizer(model, config)

    # 模拟训练
    print("收敛优化器测试:")
    print(f"初始学习率: {optimizer.get_current_lr():.8f}")

    # 模拟几个训练步骤
    for step in range(10):
        # 模拟前向传播
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.mean()

        # 反向传播
        grad_norm, clipped = optimizer.backward_step(loss)

        if step % 2 == 0:
            print(f"步骤 {step}: lr={optimizer.get_current_lr():.8f}")

    # 获取统计信息
    stats = optimizer.get_stats()
    print("\n统计信息:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # 应用权重平均
    optimizer.apply_final_averaging()


if __name__ == "__main__":
    main()
