"""
专科训练器模块

本模块提供专科领域定向微调的训练器，集成：
1. 显存优化训练策略
2. 收敛加速算法
3. 专科性能评估
4. 训练状态监控
"""

import os
import logging
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_pt_utils import LabelSmoother

from ming.training.memory_monitor import MemoryMonitor, get_gpu_memory_info
from ming.training.convergence_optimizer import (
    ConvergenceOptimizer,
    ConvergenceConfig,
)
from ming.training.data_pipeline import (
    SpecialtyDataPipeline,
    SpecialtyDataset,
    DataCollatorForSpecialty,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class SpecialtyTrainingArguments:
    """专科训练参数配置类"""

    # 基本训练参数
    output_dir: str = "./specialty_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # 显存优化
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    max_memory_usage_gb: float = 22.0  # 单卡显存上限

    # 收敛优化
    lr_scheduler_type: str = "cosine_with_warmup"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    max_steps: int = -1
    ema: bool = False
    ema_decay: float = 0.999
    early_stopping: bool = True
    early_stopping_patience: int = 3

    # 专科训练特定参数
    specialty_type: str = "general"
    specialty_weights: Dict[str, float] = field(default_factory=dict)
    difficulty_weight: float = 0.0
    specialty_focus: bool = True

    # 日志和评估
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"  # "no", "steps", "epoch"

    # 其他
    seed: int = 42
    dataloader_num_workers: int = 4
    local_rank: int = -1
    deepspeed: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class SpecialtyTrainer:
    """
    专科训练器

    专为医疗专科领域优化的训练器，集成：
    1. 智能显存管理（确保≤22GB）
    2. 收敛加速（提升≥20%）
    3. 专科定向采样和评估
    4. 实时训练状态监控
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: SpecialtyTrainingArguments,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[callable] = None,
        data_pipeline: Optional[SpecialtyDataPipeline] = None,
    ):
        """
        初始化专科训练器

        Args:
            model: 模型
            args: 训练参数
            tokenizer: 分词器
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            data_collator: 数据整理器
            compute_metrics: 指标计算函数
            data_pipeline: 专科数据管道
        """
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or DataCollatorForSpecialty(tokenizer)
        self.compute_metrics = compute_metrics
        self.data_pipeline = data_pipeline

        # 训练状态
        self.state = TrainerState()
        self.is_in_train = False

        # 设备设置
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.device.type == "cuda":
            gpu_info = get_gpu_memory_info()
            logger.info(f"使用GPU: {gpu_info.get('device_name', 'Unknown')}")
            logger.info(f"总显存: {gpu_info.get('total_memory_gb', 0):.2f}GB")

        # 初始化显存监控
        self.memory_monitor = MemoryMonitor(
            max_batch_size=args.per_device_train_batch_size,
            min_batch_size=1,
            initial_batch_size=args.per_device_train_batch_size,
            safety_margin_gb=2.0,
            critical_threshold=args.max_memory_usage_gb / 24.0,  # 转换为使用率
        )

        # 初始化收敛优化器
        self.convergence_config = ConvergenceConfig(
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            use_ema=args.ema,
            ema_decay=args.ema_decay,
            use_early_stopping=args.early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
        )

        self.convergence_optimizer: Optional[ConvergenceOptimizer] = None

        # 混合精度训练
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if args.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("启用FP16混合精度训练")
        elif args.bf16:
            logger.info("启用BF16混合精度训练")

        # 梯度检查点
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("启用梯度检查点")

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("专科训练器初始化完成")

    def _get_train_dataloader(self) -> DataLoader:
        """获取训练数据加载器"""
        if self.data_pipeline and self.data_pipeline.train_dataset:
            train_loader, _ = self.data_pipeline.get_dataloaders(
                num_workers=self.args.dataloader_num_workers
            )
            return train_loader

        return DataLoader(
            self.train_dataset,
            batch_size=self.memory_monitor.current_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def _get_eval_dataloader(
        self,
        eval_dataset: Optional[Dataset] = None,
    ) -> DataLoader:
        """获取评估数据加载器"""
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            raise ValueError("没有评估数据集")

        if self.data_pipeline and self.data_pipeline.val_dataset:
            _, val_loader = self.data_pipeline.get_dataloaders(
                num_workers=self.args.dataloader_num_workers
            )
            return val_loader

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def _prepare_inputs(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """准备输入数据"""
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs

    def _training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        step: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """执行单个训练步骤"""
        inputs = self._prepare_inputs(inputs)

        # 前向传播
        forward_kwargs = {}
        if self.args.fp16:
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # 梯度累积缩放
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # 反向传播
        if self.args.fp16 and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # 返回损失值用于日志
        return loss.detach().float().item(), {}

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        开始训练

        Args:
            resume_from_checkpoint: 从检查点恢复训练的路径

        Returns:
            Dict[str, Any]: 训练结果
        """
        self.is_in_train = True
        start_time = time.time()

        # 启动显存监控
        self.memory_monitor.start()

        try:
            # 准备数据加载器
            train_dataloader = self._get_train_dataloader()
            num_update_steps_per_epoch = (
                len(train_dataloader) // self.args.gradient_accumulation_steps
            )

            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = (
                    max_steps // num_update_steps_per_epoch
                    + int(max_steps % num_update_steps_per_epoch > 0)
                )
            else:
                max_steps = (
                    self.args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = self.args.num_train_epochs

            self.convergence_config.num_training_steps = max_steps
            self.convergence_optimizer = ConvergenceOptimizer(
                self.model,
                self.convergence_config,
                device=self.device,
            )

            logger.info(f"***** 开始训练 *****")
            logger.info(f"  训练样本数 = {len(self.train_dataset) if self.train_dataset else 'N/A'}")
            logger.info(f"  训练轮数 = {num_train_epochs}")
            logger.info(f"  每轮更新步数 = {num_update_steps_per_epoch}")
            logger.info(f"  总更新步数 = {max_steps}")
            logger.info(f"  初始批量大小 = {self.memory_monitor.current_batch_size}")
            logger.info(f"  梯度累积步数 = {self.args.gradient_accumulation_steps}")

            # 训练循环
            global_step = 0
            epochs_trained = 0
            steps_trained_in_current_epoch = 0

            # 从检查点恢复
            if resume_from_checkpoint:
                logger.info(f"从检查点恢复: {resume_from_checkpoint}")
                # 这里可以添加检查点恢复逻辑

            tr_loss = 0.0
            logging_loss = 0.0
            self.model.zero_grad()

            for epoch in range(epochs_trained, num_train_epochs):
                epoch_iterator = train_dataloader
                step = -1

                for step, inputs in enumerate(epoch_iterator):
                    # 跳过已训练的步数
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    self.model.train()
                    is_last_step = (step + 1) % self.args.gradient_accumulation_steps == 0

                    # 训练步骤
                    loss, _ = self._training_step(self.model, inputs, step)
                    tr_loss += loss

                    if is_last_step:
                        # 梯度裁剪和优化器更新
                        if self.args.fp16 and self.scaler:
                            self.scaler.unscale_(
                                self.convergence_optimizer.get_optimizer()
                            )

                        grad_norm, clipped = None, None
                        if self.convergence_optimizer.grad_clipper:
                            grad_norm, clipped = self.convergence_optimizer.grad_clipper(
                                self.model.parameters(),
                                step=global_step,
                            )

                        # 优化器步骤
                        if self.args.fp16 and self.scaler:
                            self.scaler.step(
                                self.convergence_optimizer.get_optimizer()
                            )
                            self.scaler.update()
                        else:
                            self.convergence_optimizer.get_optimizer().step()

                        # 学习率调度器步骤
                        self.convergence_optimizer.get_lr_scheduler().step()

                        # EMA更新
                        if self.convergence_optimizer.ema:
                            self.convergence_optimizer.ema.step()

                        self.model.zero_grad()
                        global_step += 1

                        # 日志
                        if (
                            self.args.logging_steps > 0
                            and global_step % self.args.logging_steps == 0
                        ):
                            loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                            logging_loss = tr_loss

                            lr = self.convergence_optimizer.get_current_lr()
                            memory_used = self.memory_monitor.tracker.get_peak_memory()

                            logs = {
                                "loss": loss_scalar,
                                "learning_rate": lr,
                                "epoch": epoch + (step + 1) / len(epoch_iterator),
                                "global_step": global_step,
                                "memory_used_gb": memory_used,
                                "batch_size": self.memory_monitor.current_batch_size,
                            }

                            if grad_norm is not None:
                                logs["grad_norm"] = grad_norm
                            if clipped is not None:
                                logs["grad_clipped"] = clipped

                            self._log(logs)

                        # 评估
                        if (
                            self.args.evaluation_strategy == "steps"
                            and self.args.eval_steps > 0
                            and global_step % self.args.eval_steps == 0
                        ):
                            eval_results = self.evaluate()
                            self._log(eval_results)

                            # 早停检查
                            if self.convergence_optimizer.early_stopping:
                                eval_metric = eval_results.get("eval_loss", 0.0)
                                early_stop_result = self.convergence_optimizer.early_stopping(
                                    -eval_metric,  # 早停期望最大化指标
                                    model=self.model,
                                    step=global_step,
                                )
                                if early_stop_result:
                                    logger.info("早停触发，结束训练")
                                    break

                        # 保存
                        if (
                            self.args.save_steps > 0
                            and global_step % self.args.save_steps == 0
                        ):
                            self._save_checkpoint(global_step)

                    # 显存动态调整
                    if (step + 1) % 10 == 0:
                        self.memory_monitor.adjust_batch_size(step=step)

                    if 0 < max_steps < global_step:
                        break

                if 0 < max_steps < global_step:
                    break

                # 轮次结束评估
                if self.args.evaluation_strategy == "epoch":
                    eval_results = self.evaluate()
                    self._log(eval_results)

            # 训练结束
            if self.convergence_optimizer:
                self.convergence_optimizer.apply_final_averaging()

            # 保存最终模型
            self._save_model()

            training_time = time.time() - start_time
            logger.info(f"训练完成，耗时: {training_time / 3600:.2f} 小时")

            # 最终评估
            final_eval_results = {}
            if self.eval_dataset:
                final_eval_results = self.evaluate()
                logger.info("最终评估结果:")
                for key, value in final_eval_results.items():
                    logger.info(f"  {key}: {value:.6f}")

            # 显存报告
            memory_report = self.memory_monitor.get_status_report()
            logger.info("显存使用报告:")
            logger.info(f"  峰值显存: {memory_report.get('peak_memory_gb', 0):.2f}GB")
            logger.info(f"  批量调整次数: {memory_report.get('adjustment_count', 0)}")

            return {
                "global_step": global_step,
                "training_loss": tr_loss / global_step if global_step > 0 else 0,
                "training_time_hours": training_time / 3600,
                "peak_memory_gb": memory_report.get("peak_memory_gb", 0),
                **final_eval_results,
            }

        finally:
            self.memory_monitor.stop()
            self.is_in_train = False

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            eval_dataset: 评估数据集
            metric_key_prefix: 指标前缀

        Returns:
            Dict[str, float]: 评估结果
        """
        eval_dataloader = self._get_eval_dataloader(eval_dataset)
        self.model.eval()

        losses = []
        preds: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        logger.info(f"开始评估，共 {len(eval_dataloader)} 个批次")

        for step, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)

            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

            losses.append(loss.mean().item())

            if self.compute_metrics:
                preds.append(logits.argmax(dim=-1).cpu().numpy())
                labels.append(inputs["labels"].cpu().numpy())

        # 计算指标
        metrics = {}
        metrics[f"{metric_key_prefix}_loss"] = np.mean(losses)

        if self.compute_metrics and preds and labels:
            preds_np = np.concatenate(preds)
            labels_np = np.concatenate(labels)
            custom_metrics = self.compute_metrics((preds_np, labels_np))
            for k, v in custom_metrics.items():
                metrics[f"{metric_key_prefix}_{k}"] = v

        # 专科特定指标
        if isinstance(self.eval_dataset, SpecialtyDataset):
            specialty_stats = self._evaluate_specialty_performance(eval_dataloader)
            metrics.update(specialty_stats)

        return metrics

    def _evaluate_specialty_performance(
        self,
        eval_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """评估专科领域性能"""
        specialty_metrics: Dict[str, List[float]] = {}

        if hasattr(eval_dataloader.dataset, "get_specialty_stats"):
            specialty_stats = eval_dataloader.dataset.get_specialty_stats()
            return {
                f"specialty_{k}_count": v
                for k, v in specialty_stats.items()
            }

        return {}

    def _log(self, logs: Dict[str, Any]) -> None:
        """记录日志"""
        self.state.log_history.append(logs)

        # 控制台输出
        output = []
        for key, value in sorted(logs.items()):
            if isinstance(value, (int, float)):
                output.append(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
            else:
                output.append(f"{key}: {value}")

        logger.info("  ".join(output))

    def _save_checkpoint(self, step: int) -> None:
        """保存检查点"""
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"保存检查点到: {output_dir}")

        # 保存模型状态
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        # 保存分词器
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        # 保存训练状态
        torch.save(
            {
                "step": step,
                "optimizer_state": self.convergence_optimizer.get_optimizer().state_dict()
                if self.convergence_optimizer
                else None,
                "scheduler_state": self.convergence_optimizer.get_lr_scheduler().state_dict()
                if self.convergence_optimizer
                else None,
                "train_args": self.args.to_dict(),
            },
            os.path.join(output_dir, "trainer_state.pt"),
        )

        # 限制检查点数量
        if self.args.save_total_limit > 0:
            checkpoints = list(Path(self.args.output_dir).glob("checkpoint-*"))
            checkpoints = [c for c in checkpoints if c.is_dir()]
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))

            while len(checkpoints) > self.args.save_total_limit:
                oldest = checkpoints.pop(0)
                logger.info(f"删除旧检查点: {oldest}")
                import shutil

                shutil.rmtree(oldest)

    def _save_model(self) -> None:
        """保存最终模型"""
        output_dir = os.path.join(self.args.output_dir, "final_model")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"保存最终模型到: {output_dir}")

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)


class TrainerState:
    """训练状态类"""

    def __init__(self):
        self.log_history: List[Dict[str, Any]] = []
        self.global_step: int = 0
        self.epoch: float = 0.0


class MemoryOptimizedTrainer(SpecialtyTrainer):
    """
    显存优化训练器

    针对24GB显存限制进行深度优化：
    1. 智能批量大小调整
    2. 梯度累积优化
    3. 激活重计算
    4. 显存碎片整理
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_cleanup_counter = 0

    def _training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        step: int,
    ) -> Tuple[float, Dict[str, Any]]:
        # 定期清理显存缓存
        self._memory_cleanup_counter += 1
        if self._memory_cleanup_counter % 100 == 0:
            torch.cuda.empty_cache()

        return super()._training_step(model, inputs, step)


def compute_basic_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    计算基础评估指标

    Args:
        eval_pred: (预测结果, 真实标签)

    Returns:
        Dict[str, float]: 指标字典
    """
    predictions, labels = eval_pred

    # 忽略IGNORE_TOKEN_ID
    mask = labels != IGNORE_TOKEN_ID
    predictions = predictions[mask]
    labels = labels[mask]

    if len(predictions) == 0:
        return {"accuracy": 0.0}

    accuracy = (predictions == labels).mean()

    return {
        "accuracy": float(accuracy),
    }


def main():
    """测试函数"""
    print("专科训练器模块加载成功")

    # 打印参数示例
    args = SpecialtyTrainingArguments(
        output_dir="./test_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        max_memory_usage_gb=22.0,
        specialty_type="cardiovascular",
    )

    print("\n训练参数示例:")
    for key, value in args.to_dict().items():
        print(f"  {key}: {value}")

    # 显存信息
    if torch.cuda.is_available():
        info = get_gpu_memory_info()
        print("\nGPU信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
