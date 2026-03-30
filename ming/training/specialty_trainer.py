"""
专科训练器
提供针对专科领域的优化训练流程
目标：专科领域EM值提升 >= 15%, 训练收敛速度提升 >= 20%
"""
import os
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float] = None
    eval_em: Optional[float] = None
    eval_f1: Optional[float] = None
    learning_rate: float = 0.0
    memory_used_gb: float = 0.0
    time_elapsed_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "eval_em": self.eval_em,
            "eval_f1": self.eval_f1,
            "learning_rate": self.learning_rate,
            "memory_used_gb": self.memory_used_gb,
            "time_elapsed_seconds": self.time_elapsed_seconds
        }


@dataclass
class TrainingResult:
    """训练结果"""
    specialty: str
    best_eval_em: float
    best_eval_f1: float
    best_step: int
    total_steps: int
    total_epochs: int
    total_time_hours: float
    peak_memory_gb: float
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    baseline_em: float = 0.0
    em_improvement: float = 0.0
    convergence_improvement: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "specialty": self.specialty,
            "best_eval_em": self.best_eval_em,
            "best_eval_f1": self.best_eval_f1,
            "best_step": self.best_step,
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "total_time_hours": self.total_time_hours,
            "peak_memory_gb": self.peak_memory_gb,
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "baseline_em": self.baseline_em,
            "em_improvement": self.em_improvement,
            "convergence_improvement": self.convergence_improvement
        }


class SpecialtyTrainer:
    """
    专科训练器
    
    功能：
    - 专科领域定向微调
    - 训练过程监控
    - 自动保存和恢复
    - 训练结果评估
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            tokenizer: 分词器
            config: 训练配置
            train_dataloader: 训练数据加载器
            eval_dataloader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_em = 0.0
        self.best_model_state = None
        
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time: Optional[float] = None
        self.peak_memory_gb = 0.0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95)
        )
    
    def _create_scheduler(self) -> Any:
        """创建学习率调度器"""
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps]
        )
    
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None
    ) -> TrainingResult:
        """
        执行训练
        
        Args:
            resume_from_checkpoint: 恢复训练的检查点路径
            
        Returns:
            TrainingResult: 训练结果
        """
        self.start_time = time.time()
        
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        self.model.train()
        
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                if self.global_step > 0 and self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(epoch, epoch_loss / num_batches)
                
                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                if self.eval_dataloader and self.global_step > 0 and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self._evaluate()
                    if eval_metrics["em"] > self.best_eval_em:
                        self.best_eval_em = eval_metrics["em"]
                        self._save_best_model()
            
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")
        
        total_time = (time.time() - self.start_time) / 3600
        
        result = TrainingResult(
            specialty=self.config.specialties[0] if self.config.specialties else "general",
            best_eval_em=self.best_eval_em,
            best_eval_f1=0.0,
            best_step=self.global_step,
            total_steps=self.global_step,
            total_epochs=self.config.num_epochs,
            total_time_hours=total_time,
            peak_memory_gb=self.peak_memory_gb,
            metrics_history=self.metrics_history
        )
        
        return result
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """执行单步训练"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config.bf16 or self.config.fp16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / self.config.gradient_accumulation_steps
        
        loss.backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        self._update_memory_stats()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def _evaluate(self) -> Dict[str, float]:
        """执行评估"""
        self.model.eval()
        total_loss = 0.0
        total_em = 0.0
        total_f1 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                em_score = self._compute_em(predictions, labels)
                total_em += em_score
                
                num_batches += 1
        
        return {
            "loss": total_loss / num_batches if num_batches > 0 else 0,
            "em": total_em / num_batches if num_batches > 0 else 0,
            "f1": total_f1 / num_batches if num_batches > 0 else 0
        }
    
    def _compute_em(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """计算Exact Match"""
        mask = labels != -100
        correct = (predictions == labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy.item()
    
    def _log_metrics(self, epoch: int, train_loss: float) -> None:
        """记录训练指标"""
        current_lr = self.optimizer.param_groups[0]["lr"]
        memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        metrics = TrainingMetrics(
            epoch=epoch,
            step=self.global_step,
            train_loss=train_loss,
            learning_rate=current_lr,
            memory_used_gb=memory_gb,
            time_elapsed_seconds=time.time() - self.start_time if self.start_time else 0
        )
        
        self.metrics_history.append(metrics)
        
        print(f"Step {self.global_step}: loss={train_loss:.4f}, lr={current_lr:.2e}, memory={memory_gb:.2f}GB")
    
    def _update_memory_stats(self) -> None:
        """更新内存统计"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.peak_memory_gb = max(self.peak_memory_gb, current_memory)
    
    def _save_checkpoint(self) -> None:
        """保存检查点"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_em": self.best_eval_em,
            "config": self.config.to_dict()
        }, checkpoint_dir / "checkpoint.pt")
        
        print(f"Checkpoint saved at step {self.global_step}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_eval_em = checkpoint.get("best_eval_em", 0.0)
        
        print(f"Resumed from checkpoint at step {self.global_step}")
    
    def _save_best_model(self) -> None:
        """保存最佳模型"""
        best_model_dir = Path(self.config.output_dir) / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), best_model_dir / "model.pt")
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(best_model_dir)
        
        print(f"Best model saved with EM: {self.best_eval_em:.4f}")
    
    def compute_baseline_comparison(
        self,
        baseline_em: float,
        baseline_epochs: int
    ) -> Tuple[float, float]:
        """
        计算与基线的对比
        
        Args:
            baseline_em: 基线EM值
            baseline_epochs: 基线训练epoch数
            
        Returns:
            (EM提升百分比, 收敛速度提升百分比)
        """
        em_improvement = ((self.best_eval_em - baseline_em) / baseline_em) * 100 if baseline_em > 0 else 0
        
        current_epochs = self.current_epoch + 1
        convergence_improvement = ((baseline_epochs - current_epochs) / baseline_epochs) * 100 if baseline_epochs > 0 else 0
        
        return em_improvement, convergence_improvement
    
    def export_training_report(
        self,
        output_path: str,
        baseline_em: float = 0.0,
        baseline_epochs: int = 0
    ) -> None:
        """
        导出训练报告
        
        Args:
            output_path: 输出路径
            baseline_em: 基线EM值
            baseline_epochs: 基线训练epoch数
        """
        em_improvement, convergence_improvement = self.compute_baseline_comparison(
            baseline_em, baseline_epochs
        )
        
        report = {
            "training_summary": {
                "specialty": self.config.specialties[0] if self.config.specialties else "general",
                "total_steps": self.global_step,
                "total_epochs": self.current_epoch + 1,
                "total_time_hours": (time.time() - self.start_time) / 3600 if self.start_time else 0,
                "peak_memory_gb": self.peak_memory_gb
            },
            "performance": {
                "best_eval_em": self.best_eval_em,
                "baseline_em": baseline_em,
                "em_improvement_percent": em_improvement,
                "convergence_improvement_percent": convergence_improvement
            },
            "config": self.config.to_dict(),
            "metrics_history": [m.to_dict() for m in self.metrics_history]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Training report saved to {output_path}")
