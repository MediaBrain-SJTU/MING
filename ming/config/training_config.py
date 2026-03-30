"""
训练配置类

提供结构化的训练配置，支持YAML序列化
"""

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """训练配置类"""

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

    # 模型和分词器
    model_name_or_path: str = ""
    tokenizer_name_or_path: Optional[str] = None
    prompt_type: str = "qwen"

    # 数据路径
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None

    # 显存优化
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    max_memory_usage_gb: float = 22.0

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

    # LoRA配置
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """从YAML文件加载配置"""
        from ming.config.config_loader import load_config

        config_dict = load_config(config_path)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        # 处理嵌套配置
        if "lora" in config_dict:
            lora_config = config_dict.pop("lora")
            for key, value in lora_config.items():
                config_dict[f"lora_{key}"] = value

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "model_name_or_path": self.model_name_or_path,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "prompt_type": self.prompt_type,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "max_memory_usage_gb": self.max_memory_usage_gb,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "ema": self.ema,
            "ema_decay": self.ema_decay,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "specialty_type": self.specialty_type,
            "specialty_weights": self.specialty_weights,
            "difficulty_weight": self.difficulty_weight,
            "specialty_focus": self.specialty_focus,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "evaluation_strategy": self.evaluation_strategy,
            "seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "local_rank": self.local_rank,
            "deepspeed": self.deepspeed,
            "lora": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules,
            },
        }
        return result

    def save_yaml(self, config_path: str) -> None:
        """保存为YAML文件"""
        from ming.config.config_loader import save_config

        save_config(self.to_dict(), config_path)

    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []

        if self.learning_rate <= 0 or self.learning_rate > 1:
            errors.append(f"学习率应在(0, 1)范围内，当前值: {self.learning_rate}")

        if self.num_train_epochs <= 0:
            errors.append(f"训练轮数应大于0，当前值: {self.num_train_epochs}")

        if self.per_device_train_batch_size <= 0:
            errors.append(f"训练批量大小应大于0，当前值: {self.per_device_train_batch_size}")

        if self.max_memory_usage_gb <= 0 or self.max_memory_usage_gb > 24:
            errors.append(f"显存限制应在(0, 24]GB范围内，当前值: {self.max_memory_usage_gb}GB")

        if self.warmup_ratio < 0 or self.warmup_ratio >= 1:
            errors.append(f"warmup比例应在[0, 1)范围内，当前值: {self.warmup_ratio}")

        if not (0 < self.lora_alpha <= 256):
            errors.append(f"LoRA alpha应在(0, 256]范围内，当前值: {self.lora_alpha}")

        if not (0 <= self.lora_dropout < 1):
            errors.append(f"LoRA dropout应在[0, 1)范围内，当前值: {self.lora_dropout}")

        return errors

    def is_valid(self) -> bool:
        """检查配置是否有效"""
        return len(self.validate()) == 0
