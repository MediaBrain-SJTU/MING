"""
训练配置模块
提供训练参数配置和管理
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
import yaml


@dataclass
class TrainingConfig:
    """
    训练配置类
    
    包含所有训练相关参数，支持YAML/JSON格式导入导出
    """
    
    model_name_or_path: str = "Qwen/Qwen-7B"
    output_dir: str = "outputs/specialty_finetuned"
    
    max_length: int = 512
    max_new_tokens: int = 256
    
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    num_epochs: int = 3
    max_steps: int = -1
    
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    num_experts: int = 4
    num_experts_per_token: int = 2
    expert_selection: str = "top_k"
    share_expert: bool = False
    router_loss_coeff: float = 0.01
    
    specialties: List[str] = field(default_factory=lambda: ["cardiovascular", "neurology"])
    
    max_memory_gb: float = 22.0
    optimize_memory: bool = True
    use_flash_attention: bool = True
    
    seed: int = 42
    dataloader_num_workers: int = 0
    
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_em"
    greater_is_better: bool = True
    
    report_to: List[str] = field(default_factory=lambda: ["none"])
    run_name: Optional[str] = None
    
    data_dir: str = "ming/eval/datasets"
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    
    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"specialty_train_{self.specialties[0] if self.specialties else 'general'}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name_or_path": self.model_name_or_path,
            "output_dir": self.output_dir,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lora_enable": self.lora_enable,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "expert_selection": self.expert_selection,
            "share_expert": self.share_expert,
            "router_loss_coeff": self.router_loss_coeff,
            "specialties": self.specialties,
            "max_memory_gb": self.max_memory_gb,
            "optimize_memory": self.optimize_memory,
            "use_flash_attention": self.use_flash_attention,
            "seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "data_dir": self.data_dir,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_yaml(self, file_path: str) -> None:
        """保存为YAML文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_yaml(cls, file_path: str) -> "TrainingConfig":
        """从YAML文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_json(self, file_path: str) -> None:
        """保存为JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, file_path: str) -> "TrainingConfig":
        """从JSON文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_effective_batch_size(self) -> int:
        """获取有效批次大小"""
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_total_steps(self, num_samples: int) -> int:
        """计算总训练步数"""
        if self.max_steps > 0:
            return self.max_steps
        
        steps_per_epoch = num_samples // self.get_effective_batch_size()
        return steps_per_epoch * self.num_epochs
    
    def get_warmup_steps(self, num_samples: int) -> int:
        """计算预热步数"""
        if self.warmup_steps > 0:
            return self.warmup_steps
        
        total_steps = self.get_total_steps(num_samples)
        return int(total_steps * self.warmup_ratio)
    
    def validate(self) -> List[str]:
        """
        验证配置参数
        
        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors = []
        
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        
        if self.num_epochs < 1 and self.max_steps < 1:
            errors.append("either num_epochs or max_steps must be >= 1")
        
        if self.max_memory_gb > 24:
            errors.append("max_memory_gb exceeds 24GB constraint")
        
        if self.lora_enable:
            if self.lora_r < 1:
                errors.append("lora_r must be >= 1")
            if self.lora_alpha < 1:
                errors.append("lora_alpha must be >= 1")
        
        if self.num_experts < 1:
            errors.append("num_experts must be >= 1")
        
        if self.num_experts_per_token > self.num_experts:
            errors.append("num_experts_per_token cannot exceed num_experts")
        
        return errors
    
    def get_memory_estimate(self) -> Dict[str, float]:
        """
        估算显存使用
        
        Returns:
            显存估算字典（GB）
        """
        base_model_memory = 14.0
        
        lora_memory = 0.5 if self.lora_enable else 0
        
        expert_memory = 0.3 * self.num_experts if self.num_experts > 1 else 0
        
        activation_memory = 2.0 * self.batch_size * (self.max_length / 512)
        
        if self.gradient_checkpointing:
            activation_memory *= 0.3
        
        if self.fp16 or self.bf16:
            memory_multiplier = 0.5
        else:
            memory_multiplier = 1.0
        
        total_memory = (base_model_memory + lora_memory + expert_memory + activation_memory) * memory_multiplier
        
        return {
            "base_model_gb": base_model_memory * memory_multiplier,
            "lora_memory_gb": lora_memory * memory_multiplier,
            "expert_memory_gb": expert_memory * memory_multiplier,
            "activation_memory_gb": activation_memory * memory_multiplier,
            "total_estimated_gb": total_memory,
            "within_constraint": total_memory <= self.max_memory_gb
        }
    
    def get_specialty_config(self, specialty: str) -> "TrainingConfig":
        """
        获取特定专科的训练配置
        
        Args:
            specialty: 专科名称
            
        Returns:
            针对该专科优化的配置
        """
        import copy
        config = copy.deepcopy(self)
        
        config.run_name = f"specialty_train_{specialty}"
        config.output_dir = f"{self.output_dir}/{specialty}"
        
        specialty_params = {
            "cardiovascular": {
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "lora_r": 16
            },
            "neurology": {
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "lora_r": 16
            },
            "hematology": {
                "learning_rate": 1.5e-5,
                "num_epochs": 4,
                "lora_r": 24
            },
            "endocrinology": {
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "lora_r": 16
            },
            "pediatrics": {
                "learning_rate": 1.5e-5,
                "num_epochs": 4,
                "lora_r": 20
            }
        }
        
        if specialty in specialty_params:
            for key, value in specialty_params[specialty].items():
                setattr(config, key, value)
        
        return config


@dataclass
class SpecialtyTrainingPlan:
    """专科训练计划"""
    
    specialty: str
    config: TrainingConfig
    train_samples: int
    val_samples: int
    estimated_time_hours: float
    estimated_memory_gb: float
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "specialty": self.specialty,
            "config": self.config.to_dict(),
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "estimated_time_hours": self.estimated_time_hours,
            "estimated_memory_gb": self.estimated_memory_gb,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, plan_dict: Dict[str, Any]) -> "SpecialtyTrainingPlan":
        config = TrainingConfig.from_dict(plan_dict["config"])
        return cls(
            specialty=plan_dict["specialty"],
            config=config,
            train_samples=plan_dict["train_samples"],
            val_samples=plan_dict["val_samples"],
            estimated_time_hours=plan_dict["estimated_time_hours"],
            estimated_memory_gb=plan_dict["estimated_memory_gb"],
            priority=plan_dict.get("priority", 1)
        )


def create_default_config(
    specialty: str = "cardiovascular",
    output_dir: str = "outputs/specialty_finetuned"
) -> TrainingConfig:
    """
    创建默认训练配置
    
    Args:
        specialty: 专科名称
        output_dir: 输出目录
        
    Returns:
        TrainingConfig实例
    """
    config = TrainingConfig(
        specialties=[specialty],
        output_dir=output_dir
    )
    return config.get_specialty_config(specialty)
