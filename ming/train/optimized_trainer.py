"""
优化的模型训练Pipeline

提供内存优化、训练加速的模型训练功能，
支持专科领域定向微调。
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from tqdm import tqdm

from ming.features.feature_extractor import FeatureExtractor, FeatureConfig
from ming.conversations import get_default_conv_template


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizedTrainingConfig:
    """优化训练配置"""
    
    # 模型配置
    model_name_or_path: str = "Qwen/Qwen1.5-7B-Chat"
    
    # 数据配置
    train_data_path: str = ""
    eval_data_path: Optional[str] = None
    max_seq_length: int = 512
    
    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 量化配置
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    
    # 训练配置
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # 内存优化配置
    gradient_checkpointing: bool = True
    max_memory_mb: int = 24000  # 24GB显存限制
    optim: str = "paged_adamw_8bit"
    
    # 保存配置
    output_dir: str = "./output"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # 专科配置
    specialty_focus: Optional[str] = None  # 如"心血管", "神经内科"
    feature_engineering: bool = True
    
    # 早停配置
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


class MedicalDataset(Dataset):
    """
    医疗数据集
    
    支持特征工程的数据集类，可根据专科领域进行数据筛选。
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        feature_extractor: Optional[FeatureExtractor] = None,
        specialty_focus: Optional[str] = None,
        prompt_type: str = "qwen"
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            feature_extractor: 特征提取器
            specialty_focus: 专科聚焦
            prompt_type: 提示类型
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = feature_extractor
        self.specialty_focus = specialty_focus
        self.prompt_type = prompt_type
        
        # 加载数据
        self.data = self._load_data(data_path)
        
        # 如果指定了专科，进行数据筛选
        if specialty_focus and feature_extractor:
            self.data = self._filter_by_specialty(self.data, specialty_focus)
            logger.info(f"筛选后数据量: {len(self.data)} (专科: {specialty_focus})")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据文件"""
        data = []
        
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
        
        return data
    
    def _filter_by_specialty(
        self, 
        data: List[Dict], 
        specialty: str
    ) -> List[Dict]:
        """根据专科筛选数据"""
        filtered = []
        
        for item in data:
            # 获取文本内容
            text = self._get_item_text(item)
            
            # 提取特征
            features = self.feature_extractor.extract(text)
            
            # 检查是否属于目标专科
            if features.specialty_features.primary_specialty == specialty:
                filtered.append(item)
            elif specialty in features.specialty_features.specialty_scores:
                score = features.specialty_features.specialty_scores[specialty]
                if score > 0.3:  # 阈值
                    filtered.append(item)
        
        return filtered
    
    def _get_item_text(self, item: Dict) -> str:
        """从数据项中提取文本"""
        if "conversations" in item:
            texts = []
            for conv in item["conversations"]:
                if "value" in conv:
                    texts.append(conv["value"])
            return " ".join(texts)
        elif "text" in item:
            return item["text"]
        elif "instruction" in item:
            return f"{item.get('instruction', '')} {item.get('input', '')} {item.get('output', '')}"
        else:
            return str(item)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据项"""
        item = self.data[idx]
        
        # 构建对话格式
        conv = get_default_conv_template(self.prompt_type).copy()
        
        if "conversations" in item:
            for conv_item in item["conversations"]:
                role = conv_item.get("from", "human")
                value = conv_item.get("value", "")
                if role == "human":
                    conv.append_message(conv.roles[0], value)
                else:
                    conv.append_message(conv.roles[1], value)
        elif "instruction" in item:
            conv.append_message(conv.roles[0], item.get("instruction", ""))
            conv.append_message(conv.roles[1], item.get("output", ""))
        
        prompt = conv.get_prompt()
        
        # 编码
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # 创建标签（用于语言模型训练）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class OptimizedTrainer:
    """
    优化的模型训练器
    
    提供内存优化、训练加速的模型训练功能。
    
    Attributes:
        config: 训练配置
        model: 模型
        tokenizer: 分词器
    
    Example:
        >>> config = OptimizedTrainingConfig()
        >>> trainer = OptimizedTrainer(config)
        >>> trainer.train()
    """
    
    def __init__(self, config: OptimizedTrainingConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        
        # 训练状态
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # 初始化
        self._setup_feature_extractor()
        self._setup_model_and_tokenizer()
    
    def _setup_feature_extractor(self) -> None:
        """设置特征提取器"""
        if self.config.feature_engineering:
            feature_config = FeatureConfig(
                extract_text_features=True,
                extract_entity_features=True,
                extract_specialty_features=True
            )
            self.feature_extractor = FeatureExtractor(feature_config)
            logger.info("特征提取器初始化完成")
    
    def _setup_model_and_tokenizer(self) -> None:
        """设置模型和分词器"""
        logger.info(f"加载模型: {self.config.model_name_or_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # 量化配置
        quantization_config = None
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, self.config.bnb_4bit_compute_dtype
                ),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("使用4-bit量化")
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            max_memory={0: f"{self.config.max_memory_mb}MB"}
        )
        
        # 准备模型用于训练
        if self.config.load_in_4bit or self.config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # 应用LoRA
        if self.config.use_lora:
            self._apply_lora()
        
        # 梯度检查点
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("启用梯度检查点")
        
        logger.info("模型初始化完成")
    
    def _apply_lora(self) -> None:
        """应用LoRA配置"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info(f"LoRA配置: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
    
    def train(self) -> Dict[str, Any]:
        """
        执行训练
        
        Returns:
            训练历史记录
        """
        # 创建数据集
        train_dataset = MedicalDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
            feature_extractor=self.feature_extractor,
            specialty_focus=self.config.specialty_focus
        )
        
        eval_dataset = None
        if self.config.eval_data_path:
            eval_dataset = MedicalDataset(
                data_path=self.config.eval_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                feature_extractor=self.feature_extractor,
                specialty_focus=self.config.specialty_focus
            )
        
        logger.info(f"训练数据量: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"评估数据量: {len(eval_dataset)}")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            optim=self.config.optim,
            fp16=True,
            report_to="none"
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(self)] if eval_dataset else []
        )
        
        # 开始训练
        logger.info("开始训练...")
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 保存最终模型
        trainer.save_model(os.path.join(self.config.output_dir, "final"))
        self.tokenizer.save_pretrained(os.path.join(self.config.output_dir, "final"))
        
        # 保存训练历史
        self.training_history = trainer.state.log_history
        
        return {
            "training_time": training_time,
            "final_loss": trainer.state.log_history[-1].get("loss", 0),
            "best_eval_loss": self.best_eval_loss,
            "history": self.training_history
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取显存使用情况"""
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
            }
        return {}


class EarlyStoppingCallback:
    """早停回调"""
    
    def __init__(self, trainer: OptimizedTrainer):
        self.trainer = trainer
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估回调"""
        eval_loss = metrics.get("eval_loss", float('inf'))
        
        if eval_loss < self.trainer.best_eval_loss - self.trainer.config.early_stopping_threshold:
            self.trainer.best_eval_loss = eval_loss
            self.trainer.patience_counter = 0
        else:
            self.trainer.patience_counter += 1
        
        if self.trainer.patience_counter >= self.trainer.config.early_stopping_patience:
            control.should_training_stop = True
            logger.info("触发早停")
        
        return control
