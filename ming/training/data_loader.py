"""
专科数据加载器
支持从内置数据集加载和预处理专科训练数据
"""
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class MedicalQASample:
    """医疗问答样本"""
    question: str
    options: Dict[str, str]
    answer: str
    answer_idx: str
    rationale: Optional[str] = None
    meta_info: Optional[str] = None
    specialty: Optional[str] = None
    difficulty: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "answer_idx": self.answer_idx,
            "rationale": self.rationale,
            "meta_info": self.meta_info,
            "specialty": self.specialty,
            "difficulty": self.difficulty,
            "features": self.features
        }


class MedicalQADataset(Dataset):
    """医疗问答数据集"""
    
    def __init__(
        self,
        samples: List[MedicalQASample],
        tokenizer: Any = None,
        max_length: int = 512,
        include_rationale: bool = True
    ):
        """
        初始化数据集
        
        Args:
            samples: 样本列表
            tokenizer: 分词器
            max_length: 最大序列长度
            include_rationale: 是否包含推理过程
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_rationale = include_rationale
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        prompt = self._build_prompt(sample)
        
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": encoding["input_ids"].squeeze(0).clone(),
                "sample_idx": idx
            }
        else:
            return {
                "prompt": prompt,
                "sample": sample,
                "sample_idx": idx
            }
    
    def _build_prompt(self, sample: MedicalQASample) -> str:
        """构建提示文本"""
        prompt = f"问题：{sample.question}\n\n"
        prompt += "选项：\n"
        for key, value in sorted(sample.options.items()):
            prompt += f"{key}. {value}\n"
        
        if self.include_rationale and sample.rationale:
            prompt += f"\n答案：{sample.answer}\n"
            prompt += f"\n解析：{sample.rationale}"
        else:
            prompt += f"\n答案：{sample.answer}"
        
        return prompt


class SpecialtyDataLoader:
    """
    专科数据加载器
    
    功能：
    - 从内置数据集加载数据
    - 按专科领域划分数据
    - 支持数据增强和平衡
    """
    
    DEFAULT_DATA_DIR = "ming/eval/datasets"
    
    SPECIALTY_MAPPING = {
        "cardiovascular": ["心血管", "心脏", "心肌", "心绞痛", "心肌梗死", "心律失常", "高血压"],
        "neurology": ["神经", "脑", "脊髓", "癫痫", "帕金森", "头痛", "眩晕"],
        "hematology": ["血液", "贫血", "血小板", "白血病", "淋巴瘤", "出血", "凝血"],
        "endocrinology": ["内分泌", "甲状腺", "糖尿病", "血糖", "胰岛素", "激素"],
        "gastroenterology": ["消化", "胃", "肠", "肝", "胆", "胰腺", "腹泻"],
        "pediatrics": ["儿科", "患儿", "儿童", "小儿", "新生儿", "麻疹", "风疹"],
        "obstetrics_gynecology": ["产科", "妇科", "妊娠", "孕妇", "子宫", "卵巢"],
        "psychiatry": ["精神", "心理", "幻觉", "妄想", "抑郁", "焦虑"],
        "immunology": ["免疫", "自身免疫", "红斑狼疮", "类风湿", "抗体"],
        "pathology": ["病理", "肿瘤", "癌", "恶性", "良性"]
    }
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        tokenizer: Any = None,
        max_length: int = 512,
        batch_size: int = 4,
        num_workers: int = 0
    ):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
            tokenizer: 分词器
            max_length: 最大序列长度
            batch_size: 批次大小
            num_workers: 数据加载线程数
        """
        self.data_dir = data_dir or self.DEFAULT_DATA_DIR
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self._samples_cache: Dict[str, List[MedicalQASample]] = {}
    
    def load_jsonl(
        self,
        file_path: str,
        specialty: Optional[str] = None
    ) -> List[MedicalQASample]:
        """
        加载JSONL格式数据
        
        Args:
            file_path: 文件路径
            specialty: 专科标签
            
        Returns:
            样本列表
        """
        samples = []
        
        try:
            import jsonlines
            with jsonlines.open(file_path) as reader:
                items = list(reader)
        except ImportError:
            with open(file_path, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f if line.strip()]
        
        for item in items:
            sample = MedicalQASample(
                question=item.get("question", ""),
                options=item.get("options", {}),
                answer=item.get("answer", ""),
                answer_idx=item.get("answer_idx", ""),
                rationale=item.get("rationale"),
                meta_info=item.get("meta_info"),
                specialty=specialty or self._detect_specialty(item)
            )
            sample.difficulty = self._detect_difficulty(item)
            samples.append(sample)
        
        return samples
    
    def load_all_datasets(
        self,
        specialties: Optional[List[str]] = None
    ) -> Dict[str, List[MedicalQASample]]:
        """
        加载所有内置数据集
        
        Args:
            specialties: 要加载的专科列表，None表示全部
            
        Returns:
            按专科划分的样本字典
        """
        data_path = Path(self.data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        all_samples: Dict[str, List[MedicalQASample]] = {}
        
        jsonl_files = list(data_path.glob("*.jsonl"))
        
        for jsonl_file in jsonl_files:
            samples = self.load_jsonl(str(jsonl_file))
            
            for sample in samples:
                spec = sample.specialty or "general"
                if spec not in all_samples:
                    all_samples[spec] = []
                all_samples[spec].append(sample)
        
        if specialties:
            all_samples = {
                k: v for k, v in all_samples.items() 
                if k in specialties
            }
        
        return all_samples
    
    def get_specialty_dataset(
        self,
        specialty: str,
        split_ratio: float = 0.8
    ) -> Tuple[MedicalQADataset, MedicalQADataset]:
        """
        获取专科数据集
        
        Args:
            specialty: 专科名称
            split_ratio: 训练集比例
            
        Returns:
            (训练数据集, 验证数据集)
        """
        if specialty not in self._samples_cache:
            all_samples = self.load_all_datasets([specialty])
            self._samples_cache[specialty] = all_samples.get(specialty, [])
        
        samples = self._samples_cache[specialty]
        
        if not samples:
            raise ValueError(f"未找到专科 {specialty} 的数据")
        
        np.random.shuffle(samples)
        split_idx = int(len(samples) * split_ratio)
        
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        train_dataset = MedicalQADataset(
            train_samples,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        val_dataset = MedicalQADataset(
            val_samples,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        return train_dataset, val_dataset
    
    def get_dataloader(
        self,
        dataset: MedicalQADataset,
        shuffle: bool = True
    ) -> DataLoader:
        """
        获取数据加载器
        
        Args:
            dataset: 数据集
            shuffle: 是否打乱
            
        Returns:
            DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def balance_samples(
        self,
        samples_by_specialty: Dict[str, List[MedicalQASample]],
        strategy: str = "oversample"
    ) -> List[MedicalQASample]:
        """
        平衡样本
        
        Args:
            samples_by_specialty: 按专科划分的样本
            strategy: 平衡策略 (oversample, undersample, none)
            
        Returns:
            平衡后的样本列表
        """
        if strategy == "none":
            all_samples = []
            for samples in samples_by_specialty.values():
                all_samples.extend(samples)
            return all_samples
        
        max_count = max(len(samples) for samples in samples_by_specialty.values())
        min_count = min(len(samples) for samples in samples_by_specialty.values())
        
        balanced_samples = []
        
        for specialty, samples in samples_by_specialty.items():
            if strategy == "oversample":
                while len(samples) < max_count:
                    samples = samples + samples[:max_count - len(samples)]
            elif strategy == "undersample":
                np.random.shuffle(samples)
                samples = samples[:min_count]
            
            balanced_samples.extend(samples)
        
        np.random.shuffle(balanced_samples)
        return balanced_samples
    
    def augment_samples(
        self,
        samples: List[MedicalQASample],
        augment_ratio: float = 0.2
    ) -> List[MedicalQASample]:
        """
        数据增强
        
        Args:
            samples: 原始样本
            augment_ratio: 增强比例
            
        Returns:
            增强后的样本列表
        """
        import copy
        
        augmented = list(samples)
        num_augment = int(len(samples) * augment_ratio)
        
        for _ in range(num_augment):
            sample = copy.deepcopy(np.random.choice(samples))
            augmented.append(sample)
        
        return augmented
    
    def _detect_specialty(self, item: Dict[str, Any]) -> str:
        """检测样本所属专科"""
        text = item.get("question", "") + " " + (item.get("rationale", "") or "")
        meta_info = item.get("meta_info", "") or ""
        
        combined_text = text + " " + meta_info
        
        specialty_scores = {}
        for specialty, keywords in self.SPECIALTY_MAPPING.items():
            score = sum(1 for kw in keywords if kw in combined_text)
            specialty_scores[specialty] = score
        
        if specialty_scores:
            max_score = max(specialty_scores.values())
            if max_score > 0:
                for specialty, score in specialty_scores.items():
                    if score == max_score:
                        return specialty
        
        return "general"
    
    def _detect_difficulty(self, item: Dict[str, Any]) -> str:
        """检测样本难度"""
        meta_info = item.get("meta_info", "") or ""
        question = item.get("question", "")
        
        if "历年真题" in meta_info:
            return "hard"
        elif "模拟试题" in meta_info:
            return "medium"
        
        hard_indicators = ["最合理", "最佳方案", "首选", "应首先"]
        for indicator in hard_indicators:
            if indicator in question:
                return "hard"
        
        easy_indicators = ["下列哪项", "以下哪项", "正确的是"]
        for indicator in easy_indicators:
            if indicator in question:
                return "easy"
        
        return "medium"
    
    def get_dataset_statistics(
        self,
        samples_by_specialty: Dict[str, List[MedicalQASample]]
    ) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Args:
            samples_by_specialty: 按专科划分的样本
            
        Returns:
            统计信息字典
        """
        stats = {
            "total_samples": sum(len(samples) for samples in samples_by_specialty.values()),
            "num_specialties": len(samples_by_specialty),
            "samples_per_specialty": {
                k: len(v) for k, v in samples_by_specialty.items()
            },
            "difficulty_distribution": {},
            "avg_question_length": 0,
            "avg_option_count": 0
        }
        
        all_samples = []
        for samples in samples_by_specialty.values():
            all_samples.extend(samples)
        
        if all_samples:
            difficulty_counts = {}
            total_q_len = 0
            total_opt_count = 0
            
            for sample in all_samples:
                diff = sample.difficulty or "unknown"
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
                total_q_len += len(sample.question)
                total_opt_count += len(sample.options)
            
            stats["difficulty_distribution"] = difficulty_counts
            stats["avg_question_length"] = total_q_len / len(all_samples)
            stats["avg_option_count"] = total_opt_count / len(all_samples)
        
        return stats
