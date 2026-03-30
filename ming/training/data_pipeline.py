"""
专科训练数据处理Pipeline

本模块提供专科领域训练数据的加载、预处理和增强功能：
1. 多源数据统一加载
2. 专科特征增强
3. 智能数据采样
4. 动态批量构建
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.trainer_pt_utils import LabelSmoother

from ming.conversations import get_default_conv_template, SeparatorStyle
from ming.feature_engineering.feature_extractor import FeatureExtractor
from ming.feature_engineering.entity_recognizer import MedicalEntityRecognizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class SpecialtySample:
    """专科样本数据类"""

    text: str
    conversations: List[Dict[str, str]]
    specialty_type: str
    difficulty: float = 1.0
    entity_count: int = 0
    feature_embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "text": self.text,
            "conversations": self.conversations,
            "specialty_type": self.specialty_type,
            "difficulty": self.difficulty,
            "entity_count": self.entity_count,
            "metadata": self.metadata,
        }
        if self.feature_embedding is not None:
            result["feature_embedding"] = self.feature_embedding.tolist()
        return result


class SpecialtyDataset(Dataset):
    """
    专科训练数据集

    支持从多种格式加载数据，并应用专科领域特定的预处理。
    """

    # 专科类型映射
    SPECIALTY_MAPPING = {
        "心血管内科": "cardiovascular",
        "cardiovascular": "cardiovascular",
        "神经内科": "neurology",
        "neurology": "neurology",
        "呼吸内科": "respiratory",
        "respiratory": "respiratory",
        "消化内科": "gastroenterology",
        "gastroenterology": "gastroenterology",
        "内分泌科": "endocrinology",
        "endocrinology": "endocrinology",
        "肾内科": "nephrology",
        "nephrology": "nephrology",
        "血液内科": "hematology",
        "hematology": "hematology",
        "风湿免疫科": "rheumatology",
        "rheumatology": "rheumatology",
        "感染科": "infectious",
        "infectious": "infectious",
        "普通外科": "general_surgery",
        "general_surgery": "general_surgery",
        "神经外科": "neurosurgery",
        "neurosurgery": "neurosurgery",
        "心胸外科": "cardiothoracic",
        "cardiothoracic": "cardiothoracic",
        "妇产科": "obstetrics_gynecology",
        "obstetrics_gynecology": "obstetrics_gynecology",
        "儿科": "pediatrics",
        "pediatrics": "pediatrics",
        "眼科": "ophthalmology",
        "ophthalmology": "ophthalmology",
        "耳鼻喉科": "otolaryngology",
        "otolaryngology": "otolaryngology",
        "皮肤科": "dermatology",
        "dermatology": "dermatology",
        "急诊科": "emergency",
        "emergency": "emergency",
        "重症医学科": "critical_care",
        "critical_care": "critical_care",
    }

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        prompt_type: str = "qwen",
        model_max_length: int = 4096,
        lazy_preprocess: bool = True,
        extract_features: bool = True,
        specialty_filter: Optional[List[str]] = None,
    ):
        """
        初始化专科数据集

        Args:
            data_path: 数据路径
            tokenizer: 分词器
            prompt_type: 提示模板类型
            model_max_length: 模型最大长度
            lazy_preprocess: 是否延迟预处理
            extract_features: 是否提取特征
            specialty_filter: 专科类型过滤器
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.model_max_length = model_max_length
        self.lazy_preprocess = lazy_preprocess
        self.extract_features = extract_features
        self.specialty_filter = specialty_filter

        # 初始化特征提取工具
        self.entity_recognizer = MedicalEntityRecognizer()
        self.feature_extractor: Optional[FeatureExtractor] = None
        if extract_features:
            self.feature_extractor = FeatureExtractor(
                entity_recognizer=self.entity_recognizer
            )

        # 加载数据
        self.raw_data = self._load_data()
        logger.info(f"加载原始数据: {len(self.raw_data)} 条")

        # 专科过滤
        if specialty_filter:
            self.raw_data = [
                d
                for d in self.raw_data
                if self._get_specialty_type(d) in specialty_filter
            ]
            logger.info(f"专科过滤后剩余: {len(self.raw_data)} 条")

        # 预处理缓存
        self.cached_data: Dict[int, Dict[str, torch.Tensor]] = {}

    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
        path = Path(self.data_path)

        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.suffix == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        elif path.suffix == ".csv":
            import csv

            data = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            return data
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

    def _get_specialty_type(self, item: Dict[str, Any]) -> str:
        """获取样本的专科类型"""
        # 尝试从metadata获取
        if "metadata" in item and isinstance(item["metadata"], dict):
            specialty = item["metadata"].get("specialty")
            if specialty:
                return self.SPECIALTY_MAPPING.get(specialty, specialty.lower())

        if "specialty" in item:
            specialty = item["specialty"]
            return self.SPECIALTY_MAPPING.get(specialty, specialty.lower())

        if "specialty_type" in item:
            specialty = item["specialty_type"]
            return self.SPECIALTY_MAPPING.get(specialty, specialty.lower())

        # 从对话内容推断
        conversations = item.get("conversations", [])
        text = ""
        if isinstance(conversations, list):
            for conv in conversations:
                if isinstance(conv, dict):
                    text += conv.get("value", conv.get("text", ""))
                elif isinstance(conv, str):
                    text += conv

        # 使用实体识别推断专科类型
        entities, _ = self.entity_recognizer.recognize(text)
        specialty_entities = [
            e.text for e in entities if e.entity_type.value == "专科"
        ]

        if specialty_entities:
            return self.SPECIALTY_MAPPING.get(
                specialty_entities[0], specialty_entities[0].lower()
            )

        # 默认
        return "general"

    def _calculate_difficulty(self, item: Dict[str, Any]) -> float:
        """计算样本难度"""
        conversations = item.get("conversations", [])
        text = ""
        for conv in conversations:
            if isinstance(conv, dict):
                text += conv.get("value", "")

        # 基于文本长度
        length_score = min(len(text) / 1000, 1.0)

        # 基于实体数量
        entities, _ = self.entity_recognizer.recognize(text)
        entity_score = min(len(entities) / 10, 1.0)

        # 基于专科关键词
        specialty_keywords = {
            "cardiovascular": {"高血压", "冠心病", "心肌梗死", "心电图", "ST段"},
            "neurology": {"脑梗死", "脑出血", "癫痫", "肌张力", "巴氏征"},
            "respiratory": {"肺炎", "哮喘", "咳嗽", "胸部CT", "氧饱和度"},
            "gastroenterology": {"胃炎", "溃疡", "肝硬化", "胃镜", "肠镜"},
            "endocrinology": {"糖尿病", "胰岛素", "甲状腺", "血糖", "糖化血红蛋白"},
        }

        specialty_count = 0
        for keywords in specialty_keywords.values():
            matches = sum(1 for kw in keywords if kw in text)
            specialty_count = max(specialty_count, matches)

        keyword_score = min(specialty_count / 5, 1.0)

        # 综合难度
        difficulty = (length_score * 0.3 + entity_score * 0.4 + keyword_score * 0.3) + 0.1
        return min(difficulty, 1.0)

    def _preprocess(
        self,
        item: Dict[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], SpecialtySample]:
        """预处理单个样本"""
        # 获取对话模板
        conv = get_default_conv_template(self.prompt_type).copy()
        if hasattr(conv, "roles"):
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        else:
            roles = {"human": "user", "gpt": "assistant"}

        system_message = getattr(conv, "system", "")

        # 特殊token
        im_start = self.tokenizer("<|im_start|>")["input_ids"][-1]
        im_end = self.tokenizer("<|im_end|>")["input_ids"][-1]
        nl_tokens = self.tokenizer("\n", add_special_tokens=False).input_ids
        _system = self.tokenizer("system", add_special_tokens=False).input_ids + nl_tokens

        conversations = item.get("conversations", [])
        if not conversations:
            conversations = item.get("items", [])

        # 确定role key名称
        if len(conversations) > 0 and isinstance(conversations[0], dict):
            role_key = "role" if "role" in conversations[0] else "from"
        else:
            role_key = "role"

        input_id, target = [], []
        system = [im_start] + _system + self.tokenizer(system_message, add_special_tokens=False).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens

        for j, sentence in enumerate(conversations):
            if not isinstance(sentence, dict):
                continue

            role = roles.get(sentence.get(role_key, "human"), "user")
            value = sentence.get("value", sentence.get("text", ""))

            _input_id = (
                self.tokenizer(role, add_special_tokens=False).input_ids
                + nl_tokens
                + self.tokenizer(value, add_special_tokens=False).input_ids
                + [im_end]
                + nl_tokens
            )
            input_id += _input_id

            if role == "<|im_start|>user":
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * len(self.tokenizer(role, add_special_tokens=False).input_ids)
                    + _input_id[len(self.tokenizer(role, add_special_tokens=False).input_ids) + 1 : -2]
                    + [im_end]
                    + nl_tokens
                )
            else:
                _target = [IGNORE_TOKEN_ID] * len(_input_id)

            target += _target

        # 截断和填充
        input_id = input_id[: self.model_max_length]
        target = target[: self.model_max_length]

        # 转换为tensor
        input_ids = torch.tensor(input_id, dtype=torch.long)
        labels = torch.tensor(target, dtype=torch.long)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 创建专科样本对象
        specialty_type = self._get_specialty_type(item)
        difficulty = self._calculate_difficulty(item)
        text = " ".join(
            [
                conv.get("value", "")
                for conv in conversations
                if isinstance(conv, dict)
            ]
        )
        entities, _ = self.entity_recognizer.recognize(text)

        sample = SpecialtySample(
            text=text,
            conversations=conversations,
            specialty_type=specialty_type,
            difficulty=difficulty,
            entity_count=len(entities),
            metadata={"source": self.data_path, "index": len(self.cached_data)},
        )

        return (
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            },
            sample,
        )

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self.cached_data:
            return self.cached_data[idx]

        item = self.raw_data[idx]
        processed, sample = self._preprocess(item)

        if not self.lazy_preprocess:
            self.cached_data[idx] = processed

        return processed

    def get_sample(self, idx: int) -> Optional[SpecialtySample]:
        """获取原始样本信息"""
        if idx >= len(self.raw_data):
            return None

        item = self.raw_data[idx]
        _, sample = self._preprocess(item)
        return sample

    def get_specialty_stats(self) -> Dict[str, int]:
        """获取专科分布统计"""
        stats: Dict[str, int] = {}
        for item in self.raw_data:
            specialty = self._get_specialty_type(item)
            stats[specialty] = stats.get(specialty, 0) + 1
        return stats


class SpecialtySampler(Sampler[int]):
    """
    专科采样器

    支持按专科类型和难度进行智能采样，提高训练效率。
    """

    def __init__(
        self,
        dataset: SpecialtyDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        difficulty_weight: float = 0.0,
    ):
        """
        初始化专科采样器

        Args:
            dataset: 专科数据集
            batch_size: 批量大小
            shuffle: 是否打乱
            weights: 专科采样权重
            temperature: 采样温度（控制探索程度）
            difficulty_weight: 难度权重
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights or {}
        self.temperature = temperature
        self.difficulty_weight = difficulty_weight

        # 按专科分组
        self.specialty_indices: Dict[str, List[int]] = {}
        self.specialty_difficulties: Dict[str, List[float]] = {}

        for idx, item in enumerate(dataset.raw_data):
            specialty = dataset._get_specialty_type(item)
            if specialty not in self.specialty_indices:
                self.specialty_indices[specialty] = []
                self.specialty_difficulties[specialty] = []
            self.specialty_indices[specialty].append(idx)
            difficulty = dataset._calculate_difficulty(item)
            self.specialty_difficulties[specialty].append(difficulty)

        # 计算采样概率
        self._update_sampling_probabilities()

    def _update_sampling_probabilities(self) -> None:
        """更新采样概率"""
        self.specialty_probs: Dict[str, float] = {}
        total_weight = 0.0

        for specialty in self.specialty_indices.keys():
            base_weight = self.weights.get(specialty, 1.0)
            count = len(self.specialty_indices[specialty])

            if self.difficulty_weight > 0:
                avg_difficulty = sum(self.specialty_difficulties[specialty]) / count
                difficulty_factor = 1.0 + (avg_difficulty - 0.5) * self.difficulty_weight
                base_weight *= difficulty_factor

            self.specialty_probs[specialty] = base_weight
            total_weight += base_weight * count

        # 归一化
        if total_weight > 0:
            for specialty in self.specialty_probs:
                count = len(self.specialty_indices[specialty])
                self.specialty_probs[specialty] = (
                    self.specialty_probs[specialty] * count / total_weight
                )

    def __iter__(self) -> Iterator[int]:
        indices = []
        remaining = {
            s: list(range(len(self.specialty_indices[s])))
            for s in self.specialty_indices
        }

        while any(remaining.values()):
            # 选择专科类型
            specialties = list(self.specialty_indices.keys())
            probs = [self.specialty_probs.get(s, 0.01) for s in specialties]

            # 应用温度
            if self.temperature != 1.0:
                probs = [p ** (1 / self.temperature) for p in probs]
                prob_sum = sum(probs)
                probs = [p / prob_sum for p in probs]

            specialty = random.choices(specialties, weights=probs, k=1)[0]

            # 从该专科选择样本
            if remaining[specialty]:
                if self.shuffle:
                    pos = random.choice(remaining[specialty])
                else:
                    pos = remaining[specialty][0]
                remaining[specialty].remove(pos)
                idx = self.specialty_indices[specialty][pos]
                indices.append(idx)

        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class DataCollatorForSpecialty:
    """
    专科训练数据整理器

    支持动态批量处理和专科特征增强。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int = 8,
        use_dynamic_padding: bool = True,
    ):
        """
        初始化数据整理器

        Args:
            tokenizer: 分词器
            pad_to_multiple_of: 填充到的倍数
            use_dynamic_padding: 是否使用动态填充
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.use_dynamic_padding = use_dynamic_padding

    def __call__(
        self,
        instances: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_masks = [instance["attention_mask"] for instance in instances]

        # 确定最大长度
        if self.use_dynamic_padding:
            max_len = max(len(ids) for ids in input_ids)
            # 向上取整到pad_to_multiple_of的倍数
            if self.pad_to_multiple_of > 1:
                max_len = (
                    (max_len + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
        else:
            max_len = max(len(ids) for ids in input_ids)

        # 填充批量
        batch_input_ids = []
        batch_labels = []
        batch_attention_masks = []

        for ids, lbl, mask in zip(input_ids, labels, attention_masks):
            padding_length = max_len - len(ids)

            if padding_length > 0:
                padded_ids = F.pad(
                    ids, (0, padding_length), value=self.tokenizer.pad_token_id
                )
                padded_labels = F.pad(
                    lbl, (0, padding_length), value=IGNORE_TOKEN_ID
                )
                padded_mask = F.pad(mask, (0, padding_length), value=False)
            else:
                padded_ids = ids[:max_len]
                padded_labels = lbl[:max_len]
                padded_mask = mask[:max_len]

            batch_input_ids.append(padded_ids)
            batch_labels.append(padded_labels)
            batch_attention_masks.append(padded_mask)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_masks),
        }


class SpecialtyDataPipeline:
    """
    专科训练数据管道

    统一管理专科训练数据的加载、预处理、采样和批量构建。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        train_data_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        prompt_type: str = "qwen",
        model_max_length: int = 4096,
        batch_size: int = 8,
        eval_batch_size: int = 16,
        specialty_weights: Optional[Dict[str, float]] = None,
        difficulty_weight: float = 0.0,
    ):
        """
        初始化数据管道

        Args:
            tokenizer: 分词器
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径
            prompt_type: 提示模板类型
            model_max_length: 模型最大长度
            batch_size: 训练批量大小
            eval_batch_size: 评估批量大小
            specialty_weights: 专科采样权重
            difficulty_weight: 难度权重
        """
        self.tokenizer = tokenizer
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.prompt_type = prompt_type
        self.model_max_length = model_max_length
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.specialty_weights = specialty_weights
        self.difficulty_weight = difficulty_weight

        # 数据集
        self.train_dataset: Optional[SpecialtyDataset] = None
        self.val_dataset: Optional[SpecialtyDataset] = None

        # 采样器
        self.train_sampler: Optional[SpecialtySampler] = None

        # 数据整理器
        self.data_collator = DataCollatorForSpecialty(
            tokenizer=tokenizer,
            use_dynamic_padding=True,
        )

    def initialize_datasets(
        self,
        specialty_filter: Optional[List[str]] = None,
    ) -> None:
        """初始化数据集"""
        if self.train_data_path:
            logger.info(f"初始化训练数据集: {self.train_data_path}")
            self.train_dataset = SpecialtyDataset(
                data_path=self.train_data_path,
                tokenizer=self.tokenizer,
                prompt_type=self.prompt_type,
                model_max_length=self.model_max_length,
                lazy_preprocess=True,
                extract_features=True,
                specialty_filter=specialty_filter,
            )

            self.train_sampler = SpecialtySampler(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                weights=self.specialty_weights,
                difficulty_weight=self.difficulty_weight,
            )

            logger.info(f"训练集大小: {len(self.train_dataset)}")
            logger.info(f"训练集专科分布: {self.train_dataset.get_specialty_stats()}")

        if self.val_data_path:
            logger.info(f"初始化验证数据集: {self.val_data_path}")
            self.val_dataset = SpecialtyDataset(
                data_path=self.val_data_path,
                tokenizer=self.tokenizer,
                prompt_type=self.prompt_type,
                model_max_length=self.model_max_length,
                lazy_preprocess=True,
                extract_features=False,
                specialty_filter=specialty_filter,
            )

            logger.info(f"验证集大小: {len(self.val_dataset)}")
            logger.info(f"验证集专科分布: {self.val_dataset.get_specialty_stats()}")

    def get_dataloaders(
        self,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> Tuple[Optional[torch.utils.data.DataLoader], Optional[torch.utils.data.DataLoader]]:
        """获取数据加载器"""
        train_loader = None
        if self.train_dataset:
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                collate_fn=self.data_collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        val_loader = None
        if self.val_dataset:
            val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        return train_loader, val_loader


def main():
    """测试函数"""
    from transformers import AutoTokenizer

    # 加载分词器
    model_path = "Qwen/Qwen-7B-Chat"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as e:
        logger.warning(f"加载分词器失败: {e}")
        logger.info("使用模拟测试")
        return

    # 创建数据管道
    pipeline = SpecialtyDataPipeline(
        tokenizer=tokenizer,
        train_data_path="ming/eval/datasets/mmedbench-02-chinese-tiny.jsonl",
        val_data_path=None,
        batch_size=4,
        model_max_length=2048,
    )

    # 初始化数据集
    pipeline.initialize_datasets()

    # 获取数据加载器
    train_loader, _ = pipeline.get_dataloaders(num_workers=0)

    if train_loader:
        # 测试批量加载
        for batch in train_loader:
            print("批量形状:")
            print(f"  input_ids: {batch['input_ids'].shape}")
            print(f"  labels: {batch['labels'].shape}")
            print(f"  attention_mask: {batch['attention_mask'].shape}")

            # 检查填充
            pad_count = (batch["input_ids"] == tokenizer.pad_token_id).sum().item()
            total_count = batch["input_ids"].numel()
            print(f"  填充token比例: {pad_count / total_count:.2%}")
            break


if __name__ == "__main__":
    main()
