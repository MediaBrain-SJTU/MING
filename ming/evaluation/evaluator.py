"""
模型评估器模块

本模块提供模型评估的核心功能：
1. 多数据集评估
2. 批量推理与结果收集
3. 指标计算与聚合
4. 评估结果持久化
"""

import json
import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
)

from ming.evaluation.metrics import EvaluationMetrics
from ming.evaluation.metrics import normalize_answer
from ming.feature_engineering.entity_recognizer import MedicalEntityRecognizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置类"""

    # 基本配置
    output_dir: str = "./evaluation_results"
    eval_batch_size: int = 8
    max_length: int = 2048
    max_new_tokens: int = 512
    num_beams: int = 1
    temperature: float = 0.0
    top_p: float = 1.0

    # 评估选项
    do_sample: bool = False
    use_cache: bool = True
    compute_entity_metrics: bool = True
    compute_bleu: bool = True
    compute_rouge: bool = True

    # 性能监控
    measure_inference_time: bool = True
    measure_memory_usage: bool = True

    # 结果保存
    save_outputs: bool = True
    save_references: bool = True
    save_metrics: bool = True

    # 复现性
    seed: int = 42
    deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


@dataclass
class EvaluationSample:
    """评估样本类"""

    id: str
    question: str
    reference: str
    specialty: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[str] = None
    inference_time_ms: Optional[float] = None
    pred_entities: Optional[List[Tuple[str, str]]] = None
    ref_entities: Optional[List[Tuple[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "id": self.id,
            "question": self.question,
            "reference": self.reference,
            "specialty": self.specialty,
            "metadata": self.metadata,
        }
        if self.prediction is not None:
            result["prediction"] = self.prediction
        if self.inference_time_ms is not None:
            result["inference_time_ms"] = self.inference_time_ms
        if self.pred_entities is not None:
            result["pred_entities"] = [
                {"text": e[0], "type": e[1]} for e in self.pred_entities
            ]
        if self.ref_entities is not None:
            result["ref_entities"] = [
                {"text": e[0], "type": e[1]} for e in self.ref_entities
            ]
        return result


@dataclass
class EvaluationResult:
    """评估结果类"""

    config: EvaluationConfig
    metrics: EvaluationMetrics
    samples: List[EvaluationSample] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    total_time_seconds: float = 0.0
    model_name: str = "unknown"
    model_path: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_info": {
                "name": self.model_name,
                "path": self.model_path,
            },
            "evaluation_info": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_time_seconds": self.total_time_seconds,
            },
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "samples": [s.to_dict() for s in self.samples] if self.config.save_outputs else [],
        }

    def save(self, filepath: str) -> None:
        """保存评估结果为JSON"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EvaluationResult":
        """从JSON加载评估结果"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = EvaluationConfig(**data.get("config", {}))
        metrics = EvaluationMetrics(**data.get("metrics", {}))

        samples = []
        for s in data.get("samples", []):
            sample = EvaluationSample(
                id=s.get("id", ""),
                question=s.get("question", ""),
                reference=s.get("reference", ""),
                specialty=s.get("specialty", "general"),
                metadata=s.get("metadata", {}),
                prediction=s.get("prediction"),
                inference_time_ms=s.get("inference_time_ms"),
            )
            if "pred_entities" in s:
                sample.pred_entities = [
                    (e["text"], e["type"]) for e in s["pred_entities"]
                ]
            if "ref_entities" in s:
                sample.ref_entities = [
                    (e["text"], e["type"]) for e in s["ref_entities"]
                ]
            samples.append(sample)

        result = cls(
            config=config,
            metrics=metrics,
            samples=samples,
            start_time=data.get("evaluation_info", {}).get("start_time", ""),
            end_time=data.get("evaluation_info", {}).get("end_time"),
            total_time_seconds=data.get("evaluation_info", {}).get("total_time_seconds", 0.0),
            model_name=data.get("model_info", {}).get("name", "unknown"),
            model_path=data.get("model_info", {}).get("path", "unknown"),
        )
        return result


class EvaluationDataset(Dataset):
    """评估数据集类"""

    def __init__(
        self,
        samples: List[EvaluationSample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 构建prompt
        prompt = self._build_prompt(sample.question)

        # 分词
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "sample_idx": idx,
        }

    def _build_prompt(self, question: str) -> str:
        """构建推理prompt"""
        return f"问题: {question}\n答案:"


class ModelEvaluator:
    """
    模型评估器

    统一管理模型评估流程，包括：
    1. 数据准备
    2. 批量推理
    3. 实体提取
    4. 指标计算
    5. 结果保存
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: EvaluationConfig,
        model_name: str = "model",
        model_path: str = "",
    ):
        """
        初始化模型评估器

        Args:
            model: 待评估模型
            tokenizer: 分词器
            config: 评估配置
            model_name: 模型名称
            model_path: 模型路径
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_name = model_name
        self.model_path = model_path

        # 设备设置
        self.device = next(model.parameters()).device
        logger.info(f"模型设备: {self.device}")

        # 实体识别器
        self.entity_recognizer = MedicalEntityRecognizer()

        # 生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            temperature=config.temperature if config.do_sample else 0.0,
            top_p=config.top_p if config.do_sample else 1.0,
            do_sample=config.do_sample,
            use_cache=config.use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("模型评估器初始化完成")

    def load_dataset(
        self,
        data_path: str,
    ) -> List[EvaluationSample]:
        """
        加载评估数据集

        Args:
            data_path: 数据路径

        Returns:
            List[EvaluationSample]: 评估样本列表
        """
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        samples = []

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        sample = self._parse_sample(data, line_num)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第 {line_num} 行JSON解析错误: {e}")
                    except Exception as e:
                        logger.warning(f"第 {line_num} 行数据解析错误: {e}")

        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
                if not isinstance(data_list, list):
                    data_list = [data_list]
                for i, data in enumerate(data_list):
                    try:
                        sample = self._parse_sample(data, i)
                        samples.append(sample)
                    except Exception as e:
                        logger.warning(f"第 {i} 个数据解析错误: {e}")

        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

        logger.info(f"从 {data_path} 加载了 {len(samples)} 个评估样本")
        return samples

    def _parse_sample(
        self,
        data: Dict[str, Any],
        idx: int,
    ) -> EvaluationSample:
        """解析单个样本数据"""
        # 尝试不同的字段名称
        question = (
            data.get("question")
            or data.get("query")
            or data.get("text")
            or data.get("input")
            or ""
        )

        reference = (
            data.get("answer")
            or data.get("response")
            or data.get("output")
            or data.get("target")
            or ""
        )

        specialty = data.get("specialty", data.get("specialty_type", "general"))

        # 从对话格式提取
        if "conversations" in data:
            convs = data["conversations"]
            if isinstance(convs, list):
                for conv in convs:
                    if isinstance(conv, dict):
                        role = conv.get("role", conv.get("from", "")).lower()
                        value = conv.get("value", conv.get("text", ""))
                        if role in ["user", "human"] and not question:
                            question = value
                        elif role in ["assistant", "gpt"] and not reference:
                            reference = value

        return EvaluationSample(
            id=str(data.get("id", idx)),
            question=question,
            reference=reference,
            specialty=specialty,
            metadata=data.get("metadata", {}),
        )

    @torch.no_grad()
    def evaluate(
        self,
        eval_samples: List[EvaluationSample],
        dataset_name: str = "evaluation",
    ) -> EvaluationResult:
        """
        执行评估

        Args:
            eval_samples: 评估样本列表
            dataset_name: 数据集名称

        Returns:
            EvaluationResult: 评估结果
        """
        start_time = time.time()
        logger.info(f"开始评估，共 {len(eval_samples)} 个样本")

        # 设置随机种子
        if self.config.deterministic:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

        # 创建数据集和数据加载器
        dataset = EvaluationDataset(
            samples=eval_samples,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.eval_batch_size,
            pin_memory=True,
        )

        # 记录初始显存使用
        initial_memory = 0.0
        if self.config.measure_memory_usage and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**3

        # 批量推理
        self.model.eval()
        results = []

        total_batches = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            batch_results = self._inference_batch(batch, batch_idx, total_batches)
            results.extend(batch_results)

        # 收集实体信息
        if self.config.compute_entity_metrics:
            self._extract_entities(results)

        # 准备指标计算数据
        predictions = [r.prediction or "" for r in results]
        references = [r.reference or "" for r in results]
        pred_entities = [r.pred_entities or [] for r in results]
        ref_entities = [r.ref_entities or [] for r in results]
        inference_times = [r.inference_time_ms or 0.0 for r in results]

        # 计算指标
        metrics = EvaluationMetrics()
        metrics.compute_all(
            predictions=predictions,
            references=references,
            pred_entities=pred_entities if self.config.compute_entity_metrics else None,
            ref_entities=ref_entities if self.config.compute_entity_metrics else None,
            inference_times=inference_times if self.config.measure_inference_time else None,
        )

        # 记录显存使用
        if self.config.measure_memory_usage and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            metrics.memory_usage_gb = peak_memory - initial_memory
            logger.info(f"评估显存使用: {metrics.memory_usage_gb:.2f}GB")

        # 构建结果
        end_time = time.time()
        total_time = end_time - start_time

        eval_result = EvaluationResult(
            config=self.config,
            metrics=metrics,
            samples=results,
            start_time=datetime.fromtimestamp(start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            total_time_seconds=total_time,
            model_name=self.model_name,
            model_path=self.model_path,
        )

        # 保存结果
        if self.config.save_metrics:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = Path(self.config.output_dir) / f"eval_{dataset_name}_{timestamp}.json"
            eval_result.save(str(result_path))

        # 打印摘要
        logger.info(f"评估完成，总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        logger.info(f"单样本平均推理时间: {metrics.inference_time_per_sample:.2f}ms")
        logger.info(f"EM分数: {metrics.em:.4f}, F1分数: {metrics.f1:.4f}")

        return eval_result

    def _inference_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        total_batches: int,
    ) -> List[EvaluationSample]:
        """执行单批量推理"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        sample_indices = batch["sample_idx"].numpy()

        batch_size = input_ids.size(0)
        input_lengths = attention_mask.sum(dim=1)

        # 计时开始
        if self.config.measure_inference_time:
            torch.cuda.synchronize()
            batch_start = time.time()

        # 生成
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
            )

        # 计时结束
        if self.config.measure_inference_time:
            torch.cuda.synchronize()
            batch_time = (time.time() - batch_start) * 1000  # 转换为毫秒
            sample_time = batch_time / batch_size
        else:
            sample_time = 0.0

        # 解码
        results = []
        for i in range(batch_size):
            sample_idx = sample_indices[i]
            input_len = input_lengths[i].item()

            # 提取生成的token（跳过输入部分）
            output_ids = outputs[i, input_len:]

            # 解码
            prediction = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # 获取原始样本
            sample = EvaluationDataset(
                [EvaluationSample("", "", "")], self.tokenizer
            ).samples[0]
            if hasattr(self, "_current_eval_samples"):
                sample = self._current_eval_samples[sample_idx]

            # 更新预测结果
            sample.prediction = prediction.strip()
            sample.inference_time_ms = sample_time

            results.append(sample)

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"已完成 {batch_idx + 1}/{total_batches} 批次")

        return results

    def _extract_entities(self, samples: List[EvaluationSample]) -> None:
        """提取预测和参考答案中的医疗实体"""
        for sample in samples:
            # 预测实体
            if sample.prediction:
                pred_entities, _ = self.entity_recognizer.recognize(sample.prediction)
                sample.pred_entities = [(e.text, e.entity_type.value) for e in pred_entities]

            # 参考实体
            if sample.reference:
                ref_entities, _ = self.entity_recognizer.recognize(sample.reference)
                sample.ref_entities = [(e.text, e.entity_type.value) for e in ref_entities]

    def compare_models(
        self,
        model_results: Dict[str, EvaluationResult],
        output_name: str = "model_comparison",
    ) -> Dict[str, Dict[str, Any]]:
        """
        比较多个模型的评估结果

        Args:
            model_results: 模型名称到评估结果的映射
            output_name: 输出文件名

        Returns:
            Dict[str, Dict[str, Any]]: 比较结果
        """
        comparison = {
            "comparison_info": {
                "timestamp": datetime.now().isoformat(),
                "models": list(model_results.keys()),
            },
            "metrics_comparison": {},
            "per_sample_comparison": [],
        }

        # 收集所有指标
        all_metrics = set()
        for result in model_results.values():
            metrics_dict = result.metrics.to_dict()
            all_metrics.update(self._flatten_metrics(metrics_dict).keys())

        # 计算每个模型的指标
        for model_name, result in model_results.items():
            metrics_dict = result.metrics.to_dict()
            flat_metrics = self._flatten_metrics(metrics_dict)

            for metric_name in all_metrics:
                if metric_name not in comparison["metrics_comparison"]:
                    comparison["metrics_comparison"][metric_name] = {}
                comparison["metrics_comparison"][metric_name][model_name] = (
                    flat_metrics.get(metric_name, 0.0)
                )

        # 单样本比较（如果有相同的样本ID）
        sample_ids = None
        for result in model_results.values():
            ids = {s.id for s in result.samples}
            if sample_ids is None:
                sample_ids = ids
            else:
                sample_ids.intersection_update(ids)

        if sample_ids:
            for sample_id in sample_ids:
                sample_data = {"sample_id": sample_id}
                for model_name, result in model_results.items():
                    sample = next((s for s in result.samples if s.id == sample_id), None)
                    if sample:
                        sample_data[f"{model_name}_prediction"] = sample.prediction
                        sample_data[f"{model_name}_em"] = (
                            1
                            if normalize_answer(sample.prediction or "")
                            == normalize_answer(sample.reference or "")
                            else 0
                        )
                        sample_data["question"] = sample.question
                        sample_data["reference"] = sample.reference
                comparison["per_sample_comparison"].append(sample_data)

        # 保存比较结果
        output_path = Path(self.config.output_dir) / f"{output_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        logger.info(f"模型比较结果已保存到: {output_path}")
        return comparison

    def _flatten_metrics(
        self,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, float]:
        """将嵌套的指标字典扁平化"""
        flat = {}
        for key, value in metrics.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_metrics(value, f"{new_key}_"))
            elif isinstance(value, (int, float)):
                flat[new_key] = float(value)
        return flat


def main():
    """测试函数"""
    print("模型评估器模块测试")

    # 创建评估配置
    config = EvaluationConfig(
        output_dir="./test_evaluation",
        eval_batch_size=2,
        max_new_tokens=100,
        compute_entity_metrics=True,
        measure_inference_time=True,
    )

    print("\n评估配置:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # 创建测试样本
    samples = [
        EvaluationSample(
            id="1",
            question="高血压的治疗方法有哪些？",
            reference="高血压的治疗包括药物治疗和生活方式改变，如低盐饮食、适量运动等。",
            specialty="cardiovascular",
        ),
        EvaluationSample(
            id="2",
            question="糖尿病患者应该如何控制饮食？",
            reference="糖尿病患者应控制碳水化合物摄入，选择低糖、高纤维食物，定时定量进餐。",
            specialty="endocrinology",
        ),
    ]

    print("\n测试样本:")
    for sample in samples:
        print(f"  ID: {sample.id}")
        print(f"  问题: {sample.question}")
        print(f"  参考: {sample.reference}")
        print(f"  专科: {sample.specialty}")
        print()


if __name__ == "__main__":
    main()
