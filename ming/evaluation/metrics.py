"""
评估指标计算模块

本模块提供多种评估指标的计算功能：
1. 精确匹配（EM）
2. F1分数
3. BLEU分数
4. ROUGE分数
5. 医疗实体识别分数
"""

import re
import string
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import Counter

import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_answer(s: str) -> str:
    """
    标准化答案文本

    包括：小写转换、标点移除、多余空格移除、冠词移除（中文不需要）

    Args:
        s: 输入文本

    Returns:
        str: 标准化后的文本
    """
    # 中文标点和英文标点
    cn_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    en_punc = string.punctuation
    all_punc = cn_punc + en_punc

    # 转换为小写
    s = s.lower()

    # 移除标点
    exclude = set(all_punc)
    s = "".join(ch for ch in s if ch not in exclude)

    # 移除多余空格
    s = re.sub(r"\s+", "", s).strip()

    return s


def compute_em_score(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> float:
    """
    计算精确匹配（Exact Match）分数

    Args:
        predictions: 预测答案列表
        references: 参考答案列表
        normalize: 是否标准化文本

    Returns:
        float: EM分数（0-1）
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"预测数量({len(predictions)})和参考数量({len(references)})不匹配"
        )

    em_count = 0
    for pred, ref in zip(predictions, references):
        if normalize:
            pred_norm = normalize_answer(pred)
            ref_norm = normalize_answer(ref)
        else:
            pred_norm = pred
            ref_norm = ref

        if pred_norm == ref_norm:
            em_count += 1

    return em_count / len(predictions) if predictions else 0.0


def compute_f1_score(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> Tuple[float, float, float]:
    """
    计算F1分数（基于token级别的匹配）

    Args:
        predictions: 预测答案列表
        references: 参考答案列表
        normalize: 是否标准化文本

    Returns:
        Tuple[float, float, float]: (平均F1, 平均精确率, 平均召回率)
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"预测数量({len(predictions)})和参考数量({len(references)})不匹配"
        )

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for pred, ref in zip(predictions, references):
        if normalize:
            pred_norm = normalize_answer(pred)
            ref_norm = normalize_answer(ref)
        else:
            pred_norm = pred
            ref_norm = ref

        # 按字符分割（中文）
        pred_tokens = list(pred_norm)
        ref_tokens = list(ref_norm)

        # 计算共现token
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            f1_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    return (
        np.mean(f1_scores) if f1_scores else 0.0,
        np.mean(precision_scores) if precision_scores else 0.0,
        np.mean(recall_scores) if recall_scores else 0.0,
    )


def compute_bleu_score(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    计算BLEU分数

    Args:
        predictions: 预测答案列表
        references: 参考答案列表（每个预测可以对应多个参考）
        max_n: 最大n-gram

    Returns:
        Dict[str, float]: BLEU分数（包括bleu-1到bleu-n和brevity_penalty）
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    if len(predictions) != len(references):
        raise ValueError(
            f"预测数量({len(predictions)})和参考数量({len(references)})不匹配"
        )

    smoothie = SmoothingFunction().method4
    bleu_scores = {f"bleu-{i + 1}": [] for i in range(max_n)}
    brevity_penalties = []

    for pred, refs in zip(predictions, references):
        # 按字符分割（中文）
        pred_tokens = list(normalize_answer(pred))
        ref_tokens_list = [list(normalize_answer(ref)) for ref in refs]

        # 计算不同n-gram的BLEU
        for n in range(1, max_n + 1):
            weights = [1.0 / n if i < n else 0.0 for i in range(max_n)]
            try:
                score = sentence_bleu(
                    ref_tokens_list,
                    pred_tokens,
                    weights=weights,
                    smoothing_function=smoothie,
                )
                bleu_scores[f"bleu-{n}"].append(score)
            except Exception as e:
                logger.debug(f"计算bleu-{n}失败: {e}")
                bleu_scores[f"bleu-{n}"].append(0.0)

        # 简洁惩罚
        ref_len = min(len(ref) for ref in ref_tokens_list)
        pred_len = len(pred_tokens)
        if pred_len > ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
        brevity_penalties.append(bp)

    result = {
        k: np.mean(v) if v else 0.0 for k, v in bleu_scores.items()
    }
    result["brevity_penalty"] = (
        np.mean(brevity_penalties) if brevity_penalties else 0.0
    )

    return result


def compute_rouge_score(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    计算ROUGE分数

    Args:
        predictions: 预测答案列表
        references: 参考答案列表
        normalize: 是否标准化文本

    Returns:
        Dict[str, Dict[str, float]]: ROUGE分数（包括rouge-1, rouge-2, rouge-l的precision, recall, f1）
    """
    rouge_scores = {
        "rouge-1": {"precision": [], "recall": [], "f1": []},
        "rouge-2": {"precision": [], "recall": [], "f1": []},
        "rouge-l": {"precision": [], "recall": [], "f1": []},
    }

    for pred, ref in zip(predictions, references):
        if normalize:
            pred_norm = normalize_answer(pred)
            ref_norm = normalize_answer(ref)
        else:
            pred_norm = pred
            ref_norm = ref

        # 按字符分割
        pred_chars = list(pred_norm)
        ref_chars = list(ref_norm)

        # ROUGE-1 (unigram)
        r1 = _compute_rouge_n(pred_chars, ref_chars, n=1)
        rouge_scores["rouge-1"]["precision"].append(r1["precision"])
        rouge_scores["rouge-1"]["recall"].append(r1["recall"])
        rouge_scores["rouge-1"]["f1"].append(r1["f1"])

        # ROUGE-2 (bigram)
        if len(pred_chars) >= 2 and len(ref_chars) >= 2:
            r2 = _compute_rouge_n(pred_chars, ref_chars, n=2)
            rouge_scores["rouge-2"]["precision"].append(r2["precision"])
            rouge_scores["rouge-2"]["recall"].append(r2["recall"])
            rouge_scores["rouge-2"]["f1"].append(r2["f1"])
        else:
            rouge_scores["rouge-2"]["precision"].append(0.0)
            rouge_scores["rouge-2"]["recall"].append(0.0)
            rouge_scores["rouge-2"]["f1"].append(0.0)

        # ROUGE-L (longest common subsequence)
        rl = _compute_rouge_l(pred_chars, ref_chars)
        rouge_scores["rouge-l"]["precision"].append(rl["precision"])
        rouge_scores["rouge-l"]["recall"].append(rl["recall"])
        rouge_scores["rouge-l"]["f1"].append(rl["f1"])

    # 计算平均值
    result = {}
    for rouge_type, scores in rouge_scores.items():
        result[rouge_type] = {
            metric: np.mean(values) if values else 0.0
            for metric, values in scores.items()
        }

    return result


def _compute_rouge_n(
    pred: List[str],
    ref: List[str],
    n: int,
) -> Dict[str, float]:
    """计算ROUGE-N分数"""
    if n <= 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def get_ngrams(tokens: List[str], n: int) -> Counter:
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
        return Counter(ngrams)

    pred_ngrams = get_ngrams(pred, n)
    ref_ngrams = get_ngrams(ref, n)

    if not pred_ngrams and not ref_ngrams:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = pred_ngrams & ref_ngrams
    num_intersection = sum(intersection.values())

    precision = num_intersection / sum(pred_ngrams.values())
    recall = num_intersection / sum(ref_ngrams.values())
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def _compute_rouge_l(
    pred: List[str],
    ref: List[str],
) -> Dict[str, float]:
    """计算ROUGE-L分数（基于最长公共子序列）"""
    if not pred and not ref:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # 计算LCS长度
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_length = dp[m][n]

    precision = lcs_length / m
    recall = lcs_length / n
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_medical_entity_score(
    pred_entities: List[List[Tuple[str, str]]],
    ref_entities: List[List[Tuple[str, str]]],
) -> Dict[str, float]:
    """
    计算医疗实体识别分数

    Args:
        pred_entities: 预测实体列表，每个元素是(实体文本, 实体类型)的列表
        ref_entities: 参考实体列表，每个元素是(实体文本, 实体类型)的列表

    Returns:
        Dict[str, float]: 实体识别指标（precision, recall, f1, 各类型f1）
    """
    if len(pred_entities) != len(ref_entities):
        raise ValueError(
            f"预测数量({len(pred_entities)})和参考数量({len(ref_entities)})不匹配"
        )

    entity_types = set()
    for entities in ref_entities:
        for _, ent_type in entities:
            entity_types.add(ent_type)

    type_correct: Dict[str, int] = {t: 0 for t in entity_types}
    type_pred: Dict[str, int] = {t: 0 for t in entity_types}
    type_ref: Dict[str, int] = {t: 0 for t in entity_types}

    total_correct = 0
    total_pred = 0
    total_ref = 0

    for pred_ents, ref_ents in zip(pred_entities, ref_entities):
        # 转换为集合便于比较
        pred_set = set(pred_ents)
        ref_set = set(ref_ents)

        # 统计总体
        correct = len(pred_set & ref_set)
        total_correct += correct
        total_pred += len(pred_set)
        total_ref += len(ref_set)

        # 统计各类型
        for ent_text, ent_type in pred_set:
            type_pred[ent_type] = type_pred.get(ent_type, 0) + 1
        for ent_text, ent_type in ref_set:
            type_ref[ent_type] = type_ref.get(ent_type, 0) + 1
        for ent_text, ent_type in pred_set & ref_set:
            type_correct[ent_type] = type_correct.get(ent_type, 0) + 1

    # 计算总体指标
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_ref if total_ref > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # 计算各类型F1
    for ent_type in entity_types:
        tp = type_correct[ent_type]
        p = type_pred[ent_type]
        r = type_ref[ent_type]

        type_precision = tp / p if p > 0 else 0.0
        type_recall = tp / r if r > 0 else 0.0
        type_f1 = (
            2 * type_precision * type_recall / (type_precision + type_recall)
            if (type_precision + type_recall) > 0
            else 0.0
        )

        result[f"{ent_type}_f1"] = type_f1

    return result


@dataclass
class EvaluationMetrics:
    """
    评估指标集合类

    统一管理所有评估指标的计算和存储
    """

    em: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    bleu: Dict[str, float] = None
    rouge: Dict[str, Dict[str, float]] = None
    entity_metrics: Dict[str, float] = None
    inference_time_per_sample: float = 0.0
    memory_usage_gb: float = 0.0
    sample_count: int = 0

    def __post_init__(self):
        if self.bleu is None:
            self.bleu = {}
        if self.rouge is None:
            self.rouge = {}
        if self.entity_metrics is None:
            self.entity_metrics = {}

    def compute_all(
        self,
        predictions: List[str],
        references: List[str],
        pred_entities: Optional[List[List[Tuple[str, str]]]] = None,
        ref_entities: Optional[List[List[Tuple[str, str]]]] = None,
        inference_times: Optional[List[float]] = None,
    ) -> None:
        """
        计算所有评估指标

        Args:
            predictions: 预测答案列表
            references: 参考答案列表
            pred_entities: 预测实体列表
            ref_entities: 参考实体列表
            inference_times: 每个样本的推理时间（毫秒）
        """
        self.sample_count = len(predictions)

        # EM分数
        self.em = compute_em_score(predictions, references)

        # F1分数
        self.f1, self.precision, self.recall = compute_f1_score(predictions, references)

        # BLEU分数
        references_list = [[ref] for ref in references]
        self.bleu = compute_bleu_score(predictions, references_list)

        # ROUGE分数
        self.rouge = compute_rouge_score(predictions, references)

        # 实体识别分数
        if pred_entities and ref_entities:
            self.entity_metrics = compute_medical_entity_score(
                pred_entities, ref_entities
            )

        # 推理时间
        if inference_times:
            self.inference_time_per_sample = (
                np.mean(inference_times) if inference_times else 0.0
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "em": self.em,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "bleu": self.bleu,
            "rouge": self.rouge,
            "entity_metrics": self.entity_metrics,
            "inference_time_per_sample_ms": self.inference_time_per_sample,
            "memory_usage_gb": self.memory_usage_gb,
            "sample_count": self.sample_count,
        }

    def summary(self) -> str:
        """生成指标摘要"""
        lines = [
            "=" * 50,
            "评估指标摘要",
            "=" * 50,
            f"样本数量: {self.sample_count}",
            f"EM分数: {self.em:.4f}",
            f"F1分数: {self.f1:.4f}",
            f"精确率: {self.precision:.4f}",
            f"召回率: {self.recall:.4f}",
        ]

        if self.bleu:
            lines.append("")
            lines.append("BLEU分数:")
            for n in range(1, 5):
                bleu_n = self.bleu.get(f"bleu-{n}", 0.0)
                lines.append(f"  bleu-{n}: {bleu_n:.4f}")

        if self.rouge:
            lines.append("")
            lines.append("ROUGE分数:")
            for rouge_type in ["rouge-1", "rouge-2", "rouge-l"]:
                scores = self.rouge.get(rouge_type, {})
                lines.append(
                    f"  {rouge_type}: P={scores.get('precision', 0):.4f}, "
                    f"R={scores.get('recall', 0):.4f}, "
                    f"F1={scores.get('f1', 0):.4f}"
                )

        if self.entity_metrics:
            lines.append("")
            lines.append("医疗实体识别:")
            lines.append(f"  总体F1: {self.entity_metrics.get('f1', 0):.4f}")

        if self.inference_time_per_sample > 0:
            lines.append("")
            lines.append("性能指标:")
            lines.append(f"  单样本推理时间: {self.inference_time_per_sample:.2f}ms")

        if self.memory_usage_gb > 0:
            lines.append(f"  显存使用: {self.memory_usage_gb:.2f}GB")

        lines.append("=" * 50)
        return "\n".join(lines)


def main():
    """测试函数"""
    print("评估指标计算模块测试")

    # 测试数据
    predictions = [
        "高血压的治疗方法包括药物治疗和生活方式改变",
        "糖尿病患者需要控制血糖水平",
    ]
    references = [
        "高血压的治疗包括药物治疗和生活方式改变",
        "糖尿病患者需要控制血糖",
    ]

    print("\n测试数据:")
    for i, (p, r) in enumerate(zip(predictions, references)):
        print(f"样本 {i + 1}:")
        print(f"  预测: {p}")
        print(f"  参考: {r}")

    # 测试EM分数
    em = compute_em_score(predictions, references)
    print(f"\nEM分数: {em:.4f}")

    # 测试F1分数
    f1, prec, rec = compute_f1_score(predictions, references)
    print(f"F1分数: {f1:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}")

    # 测试BLEU分数
    references_list = [[r] for r in references]
    bleu = compute_bleu_score(predictions, references_list, max_n=2)
    print("BLEU分数:")
    for k, v in bleu.items():
        print(f"  {k}: {v:.4f}")

    # 测试ROUGE分数
    rouge = compute_rouge_score(predictions, references)
    print("ROUGE分数:")
    for rouge_type, scores in rouge.items():
        print(
            f"  {rouge_type}: P={scores['precision']:.4f}, "
            f"R={scores['recall']:.4f}, F1={scores['f1']:.4f}"
        )

    # 测试实体识别分数
    pred_entities = [
        [("高血压", "疾病"), ("药物治疗", "治疗")],
        [("糖尿病", "疾病"), ("血糖", "检查")],
    ]
    ref_entities = [
        [("高血压", "疾病"), ("药物治疗", "治疗"), ("生活方式改变", "治疗")],
        [("糖尿病", "疾病"), ("血糖", "检查")],
    ]

    entity_scores = compute_medical_entity_score(pred_entities, ref_entities)
    print("\n实体识别分数:")
    for k, v in entity_scores.items():
        print(f"  {k}: {v:.4f}")

    # 测试EvaluationMetrics类
    metrics = EvaluationMetrics()
    metrics.compute_all(
        predictions,
        references,
        pred_entities=pred_entities,
        ref_entities=ref_entities,
        inference_times=[120.5, 95.3],
    )
    print("\n" + metrics.summary())


if __name__ == "__main__":
    main()
