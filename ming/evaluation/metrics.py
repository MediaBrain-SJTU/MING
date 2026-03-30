"""
评估指标计算模块
提供多种评估指标的计算方法
"""
import re
import string
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter
import math


@dataclass
class MetricResult:
    """指标计算结果"""
    metric_name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "details": self.details
        }


class BaseMetric(ABC):
    """评估指标基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> MetricResult:
        """
        计算指标
        
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            
        Returns:
            MetricResult: 指标结果
        """
        pass
    
    def normalize_text(self, text: str) -> str:
        """标准化文本"""
        text = text.lower()
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        return list(text)


class ExactMatchMetric(BaseMetric):
    """
    精确匹配指标
    
    计算预测与参考答案完全匹配的比例
    """
    
    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
        normalize_whitespace: bool = True
    ):
        super().__init__("exact_match")
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.normalize_whitespace = normalize_whitespace
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> MetricResult:
        """计算精确匹配率"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
        
        matches = 0
        details = {"matches": [], "mismatches": []}
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            normalized_pred = self._normalize(pred)
            normalized_ref = self._normalize(ref)
            
            if normalized_pred == normalized_ref:
                matches += 1
                details["matches"].append({
                    "index": i,
                    "prediction": pred,
                    "reference": ref
                })
            else:
                details["mismatches"].append({
                    "index": i,
                    "prediction": pred,
                    "reference": ref,
                    "normalized_pred": normalized_pred,
                    "normalized_ref": normalized_ref
                })
        
        em_score = matches / len(references) if references else 0.0
        
        return MetricResult(
            metric_name=self.name,
            value=em_score,
            details={
                "total_samples": len(references),
                "correct_matches": matches,
                "match_details": details
            }
        )
    
    def _normalize(self, text: str) -> str:
        """标准化文本"""
        if self.ignore_case:
            text = text.lower()
        
        if self.ignore_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            chinese_punctuation = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—'
            text = text.translate(str.maketrans('', '', chinese_punctuation))
        
        if self.normalize_whitespace:
            text = ' '.join(text.split())
        
        return text.strip()


class F1Metric(BaseMetric):
    """
    F1分数指标
    
    计算词级别的精确率、召回率和F1分数
    """
    
    def __init__(
        self,
        average: str = "micro",
        ignore_case: bool = True
    ):
        super().__init__("f1")
        self.average = average
        self.ignore_case = ignore_case
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> MetricResult:
        """计算F1分数"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
        
        all_precision = []
        all_recall = []
        all_f1 = []
        details = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            pred_tokens = set(self._tokenize(pred))
            ref_tokens = set(self._tokenize(ref))
            
            if not pred_tokens and not ref_tokens:
                precision = recall = f1 = 1.0
            elif not pred_tokens:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            elif not ref_tokens:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                common = pred_tokens & ref_tokens
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            
            details.append({
                "index": i,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pred_tokens": list(pred_tokens),
                "ref_tokens": list(ref_tokens)
            })
        
        if self.average == "micro":
            avg_precision = sum(all_precision) / len(all_precision)
            avg_recall = sum(all_recall) / len(all_recall)
            avg_f1 = sum(all_f1) / len(all_f1)
        elif self.average == "macro":
            avg_precision = sum(all_precision) / len(all_precision)
            avg_recall = sum(all_recall) / len(all_recall)
            avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        else:
            avg_precision = sum(all_precision) / len(all_precision)
            avg_recall = sum(all_recall) / len(all_recall)
            avg_f1 = sum(all_f1) / len(all_f1)
        
        return MetricResult(
            metric_name=self.name,
            value=avg_f1,
            details={
                "average": self.average,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "per_sample_details": details
            }
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.ignore_case:
            text = text.lower()
        
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        tokens = []
        current_word = ""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                tokens.append(char)
            elif char.isalnum():
                current_word += char
            else:
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
        
        if current_word:
            tokens.append(current_word)
        
        return tokens


class ROUGEMetric(BaseMetric):
    """
    ROUGE指标
    
    计算ROUGE-1, ROUGE-2, ROUGE-L
    """
    
    def __init__(
        self,
        rouge_types: Optional[List[str]] = None,
        ignore_case: bool = True
    ):
        super().__init__("rouge")
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self.ignore_case = ignore_case
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> MetricResult:
        """计算ROUGE分数"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
        
        results = {rouge_type: [] for rouge_type in self.rouge_types}
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            if "rouge1" in self.rouge_types:
                score = self._compute_ngram_f1(pred_tokens, ref_tokens, 1)
                results["rouge1"].append(score)
            
            if "rouge2" in self.rouge_types:
                score = self._compute_ngram_f1(pred_tokens, ref_tokens, 2)
                results["rouge2"].append(score)
            
            if "rougeL" in self.rouge_types:
                score = self._compute_lcs_f1(pred_tokens, ref_tokens)
                results["rougeL"].append(score)
        
        avg_results = {
            rouge_type: sum(scores) / len(scores) if scores else 0.0
            for rouge_type, scores in results.items()
        }
        
        return MetricResult(
            metric_name=self.name,
            value=avg_results.get("rougeL", 0.0),
            details={
                "rouge1": avg_results.get("rouge1", 0.0),
                "rouge2": avg_results.get("rouge2", 0.0),
                "rougeL": avg_results.get("rougeL", 0.0),
                "per_sample_scores": results
            }
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.ignore_case:
            text = text.lower()
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        return text.split()
    
    def _compute_ngram_f1(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str],
        n: int
    ) -> float:
        """计算n-gram F1分数"""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if not pred_ngrams or not ref_ngrams:
            return 0.0
        
        common = pred_ngrams & ref_ngrams
        precision = len(common) / len(pred_ngrams)
        recall = len(common) / len(ref_ngrams)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """获取n-gram"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams
    
    def _compute_lcs_f1(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str]
    ) -> float:
        """计算最长公共子序列F1分数"""
        lcs_len = self._lcs_length(pred_tokens, ref_tokens)
        
        if lcs_len == 0:
            return 0.0
        
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]


class BLEUMetric(BaseMetric):
    """
    BLEU指标
    
    计算BLEU分数
    """
    
    def __init__(
        self,
        max_n: int = 4,
        smooth: bool = True
    ):
        super().__init__("bleu")
        self.max_n = max_n
        self.smooth = smooth
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> MetricResult:
        """计算BLEU分数"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
        
        scores = []
        details = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            score = self._compute_bleu(pred_tokens, ref_tokens)
            scores.append(score)
            details.append({
                "prediction": pred,
                "reference": ref,
                "bleu_score": score
            })
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return MetricResult(
            metric_name=self.name,
            value=avg_score,
            details={
                "max_n": self.max_n,
                "smooth": self.smooth,
                "per_sample_scores": scores,
                "per_sample_details": details
            }
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        text = text.lower()
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        return text.split()
    
    def _compute_bleu(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str]
    ) -> float:
        """计算单个样本的BLEU分数"""
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        brevity_penalty = 1.0
        if len(pred_tokens) < len(ref_tokens):
            brevity_penalty = math.exp(1 - len(ref_tokens) / len(pred_tokens))
        
        precisions = []
        for n in range(1, self.max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            if not pred_ngrams:
                precisions.append(0.0)
                continue
            
            matches = 0
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            if self.smooth:
                precision = (matches + 1) / (len(pred_ngrams) + 1)
            else:
                precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
            
            precisions.append(precision)
        
        if not precisions or all(p == 0 for p in precisions):
            return 0.0
        
        log_precisions = [math.log(p) for p in precisions if p > 0]
        if not log_precisions:
            return 0.0
        
        avg_log_precision = sum(log_precisions) / len(log_precisions)
        
        return brevity_penalty * math.exp(avg_log_precision)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """获取n-gram"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams


class MedicalAccuracyMetric(BaseMetric):
    """
    医疗准确性指标
    
    评估医疗问答的准确性，包括：
    - 诊断准确性
    - 治疗方案准确性
    - 药物推荐准确性
    """
    
    def __init__(
        self,
        entity_recognizer: Optional[Any] = None,
        check_medical_entities: bool = True
    ):
        super().__init__("medical_accuracy")
        self.entity_recognizer = entity_recognizer
        self.check_medical_entities = check_medical_entities
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        questions: Optional[List[str]] = None,
        **kwargs
    ) -> MetricResult:
        """计算医疗准确性"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
        
        results = {
            "exact_match": [],
            "entity_match": [],
            "option_match": []
        }
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            exact_match = self._normalize(pred) == self._normalize(ref)
            results["exact_match"].append(exact_match)
            
            if self.entity_recognizer and self.check_medical_entities:
                pred_entities = self._extract_medical_entities(pred)
                ref_entities = self._extract_medical_entities(ref)
                entity_match = len(pred_entities & ref_entities) > 0
                results["entity_match"].append(entity_match)
            
            option_match = self._check_option_match(pred, ref)
            results["option_match"].append(option_match)
        
        exact_match_rate = sum(results["exact_match"]) / len(results["exact_match"])
        option_match_rate = sum(results["option_match"]) / len(results["option_match"])
        
        entity_match_rate = 0.0
        if results["entity_match"]:
            entity_match_rate = sum(results["entity_match"]) / len(results["entity_match"])
        
        overall_accuracy = (exact_match_rate + option_match_rate + entity_match_rate) / 3
        
        return MetricResult(
            metric_name=self.name,
            value=overall_accuracy,
            details={
                "exact_match_rate": exact_match_rate,
                "option_match_rate": option_match_rate,
                "entity_match_rate": entity_match_rate,
                "per_sample_results": results
            }
        )
    
    def _normalize(self, text: str) -> str:
        """标准化文本"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return text
    
    def _extract_medical_entities(self, text: str) -> set:
        """提取医疗实体"""
        if self.entity_recognizer:
            result = self.entity_recognizer.recognize(text)
            return set(e.text for e in result.entities)
        
        medical_keywords = [
            "诊断", "治疗", "药物", "检查", "症状", "疾病",
            "手术", "用药", "剂量", "疗程"
        ]
        entities = set()
        for keyword in medical_keywords:
            if keyword in text:
                entities.add(keyword)
        return entities
    
    def _check_option_match(self, prediction: str, reference: str) -> bool:
        """检查选项匹配"""
        pred_options = re.findall(r'[A-Da-d]', prediction)
        ref_options = re.findall(r'[A-Da-d]', reference)
        
        if pred_options and ref_options:
            return pred_options[0].upper() == ref_options[0].upper()
        
        return False


class CompositeMetric(BaseMetric):
    """
    组合指标
    
    组合多个指标计算综合得分
    """
    
    def __init__(
        self,
        metrics: List[BaseMetric],
        weights: Optional[List[float]] = None
    ):
        super().__init__("composite")
        self.metrics = metrics
        self.weights = weights or [1.0 / len(metrics)] * len(metrics)
        
        if len(self.weights) != len(self.metrics):
            raise ValueError("权重数量与指标数量不匹配")
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> MetricResult:
        """计算组合指标"""
        results = {}
        weighted_sum = 0.0
        
        for metric, weight in zip(self.metrics, self.weights):
            result = metric.compute(predictions, references, **kwargs)
            results[metric.name] = result.to_dict()
            weighted_sum += result.value * weight
        
        return MetricResult(
            metric_name=self.name,
            value=weighted_sum,
            details={
                "component_metrics": results,
                "weights": self.weights
            }
        )
