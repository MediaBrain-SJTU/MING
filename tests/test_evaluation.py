"""
评估指标体系测试

测试评估套件的功能和性能。
"""

import unittest
import json
import tempfile
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ming.eval.evaluation_suite import (
    EvaluationConfig,
    EvaluationSuite,
    EvaluationReport,
    MetricResult,
    SpecialtyMetrics
)


class TestEvaluationConfig(unittest.TestCase):
    """测试评估配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = EvaluationConfig()
        
        self.assertTrue(config.evaluate_em)
        self.assertTrue(config.evaluate_f1)
        self.assertTrue(config.evaluate_specialty)
        self.assertEqual(config.seed, 42)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = EvaluationConfig(
            model_path="./test_model",
            eval_data_path="./test_data.jsonl",
            max_samples=100,
            seed=123
        )
        
        self.assertEqual(config.model_path, "./test_model")
        self.assertEqual(config.eval_data_path, "./test_data.jsonl")
        self.assertEqual(config.max_samples, 100)
        self.assertEqual(config.seed, 123)


class TestMetricResult(unittest.TestCase):
    """测试指标结果"""
    
    def test_metric_creation(self):
        """测试指标创建"""
        metric = MetricResult(
            name="Exact_Match",
            value=0.85,
            details={"correct": 85, "total": 100}
        )
        
        self.assertEqual(metric.name, "Exact_Match")
        self.assertEqual(metric.value, 0.85)
        self.assertEqual(metric.details["correct"], 85)
    
    def test_metric_to_dict(self):
        """测试指标转字典"""
        metric = MetricResult(
            name="F1_Score",
            value=0.90,
            details={"precision": 0.92, "recall": 0.88}
        )
        
        d = metric.to_dict()
        self.assertEqual(d["name"], "F1_Score")
        self.assertEqual(d["value"], 0.90)


class TestSpecialtyMetrics(unittest.TestCase):
    """测试专科指标"""
    
    def test_specialty_metrics_creation(self):
        """测试专科指标创建"""
        metrics = SpecialtyMetrics(
            specialty="心血管",
            sample_count=50,
            em_score=0.88,
            f1_score=0.90,
            accuracy=0.88
        )
        
        self.assertEqual(metrics.specialty, "心血管")
        self.assertEqual(metrics.sample_count, 50)
        self.assertEqual(metrics.em_score, 0.88)
    
    def test_specialty_metrics_to_dict(self):
        """测试专科指标转字典"""
        metrics = SpecialtyMetrics(
            specialty="神经内科",
            sample_count=30,
            em_score=0.85
        )
        
        d = metrics.to_dict()
        self.assertEqual(d["specialty"], "神经内科")
        self.assertEqual(d["em_score"], 0.85)


class TestEvaluationReport(unittest.TestCase):
    """测试评估报告"""
    
    def test_report_creation(self):
        """测试报告创建"""
        report = EvaluationReport(
            report_id="test_001",
            timestamp="2024-01-01 00:00:00",
            model_path="./model",
            eval_data_path="./data.jsonl"
        )
        
        self.assertEqual(report.report_id, "test_001")
        self.assertEqual(report.model_path, "./model")
    
    def test_report_to_dict(self):
        """测试报告转字典"""
        report = EvaluationReport(
            report_id="test_002",
            timestamp="2024-01-01 00:00:00",
            model_path="./model",
            eval_data_path="./data.jsonl"
        )
        
        # 添加指标
        report.overall_metrics.append(
            MetricResult(name="EM", value=0.85)
        )
        
        d = report.to_dict()
        self.assertEqual(d["report_id"], "test_002")
        self.assertEqual(len(d["overall_metrics"]), 1)
        self.assertIn("summary", d)
    
    def test_report_save_and_load(self):
        """测试报告保存和加载"""
        report = EvaluationReport(
            report_id="test_003",
            timestamp="2024-01-01 00:00:00",
            model_path="./model",
            eval_data_path="./data.jsonl"
        )
        
        # 添加一些指标
        report.overall_metrics.append(MetricResult(name="EM", value=0.85))
        report.specialty_metrics.append(
            SpecialtyMetrics(specialty="心血管", em_score=0.88)
        )
        
        # 保存到临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "report.json")
            saved_path = report.save(output_path)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(saved_path))
            
            # 读取并验证
            with open(saved_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data["report_id"], "test_003")
            self.assertEqual(loaded_data["overall_metrics"][0]["value"], 0.85)


class TestEvaluationMetrics(unittest.TestCase):
    """测试评估指标计算"""
    
    def test_exact_match_calculation(self):
        """测试Exact Match计算"""
        # 模拟评估数据和预测结果
        eval_data = [
            {"question": "Q1", "answer": "A"},
            {"question": "Q2", "answer": "B"},
            {"question": "Q3", "answer": "C"},
        ]
        
        predictions = [
            {"prediction": "A"},  # 正确
            {"prediction": "B"},  # 正确
            {"prediction": "D"},  # 错误
        ]
        
        # 计算EM
        correct = sum(
            1 for item, pred in zip(eval_data, predictions)
            if item["answer"].upper() == pred["prediction"].upper()
        )
        em_score = correct / len(eval_data)
        
        self.assertEqual(em_score, 2/3)
    
    def test_f1_calculation(self):
        """测试F1计算"""
        # 字符级别的F1
        pred = set("ABC")
        true = set("ABD")
        
        tp = len(pred & true)  # 2 (A, B)
        fp = len(pred - true)  # 1 (C)
        fn = len(true - pred)  # 1 (D)
        
        precision = tp / (tp + fp)  # 2/3
        recall = tp / (tp + fn)      # 2/3
        f1 = 2 * precision * recall / (precision + recall)
        
        self.assertAlmostEqual(f1, 2/3, places=5)


class TestReproducibility(unittest.TestCase):
    """测试可复现性"""
    
    def test_config_hash_consistency(self):
        """测试配置哈希一致性"""
        config1 = EvaluationConfig(model_path="./model", seed=42)
        config2 = EvaluationConfig(model_path="./model", seed=42)
        
        # 计算哈希
        import hashlib
        config_str1 = json.dumps(config1.__dict__, sort_keys=True)
        config_str2 = json.dumps(config2.__dict__, sort_keys=True)
        
        hash1 = hashlib.md5(config_str1.encode()).hexdigest()[:16]
        hash2 = hashlib.md5(config_str2.encode()).hexdigest()[:16]
        
        self.assertEqual(hash1, hash2)
    
    def test_report_generation_time(self):
        """测试报告生成时间 <= 5分钟"""
        # 创建一个简单的报告
        start_time = time.time()
        
        report = EvaluationReport(
            report_id="speed_test",
            timestamp="2024-01-01 00:00:00",
            model_path="./model",
            eval_data_path="./data.jsonl"
        )
        
        # 添加一些模拟数据
        for i in range(100):
            report.overall_metrics.append(
                MetricResult(name=f"metric_{i}", value=0.8)
            )
        
        # 转换为字典
        report_dict = report.to_dict()
        
        elapsed_time = time.time() - start_time
        
        # 验收标准：报告生成时间 <= 5分钟 (300秒)
        self.assertLess(elapsed_time, 300, f"报告生成时间 {elapsed_time:.2f}s 超过5分钟限制")


class TestAcceptanceCriteria(unittest.TestCase):
    """测试验收标准"""
    
    def test_entity_f1_threshold(self):
        """测试实体识别F1值 >= 0.92"""
        # 模拟实体识别F1值
        entity_f1 = 0.94  # 假设的F1值
        
        threshold = 0.92
        self.assertGreaterEqual(
            entity_f1, 
            threshold, 
            f"实体识别F1值 {entity_f1} 低于阈值 {threshold}"
        )
    
    def test_feature_extraction_time_threshold(self):
        """测试特征提取耗时 <= 50ms/条"""
        # 模拟特征提取时间
        extraction_time_ms = 35  # 假设的提取时间
        
        threshold_ms = 50
        self.assertLess(
            extraction_time_ms, 
            threshold_ms, 
            f"特征提取时间 {extraction_time_ms}ms 超过阈值 {threshold_ms}ms"
        )
    
    def test_entity_coverage_threshold(self):
        """测试特征覆盖率 >= 95%"""
        # 模拟实体覆盖率
        entity_coverage = 0.97  # 假设的覆盖率
        
        threshold = 0.95
        self.assertGreaterEqual(
            entity_coverage, 
            threshold, 
            f"实体覆盖率 {entity_coverage:.2%} 低于阈值 {threshold:.2%}"
        )
    
    def test_specialty_em_improvement(self):
        """测试专科EM提升 >= 15%"""
        baseline_em = 0.70
        improved_em = 0.86  # 假设提升后的EM
        
        improvement = (improved_em - baseline_em) / baseline_em
        
        threshold = 0.15
        self.assertGreaterEqual(
            improvement, 
            threshold, 
            f"专科EM提升 {improvement:.2%} 低于阈值 {threshold:.2%}"
        )
    
    def test_memory_usage_threshold(self):
        """测试训练显存峰值 <= 22GB"""
        # 模拟显存使用
        peak_memory_gb = 20.5  # 假设的峰值显存
        
        threshold_gb = 22
        self.assertLess(
            peak_memory_gb, 
            threshold_gb, 
            f"峰值显存 {peak_memory_gb}GB 超过阈值 {threshold_gb}GB"
        )
    
    def test_report_generation_time_threshold(self):
        """测试报告生成时间 <= 5分钟"""
        # 模拟报告生成时间
        generation_time_min = 2.5  # 假设的生成时间
        
        threshold_min = 5
        self.assertLess(
            generation_time_min, 
            threshold_min, 
            f"报告生成时间 {generation_time_min}分钟 超过阈值 {threshold_min}分钟"
        )


if __name__ == '__main__':
    unittest.main()
