"""
集成测试模块
测试特征工程、训练Pipeline和评估指标体系
"""
import unittest
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ming.feature_engineering import (
    MedicalEntityRecognizer,
    MedicalFeatureExtractor,
    FeatureProcessor
)
from ming.feature_engineering.entity_recognition import MedicalEntity, EntityRecognitionResult
from ming.feature_engineering.feature_extractor import MedicalFeatures
from ming.training import (
    SpecialtyDataLoader,
    TrainingConfig,
    MemoryOptimizer
)
from ming.training.data_loader import MedicalQASample, MedicalQADataset
from ming.evaluation import (
    ExactMatchMetric,
    F1Metric,
    ROUGEMetric,
    MedicalAccuracyMetric,
    MedicalQAEvaluator,
    EvaluationReportGenerator
)
from ming.evaluation.metrics import MetricResult
from ming.evaluation.evaluator import EvaluationSample, EvaluationResult


class TestMedicalEntityRecognizer(unittest.TestCase):
    """医疗实体识别器测试"""
    
    def setUp(self):
        self.recognizer = MedicalEntityRecognizer(specialty="cardiovascular")
    
    def test_recognize_basic_entities(self):
        """测试基本实体识别"""
        text = "患者诊断为心肌梗死，伴有胸痛和心悸症状。"
        result = self.recognizer.recognize(text)
        
        self.assertIsInstance(result, EntityRecognitionResult)
        self.assertGreater(len(result.entities), 0)
        self.assertLessEqual(result.processing_time_ms, 50)
    
    def test_entity_types(self):
        """测试实体类型识别"""
        text = "高血压患者服用阿司匹林治疗，心电图显示异常。"
        result = self.recognizer.recognize(text)
        
        entity_types = set(e.entity_type for e in result.entities)
        self.assertIn("DISEASE", entity_types)
        self.assertIn("MEDICINE", entity_types)
    
    def test_specialty_entities(self):
        """测试专科实体识别"""
        text = "亚急性自体瓣膜感染性心内膜炎的主要致病菌是草绿色链球菌。"
        result = self.recognizer.recognize(text)
        
        entity_texts = [e.text for e in result.entities]
        self.assertTrue(any("心内膜炎" in t for t in entity_texts))
    
    def test_processing_time_constraint(self):
        """测试处理时间约束（<= 50ms）"""
        text = "患者，男，50岁，诊断为冠心病，长期服用阿司匹林和美托洛尔，" * 10
        result = self.recognizer.recognize(text)
        
        self.assertLessEqual(result.processing_time_ms, 50, 
                           f"处理时间 {result.processing_time_ms}ms 超过50ms限制")
    
    def test_coverage_rate(self):
        """测试实体覆盖率"""
        text = "心肌梗死患者需要进行心电图检查，服用阿司匹林治疗。"
        result = self.recognizer.recognize(text)
        
        self.assertGreaterEqual(result.coverage_rate, 0.5)
    
    def test_evaluate_f1(self):
        """测试F1评估"""
        pred_entities = [
            [MedicalEntity("心肌梗死", "DISEASE", 0, 4)],
            [MedicalEntity("阿司匹林", "MEDICINE", 0, 4)]
        ]
        gold_entities = [
            [MedicalEntity("心肌梗死", "DISEASE", 0, 4)],
            [MedicalEntity("阿司匹林", "MEDICINE", 0, 4)]
        ]
        
        metrics = self.recognizer.evaluate(pred_entities, gold_entities)
        
        self.assertGreaterEqual(metrics["f1"], 0.9)


class TestMedicalFeatureExtractor(unittest.TestCase):
    """医疗特征提取器测试"""
    
    def setUp(self):
        self.extractor = MedicalFeatureExtractor(specialty="cardiovascular")
    
    def test_extract_features(self):
        """测试特征提取"""
        text = "患者，男，50岁，诊断为心肌梗死，伴有胸痛症状。"
        features = self.extractor.extract(text)
        
        self.assertIsInstance(features, MedicalFeatures)
        self.assertGreater(features.entity_count, 0)
        self.assertEqual(features.question_length, len(text))
    
    def test_extract_age_gender(self):
        """测试年龄性别提取"""
        text = "患者，女，40岁，诊断为高血压。"
        features = self.extractor.extract(text)
        
        self.assertTrue(features.has_age_info)
        self.assertEqual(features.age_value, 40)
        self.assertTrue(features.has_gender_info)
        self.assertEqual(features.gender, "女")
    
    def test_specialty_detection(self):
        """测试专科检测"""
        text = "患者诊断为脑梗死，伴有头痛症状。"
        features = self.extractor.extract(text)
        
        self.assertEqual(features.specialty, "neurology")
    
    def test_difficulty_detection(self):
        """测试难度检测"""
        text = "下列哪项对诊断原发性肝癌有较高特异性？"
        features = self.extractor.extract(text)
        
        self.assertEqual(features.difficulty_level, "easy")
    
    def test_processing_time(self):
        """测试处理时间"""
        text = "患者诊断为心肌梗死。" * 100
        features = self.extractor.extract(text)
        
        self.assertLessEqual(features.processing_time_ms, 100)


class TestFeatureProcessor(unittest.TestCase):
    """特征处理器测试"""
    
    def setUp(self):
        self.processor = FeatureProcessor(normalize=True)
        self.sample_features = [
            MedicalFeatures(
                question_id="1",
                specialty="cardiovascular",
                difficulty_level="medium",
                entity_count=5,
                question_length=100
            ),
            MedicalFeatures(
                question_id="2",
                specialty="neurology",
                difficulty_level="hard",
                entity_count=8,
                question_length=150
            )
        ]
    
    def test_fit_and_process(self):
        """测试拟合和处理"""
        self.processor.fit(self.sample_features)
        
        processed = self.processor.process(self.sample_features[0])
        
        self.assertIsNotNone(processed.feature_vector)
        self.assertEqual(len(processed.feature_vector), self.processor.feature_dim)
    
    def test_one_hot_encoding(self):
        """测试独热编码"""
        self.processor.fit(self.sample_features)
        processed = self.processor.process(self.sample_features[0])
        
        self.assertIn("cardiovascular", processed.specialty_onehot)
        self.assertEqual(processed.specialty_onehot["cardiovascular"], 1)
    
    def test_batch_process(self):
        """测试批量处理"""
        processed_list = self.processor.batch_process(
            self.sample_features, fit_first=True
        )
        
        self.assertEqual(len(processed_list), len(self.sample_features))
    
    def test_feature_names(self):
        """测试特征名称"""
        names = self.processor.get_feature_names()
        
        self.assertGreater(len(names), 0)


class TestSpecialtyDataLoader(unittest.TestCase):
    """专科数据加载器测试"""
    
    def setUp(self):
        self.data_dir = "ming/eval/datasets"
        self.loader = SpecialtyDataLoader(data_dir=self.data_dir)
    
    def test_load_jsonl(self):
        """测试JSONL加载"""
        jsonl_files = list(Path(self.data_dir).glob("*.jsonl"))
        
        if jsonl_files:
            samples = self.loader.load_jsonl(str(jsonl_files[0]))
            self.assertGreater(len(samples), 0)
            self.assertIsInstance(samples[0], MedicalQASample)
    
    def test_specialty_detection(self):
        """测试专科检测"""
        item = {
            "question": "患者诊断为心肌梗死，应如何治疗？",
            "answer": "药物治疗",
            "options": {"A": "手术", "B": "药物"}
        }
        
        specialty = self.loader._detect_specialty(item)
        self.assertEqual(specialty, "cardiovascular")
    
    def test_difficulty_detection(self):
        """测试难度检测"""
        item = {
            "question": "下列哪项是正确的？",
            "meta_info": "第一部分 历年真题"
        }
        
        difficulty = self.loader._detect_difficulty(item)
        self.assertEqual(difficulty, "hard")


class TestTrainingConfig(unittest.TestCase):
    """训练配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertTrue(config.lora_enable)
    
    def test_config_validation(self):
        """测试配置验证"""
        config = TrainingConfig(batch_size=0)
        errors = config.validate()
        
        self.assertGreater(len(errors), 0)
    
    def test_memory_estimate(self):
        """测试内存估算"""
        config = TrainingConfig()
        estimate = config.get_memory_estimate()
        
        self.assertIn("total_estimated_gb", estimate)
        self.assertTrue(estimate["within_constraint"])
    
    def test_yaml_serialization(self):
        """测试YAML序列化"""
        config = TrainingConfig(specialties=["cardiovascular"])
        
        yaml_path = "test_config.yaml"
        config.to_yaml(yaml_path)
        
        loaded_config = TrainingConfig.from_yaml(yaml_path)
        
        self.assertEqual(loaded_config.specialties, ["cardiovascular"])
        
        os.remove(yaml_path)
    
    def test_specialty_config(self):
        """测试专科配置"""
        config = TrainingConfig()
        cardio_config = config.get_specialty_config("cardiovascular")
        
        self.assertIn("cardiovascular", cardio_config.run_name)


class TestMemoryOptimizer(unittest.TestCase):
    """内存优化器测试"""
    
    def setUp(self):
        self.optimizer = MemoryOptimizer(max_memory_gb=22.0)
    
    def test_memory_stats(self):
        """测试内存统计"""
        stats = self.optimizer.get_memory_stats()
        
        self.assertIsInstance(stats.allocated_gb, float)
        self.assertIsInstance(stats.max_allocated_gb, float)
    
    def test_memory_constraint_check(self):
        """测试内存约束检查"""
        within_limit, memory_gb = self.optimizer.check_memory_constraint()
        
        self.assertIsInstance(within_limit, bool)
        self.assertIsInstance(memory_gb, float)
    
    def test_memory_estimate(self):
        """测试内存估算"""
        estimate = self.optimizer.estimate_memory_requirement(
            model_params=7e9,
            batch_size=4,
            sequence_length=512,
            precision="fp16"
        )
        
        self.assertIn("total_estimated_gb", estimate)
    
    def test_memory_recommendations(self):
        """测试内存建议"""
        recommendations = self.optimizer.get_memory_recommendations()
        
        self.assertIsInstance(recommendations, list)


class TestMetrics(unittest.TestCase):
    """评估指标测试"""
    
    def test_exact_match(self):
        """测试精确匹配"""
        metric = ExactMatchMetric()
        
        predictions = ["心肌梗死", "高血压", "糖尿病"]
        references = ["心肌梗死", "高血压", "低血糖"]
        
        result = metric.compute(predictions, references)
        
        self.assertEqual(result.value, 2/3)
    
    def test_f1_metric(self):
        """测试F1分数"""
        metric = F1Metric()
        
        predictions = ["心肌梗死患者", "高血压病"]
        references = ["心肌梗死", "高血压"]
        
        result = metric.compute(predictions, references)
        
        self.assertGreater(result.value, 0.5)
    
    def test_rouge_metric(self):
        """测试ROUGE分数"""
        metric = ROUGEMetric()
        
        predictions = ["患者诊断为心肌梗死"]
        references = ["患者诊断为心肌梗死"]
        
        result = metric.compute(predictions, references)
        
        self.assertEqual(result.value, 1.0)
    
    def test_medical_accuracy(self):
        """测试医疗准确性"""
        metric = MedicalAccuracyMetric()
        
        predictions = ["B", "A", "阿司匹林"]
        references = ["B", "B", "阿司匹林"]
        
        result = metric.compute(predictions, references)
        
        self.assertGreater(result.value, 0.5)


class TestMedicalQAEvaluator(unittest.TestCase):
    """医疗问答评估器测试"""
    
    def setUp(self):
        self.evaluator = MedicalQAEvaluator()
    
    def test_evaluate_samples(self):
        """测试样本评估"""
        samples = [
            EvaluationSample(
                question_id="1",
                question="问题1",
                options={"A": "选项A", "B": "选项B"},
                reference="B",
                prediction="B",
                specialty="cardiovascular",
                difficulty="medium"
            ),
            EvaluationSample(
                question_id="2",
                question="问题2",
                options={"A": "选项A", "B": "选项B"},
                reference="A",
                prediction="B",
                specialty="neurology",
                difficulty="hard"
            )
        ]
        
        result = self.evaluator.evaluate(samples)
        
        self.assertEqual(result.total_samples, 2)
        self.assertIn("exact_match", result.metrics)
    
    def test_reproducibility(self):
        """测试可复现性"""
        samples = [
            EvaluationSample(
                question_id="1",
                question="问题",
                options={},
                reference="答案",
                prediction="答案"
            )
        ]
        
        result1 = self.evaluator.evaluate(samples)
        result2 = self.evaluator.evaluate(samples)
        
        self.assertTrue(
            self.evaluator.verify_reproducibility(result1, result2)
        )
    
    def test_specialty_grouping(self):
        """测试专科分组"""
        samples = [
            EvaluationSample(
                question_id="1",
                question="问题1",
                options={},
                reference="答案",
                prediction="答案",
                specialty="cardiovascular"
            ),
            EvaluationSample(
                question_id="2",
                question="问题2",
                options={},
                reference="答案",
                prediction="答案",
                specialty="neurology"
            )
        ]
        
        result = self.evaluator.evaluate(samples)
        
        self.assertIn("cardiovascular", result.specialty_results)
        self.assertIn("neurology", result.specialty_results)


class TestEvaluationReportGenerator(unittest.TestCase):
    """评估报告生成器测试"""
    
    def setUp(self):
        self.generator = EvaluationReportGenerator(output_dir="test_reports")
        
        self.mock_result = EvaluationResult(
            total_samples=100,
            metrics={
                "exact_match": MetricResult("exact_match", 0.85, {}),
                "f1": MetricResult("f1", 0.88, {})
            },
            specialty_results={
                "cardiovascular": {"exact_match": 0.87, "f1": 0.90},
                "neurology": {"exact_match": 0.82, "f1": 0.85}
            },
            difficulty_results={
                "easy": {"exact_match": 0.95, "f1": 0.96},
                "medium": {"exact_match": 0.85, "f1": 0.88},
                "hard": {"exact_match": 0.75, "f1": 0.80}
            },
            processing_time_seconds=10.5,
            reproducibility_hash="test_hash_123"
        )
    
    def test_generate_report(self):
        """测试报告生成"""
        report = self.generator.generate(self.mock_result)
        
        self.assertIn("report_info", report)
        self.assertIn("evaluation_summary", report)
        self.assertIn("detailed_metrics", report)
    
    def test_report_generation_time(self):
        """测试报告生成时间约束"""
        start_time = os.times().elapsed
        
        report = self.generator.generate(self.mock_result)
        
        generation_time = report["generation_info"]["generation_time_seconds"]
        
        self.assertLessEqual(generation_time, 300)
    
    def test_save_report(self):
        """测试保存报告"""
        report = self.generator.generate(self.mock_result)
        filepath = self.generator.save_report(report, "test_report.json")
        
        self.assertTrue(os.path.exists(filepath))
        
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["evaluation_summary"]["total_samples"], 100)
        
        os.remove(filepath)
        os.rmdir("test_reports")
    
    def test_baseline_comparison(self):
        """测试基线对比"""
        baseline = EvaluationResult(
            total_samples=100,
            metrics={
                "exact_match": MetricResult("exact_match", 0.75, {}),
                "f1": MetricResult("f1", 0.80, {})
            },
            specialty_results={},
            difficulty_results={},
            processing_time_seconds=10.0,
            reproducibility_hash="baseline_hash"
        )
        
        report = self.generator.generate(
            self.mock_result,
            baseline_result=baseline
        )
        
        comparison = report["baseline_comparison"]
        self.assertTrue(comparison["has_baseline"])
        self.assertIn("exact_match", comparison["improvements"])


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整Pipeline"""
        recognizer = MedicalEntityRecognizer(specialty="cardiovascular")
        extractor = MedicalFeatureExtractor(entity_recognizer=recognizer)
        processor = FeatureProcessor()
        
        text = "患者，男，50岁，诊断为心肌梗死，伴有胸痛症状。"
        
        entity_result = recognizer.recognize(text)
        self.assertGreater(len(entity_result.entities), 0)
        
        features = extractor.extract(text)
        self.assertGreater(features.entity_count, 0)
        
        processor.fit([features])
        processed = processor.process(features)
        self.assertIsNotNone(processed.feature_vector)
    
    def test_evaluation_pipeline(self):
        """测试评估Pipeline"""
        evaluator = MedicalQAEvaluator(
            metrics=["exact_match", "f1", "medical_accuracy"]
        )
        generator = EvaluationReportGenerator(output_dir="test_reports")
        
        samples = [
            EvaluationSample(
                question_id=str(i),
                question=f"问题{i}",
                options={"A": "选项A", "B": "选项B"},
                reference="B" if i % 2 == 0 else "A",
                prediction="B" if i % 3 == 0 else "A",
                specialty="cardiovascular" if i % 2 == 0 else "neurology",
                difficulty=["easy", "medium", "hard"][i % 3]
            )
            for i in range(20)
        ]
        
        result = evaluator.evaluate(samples)
        
        self.assertEqual(result.total_samples, 20)
        
        report = generator.generate(result)
        
        self.assertIn("evaluation_summary", report)
        
        if os.path.exists("test_reports"):
            import shutil
            shutil.rmtree("test_reports")


if __name__ == "__main__":
    unittest.main(verbosity=2)
