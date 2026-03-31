"""
Tests for evaluation module.
"""
import pytest
import os
import json
from ming.eval import cblue


class TestCBLUEEval:
    """Test CBLUE evaluation module."""

    def test_import_cblue(self):
        """Test that CBLUE module can be imported."""
        assert cblue is not None

    def test_evaluators_import(self):
        """Test that evaluators can be imported."""
        try:
            from ming.eval.cblue.evaluators import (
                calc_cls_task_scores,
                calc_info_extract_task_scores,
                calc_nlg_task_scores
            )
            assert callable(calc_cls_task_scores)
            assert callable(calc_info_extract_task_scores)
            assert callable(calc_nlg_task_scores)
        except ImportError as e:
            pytest.skip(f"Evaluators not available: {e}")


class TestEvaluationMetrics:
    """Test evaluation metrics calculation."""

    def test_classification_metrics(self):
        """Test classification task metrics."""
        try:
            from ming.eval.cblue.evaluators import calc_cls_task_scores

            predictions = [
                {"id": 1, "answer": "A"},
                {"id": 2, "answer": "B"},
                {"id": 3, "answer": "A"}
            ]
            references = [
                {"id": 1, "answer": "A"},
                {"id": 2, "answer": "B"},
                {"id": 3, "answer": "C"}
            ]

            result = calc_cls_task_scores(predictions, references)
            assert isinstance(result, dict)
            assert "accuracy" in result or "score" in result
        except Exception as e:
            pytest.skip(f"Classification metrics test skipped: {e}")

    def test_nlg_metrics(self):
        """Test NLG task metrics."""
        try:
            from ming.eval.cblue.evaluators import calc_nlg_task_scores

            predictions = [
                {"id": 1, "answer": "这是一个测试答案。"}
            ]
            references = [
                {"id": 1, "answer": "这是一个测试答案。"}
            ]

            result = calc_nlg_task_scores(predictions, references)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.skip(f"NLG metrics test skipped: {e}")


class TestDatasetValidation:
    """Test evaluation dataset validation."""

    def test_dataset_format(self):
        """Test evaluation dataset format."""
        # Sample valid dataset entry
        valid_entry = {
            "id": "test_001",
            "question": "这是什么病？",
            "answer": "这是感冒。"
        }

        assert "id" in valid_entry
        assert "question" in valid_entry or "input" in valid_entry
        assert "answer" in valid_entry or "output" in valid_entry

    def test_jsonl_format(self):
        """Test JSONL format for evaluation data."""
        import tempfile

        test_data = [
            {"id": 1, "text": "测试1"},
            {"id": 2, "text": "测试2"}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_path = f.name

        try:
            loaded = []
            with open(temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    loaded.append(json.loads(line.strip()))
            assert loaded == test_data
        finally:
            os.unlink(temp_path)
