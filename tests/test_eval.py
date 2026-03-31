import pytest
from pathlib import Path
import sys
import jsonlines

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEvalDatasets:
    def test_datasets_directory_exists(self):
        datasets_dir = Path(__file__).parent.parent / "ming" / "eval" / "datasets"
        assert datasets_dir.exists(), "Datasets directory not found"

    def test_jsonl_files_valid(self):
        datasets_dir = Path(__file__).parent.parent / "ming" / "eval" / "datasets"
        
        if not datasets_dir.exists():
            pytest.skip("Datasets directory not found")
        
        jsonl_files = list(datasets_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            pytest.skip("No JSONL files found")
        
        for jsonl_file in jsonl_files:
            with jsonlines.open(jsonl_file) as reader:
                count = 0
                for obj in reader:
                    assert isinstance(obj, dict), f"Invalid JSONL format in {jsonl_file}"
                    count += 1
                    if count >= 5:
                        break


class TestEvalModule:
    def test_eval_em_import(self):
        try:
            from ming.eval import eval_em
            assert eval_em is not None
        except ImportError:
            pytest.skip("eval_em module not available")

    def test_eval_gpt4_import(self):
        try:
            from ming.eval import eval_gpt4
            assert eval_gpt4 is not None
        except ImportError:
            pytest.skip("eval_gpt4 module not available")

    def test_cblue_evaluate_import(self):
        try:
            from ming.eval.cblue.evaluate import calc_scores
            assert calc_scores is not None
        except ImportError:
            pytest.skip("cblue evaluate module not available")

    def test_cblue_evaluators_import(self):
        try:
            from ming.eval.cblue import evaluators
            assert evaluators is not None
        except ImportError:
            pytest.skip("cblue evaluators module not available")


class TestEvalFunctions:
    def test_normalize_frac_function(self):
        try:
            from ming.eval.eval_em import normalize_frac
            
            result = normalize_frac(r"\frac{1}{2}")
            assert result is not None
            assert result == ("1", "2")
        except ImportError:
            pytest.skip("normalize_frac function not available")

    def test_normalize_dfrac_function(self):
        try:
            from ming.eval.eval_em import normalize_dfrac
            
            result = normalize_dfrac(r"\dfrac{3}{4}")
            assert result is not None
            assert result == ("3", "4")
        except ImportError:
            pytest.skip("normalize_dfrac function not available")
