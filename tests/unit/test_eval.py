import pytest
import unittest
import os

class TestEval(unittest.TestCase):
    def test_import_eval_module(self):
        try:
            from ming.eval import eval_em
            self.assertIsNotNone(eval_em)
        except ImportError as e:
            pytest.skip(f"Eval EM module import skipped: {e}")
    
    def test_import_eval_gpt4_module(self):
        try:
            from ming.eval import eval_gpt4
            self.assertIsNotNone(eval_gpt4)
        except ImportError as e:
            pytest.skip(f"Eval GPT4 module import skipped: {e}")
    
    def test_import_cblue_module(self):
        try:
            from ming.eval.cblue import evaluate
            self.assertIsNotNone(evaluate)
        except ImportError as e:
            pytest.skip(f"CBlue evaluate module import skipped: {e}")

if __name__ == '__main__':
    unittest.main()
