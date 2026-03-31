import pytest
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDeepSpeedConfig:
    def test_zero2_config_exists(self):
        config_path = Path(__file__).parent.parent / "scripts" / "zero2.json"
        assert config_path.exists(), "zero2.json config file not found"

    def test_zero3_config_exists(self):
        config_path = Path(__file__).parent.parent / "scripts" / "zero3.json"
        assert config_path.exists(), "zero3.json config file not found"

    def test_zero3_offload_config_exists(self):
        config_path = Path(__file__).parent.parent / "scripts" / "zero3_offload.json"
        assert config_path.exists(), "zero3_offload.json config file not found"

    def test_zero2_config_valid(self):
        config_path = Path(__file__).parent.parent / "scripts" / "zero2.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 2
        assert "fp16" in config or "bf16" in config

    def test_zero3_config_valid(self):
        config_path = Path(__file__).parent.parent / "scripts" / "zero3.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 3
        assert "fp16" in config or "bf16" in config


class TestTrainModule:
    def test_train_module_import(self):
        try:
            from ming.train import train
            assert train is not None
        except ImportError:
            pytest.skip("Train module not available")

    def test_trainer_import(self):
        try:
            from ming.train.trainer import MINGTrainer
            assert MINGTrainer is not None
        except ImportError:
            pytest.skip("MINGTrainer not available")

    def test_train_script_syntax(self):
        train_path = Path(__file__).parent.parent / "ming" / "train" / "train.py"
        assert train_path.exists()
        
        with open(train_path) as f:
            content = f.read()
        
        compile(content, train_path, 'exec')


class TestModelModule:
    def test_model_builder_import(self):
        try:
            from ming.model.builder import load_pretrained_model
            assert load_pretrained_model is not None
        except ImportError:
            pytest.skip("Model builder not available")

    def test_model_utils_import(self):
        try:
            from ming.model.utils import get_mixoflora_model
            assert get_mixoflora_model is not None
        except ImportError:
            pytest.skip("Model utils not available")
