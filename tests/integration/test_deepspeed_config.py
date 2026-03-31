import pytest
import json
import os
from pathlib import Path

class TestDeepSpeedConfig:
    def test_deepspeed_configs_exist(self, project_root):
        scripts_dir = os.path.join(project_root, 'scripts')
        config_files = list(Path(scripts_dir).glob('*.json'))
        assert len(config_files) > 0, "No DeepSpeed config files found"
    
    def test_deepspeed_config_valid_json(self, project_root):
        scripts_dir = os.path.join(project_root, 'scripts')
        config_files = list(Path(scripts_dir).glob('*.json'))
        
        for config_file in config_files:
            with open(config_file, 'r', encoding='utf-8') as f:
                try:
                    config = json.load(f)
                    assert isinstance(config, dict)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {config_file}: {e}")
    
    def test_deepspeed_config_required_fields(self, project_root):
        scripts_dir = os.path.join(project_root, 'scripts')
        config_files = list(Path(scripts_dir).glob('*.json'))
        
        required_fields = ['train_batch_size', 'gradient_accumulation_steps']
        
        for config_file in config_files:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                for field in required_fields:
                    assert field in config or 'train_micro_batch_size_per_gpu' in config, \
                        f"Missing required field {field} in {config_file}"
