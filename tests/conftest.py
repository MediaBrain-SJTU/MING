import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def project_root():
    return Path(__file__).parent


@pytest.fixture
def sample_prompt():
    return "患者主诉头痛三天，伴有发热症状。"


@pytest.fixture
def sample_medical_text():
    return """
    病历摘要：
    患者男性，45岁，主诉头痛三天。
    现病史：患者三天前无明显诱因出现头痛，呈持续性钝痛。
    既往史：高血压病史5年。
    诊断：偏头痛
    """


@pytest.fixture
def mock_model_config():
    return {
        "model_name": "test-model",
        "max_new_tokens": 256,
        "temperature": 1.0,
        "top_p": 0.9,
    }
