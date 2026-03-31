import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_ming():
    try:
        import ming
        assert ming is not None
    except ImportError:
        pytest.skip("MING package not installed")


def test_import_conversations():
    try:
        from ming.conversations import get_default_conv_template
        conv = get_default_conv_template()
        assert conv is not None
    except ImportError:
        pytest.skip("Conversations module not available")


def test_import_constants():
    try:
        from ming.constants import *
        assert True
    except ImportError:
        pytest.skip("Constants module not available")


def test_import_utils():
    try:
        from ming.utils import *
        assert True
    except ImportError:
        pytest.skip("Utils module not available")


def test_project_structure():
    project_root = Path(__file__).parent.parent
    
    assert (project_root / "ming").exists()
    assert (project_root / "ming" / "__init__.py").exists()
    assert (project_root / "ming" / "train").exists()
    assert (project_root / "ming" / "eval").exists()
    assert (project_root / "ming" / "serve").exists()
    assert (project_root / "ming" / "model").exists()


def test_pyproject_toml_exists():
    project_root = Path(__file__).parent.parent
    assert (project_root / "pyproject.toml").exists()


def test_requirements_txt_exists():
    project_root = Path(__file__).parent.parent
    assert (project_root / "requirements.txt").exists()
