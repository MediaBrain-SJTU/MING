import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def test_data_dir(project_root):
    return os.path.join(project_root, 'tests', 'data')
