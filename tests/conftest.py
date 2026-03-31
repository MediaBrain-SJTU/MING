"""
Pytest configuration and fixtures for MING tests.
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture(scope="session")
def sample_conversation():
    """Return a sample conversation for testing."""
    return {
        "system": "你是一个专业的医疗助手。",
        "messages": [
            {"role": "user", "content": "你好，我最近头痛怎么办？"},
            {"role": "assistant", "content": "头痛可能由多种原因引起，建议..."}
        ]
    }


@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path
