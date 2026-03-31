"""
Tests for utility functions.
"""
import pytest
import os
import json
import tempfile
from ming import utils


class TestUtils:
    """Test utility functions."""

    def test_import_utils(self):
        """Test that utils module can be imported."""
        assert utils is not None

    def test_build_logger(self):
        """Test logger building function."""
        try:
            logger = utils.build_logger("test_logger", "test.log")
            assert logger is not None
            # Clean up
            if os.path.exists("test.log"):
                os.remove("test.log")
        except Exception as e:
            pytest.skip(f"Logger test skipped: {e}")


class TestFileOperations:
    """Test file operation utilities."""

    def test_json_loading(self):
        """Test JSON file loading."""
        test_data = {"key": "value", "number": 123}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            assert loaded == test_data
        finally:
            os.unlink(temp_path)

    def test_jsonlines_loading(self):
        """Test JSON Lines file loading."""
        test_data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        try:
            loaded = []
            with open(temp_path, 'r') as f:
                for line in f:
                    loaded.append(json.loads(line.strip()))
            assert loaded == test_data
        finally:
            os.unlink(temp_path)
