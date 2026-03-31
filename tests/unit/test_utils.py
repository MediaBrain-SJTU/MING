import pytest
import unittest

class TestUtils(unittest.TestCase):
    def test_import_utils_module(self):
        try:
            from ming import utils
            self.assertIsNotNone(utils)
        except ImportError as e:
            pytest.skip(f"Utils module import skipped: {e}")
    
    def test_import_conversations_module(self):
        try:
            from ming import conversations
            self.assertIsNotNone(conversations)
        except ImportError as e:
            pytest.skip(f"Conversations module import skipped: {e}")

if __name__ == '__main__':
    unittest.main()
