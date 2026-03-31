"""
Tests for conversation handling module.
"""
import pytest
from ming.conversations import (
    conv_templates,
    get_default_conv_template,
    SeparatorStyle
)


class TestConversations:
    """Test conversation template functionality."""

    def test_conv_templates_exist(self):
        """Test that conversation templates are defined."""
        assert isinstance(conv_templates, dict)
        assert len(conv_templates) > 0

    def test_get_default_conv_template(self):
        """Test getting default conversation template."""
        # Test with 'ming' template
        conv = get_default_conv_template("ming")
        assert conv is not None
        assert hasattr(conv, 'system')
        assert hasattr(conv, 'roles')
        assert hasattr(conv, 'messages')

    def test_separator_style_enum(self):
        """Test SeparatorStyle enum values."""
        assert hasattr(SeparatorStyle, 'SINGLE')
        assert hasattr(SeparatorStyle, 'TWO')
        assert hasattr(SeparatorStyle, 'MPT')

    def test_conversation_copy(self):
        """Test that conversation can be copied."""
        conv = get_default_conv_template("ming")
        conv_copy = conv.copy()
        assert conv_copy is not None
        assert conv_copy.system == conv.system

    def test_append_message(self):
        """Test appending messages to conversation."""
        conv = get_default_conv_template("ming")
        initial_len = len(conv.messages)
        conv.append_message(conv.roles[0], "Test message")
        assert len(conv.messages) == initial_len + 1

    def test_get_prompt(self):
        """Test getting prompt from conversation."""
        conv = get_default_conv_template("ming")
        conv.append_message(conv.roles[0], "Hello")
        prompt = conv.get_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestConversationFormats:
    """Test different conversation formats."""

    def test_ming_template_structure(self):
        """Test MING template has correct structure."""
        conv = get_default_conv_template("ming")
        assert conv.name == "ming"
        assert conv.system is not None
        assert len(conv.roles) == 2

    def test_empty_conversation(self):
        """Test behavior with empty conversation."""
        conv = get_default_conv_template("ming")
        prompt = conv.get_prompt()
        assert isinstance(prompt, str)
