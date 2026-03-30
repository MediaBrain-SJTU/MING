#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具模块

提供通用工具函数
"""

from ming.utils.encoding import setup_utf8_output, safe_print, to_unicode
from ming.utils.console import setup_console, print_utf8, create_utf8_stdout

__all__ = [
    'setup_utf8_output',
    'safe_print',
    'to_unicode',
    'setup_console',
    'print_utf8',
    'create_utf8_stdout',
]
