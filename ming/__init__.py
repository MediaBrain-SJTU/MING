#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MING-7B 中文医疗大模型包

提供UTF-8编码支持，解决Windows控制台乱码问题
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "MING-7B Team"


def setup_console():
    """
    配置控制台环境，解决Windows下中文乱码问题

    1. 设置控制台代码页为UTF-8
    2. 配置标准输出流使用UTF-8编码
    """
    try:
        from ming.utils.console import setup_console
        setup_console()
    except ImportError:
        # 如果工具模块还未初始化，直接设置
        import sys
        import os
        import io
        
        if sys.platform.startswith('win'):
            # 设置环境变量
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleOutputCP(65001)
                kernel32.SetConsoleCP(65001)
            except Exception:
                pass
            
            try:
                if (hasattr(sys.stdout, 'buffer') and 
                    not isinstance(sys.stdout, io.TextIOWrapper)):
                    sys.stdout = io.TextIOWrapper(
                        sys.stdout.buffer,
                        encoding='utf-8',
                        line_buffering=True,
                        errors='replace'
                    )
            except (ValueError, AttributeError):
                pass
            
            try:
                if (hasattr(sys.stderr, 'buffer') and 
                    not isinstance(sys.stderr, io.TextIOWrapper)):
                    sys.stderr = io.TextIOWrapper(
                        sys.stderr.buffer,
                        encoding='utf-8',
                        line_buffering=True,
                        errors='replace'
                    )
            except (ValueError, AttributeError):
                pass


# 兼容旧接口
setup_encoding = setup_console
