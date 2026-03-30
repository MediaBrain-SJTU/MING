#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
控制台工具模块

解决Windows控制台中文乱码问题，支持多种终端环境
"""

import sys
import os
import io


def setup_console():
    """
    设置控制台环境，解决中文乱码问题

    在Windows系统上，设置代码页为UTF-8，并配置标准输出流使用UTF-8编码。
    在其他系统上不做任何操作。
    """
    if sys.platform.startswith('win'):
        # 尝试设置控制台代码页为UTF-8
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 设置控制台输出代码页为UTF-8 (65001)
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass

        # 设置环境变量，指示使用UTF-8编码
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # 重新配置标准输出流
        try:
            # 检查stdout是否需要重新配置
            if (hasattr(sys.stdout, 'buffer') and 
                not isinstance(sys.stdout, io.TextIOWrapper)):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding='utf-8',
                    line_buffering=True,
                    errors='replace'
                )
            elif isinstance(sys.stdout, io.TextIOWrapper):
                # 如果已经是TextIOWrapper，确保编码是utf-8
                if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
                    sys.stdout = io.TextIOWrapper(
                        sys.stdout.buffer,
                        encoding='utf-8',
                        line_buffering=True,
                        errors='replace'
                    )
        except (ValueError, AttributeError):
            pass

        try:
            # 检查stderr是否需要重新配置
            if (hasattr(sys.stderr, 'buffer') and 
                not isinstance(sys.stderr, io.TextIOWrapper)):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding='utf-8',
                    line_buffering=True,
                    errors='replace'
                )
            elif isinstance(sys.stderr, io.TextIOWrapper):
                # 如果已经是TextIOWrapper，确保编码是utf-8
                if sys.stderr.encoding.lower() not in ('utf-8', 'utf8'):
                    sys.stderr = io.TextIOWrapper(
                        sys.stderr.buffer,
                        encoding='utf-8',
                        line_buffering=True,
                        errors='replace'
                    )
        except (ValueError, AttributeError):
            pass


def create_utf8_stdout():
    """
    创建一个UTF-8编码的标准输出包装器

    Returns:
        io.TextIOWrapper: UTF-8编码的标准输出包装器
    """
    if hasattr(sys.stdout, 'buffer'):
        return io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            line_buffering=True,
            errors='replace'
        )
    return sys.stdout


def print_utf8(*args, **kwargs):
    """
    使用UTF-8编码打印输出

    Args:
        *args: 要打印的参数
        **kwargs: 打印的关键字参数
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 如果出现编码错误，手动编码后再打印
        file = kwargs.get('file', sys.stdout)
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        
        output = sep.join(str(arg) for arg in args) + end
        try:
            file.write(output.encode('utf-8', errors='replace').decode('utf-8'))
        except Exception:
            # 最后尝试：使用ASCII替代
            ascii_output = output.encode('ascii', errors='replace').decode('ascii')
            file.write(ascii_output)
        file.flush()
