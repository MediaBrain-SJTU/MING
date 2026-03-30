#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
编码处理工具模块

解决Windows控制台中文乱码问题，统一使用UTF-8编码输出
"""

import sys
import io


def setup_utf8_output():
    """
    设置UTF-8输出编码

    在Windows系统上，将标准输出和标准错误设置为UTF-8编码，
    解决中文乱码问题。在其他系统上不做任何操作。
    """
    if sys.platform.startswith('win'):
        # 仅在Windows系统上设置
        try:
            # 检查stdout是否可用
            if (hasattr(sys.stdout, 'buffer') and 
                not isinstance(sys.stdout, io.TextIOWrapper)):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding='utf-8',
                    line_buffering=True,
                    errors='replace'
                )
        except (ValueError, AttributeError):
            # 如果stdout已关闭或不可用，跳过设置
            pass
        
        try:
            # 检查stderr是否可用
            if (hasattr(sys.stderr, 'buffer') and 
                not isinstance(sys.stderr, io.TextIOWrapper)):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding='utf-8',
                    line_buffering=True,
                    errors='replace'
                )
        except (ValueError, AttributeError):
            # 如果stderr已关闭或不可用，跳过设置
            pass


def safe_print(*args, **kwargs):
    """
    安全打印函数，避免编码错误

    Args:
        *args: 要打印的参数
        **kwargs: 打印的关键字参数
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 如果出现编码错误，尝试使用替代字符
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                new_args.append(arg.encode('utf-8', errors='replace').decode('utf-8'))
            else:
                new_args.append(arg)
        print(*new_args, **kwargs)


def to_unicode(s, encoding='utf-8'):
    """
    将字符串转换为Unicode

    Args:
        s: 输入字符串
        encoding: 源编码

    Returns:
        str: Unicode字符串
    """
    if isinstance(s, bytes):
        return s.decode(encoding, errors='replace')
    return str(s)
