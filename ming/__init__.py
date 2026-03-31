"""Medical GPT - 中文医疗大语言模型"""

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("ming")
except PackageNotFoundError:
    __version__ = "1.1.3"
