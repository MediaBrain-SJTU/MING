# Windows 控制台中文乱码解决方案

## 问题描述

在Windows环境下运行Python脚本时，中文输出可能出现乱码。这是因为Windows控制台默认使用GBK或GB2312编码，而Python输出的是UTF-8编码。

## 解决方案

### 方法一：使用项目提供的自动配置（推荐）

项目已经内置了编码自动配置功能，只需在脚本最开始导入并调用：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必须在最开始调用
import ming
ming.setup_console()

# 之后的代码中文输出将正常显示
print("测试中文输出")
```

### 方法二：手动设置环境变量（命令行执行时）

在运行Python脚本前设置环境变量：

```cmd
:: cmd.exe
set PYTHONIOENCODING=utf-8
chcp 65001
python your_script.py
```

```powershell
# PowerShell
$env:PYTHONIOENCODING='utf-8'
chcp 65001
python your_script.py
```

### 方法三：使用批处理文件运行

可以创建批处理文件（如`run.bat`）：

```batch
@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
python your_script.py
pause
```

### 方法四：修改控制台属性（永久解决）

1. 打开cmd.exe
2. 右键标题栏 -> 属性
3. 字体 -> 选择"Lucida Console"或"Consolas"
4. 确认后关闭

## 项目中已实现的功能

### 1. ming.utils.console 模块

提供以下功能：
- `setup_console()`: 自动配置控制台为UTF-8编码
- `print_utf8()`: 安全打印函数，处理编码错误
- `create_utf8_stdout()`: 创建UTF-8编码的标准输出包装器

### 2. ming.utils.encoding 模块

提供基础编码处理函数：
- `setup_utf8_output()`: 设置标准输出为UTF-8编码
- `safe_print()`: 安全打印函数
- `to_unicode()`: 字符串转Unicode

### 3. 包级别的自动配置

在`ming/__init__.py`中提供了`setup_console()`函数，可直接调用。

## 已修改的文件清单

为了确保整个项目的中文输出正常，以下文件已做了编码相关的修改：

1. **`ming/__init__.py`**
   - 添加了`setup_console()`函数
   - 提供编码自动配置功能

2. **`ming/main.py`**
   - 在脚本最开始设置UTF-8编码
   - 修改日志Handler使用UTF-8编码

3. **`ming/utils/encoding.py`** (新建)
   - 基础编码处理工具

4. **`ming/utils/console.py`** (新建)
   - Windows控制台编码配置工具

5. **`ming/utils/__init__.py`** (新建)
   - 工具模块的导出

## 使用建议

1. **开发环境**：推荐使用VS Code、PyCharm等IDE，它们的终端默认支持UTF-8

2. **生产环境**：使用方法二或方法三设置环境变量后运行

3. **日志输出**：确保日志配置使用UTF-8编码，项目中的`main.py`已做相应修改

## 测试验证

可以运行以下脚本验证编码设置是否生效：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ming
ming.setup_console()

print("=" * 50)
print("MING-7B 中文显示测试")
print("=" * 50)
print()
print("纯中文测试：高血压、糖尿病、冠心病")
print("中英文混合：患者Patient有高血压Hypertension病史")
print("[OK] 测试完成")
```

## 注意事项

1. 所有Python脚本文件都应该使用`UTF-8 无BOM`编码保存

2. 在脚本最开始进行编码设置，确保所有后续输出都使用正确的编码

3. 如果在IDE中运行正常但在控制台中乱码，请检查方法二的环境变量设置

4. Trae IDE等在线环境的终端可能有特殊的编码限制，以真实本地环境测试为准
