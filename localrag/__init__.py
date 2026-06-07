"""localrag — 离线本地 RAG 聊天助手

一条命令启动，不吃配置，纯 CPU 运行。
pip install localrag && localrag start
"""

import sys
import os

__version__ = "1.0.0"
__author__ = "dmx"

# Windows: 强制 UTF-8 输出，避免 emoji 打印报错
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # 子进程或重定向时可能失败，忽略
