"""localrag 启动器 — 解压即用

PyInstaller 打包后 sys.executable 指向 exe，exe 旁边就是 models/ 目录。
"""

import sys
import os

# PyInstaller: 以 exe 所在目录为工作目录
if getattr(sys, 'frozen', False):
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))

os.chdir(ROOT)

# Windows UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# HF 镜像（首次下载时用，已有本地模型可跳过）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from localrag.app import main

if __name__ == "__main__":
    main()
