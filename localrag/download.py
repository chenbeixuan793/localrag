"""模型下载模块 — 自动下载 GGUF 模型 + llama-server + 嵌入模型"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Callable
from urllib.request import urlopen, Request, urlretrieve
from urllib.error import URLError

from .config import MODELS_DIR, get_config
from . import __version__

HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"
LLAMA_CPP_RELEASES = "https://github.com/ggerganov/llama.cpp/releases"


def _get_hf_url(repo: str, filename: str, use_mirror: bool = True) -> str:
    base = HF_MIRROR if use_mirror else HF_OFFICIAL
    return f"{base}/{repo}/resolve/main/{filename}"


def _download_with_progress(url: str, dest: Path, label: str = "") -> bool:
    """下载文件，带进度条"""
    if dest.exists():
        print(f"  ✅ 已存在: {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    print(f"  📥 下载 {label or dest.name}...")

    try:
        req = Request(url, headers={"User-Agent": f"localrag/{__version__}"})
        with urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        print(f"\r  [{bar}] {pct}% ({downloaded//(1024*1024)}/{total//(1024*1024)} MB)", end="", flush=True)

        print()  # 换行
        tmp.rename(dest)
        return True

    except URLError as e:
        print(f"\n  ❌ 下载失败: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def download_chat_model(use_mirror: bool = True) -> Optional[Path]:
    """下载 GGUF 聊天模型"""
    config = get_config()
    filename = config.model.chat_model_file
    repo = config.model.chat_model_repo
    dest = MODELS_DIR / filename

    if dest.exists():
        print(f"  ✅ 聊天模型已存在: {filename} ({dest.stat().st_size // (1024*1024)} MB)")
        return dest

    print(f"\n📥 下载聊天模型: {filename}")
    print(f"   大小约 1-2GB，首次需要几分钟...")

    url = _get_hf_url(repo, filename, use_mirror)
    ok = _download_with_progress(url, dest, label="聊天模型")

    if not ok and use_mirror:
        print("  ⚠️  镜像失败，尝试官方源...")
        url = _get_hf_url(repo, filename, use_mirror=False)
        ok = _download_with_progress(url, dest, label="聊天模型 (官方源)")

    if not ok:
        print(f"  ❌ 聊天模型下载失败")
        print(f"  手动下载: https://huggingface.co/{repo}/resolve/main/{filename}")
        print(f"  放到: {dest}")
        return None

    print(f"  ✅ 聊天模型下载完成! ({dest.stat().st_size // (1024*1024)} MB)")
    return dest


# GitHub 代理列表（国内用户）
GITHUB_PROXIES = [
    "https://gh-proxy.com/",
    "https://gh.llkk.cc/",
    "https://ghp.ci/",
]


def _github_proxy_url(original_url: str) -> str:
    """给 GitHub URL 加代理前缀"""
    return f"https://gh-proxy.com/{original_url}"


def _find_llama_server_bin() -> Optional[Path]:
    """查找已安装的 llama-server"""
    for name in ["llama-server.exe", "llama-server"]:
        p = MODELS_DIR / name
        if p.exists():
            return p
    return None


def download_llama_server() -> Optional[Path]:
    """下载 llama-server 预编译二进制（Windows/Linux）"""
    existing = _find_llama_server_bin()
    if existing:
        print(f"  ✅ llama-server 已存在: {existing.name}")
        return existing

    # 获取最新版本号
    latest_tag = _get_latest_llama_tag()
    if not latest_tag:
        print("  ❌ 无法获取 llama.cpp 最新版本")
        return None

    if sys.platform == "win32":
        asset_name = f"llama-{latest_tag}-bin-win-cpu-x64.zip"
    elif sys.platform.startswith("linux"):
        asset_name = f"llama-{latest_tag}-bin-ubuntu-x64.zip"
    else:
        print(f"  ⚠️  平台 {sys.platform} 不支持自动下载")
        return None

    url = f"https://github.com/ggerganov/llama.cpp/releases/download/{latest_tag}/{asset_name}"
    proxy_url = _github_proxy_url(url)

    print(f"  📥 下载 llama-server ({latest_tag})...")
    dest_dir = MODELS_DIR / "_llama_extract"
    dest_dir.mkdir(exist_ok=True)
    zip_path = dest_dir / asset_name

    if not _download_with_progress(proxy_url, zip_path, label="llama-server"):
        shutil.rmtree(dest_dir, ignore_errors=True)
        print(f"  ❌ 下载失败，请手动下载解压到: {MODELS_DIR}")
        return None

    # 解压所有文件
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if not zf.getinfo(name).is_dir():
                zf.extract(name, dest_dir)
                src = dest_dir / name
                tgt = MODELS_DIR / Path(name).name
                try:
                    shutil.copy2(str(src), str(tgt))
                except Exception:
                    pass

    # 清理
    shutil.rmtree(dest_dir, ignore_errors=True)

    result = _find_llama_server_bin()
    if result:
        print(f"  ✅ llama-server 就绪!")
        return result
    print("  ❌ 解压后未找到 llama-server")
    return None


def _get_latest_llama_tag() -> Optional[str]:
    """获取 llama.cpp 最新版本号"""
    try:
        import urllib.request, json
        req = Request(
            "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest",
            headers={"User-Agent": f"localrag/{__version__}"},
        )
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("tag_name")
    except Exception as e:
        print(f"  ⚠️  获取版本号失败: {e}")
        # 返回一个已知的版本号作为 fallback
        return "b9442"


def ensure_models(use_mirror: bool = True) -> tuple[Optional[Path], Optional[Path]]:
    """确保所有模型就绪

    Returns:
        (聊天模型路径, 嵌入模型路径/名称)
    """
    config = get_config()
    if not config.model.auto_download:
        chat = MODELS_DIR / config.model.chat_model_file
        if not chat.exists():
            raise FileNotFoundError(f"聊天模型不存在: {chat}")
        return chat, config.model.embed_model_repo

    print("\n🔍 检查本地模型...")

    # 1. 聊天模型
    chat_path = download_chat_model(use_mirror)

    # 2. llama-server (如果 llama-cpp-python 不可用则需要)
    try:
        import llama_cpp
        print("  ✅ llama-cpp-python 可用，无需下载 llama-server")
    except ImportError:
        print("\n  llama-cpp-python 不可用，下载 llama-server...")
        download_llama_server()

    # 3. 嵌入模型
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  ✅ 嵌入模型将自动下载: {config.model.embed_model_repo}")
    except ImportError:
        raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

    print("\n✅ 所有模型就绪!\n")
    return chat_path, config.model.embed_model_repo
