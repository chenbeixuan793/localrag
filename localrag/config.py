"""配置管理中心 — 一切都从这里调参"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# 项目根目录（兼容 PyInstaller 打包）
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后，exe 所在目录就是根目录
    ROOT_DIR = Path(os.path.dirname(os.path.abspath(sys.executable)))
else:
    ROOT_DIR = Path(__file__).parent.parent

MODELS_DIR = ROOT_DIR / "models"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
CONFIG_FILE = ROOT_DIR / "config.json"

# 确保目录存在
MODELS_DIR.mkdir(exist_ok=True)
KNOWLEDGE_DIR.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """模型配置"""
    # 聊天模型
    chat_model_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    chat_model_file: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    chat_n_ctx: int = 16384          # llama-server 启动上下文（给足够大，实际由UI滑块控制）
    chat_n_threads: int = 8          # CPU 线程数（物理核心数）
    chat_n_batch: int = 1024         # 批处理大小（大batch加速prompt处理）

    # 嵌入模型
    embed_model_repo: str = "BAAI/bge-small-zh-v1.5"
    embed_device: str = "cpu"

    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1

    # 自动下载
    auto_download: bool = True


@dataclass
class RAGConfig:
    """RAG 配置"""
    chunk_size: int = 500            # 分块大小（字符）
    chunk_overlap: int = 100         # 块重叠
    top_k: int = 5                   # 检索返回数
    score_threshold: float = 0.3     # 相似度阈值
    collection_name: str = "localrag_knowledge"


@dataclass
class UIConfig:
    """UI 配置"""
    title: str = "💫 离线 AI 聊天助手"
    theme: str = "soft"
    server_port: int = 7860
    share: bool = False
    inbrowser: bool = True
    show_logo: bool = True
    dark_mode: bool = True


@dataclass
class AppConfig:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def save(self, path: Optional[Path] = None):
        path = path or CONFIG_FILE
        d = {"model": asdict(self.model), "rag": asdict(self.rag), "ui": asdict(self.ui)}
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        path = path or CONFIG_FILE
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                model=ModelConfig(**data.get("model", {})),
                rag=RAGConfig(**data.get("rag", {})),
                ui=UIConfig(**data.get("ui", {})),
            )
        return cls()

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量覆盖配置"""
        config = cls.load()
        for key in ["temperature", "top_p", "top_k", "max_tokens", "chat_n_threads", "chat_n_ctx"]:
            if env_val := os.environ.get(f"LOCALRAG_{key.upper()}"):
                if hasattr(config.model, key):
                    setattr(config.model, key, type(getattr(config.model, key))(env_val))
        return config


# 全局单例
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config():
    global _config
    _config = None
