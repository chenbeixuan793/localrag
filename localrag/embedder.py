"""嵌入模型封装 — sentence-transformers，用于文档向量化

优先加载本地缓存的模型，离线可用。
"""

from typing import Optional
from pathlib import Path

from .config import get_config, AppConfig, MODELS_DIR

# 本地嵌入模型缓存路径
LOCAL_EMBED_DIR = MODELS_DIR / "bge-small-zh-v1.5"


class Embedder:
    """文本嵌入模型封装"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self._model = None

    def load(self, use_mirror: bool = True) -> bool:
        """加载嵌入模型

        - 优先使用本地缓存的模型（models/bge-small-zh-v1.5/）
        - 本地没有则从 HuggingFace 下载
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        import os

        # 优先使用本地模型（解压即用）
        if LOCAL_EMBED_DIR.exists() and (LOCAL_EMBED_DIR / "model.safetensors").exists():
            repo = str(LOCAL_EMBED_DIR.resolve())
            print(f"[embedder] Loading local: {repo}")
        else:
            # 从 HuggingFace 下载
            if use_mirror:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            repo = self.config.model.embed_model_repo
            print(f"[embedder] Downloading: {repo}")

        device = self.config.model.embed_device
        self._model = SentenceTransformer(repo, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        print(f"[embedder] Dimension: {dim}")
        print("[embedder] Ready!")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if not self.is_loaded:
            raise RuntimeError("嵌入模型未加载，请先调用 load()")
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """将单个查询文本转换为向量"""
        result = self.embed([text])
        return result[0]

    @property
    def dimension(self) -> int:
        """嵌入维度"""
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 512  # bge-small-zh 默认


# 全局单例
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
