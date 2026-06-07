"""向量数据库封装 — ChromaDB，零配置纯本地"""

from typing import Optional, List

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import get_config, KNOWLEDGE_DIR, AppConfig


class VectorDB:
    """ChromaDB 向量存储封装"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self._client = None
        self._collection = None

    def load(self) -> bool:
        """初始化 ChromaDB"""
        persist_dir = str(KNOWLEDGE_DIR / "chroma_db")

        print(f"📚 初始化向量数据库: ChromaDB")
        print(f"   存储路径: {persist_dir}")

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        collection_name = self.config.rag.collection_name
        try:
            self._collection = self._client.get_collection(name=collection_name)
            doc_count = self._collection.count()
            print(f"   已有知识库: {doc_count} 个文档块")
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print("   创建新的知识库")

        print("✅ 向量数据库就绪!")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._client is not None and self._collection is not None

    @property
    def collection(self):
        if not self.is_loaded:
            raise RuntimeError("向量数据库未初始化，请先调用 load()")
        return self._collection

    @property
    def count(self) -> int:
        """文档块总数"""
        return self.collection.count() if self.is_loaded else 0

    def add(self, ids: List[str], documents: List[str], embeddings: List[List[float]],
            metadatas: Optional[List[dict]] = None):
        """添加文档块到向量库"""
        if not self.is_loaded:
            raise RuntimeError("向量数据库未初始化")
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embedding: List[float], top_k: Optional[int] = None) -> dict:
        """语义搜索

        Returns:
            ChromaDB 查询结果
        """
        if not self.is_loaded:
            raise RuntimeError("向量数据库未初始化")
        top_k = top_k or self.config.rag.top_k
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def search_text(self, query_embedding: List[float], top_k: Optional[int] = None,
                    threshold: Optional[float] = None) -> list[str]:
        """语义搜索，返回纯文本结果（带阈值过滤）"""
        threshold = threshold if threshold is not None else self.config.rag.score_threshold
        results = self.search(query_embedding, top_k)

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        filtered = []
        for doc, dist in zip(documents, distances):
            # ChromaDB 用余弦距离，1 - distance = similarity
            similarity = 1.0 - dist
            if similarity >= threshold:
                filtered.append(doc)

        return filtered

    def delete_all(self):
        """清空知识库"""
        if not self.is_loaded:
            return
        # 删除并重建
        name = self.config.rag.collection_name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        print("🗑️  知识库已清空")

    def get_all_docs(self, limit: int = 200) -> list[dict]:
        """获取文档块信息（默认上限 200，避免页面卡死）"""
        if not self.is_loaded:
            return []
        result = self.collection.get(include=["documents", "metadatas"], limit=limit)
        return [
            {"id": id_, "doc": doc, "meta": meta}
            for id_, doc, meta in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
            )
        ]


    # ======== 人格私有知识库 ========

    def get_persona_collection(self, persona_key: str) -> Optional[object]:
        """获取人格私有知识库 collection，不存在则返回 None"""
        if not self.is_loaded:
            return None
        name = f"persona_{persona_key}"
        try:
            return self._client.get_collection(name=name)
        except Exception:
            return None

    def create_persona_kb(self, persona_key: str, documents: list[str],
                          embeddings: list[list[float]]) -> int:
        """创建或重置人格私有知识库，返回块数"""
        name = f"persona_{persona_key}"
        try:
            self._client.delete_collection(name=name)
        except Exception:
            pass
        col = self._client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        ids = [f"pkb_{persona_key}_{i}" for i in range(len(documents))]
        metadatas = [{"source": f"persona:{persona_key}", "chunk_index": i}
                      for i in range(len(documents))]
        col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return len(documents)

    def search_persona(self, persona_key: str, query_embedding: list[float],
                       top_k: int = 5) -> list[str]:
        """搜索人格私有知识库"""
        col = self.get_persona_collection(persona_key)
        if not col:
            return []
        results = col.query(query_embeddings=[query_embedding], n_results=top_k,
                          include=["documents"])
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d]


# 全局单例
_vectordb: Optional[VectorDB] = None


def get_vectordb() -> VectorDB:
    global _vectordb
    if _vectordb is None:
        _vectordb = VectorDB()
    return _vectordb
