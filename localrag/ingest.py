"""文档导入模块 — 支持 PDF / Word / Markdown / TXT，自动分块 + 向量化"""

import re
import uuid
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple

from .config import get_config, AppConfig
from .embedder import get_embedder
from .vectordb import get_vectordb


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """按句子边界智能分块"""
    # 先按段落分
    paragraphs = re.split(r'\n{2,}', text.strip())
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果当前块 + 新段不超过限制，合并
        if len(current) + len(para) < chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            # 当前块已满，保存
            if current:
                chunks.append(current)
            # 新段太长，按句子切
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[。！？!?；;])\s*', para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current) + len(sent) < chunk_size:
                        current = (current + sent).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para

    if current:
        chunks.append(current)

    # 添加重叠（简单实现：每个块尾部 + 下一块头部）
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_tail = chunks[i-1][-chunk_overlap:]
                chunk = prev_tail + "\n...\n" + chunk
            overlapped.append(chunk)
        chunks = overlapped

    return chunks


def _read_pdf(path: Path) -> str:
    """读取 PDF 文件"""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t)
    return "\n\n".join(texts)


def _read_docx(path: Path) -> str:
    """读取 Word 文件"""
    from docx import Document
    doc = Document(str(path))
    texts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(texts)


def _read_markdown(path: Path) -> str:
    """读取 Markdown 文件（去掉标记符号）"""
    import markdown
    from io import StringIO
    from html.parser import HTMLParser

    class MLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = StringIO()
        def handle_data(self, d):
            self.text.write(d)
        def get_data(self):
            return self.text.getvalue()

    html = markdown.markdown(path.read_text(encoding="utf-8"))
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def _read_txt(path: Path) -> str:
    """读取纯文本文件"""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_file(path: Path) -> Tuple[str, str]:
    """读取任意支持的文件，返回 (完整文本, 文件类型)"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path), "pdf"
    elif suffix in (".docx", ".doc"):
        return _read_docx(path), "docx"
    elif suffix in (".md", ".markdown"):
        return _read_markdown(path), "md"
    elif suffix in (".txt", ".text", ".log", ".csv", ".json", ".xml", ".yaml", ".yml", ".py", ".js", ".html", ".css"):
        return _read_txt(path), "txt"
    else:
        # 尝试当文本文件读
        try:
            return _read_txt(path), "txt"
        except Exception:
            raise ValueError(f"不支持的文件格式: {suffix}")


def ingest_file(path: Path, config: Optional[AppConfig] = None,
                chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> int:
    """导入单个文件到知识库

    Args:
        path: 文件路径
        config: 配置
        chunk_size: 分块大小
        chunk_overlap: 块重叠

    Returns:
        导入的文档块数量
    """
    config = config or get_config()
    chunk_size = chunk_size or config.rag.chunk_size
    chunk_overlap = chunk_overlap or config.rag.chunk_overlap

    embedder = get_embedder()
    vectordb = get_vectordb()

    # 读取文件
    full_text, file_type = _read_file(path)

    if not full_text.strip():
        print(f"  ⚠️  文件为空: {path.name}")
        return 0

    # 分块
    chunks = _split_text(full_text, chunk_size, chunk_overlap)

    if not chunks:
        print(f"  ⚠️  无法从文件中提取文本: {path.name}")
        return 0

    # 生成向量
    embeddings = embedder.embed(chunks)

    # 生成唯一 ID
    ids = [f"{_hash_text(path.name)}_{_hash_text(c)}_{uuid.uuid4().hex[:6]}" for c in chunks]

    # 元数据
    metadatas = [
        {
            "source": path.name,
            "source_path": str(path.absolute()),
            "type": file_type,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_count": len(c),
        }
        for i, c in enumerate(chunks)
    ]

    # 写入向量库
    vectordb.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    return len(chunks)


def ingest_directory(dir_path: Path, config: Optional[AppConfig] = None,
                     chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None,
                     recursive: bool = True) -> int:
    """导入整个目录中的所有支持文件

    Returns:
        总共导入的文档块数
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")

    supported = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt", ".text",
                 ".log", ".csv", ".json", ".xml", ".yaml", ".yml"}

    files = list(dir_path.rglob("*") if recursive else dir_path.glob("*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in supported]

    total = 0
    for f in files:
        try:
            n = ingest_file(f, config, chunk_size, chunk_overlap)
            total += n
            print(f"  ✅ {f.name}: {n} 块")
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")

    return total


def ingest_text(text: str, source_name: str = "manual",
                config: Optional[AppConfig] = None) -> int:
    """直接导入文本到知识库

    Args:
        text: 文本内容
        source_name: 来源名称
        config: 配置

    Returns:
        导入的块数
    """
    config = config or get_config()
    embedder = get_embedder()
    vectordb = get_vectordb()

    chunks = _split_text(text, config.rag.chunk_size, config.rag.chunk_overlap)
    embeddings = embedder.embed(chunks)
    ids = [f"text_{source_name}_{_hash_text(c)}_{uuid.uuid4().hex[:6]}" for c in chunks]
    metadatas = [
        {"source": source_name, "type": "text", "chunk_index": i, "total_chunks": len(chunks)}
        for i in range(len(chunks))
    ]
    vectordb.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)
