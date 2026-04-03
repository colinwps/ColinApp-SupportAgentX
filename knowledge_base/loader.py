"""
knowledge_base/loader.py
本地知识库加载器：读取 docs/ 目录下的文档，向量化后存入 ChromaDB
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config.settings import settings, get_embedding_model


def load_documents(docs_dir: str | None = None) -> List[Document]:
    """
    从本地目录加载 .txt / .md 文档
    可扩展：加入 PDF / Word / Excel 解析
    """
    docs_dir = Path(docs_dir or settings.KNOWLEDGE_BASE_DIR)
    if not docs_dir.exists():
        print(f"[知识库] 目录不存在，已创建: {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        return []

    documents = []
    supported_ext = {".txt", ".md"}

    for file_path in sorted(docs_dir.rglob("*")):
        if file_path.suffix.lower() not in supported_ext:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": file_path.suffix,
                },
            )
            documents.append(doc)
            print(f"[知识库] 已加载: {file_path.name} ({len(text)} 字符)")
        except Exception as e:
            print(f"[知识库] 加载失败 {file_path.name}: {e}")

    print(f"[知识库] 共加载 {len(documents)} 个文档")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """将文档切分为适合检索的小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # 每块最大字符数
        chunk_overlap=50,        # 块间重叠，保留上下文
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[知识库] 切分为 {len(chunks)} 个文本块")
    return chunks


def build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    构建或加载向量数据库
    - 首次运行：加载文档 → 向量化 → 持久化存储
    - 后续运行：直接从磁盘加载，无需重复向量化
    """
    persist_dir = settings.CHROMA_PERSIST_DIR
    embedding = get_embedding_model()

    # 已有向量库且不强制重建，直接加载
    if os.path.exists(persist_dir) and not force_rebuild:
        print(f"[知识库] 加载已有向量库: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
            collection_name="customer_service_kb",
        )
        count = vectorstore._collection.count()
        if count > 0:
            print(f"[知识库] 已有 {count} 条向量记录")
            return vectorstore
        print("[知识库] 向量库为空，重新构建...")

    # 构建新向量库
    documents = load_documents()
    if not documents:
        print("[知识库] 没有文档可加载，返回空向量库")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
            collection_name="customer_service_kb",
        )

    chunks = split_documents(documents)
    print("[知识库] 向量化中，首次可能需要下载 Embedding 模型...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir,
        collection_name="customer_service_kb",
    )
    print(f"[知识库] 向量库构建完成，共 {len(chunks)} 条")
    return vectorstore
