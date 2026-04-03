"""
knowledge_base/retriever.py
RAG 检索器：对用户问题进行语义检索，返回相关知识片段
"""

from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import settings
from .loader import build_or_load_vectorstore

# 单例向量库
_vectorstore: Chroma | None = None


def get_vectorstore(force_rebuild: bool = False) -> Chroma:
    global _vectorstore
    if _vectorstore is None or force_rebuild:
        _vectorstore = build_or_load_vectorstore(force_rebuild)
    return _vectorstore


def retrieve(query: str, top_k: int | None = None) -> List[Document]:
    """
    根据用户问题检索最相关的知识片段

    Args:
        query: 用户问题
        top_k: 返回文档数量，默认使用配置值

    Returns:
        相关文档列表（按相似度降序）
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K
    vs = get_vectorstore()

    # 相似度检索（返回 (doc, score) 元组）
    results = vs.similarity_search_with_relevance_scores(query, k=top_k)

    # 过滤低相似度结果
    filtered = [
        doc for doc, score in results
        if score >= settings.CONFIDENCE_THRESHOLD
    ]

    if not filtered and results:
        # 如果全部低于阈值，至少返回最相关的一条
        filtered = [results[0][0]]

    return filtered


def format_context(docs: List[Document]) -> str:
    """将检索到的文档格式化为 Prompt 上下文"""
    if not docs:
        return "（知识库中未找到相关信息）"

    parts = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "未知来源")
        parts.append(f"【参考资料{i}】来源: {filename}\n{doc.page_content.strip()}")

    return "\n\n".join(parts)


def add_documents_to_kb(texts: List[str], metadatas: List[dict] | None = None):
    """
    动态向知识库添加新文档（无需重启）

    Args:
        texts: 文本内容列表
        metadatas: 对应元数据列表
    """
    from langchain_core.documents import Document
    vs = get_vectorstore()
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(texts, metadatas or [{}] * len(texts))
    ]
    vs.add_documents(docs)
    print(f"[知识库] 已动态添加 {len(docs)} 条文档")
