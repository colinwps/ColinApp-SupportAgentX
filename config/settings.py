"""
config/settings.py
统一配置中心 + 多厂商模型工厂
支持：Anthropic / OpenAI / DeepSeek / 通义千问 / Ollama
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


# ──────────────────────────────────────────
# 基础配置
# ──────────────────────────────────────────
class Settings:
    # 模型提供商
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")

    # 各厂商 API Key
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")

    # 各厂商模型名称
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-max")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # 知识库
    KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base/docs")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./knowledge_base/.chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # Agent 行为
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))


settings = Settings()


# ──────────────────────────────────────────
# 多厂商模型工厂
# ──────────────────────────────────────────
@lru_cache(maxsize=1)
def get_llm(provider: str | None = None, temperature: float = 0.1) -> BaseChatModel:
    """
    统一模型工厂，根据 provider 返回对应 LLM 实例。

    用法:
        llm = get_llm()                      # 使用 .env 配置的默认厂商
        llm = get_llm("openai")              # 强制指定 OpenAI
        llm = get_llm("ollama")              # 本地 Ollama
    """
    provider = (provider or settings.LLM_PROVIDER).lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=4096,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )

    elif provider == "deepseek":
        # DeepSeek 兼容 OpenAI 接口
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
        )

    elif provider == "qwen":
        # 通义千问兼容 OpenAI 接口
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.QWEN_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"不支持的模型提供商: '{provider}'。"
            f"可选值: anthropic, openai, deepseek, qwen, ollama"
        )


def get_embedding_model():
    """返回本地 Embedding 模型（HuggingFace，中文友好）"""
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
