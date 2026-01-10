"""Extensions knowledge module."""

from .retriever import (
    KnowledgeRetriever,
    Snippet,
    RetrievalResult,
    retrieve,
    get_retriever,
    RerankerFunc,
)
from .rag_injector import (
    RAGContext,
    RAGPromptMixin,
    inject_rag_context,
    inject_rag_into_prompts_sync,
    ensure_rag_grounded,
    format_snippets,
)
from .base_rag_agent import RAGBaseAgent

__all__ = [
    "KnowledgeRetriever",
    "Snippet",
    "RetrievalResult",
    "retrieve",
    "get_retriever",
    "RerankerFunc",
    "RAGContext",
    "RAGPromptMixin",
    "inject_rag_context",
    "inject_rag_into_prompts_sync",
    "ensure_rag_grounded",
    "format_snippets",
    "RAGBaseAgent",
]
