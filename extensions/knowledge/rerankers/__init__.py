"""
Rerankers Package
=================

Reranking modules for improving retrieval quality.

Available rerankers:
- CrossEncoderReranker: Uses sentence-transformers cross-encoders
- BgeReranker: Uses BAAI bge reranker models

Usage:
    from extensions.knowledge.rerankers import create_reranker

    reranker = create_reranker()  # Loads from RERANK_* env vars
    results = reranker.rerank(query, snippets)
"""

from .base import (
    BaseReranker,
    NoOpReranker,
    RerankConfig,
    RerankResult,
    RerankStats,
    Reranker,
    create_reranker,
)
from .cross_encoder import BgeReranker, CrossEncoderReranker

__all__ = [
    "BaseReranker",
    "NoOpReranker",
    "RerankConfig",
    "RerankResult",
    "RerankStats",
    "Reranker",
    "create_reranker",
    "CrossEncoderReranker",
    "BgeReranker",
]
