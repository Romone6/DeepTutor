from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from data.knowledge_bases import get_kb_path, get_embedder
from data.knowledge_bases.indexer import load_index, search_index
from data.knowledge_bases.chunker import Chunk

if TYPE_CHECKING:
    from extensions.knowledge.rerankers.base import RerankConfig, RerankResult, create_reranker

try:
    from .rerankers.base import (
        RerankConfig as BaseRerankConfig,
        RerankResult,
        create_reranker,
    )

    RERANKERS_AVAILABLE = True
except ImportError:
    RERANKERS_AVAILABLE = False
    BaseRerankConfig = None
    RerankResult = None
    create_reranker = None
    RerankConfig = None


@dataclass
class Snippet:
    text: str
    metadata: dict
    score: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class RetrievalResult:
    query: str
    snippets: list[Snippet]
    kb_name: str
    total_chunks: int = 0
    reranked: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "kb_name": self.kb_name,
            "total_chunks": self.total_chunks,
            "reranked": self.reranked,
            "snippets": [s.to_dict() for s in self.snippets],
            "error": self.error,
        }


RerankerFunc = Callable[[str, list[Snippet]], list[Snippet]]


def default_reranker(query: str, snippets: list[Snippet]) -> list[Snippet]:
    """Default reranker - returns snippets in original order (no-op)."""
    return snippets


class KnowledgeRetriever:
    """Retriever for knowledge base snippets with optional reranking."""

    def __init__(
        self,
        reranker: RerankerFunc | None = None,
        rerank_config=None,
    ):
        """
        Initialize the retriever.

        Args:
            reranker: Custom reranker function (deprecated, use rerank_config)
            rerank_config: Configuration for reranking
        """
        self.reranker = reranker or default_reranker
        self._index_cache: dict[str, tuple] = {}
        self._rerank_config = rerank_config
        if self._rerank_config is None:
            if BaseRerankConfig is not None:
                self._rerank_config = BaseRerankConfig.from_env()
            else:
                self._rerank_config = None
        self._reranker_instance = None

    def _get_reranker(self):
        """Lazy load the reranker instance."""
        if self._reranker_instance is None:
            config = self._rerank_config
            if config is not None and getattr(config, "enabled", False):
                try:
                    from .rerankers import create_reranker

                    self._reranker_instance = create_reranker(config)
                except ImportError:
                    self._reranker_instance = None
            else:
                self._reranker_instance = None

        return self._reranker_instance

    def _get_index(self, kb_path: Path):
        """Load and cache the index for a knowledge base."""
        cache_key = str(kb_path)

        if cache_key not in self._index_cache:
            index_dir = kb_path / "vector_index"
            index, chunks = load_index(index_dir)
            self._index_cache[cache_key] = (index, chunks)

        return self._index_cache[cache_key]

    async def retrieve(
        self,
        kb_name: str,
        query: str,
        top_k: int | None = None,
        use_rerank: bool | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant snippets from a knowledge base.

        Args:
            kb_name: Name of the knowledge base (e.g., 'hsc_math_adv')
            query: Query string to search for
            top_k: Maximum number of snippets to return (after reranking)
            use_rerank: Override for reranking (default: use config)

        Returns:
            RetrievalResult with snippets containing text, metadata, and scores
        """
        kb_path = get_kb_path(kb_name)

        if kb_path is None:
            return RetrievalResult(
                query=query,
                snippets=[],
                kb_name=kb_name,
                error=f"Knowledge base not found: {kb_name}",
            )

        if not kb_path.exists():
            return RetrievalResult(
                query=query,
                snippets=[],
                kb_name=kb_name,
                error=f"Knowledge base path does not exist: {kb_path}",
            )

        index_dir = kb_path / "vector_index"
        if not index_dir.exists():
            return RetrievalResult(
                query=query,
                snippets=[],
                kb_name=kb_name,
                total_chunks=0,
                error="Knowledge base has not been indexed yet",
            )

        try:
            index, chunks = self._get_index(kb_path)

            if not chunks:
                return RetrievalResult(
                    query=query,
                    snippets=[],
                    kb_name=kb_name,
                    total_chunks=0,
                    error="Knowledge base is empty",
                )

            embedder = get_embedder()
            if embedder is None:
                return RetrievalResult(
                    query=query,
                    snippets=[],
                    kb_name=kb_name,
                    total_chunks=len(chunks),
                    error="No embedding provider configured",
                )

            query_embedding = await embedder.embed([query])
            if not query_embedding:
                return RetrievalResult(
                    query=query,
                    snippets=[],
                    kb_name=kb_name,
                    total_chunks=len(chunks),
                    error="Failed to generate query embedding",
                )

            config = self._rerank_config
            should_rerank = (
                use_rerank
                if use_rerank is not None
                else (config.enabled if config and hasattr(config, "enabled") else False)
            )

            retrieve_top_k = (
                config.retrieve_top_k
                if should_rerank and config and hasattr(config, "retrieve_top_k")
                else (top_k or 12)
            )

            if index is None:
                results = self._basic_search(chunks, query, retrieve_top_k)
            else:
                results = search_index(index, chunks, query_embedding[0], retrieve_top_k)

            snippets = [
                Snippet(
                    text=chunk.content,
                    metadata=chunk.metadata,
                    score=score,
                )
                for chunk, score in results
            ]

            reranked = False
            rerank_stats = None

            if should_rerank and len(snippets) > (top_k or 6):
                reranker = self._get_reranker()
                if reranker is not None:
                    from .rerankers import RerankResult

                    config = self._rerank_config
                    final_top_k = top_k or (
                        config.top_k if config and hasattr(config, "top_k") else 6
                    )
                    rerank_results = reranker.rerank(query, snippets, top_k=final_top_k)

                    if rerank_results:
                        reranked = True
                        snippets = [r.snippet for r in rerank_results]

                        for i, snippet in enumerate(snippets):
                            original_score = snippet.score
                            new_score = rerank_results[i].rerank_score
                            snippet.score = new_score

                        rerank_stats = reranker.get_stats()

            return RetrievalResult(
                query=query,
                snippets=snippets,
                kb_name=kb_name,
                total_chunks=len(chunks),
                reranked=reranked,
            )

        except Exception as e:
            return RetrievalResult(
                query=query,
                snippets=[],
                kb_name=kb_name,
                error=f"Retrieval failed: {str(e)}",
            )

    def _basic_search(
        self,
        chunks: list[Chunk],
        query: str,
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Basic keyword-based search when vector index is unavailable."""
        query_words = set(query.lower().split())
        scored_chunks = []

        for chunk in chunks:
            content_lower = chunk.content.lower()
            matches = sum(1 for word in query_words if word in content_lower)
            if matches > 0:
                score = matches / len(query_words) if query_words else 0.0
                scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def clear_cache(self) -> None:
        """Clear the index cache."""
        self._index_cache.clear()


_retriever: KnowledgeRetriever | None = None


def get_retriever(reranker: RerankerFunc | None = None) -> KnowledgeRetriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None or reranker is not None:
        _retriever = KnowledgeRetriever(reranker=reranker)
    return _retriever


async def retrieve(
    kb_name: str,
    query: str,
    top_k: int | None = None,
    reranker: RerankerFunc | None = None,
    use_rerank: bool | None = None,
) -> RetrievalResult:
    """Retrieve relevant snippets from a knowledge base.

    This is a convenience function that wraps the KnowledgeRetriever.

    Args:
        kb_name: Name of the knowledge base (e.g., 'hsc_math_adv')
        query: Query string to search for
        top_k: Maximum number of snippets to return (after reranking)
        reranker: Optional custom reranking function (deprecated)
        use_rerank: Override for reranking (default: use RERANK_ENABLED env)

    Returns:
        RetrievalResult with snippets containing text, metadata, and scores
    """
    retriever = get_retriever(reranker=reranker)
    return await retriever.retrieve(kb_name, query, top_k=top_k, use_rerank=use_rerank)
