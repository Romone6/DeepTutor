"""
Base Reranker Interface
=======================

Abstract base class and protocol for reranking retrieved snippets.

Rerankers improve retrieval quality by scoring query-document pairs
and returning the most relevant results.

Usage:
    from extensions.knowledge.rerankers.base import Reranker, RerankResult

    class MyReranker(Reranker):
        def __init__(self, model_name: str = "model"):
            super().__init__(model_name)
            self.load_model()

        def rerank(self, query: str, snippets: list[Snippet]) -> list[RerankResult]:
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, TYPE_CHECKING

logger = logging.getLogger("Reranker")

if TYPE_CHECKING:
    from extensions.knowledge.retriever import Snippet


@dataclass
class RerankResult:
    """Result from reranking a single snippet."""

    snippet: Snippet
    rerank_score: float
    original_rank: int
    new_rank: int

    def to_dict(self) -> dict:
        return {
            "text": self.snippet.text,
            "metadata": self.snippet.metadata,
            "rerank_score": self.rerank_score,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
        }


@dataclass
class RerankConfig:
    """Configuration for reranking."""

    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    device: str = "auto"
    batch_size: int = 32
    top_k: int = 6
    retrieve_top_k: int = 30
    min_score_threshold: float = 0.0
    normalize_scores: bool = True

    @classmethod
    def from_env(cls) -> "RerankConfig":
        """Create config from environment variables."""
        import os

        return cls(
            enabled=os.getenv("RERANK_ENABLED", "").lower() in ("true", "1", "yes"),
            model_name=os.getenv("RERANK_MODEL", cls.model_name),
            device=os.getenv("RERANK_DEVICE", cls.device),
            batch_size=int(os.getenv("RERANK_BATCH_SIZE", str(cls.batch_size))),
            top_k=int(os.getenv("RERANK_TOP_K", str(cls.top_k))),
            retrieve_top_k=int(os.getenv("RERANK_RETRIEVE_TOP_K", str(cls.retrieve_top_k))),
        )


@dataclass
class RerankStats:
    """Statistics from a reranking operation."""

    query: str
    input_snippets: int
    output_snippets: int
    processing_time_ms: float
    model_name: str
    device: str
    rerank_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "input_snippets": self.input_snippets,
            "output_snippets": self.output_snippets,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "device": self.device,
            "rerank_scores": self.rerank_scores,
        }


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking snippets based on query relevance."""

    @property
    def model_name(self) -> str:
        """Name of the reranker model."""
        ...

    @property
    def device(self) -> str:
        """Device the model is running on (cpu, cuda, etc.)."""
        ...

    def rerank(
        self, query: str, snippets: list[Snippet], top_k: int | None = None
    ) -> list[RerankResult]:
        """
        Rerank snippets based on relevance to query.

        Args:
            query: The search query
            snippets: List of snippets to rerank
            top_k: Number of top results to return (default: config.top_k)

        Returns:
            List of RerankResult sorted by rerank_score descending
        """
        ...

    def rerank_batch(
        self, queries: list[str], snippets_list: list[list[Snippet]], top_k: int | None = None
    ) -> list[list[RerankResult]]:
        """
        Rerank multiple query-snippet sets batched.

        Args:
            queries: List of queries
            snippets_list: List of snippet lists, one per query
            top_k: Number of top results per query

        Returns:
            List of rerank result lists
        """
        ...

    def get_stats(self) -> RerankStats | None:
        """Get statistics from the last reranking operation."""
        ...


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the reranker.

        Args:
            model_name: Name or path of the model
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self._model_name = model_name
        self._device = device
        self._stats: RerankStats | None = None
        self._model_loaded = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @abstractmethod
    def load_model(self) -> None:
        """Load the reranker model."""
        ...

    @abstractmethod
    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """
        Score query-text pairs.

        Args:
            query: The search query
            texts: List of texts to score

        Returns:
            List of relevance scores
        """
        ...

    def rerank(
        self, query: str, snippets: list[Snippet], top_k: int | None = None
    ) -> list[RerankResult]:
        """
        Rerank snippets based on relevance to query.

        Args:
            query: The search query
            snippets: List of snippets to rerank
            top_k: Number of top results to return

        Returns:
            List of RerankResult sorted by rerank_score descending
        """
        if not snippets:
            return []

        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True

        import time

        start_time = time.time()

        texts = [s.text for s in snippets]
        scores = self._score_pairs(query, texts)

        results = [
            RerankResult(
                snippet=snippets[i],
                rerank_score=scores[i],
                original_rank=i,
                new_rank=0,
            )
            for i in range(len(snippets))
        ]

        results.sort(key=lambda x: x.rerank_score, reverse=True)

        for i, result in enumerate(results):
            result.new_rank = i

        if top_k is not None:
            results = results[:top_k]

        processing_time_ms = (time.time() - start_time) * 1000

        self._stats = RerankStats(
            query=query,
            input_snippets=len(snippets),
            output_snippets=len(results),
            processing_time_ms=processing_time_ms,
            model_name=self._model_name,
            device=self._device,
            rerank_scores=[r.rerank_score for r in results],
        )

        return results

    def rerank_batch(
        self, queries: list[str], snippets_list: list[list[Snippet]], top_k: int | None = None
    ) -> list[list[RerankResult]]:
        """
        Rerank multiple query-snippet sets batched.

        Args:
            queries: List of queries
            snippets_list: List of snippet lists, one per query
            top_k: Number of top results per query

        Returns:
            List of rerank result lists
        """
        if top_k is None:
            top_k = 6

        results = []
        for query, snippets in zip(queries, snippets_list):
            reranked = self.rerank(query, snippets, top_k)
            results.append(reranked)

        return results

    def get_stats(self) -> RerankStats | None:
        """Get statistics from the last reranking operation."""
        return self._stats

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self._model_name}, device={self._device})"


class NoOpReranker(BaseReranker):
    """No-op reranker that returns snippets in original order."""

    def __init__(self):
        super().__init__("no-op", "cpu")

    def load_model(self) -> None:
        pass

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        return [1.0] * len(texts)


def create_reranker(config: RerankConfig | None = None) -> Reranker:
    """
    Factory function to create a reranker based on configuration.

    Args:
        config: RerankConfig with settings. If None, loads from environment.

    Returns:
        Configured Reranker instance
    """
    if config is None:
        config = RerankConfig.from_env()

    if not config.enabled:
        logger.info("Reranking disabled, using NoOpReranker")
        return NoOpReranker()

    try:
        from .cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model_name=config.model_name,
            device=config.device,
            batch_size=config.batch_size,
        )
        logger.info(f"Created CrossEncoderReranker with model: {config.model_name}")
        return reranker

    except ImportError as e:
        logger.warning(f"Could not create CrossEncoderReranker: {e}")
        logger.info("Falling back to NoOpReranker")
        return NoOpReranker()


__all__ = [
    "Reranker",
    "BaseReranker",
    "RerankResult",
    "RerankConfig",
    "RerankStats",
    "NoOpReranker",
    "create_reranker",
]
