"""
Cross-Encoder Reranker
======================

Local reranker using sentence-transformers cross-encoder models.

Uses cross-encoders for better query-document relevance scoring than
bi-encoder embeddings. Supports GPU acceleration when available.

Models:
- cross-encoder/ms-marco-MiniLM-L-12-v2: Fast, good quality (default)
- cross-encoder/ms-marco-MiniLM-L-6-v2: Faster, lighter
- cross-encoder/ms-marco-TinyBERT-L-2: Lightest option
- cross-encoder/ms-marco-MiniLM-L-12-H384-uncased: Higher quality

Usage:
    from extensions.knowledge.rerankers import CrossEncoderReranker

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        device="auto",
    )

    results = reranker.rerank("What is photosynthesis?", snippets)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .base import BaseReranker, RerankConfig

logger = logging.getLogger("CrossEncoderReranker")


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using cross-encoder models from sentence-transformers.

    Cross-encoders process the query and document together, allowing
    for more accurate relevance scoring than bi-encoder embeddings.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        truncation: bool = True,
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('auto', 'cpu', 'cuda')
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
        """
        super().__init__(model_name, device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._truncation = truncation
        self._model = None
        self._tokenizer = None
        self._device_type = "cpu"

    def load_model(self) -> None:
        """Load the cross-encoder model and tokenizer."""
        try:
            from sentence_transformers import CrossEncoder as STCrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading cross-encoder model: {self._model_name}")

        device = self._get_device()
        self._device_type = device.type if hasattr(device, "type") else str(device)

        try:
            self._model = STCrossEncoder(
                model_name=self._model_name,
                device=device,
                max_length=self._max_length,
                tokenizer_args={"truncation": self._truncation},
            )
            self._model_loaded = True
            logger.info(f"Cross-encoder loaded successfully on {self._device_type}")

        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise

    def _get_device(self) -> torch.device:
        """Determine the best device to use."""
        if self._device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

        return torch.device(self._device)

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """
        Score query-text pairs using cross-encoder.

        Args:
            query: The search query
            texts: List of texts to score

        Returns:
            List of relevance scores
        """
        if not texts:
            return []

        if self._model is None:
            self.load_model()
            self._model_loaded = True

        pairs = [[query, text] for text in texts]

        try:
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().float().tolist()

            scores = [float(s) for s in scores]

            return scores

        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return [0.0] * len(texts)

    def rerank(self, query: str, snippets: list, top_k: int | None = None) -> list:
        """
        Rerank snippets with additional metadata.

        Args:
            query: The search query
            snippets: List of snippets with text and metadata
            top_k: Number of top results to return

        Returns:
            List of reranked results with cross-encoder scores
        """
        from .base import RerankResult

        if not snippets:
            return []

        import time

        start_time = time.time()

        texts = [getattr(s, "text", str(s)) for s in snippets]
        scores = self._score_pairs(query, texts)

        results = []
        for i, (snippet, score) in enumerate(zip(snippets, scores)):
            results.append(
                RerankResult(
                    snippet=snippet,
                    rerank_score=score,
                    original_rank=i,
                    new_rank=0,
                )
            )

        results.sort(key=lambda x: x.rerank_score, reverse=True)

        for i, result in enumerate(results):
            result.new_rank = i

        if top_k is not None:
            results = results[:top_k]

        processing_time_ms = (time.time() - start_time) * 1000

        from .base import RerankStats

        self._stats = RerankStats(
            query=query,
            input_snippets=len(snippets),
            output_snippets=len(results),
            processing_time_ms=processing_time_ms,
            model_name=self._model_name,
            device=self._device_type,
            rerank_scores=[r.rerank_score for r in results],
        )

        return results

    def get_config(self) -> RerankConfig:
        """Get current configuration."""
        return RerankConfig(
            enabled=True,
            model_name=self._model_name,
            device=self._device,
            batch_size=self._batch_size,
        )

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker("
            f"model={self._model_name}, "
            f"device={self._device_type}, "
            f"batch_size={self._batch_size})"
        )


class BgeReranker(BaseReranker):
    """
    Reranker using BAAI/bge reranker models.

    BGE rerankers are high-quality models trained for instruction-based
    reranking. They often outperform cross-encoders for complex queries.

    Models:
    - BAAI/bge-reranker-base
    - BAAI/bge-reranker-large
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        """
        Initialize the BGE reranker.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('auto', 'cpu', 'cuda')
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        super().__init__(model_name, device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device_type = "cpu"

    def load_model(self) -> None:
        """Load the BGE reranker model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for BgeReranker. "
                "Install with: pip install transformers torch"
            ) from e

        logger.info(f"Loading BGE reranker model: {self._model_name}")

        device = self._get_device()
        self._device_type = device.type if hasattr(device, "type") else str(device)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._model.to(device)
            self._model.eval()
            self._model_loaded = True
            logger.info(f"BGE reranker loaded successfully on {self._device_type}")

        except Exception as e:
            logger.error(f"Failed to load BGE reranker: {e}")
            raise

    def _get_device(self) -> torch.device:
        """Determine the best device to use."""
        if self._device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

        return torch.device(self._device)

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Score query-text pairs using BGE reranker."""
        if not texts:
            return []

        import torch
        from transformers import AutoTokenizer

        if self._model is None:
            self.load_model()
            self._model_loaded = True

        device = next(self._model.parameters()).device

        pairs = [[query, text] for text in texts]

        try:
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                scores = self._model(**inputs).logits.squeeze(-1)

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().float().tolist()

            return [float(s) for s in scores]

        except Exception as e:
            logger.error(f"BGE reranker scoring failed: {e}")
            return [0.0] * len(texts)

    def __repr__(self) -> str:
        return (
            f"BgeReranker("
            f"model={self._model_name}, "
            f"device={self._device_type}, "
            f"batch_size={self._batch_size})"
        )


__all__ = ["CrossEncoderReranker", "BgeReranker"]
