"""
Budgeted LLM Service
====================

Context-aware LLM service with automatic budgeting and provider fallback.

Features:
- Automatic context budgeting before LLM calls
- Provider fallback chains (local → cloud)
- Token usage tracking and warnings
- Route-specific optimization

Usage:
    from src.services.llm.budgeted import budgeted_llm, get_budgeted_client

    result = await budgeted_llm(
        prompt="Solve this math problem...",
        system_prompt="You are a tutor...",
        route="solve",
        allow_fallback=True,
    )

    print(result.response)
    print(result.budget_stats)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.logging import get_logger

logger = logging.getLogger("BudgetedLLM")


@dataclass
class BudgetedResult:
    """Result from a budgeted LLM call."""

    response: str
    within_budget: bool
    total_tokens: int
    system_tokens: int
    history_tokens: int
    rag_tokens: int
    reserved_tokens: int
    provider: str = ""
    latency_ms: float = 0.0
    trimmed_items: List[Dict[str, Any]] = field(default_factory=list)
    compression_ratio: float = 0.0
    warnings: List[str] = field(default_factory=list)
    from_fallback: bool = False
    fallback_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "within_budget": self.within_budget,
            "tokens": {
                "total": self.total_tokens,
                "system": self.system_tokens,
                "history": self.history_tokens,
                "rag": self.rag_tokens,
                "reserved": self.reserved_tokens,
            },
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "trimmed_items": self.trimmed_items,
            "compression_ratio": self.compression_ratio,
            "warnings": self.warnings,
            "from_fallback": self.from_fallback,
            "fallback_attempts": self.fallback_attempts,
        }


class BudgetedLLMService:
    """
    LLM service with context budgeting and provider fallback.

    Integrates:
    - ContextBudgeter for token budgeting
    - HybridProviderManager for fallback chains
    - Route-specific optimization
    """

    def __init__(self, enable_fallback: bool = True, log_requests: bool = True):
        """
        Initialize budgeted LLM service.

        Args:
            enable_fallback: Enable provider fallback on failure
            log_requests: Log all LLM requests
        """
        self.enable_fallback = enable_fallback
        self.log_requests = log_requests
        self._request_count = 0
        self._total_tokens = 0

        self._init_extensions()

    def _init_extensions(self):
        """Initialize extension modules if available."""
        try:
            from extensions.router.provider_manager import (
                HybridProviderManager,
                get_provider_manager,
            )
            from extensions.utils.context_budgeter import ContextBudgeter, ContextItem

            self.provider_manager = get_provider_manager()
            self.budgeter = ContextBudgeter()
            self._extensions_available = True
            logger.info("BudgetedLLM: Extensions loaded successfully")
        except ImportError as e:
            self._extensions_available = False
            self.provider_manager = None
            self.budgeter = None
            logger.warning(f"BudgetedLLM: Extensions not available ({e})")

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        rag_snippets: Optional[List[Dict[str, Any]]] = None,
        route: str = "general",
        preferred_provider: Optional[str] = None,
        allow_fallback: bool = True,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> BudgetedResult:
        """
        Complete LLM request with context budgeting and fallback.

        Args:
            prompt: User prompt
            system_prompt: Optional system message
            history: Conversation history
            rag_snippets: RAG context with content, score, token_count
            route: Route name for budgeting (solve, chat, research, etc.)
            preferred_provider: Preferred provider binding
            allow_fallback: Enable fallback on failure
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            **kwargs: Additional parameters

        Returns:
            BudgetedResult with response and metadata
        """
        start_time = time.time()
        self._request_count += 1

        if not self._extensions_available:
            return await self._standard_complete(
                prompt, system_prompt, history, max_tokens, temperature, **kwargs
            )

        try:
            budget_result = self._apply_budget(
                route=route,
                system_prompt=system_prompt,
                history=history,
                rag_snippets=rag_snippets,
            )

            provider_result = await self._call_with_fallback(
                prompt=budget_result.get("trimmed_prompt", prompt),
                system_prompt=budget_result.get("trimmed_system", system_prompt),
                history=budget_result.get("trimmed_history", history),
                preferred_provider=preferred_provider,
                allow_fallback=allow_fallback,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            if provider_result.success:
                self._total_tokens += provider_result.latency_ms // 10

                return BudgetedResult(
                    response=provider_result.response,
                    within_budget=budget_result["within_budget"],
                    total_tokens=budget_result["total_tokens"],
                    system_tokens=budget_result["system_tokens"],
                    history_tokens=budget_result["history_tokens"],
                    rag_tokens=budget_result["rag_tokens"],
                    reserved_tokens=budget_result["reserved_tokens"],
                    provider=provider_result.provider,
                    latency_ms=latency_ms,
                    trimmed_items=budget_result["trimmed_items"],
                    compression_ratio=budget_result["compression_ratio"],
                    warnings=budget_result["warnings"],
                    from_fallback=provider_result.attempt > 0,
                    fallback_attempts=provider_result.attempt + 1,
                )
            else:
                return BudgetedResult(
                    response="",
                    within_budget=budget_result["within_budget"],
                    total_tokens=budget_result["total_tokens"],
                    system_tokens=budget_result["system_tokens"],
                    history_tokens=budget_result["history_tokens"],
                    rag_tokens=budget_result["rag_tokens"],
                    reserved_tokens=budget_result["reserved_tokens"],
                    provider=provider_result.provider,
                    latency_ms=latency_ms,
                    trimmed_items=budget_result["trimmed_items"],
                    compression_ratio=budget_result["compression_ratio"],
                    warnings=budget_result["warnings"]
                    + [f"Provider error: {provider_result.error}"],
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"BudgetedLLM complete failed: {e}")

            return BudgetedResult(
                response="",
                within_budget=True,
                total_tokens=0,
                system_tokens=0,
                history_tokens=0,
                rag_tokens=0,
                reserved_tokens=0,
                latency_ms=latency_ms,
                warnings=[f"Error: {str(e)}"],
            )

    async def _standard_complete(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> BudgetedResult:
        """Standard completion without extensions."""
        from src.services.llm.client import get_llm_client
        from src.services.llm.config import get_llm_config

        config = get_llm_config()
        client = get_llm_client(config)

        response = await client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return BudgetedResult(
            response=response,
            within_budget=True,
            total_tokens=len(prompt) // 4,
            system_tokens=len(system_prompt) // 4 if system_prompt else 0,
            history_tokens=sum(len(m.get("content", "")) // 4 for m in (history or [])),
            rag_tokens=0,
            reserved_tokens=500,
            provider=config.binding,
        )

    def _apply_budget(
        self,
        route: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        rag_snippets: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Apply context budgeting to all inputs."""
        from extensions.utils.context_budgeter import ContextBudgeter, ContextItem

        budgeter = ContextBudgeter()

        snippets = [
            ContextItem(
                content=s.get("text", s.get("content", "")),
                token_count=s.get("token_count", 0),
                score=s.get("score", 0.0),
                metadata=s.get("metadata", {}),
                item_type="snippet",
            )
            for s in (rag_snippets or [])
        ]

        result = budgeter.apply_budget(
            route=route,
            system_prompt=system_prompt or "",
            chat_history=history or [],
            rag_snippets=snippets,
        )

        return {
            "within_budget": result.within_budget,
            "total_tokens": result.total_tokens,
            "system_tokens": result.system_tokens,
            "history_tokens": result.history_tokens,
            "rag_tokens": result.rag_tokens,
            "reserved_tokens": result.reserved_tokens,
            "trimmed_prompt": None,
            "trimmed_system": system_prompt,
            "trimmed_history": history,
            "trimmed_items": [
                {"reason": t.reason, "tokens": t.token_reduction, "type": t.item_type}
                for t in result.trimmed_items
            ],
            "compression_ratio": result.compression_ratio,
            "warnings": result.warnings,
        }

    async def _call_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        preferred_provider: Optional[str],
        allow_fallback: bool,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> "ProviderResult":
        """Call LLM with fallback support."""
        from extensions.router.provider_manager import get_provider_response

        result = await get_provider_response(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            preferred_provider=preferred_provider,
            allow_fallback=allow_fallback,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "request_count": self._request_count,
            "total_tokens_estimate": self._total_tokens,
            "extensions_available": self._extensions_available,
            "fallback_enabled": self.enable_fallback,
        }

    def reset_stats(self) -> None:
        """Reset service statistics."""
        self._request_count = 0
        self._total_tokens = 0


_budgeted_service: Optional[BudgetedLLMService] = None


def get_budgeted_service() -> BudgetedLLMService:
    """Get or create the budgeted LLM service."""
    global _budgeted_service
    if _budgeted_service is None:
        _budgeted_service = BudgetedLLMService()
    return _budgeted_service


async def budgeted_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    rag_snippets: Optional[List[Dict[str, Any]]] = None,
    route: str = "general",
    preferred_provider: Optional[str] = None,
    allow_fallback: bool = True,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BudgetedResult:
    """
    Convenience function for budgeted LLM calls.

    Args:
        prompt: User prompt
        system_prompt: Optional system message
        history: Conversation history
        rag_snippets: RAG context with content, score, token_count
        route: Route name for budgeting
        preferred_provider: Preferred provider binding
        allow_fallback: Enable fallback on failure
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        **kwargs: Additional parameters

    Returns:
        BudgetedResult with response and metadata
    """
    service = get_budgeted_service()
    return await service.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        history=history,
        rag_snippets=rag_snippets,
        route=route,
        preferred_provider=preferred_provider,
        allow_fallback=allow_fallback,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )


__all__ = [
    "BudgetedLLMService",
    "BudgetedResult",
    "get_budgeted_service",
    "budgeted_llm",
]
