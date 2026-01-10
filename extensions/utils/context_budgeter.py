"""
Context Budgeting Guardrail

Prevents context collapse by estimating token counts and automatically
compressing/trimming content before LLM calls.

Features:
- Token estimation for text, system prompts, chat history, RAG snippets
- Route-specific budgets (solve/guide/question/research/co-writer)
- Smart compression: keep high-score snippets, shortest passages
- Detailed logging of all trimming operations

Usage:
    from extensions.utils import ContextBudgeter, ContextItem, RouteBudgets

    budgeter = ContextBudgeter()
    result = budgeter.apply_budget(
        route="solve",
        system_prompt="You are a helpful tutor...",
        chat_history=[{"role": "user", "content": "..."}],
        rag_snippets=[ContextItem(content="...", token_count=100, score=0.9)],
    )

    if result.trimmed_items:
        for item in result.trimmed_items:
            print(f"Trimmed: {item.reason}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("ContextBudgeter")


class Route(str, Enum):
    """DeepTutor routes that have context budgets."""

    SOLVE = "solve"
    GUIDE = "guide"
    QUESTION = "question"
    RESEARCH = "research"
    CO_WRITER = "co_writer"
    CHAT = "chat"
    GENERAL = "general"


@dataclass
class RouteBudget:
    """Budget limits for a specific route."""

    route: str
    max_total_tokens: int = 8000
    max_system_tokens: int = 2000
    max_history_tokens: int = 3000
    max_rag_tokens: int = 4000
    reserve_tokens: int = 500  # Reserved for response
    compression_aggressiveness: float = 0.3  # 0.0 = minimal, 1.0 = aggressive

    @classmethod
    def for_route(cls, route: str) -> "RouteBudget":
        """Get budget for a specific route."""
        budgets = {
            Route.SOLVE: cls(
                route=Route.SOLVE.value,
                max_total_tokens=12000,
                max_system_tokens=1500,
                max_history_tokens=2000,
                max_rag_tokens=8000,
                reserve_tokens=1000,
                compression_aggressiveness=0.4,
            ),
            Route.GUIDE: cls(
                route=Route.GUIDE.value,
                max_total_tokens=8000,
                max_system_tokens=2000,
                max_history_tokens=3000,
                max_rag_tokens=4000,
                reserve_tokens=500,
                compression_aggressiveness=0.3,
            ),
            Route.QUESTION: cls(
                route=Route.QUESTION.value,
                max_total_tokens=6000,
                max_system_tokens=1000,
                max_history_tokens=1500,
                max_rag_tokens=4000,
                reserve_tokens=500,
                compression_aggressiveness=0.2,
            ),
            Route.RESEARCH: cls(
                route=Route.RESEARCH.value,
                max_total_tokens=16000,
                max_system_tokens=1500,
                max_history_tokens=2000,
                max_rag_tokens=12000,
                reserve_tokens=1000,
                compression_aggressiveness=0.5,
            ),
            Route.CO_WRITER: cls(
                route=Route.CO_WRITER.value,
                max_total_tokens=10000,
                max_system_tokens=2500,
                max_history_tokens=4000,
                max_rag_tokens=5000,
                reserve_tokens=500,
                compression_aggressiveness=0.2,
            ),
            Route.CHAT: cls(
                route=Route.CHAT.value,
                max_total_tokens=8000,
                max_system_tokens=1000,
                max_history_tokens=6000,
                max_rag_tokens=2000,
                reserve_tokens=500,
                compression_aggressiveness=0.3,
            ),
            Route.GENERAL: cls(
                route=Route.GENERAL.value,
                max_total_tokens=8000,
                max_system_tokens=1500,
                max_history_tokens=3000,
                max_rag_tokens=4000,
                reserve_tokens=500,
                compression_aggressiveness=0.3,
            ),
        }
        return budgets.get(route, budgets[Route.GENERAL])


@dataclass
class ContextItem:
    """A single item of context (snippet, message, etc.)."""

    content: str
    token_count: int = 0
    score: float = 0.0  # Relevance score for RAG
    metadata: Dict[str, Any] = field(default_factory=dict)
    role: str = ""  # For chat messages
    item_type: str = "snippet"  # "snippet", "message", "system"

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


@dataclass
class TrimmedItem:
    """Record of a trimmed/removed item."""

    original_content: str
    trimmed_content: str
    reason: str
    token_reduction: int
    item_type: str
    score: float = 0.0


@dataclass
class BudgetResult:
    """Result of applying budget constraints."""

    within_budget: bool
    total_tokens: int
    system_tokens: int
    history_tokens: int
    rag_tokens: int
    reserved_tokens: int
    available_for_content: int
    trimmed_items: List[TrimmedItem] = field(default_factory=list)
    compression_ratio: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "within_budget": self.within_budget,
            "total_tokens": self.total_tokens,
            "system_tokens": self.system_tokens,
            "history_tokens": self.history_tokens,
            "rag_tokens": self.rag_tokens,
            "reserved_tokens": self.reserved_tokens,
            "available_for_content": self.available_for_content,
            "trimmed_items": [
                {
                    "reason": t.reason,
                    "token_reduction": t.token_reduction,
                    "item_type": t.item_type,
                }
                for t in self.trimmed_items
            ],
            "compression_ratio": self.compression_ratio,
            "warnings": self.warnings,
        }


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses a simple approximation: ~4 characters per token for English text.
    More accurate for models like GPT-4, but sufficient for budgeting.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    text = text.strip()

    rough_estimate = len(text) // 4

    adjustment = 0
    for sep in [". ", ", ", "; ", ": ", "\n\n", "\n"]:
        adjustment += text.count(sep)

    token_count = max(1, rough_estimate + adjustment // 2)

    return token_count


def estimate_message_tokens(role: str, content: str) -> int:
    """
    Estimate tokens for a chat message including role overhead.

    Args:
        role: Message role (user, assistant, system)
        content: Message content

    Returns:
        Estimated token count
    """
    role_overhead = {"system": 3, "user": 4, "assistant": 5}.get(role.lower(), 4)
    content_tokens = estimate_tokens(content)
    return role_overhead + content_tokens


class ContextBudgeter:
    """
    Manages context budgets for LLM calls.

    Features:
    - Token estimation for all context types
    - Route-specific budget limits
    - Smart compression of RAG snippets
    - Chat history summarization when needed
    - Detailed logging of all trimming
    """

    def __init__(
        self,
        default_budget: Optional[RouteBudget] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize context budgeter.

        Args:
            default_budget: Default budget if route not specified
            logger_instance: Custom logger instance
        """
        self.default_budget = default_budget or RouteBudget.for_route(Route.GENERAL.value)
        self.logger = logger_instance or logger
        self._trim_count = 0
        self._total_trimmed_tokens = 0

    def apply_budget(
        self,
        route: str,
        system_prompt: str = "",
        chat_history: List[Dict[str, str]] | List[ContextItem] = [],
        rag_snippets: List[ContextItem] = [],
        additional_context: List[ContextItem] = [],
        custom_budget: Optional[RouteBudget] = None,
    ) -> BudgetResult:
        """
        Apply budget constraints to all context.

        Args:
            route: Route name (solve/guide/question/research/co_writer)
            system_prompt: System prompt text
            chat_history: List of chat messages
            rag_snippets: List of RAG context items
            additional_context: Other context items
            custom_budget: Override budget for this call

        Returns:
            BudgetResult with trimmed context and metadata
        """
        budget = custom_budget or RouteBudget.for_route(route)
        trimmed_items: List[TrimmedItem] = []
        warnings: List[str] = []

        self.logger.info(
            f"Applying budget for route '{route}': max={budget.max_total_tokens}tokens"
        )

        budget_result = self._apply_budget_internal(
            budget=budget,
            system_prompt=system_prompt,
            chat_history=chat_history,
            rag_snippets=rag_snippets,
            additional_context=additional_context,
            trimmed_items=trimmed_items,
            warnings=warnings,
        )

        return budget_result

    def _apply_budget_internal(
        self,
        budget: RouteBudget,
        system_prompt: str,
        chat_history: List[Dict[str, str]] | List[ContextItem],
        rag_snippets: List[ContextItem],
        additional_context: List[ContextItem],
        trimmed_items: List[TrimmedItem],
        warnings: List[str],
    ) -> BudgetResult:
        """Internal implementation of budget application."""
        reserved = budget.reserve_tokens
        max_total = budget.max_total_tokens
        max_system = budget.max_system_tokens
        max_history = budget.max_history_tokens
        max_rag = budget.max_rag_tokens

        processed_system = system_prompt
        processed_history = chat_history
        processed_rag = rag_snippets
        processed_additional = additional_context

        system_tokens = estimate_tokens(processed_system)
        if system_tokens > max_system:
            self.logger.warning(f"System prompt exceeds budget: {system_tokens} > {max_system}")
            processed_system, trimmed = self._trim_text(
                processed_system,
                max_system,
                "System prompt too long",
                "system",
            )
            system_tokens = len(processed_system) // 4
            trimmed_items.extend(trimmed)

        history_items = self._prepare_history(processed_history)
        history_tokens = sum(
            estimate_message_tokens(m.get("role", ""), m.get("content", "")) for m in history_items
        )
        if history_tokens > max_history:
            self.logger.warning(f"Chat history exceeds budget: {history_tokens} > {max_history}")
            processed_history, trimmed = self._trim_history(
                history_items,
                max_history,
            )
            history_tokens = sum(
                estimate_message_tokens(m.get("role", ""), m.get("content", ""))
                for m in processed_history
            )
            trimmed_items.extend(trimmed)

        rag_tokens = sum(s.token_count for s in processed_rag)
        if rag_tokens > max_rag:
            self.logger.warning(f"RAG snippets exceed budget: {rag_tokens} > {max_rag}")
            processed_rag, trimmed = self._trim_rag(
                processed_rag,
                max_rag,
                budget.compression_aggressiveness,
            )
            rag_tokens = sum(s.token_count for s in processed_rag)
            trimmed_items.extend(trimmed)

        additional_tokens = sum(s.token_count for s in processed_additional)
        total_used = system_tokens + history_tokens + rag_tokens + additional_tokens + reserved
        available = max(0, max_total - total_used)

        within_budget = total_used <= max_total

        if not within_budget:
            warnings.append(f"Total tokens ({total_used}) exceeds budget ({max_total})")

        compression_ratio = 0.0
        if trimmed_items:
            original_tokens = total_used + sum(t.token_reduction for t in trimmed_items)
            compression_ratio = (
                sum(t.token_reduction for t in trimmed_items) / original_tokens
                if original_tokens > 0
                else 0
            )

        self._trim_count += len(trimmed_items)
        for item in trimmed_items:
            self._total_trimmed_tokens += item.token_reduction

        return BudgetResult(
            within_budget=within_budget,
            total_tokens=total_used,
            system_tokens=system_tokens,
            history_tokens=history_tokens,
            rag_tokens=rag_tokens,
            reserved_tokens=reserved,
            available_for_content=available,
            trimmed_items=trimmed_items,
            compression_ratio=compression_ratio,
            warnings=warnings,
        )

    def _trim_text(
        self,
        text: str,
        max_tokens: int,
        reason: str,
        item_type: str,
    ) -> tuple[str, List[TrimmedItem]]:
        """Trim text to fit within token budget."""
        current_tokens = estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text, []

        target_chars = max_tokens * 4
        trimmed_text = text[:target_chars]

        trimmed_item = TrimmedItem(
            original_content=text,
            trimmed_content=trimmed_text,
            reason=reason,
            token_reduction=current_tokens - max_tokens,
            item_type=item_type,
        )

        self.logger.info(f"Trimmed {item_type}: -{trimmed_item.token_reduction} tokens")
        return trimmed_text, [trimmed_item]

    def _prepare_history(
        self,
        history: List[Dict[str, str]] | List[ContextItem],
    ) -> List[Dict[str, str]]:
        """Convert history to standard format."""
        result = []
        for item in history:
            if isinstance(item, ContextItem):
                result.append(
                    {
                        "role": item.role or "user",
                        "content": item.content,
                    }
                )
            else:
                result.append(
                    {
                        "role": item.get("role", "user"),
                        "content": item.get("content", ""),
                    }
                )
        return result

    def _trim_history(
        self,
        history: List[Dict[str, str]],
        max_tokens: int,
    ) -> tuple[List[Dict[str, str]], List[TrimmedItem]]:
        """Trim chat history to fit within token budget."""
        trimmed_items: List[TrimmedItem] = []

        current_tokens = sum(
            estimate_message_tokens(m.get("role", ""), m.get("content", "")) for m in history
        )
        if current_tokens <= max_tokens:
            return history, trimmed_items

        kept: List[Dict[str, str]] = []
        removed: List[Dict[str, str]] = []

        for msg in reversed(history):
            msg_tokens = estimate_message_tokens(msg.get("role", ""), msg.get("content", ""))
            if current_tokens <= max_tokens:
                kept.insert(0, msg)
            else:
                removed.insert(0, msg)
                current_tokens -= msg_tokens

        if removed:
            kept_tokens = sum(
                estimate_message_tokens(m.get("role", ""), m.get("content", "")) for m in kept
            )
            trimmed_items.append(
                TrimmedItem(
                    original_content=f"[{len(removed)} messages]",
                    trimmed_content=f"[{len(removed)} messages trimmed]",
                    reason="Chat history exceeds budget",
                    token_reduction=kept_tokens - current_tokens if removed else 0,
                    item_type="history",
                )
            )
            self.logger.info(f"Trimmed history: {len(removed)} messages removed")

        return kept, trimmed_items

    def _trim_rag(
        self,
        snippets: List[ContextItem],
        max_tokens: int,
        aggressiveness: float = 0.3,
    ) -> tuple[List[ContextItem], List[TrimmedItem]]:
        """Trim RAG snippets to fit within token budget."""
        trimmed_items: List[TrimmedItem] = []

        if not snippets:
            return snippets, trimmed_items

        current_tokens = sum(s.token_count for s in snippets)
        if current_tokens <= max_tokens:
            return snippets, trimmed_items

        sorted_snippets = sorted(snippets, key=lambda s: (s.score, len(s.content)))

        kept: List[ContextItem] = []

        for snippet in sorted_snippets:
            if current_tokens <= max_tokens:
                kept.append(snippet)
            else:
                current_tokens -= snippet.token_count

        final_snippets = sorted(kept, key=lambda s: s.score, reverse=True)

        trimmed_count = len(snippets) - len(final_snippets)
        if trimmed_count > 0:
            trimmed_tokens = current_tokens
            trimmed_items.append(
                TrimmedItem(
                    original_content=f"[{trimmed_count} snippets]",
                    trimmed_content=f"[{trimmed_count} low-score snippets removed]",
                    reason="RAG snippets exceed budget",
                    token_reduction=trimmed_tokens,
                    item_type="rag",
                    score=sorted_snippets[trimmed_count].score
                    if trimmed_count < len(sorted_snippets)
                    else 0,
                )
            )
            self.logger.info(
                f"Trimmed RAG: {trimmed_count} snippets removed ({trimmed_tokens} tokens)"
            )

        return final_snippets, trimmed_items

    def compress_rag_by_score(
        self,
        snippets: List[ContextItem],
        target_tokens: int,
        min_score_keep: float = 0.3,
    ) -> tuple[List[ContextItem], List[TrimmedItem]]:
        """
        Compress RAG snippets by dropping low-score items first.

        Args:
            snippets: List of RAG snippets with scores
            target_tokens: Maximum tokens to keep
            min_score_keep: Minimum score to keep a snippet

        Returns:
            Tuple of (compressed_snippets, trimmed_items)
        """
        return self._trim_rag(
            [s for s in snippets if s.score >= min_score_keep],
            target_tokens,
            aggressiveness=0.5,
        )

    def get_stats(self) -> Dict[str, int]:
        """Get budgeter statistics."""
        return {
            "total_trim_operations": self._trim_count,
            "total_trimmed_tokens": self._total_trimmed_tokens,
        }

    def reset_stats(self) -> None:
        """Reset budgeter statistics."""
        self._trim_count = 0
        self._total_trimmed_tokens = 0


def apply_context_budget(
    route: str,
    system_prompt: str = "",
    chat_history: List[Dict[str, str]] = [],
    rag_snippets: List[Dict[str, Any]] = [],
) -> Dict[str, Any]:
    """
    Convenience function to apply context budget.

    Args:
        route: Route name
        system_prompt: System prompt
        chat_history: Chat history
        rag_snippets: RAG snippets (converted to ContextItem)

    Returns:
        Dictionary with budget result and processed context
    """
    budgeter = ContextBudgeter()

    snippets = [
        ContextItem(
            content=s.get("text", s.get("content", "")),
            token_count=s.get("token_count", estimate_tokens(s.get("text", ""))),
            score=s.get("score", 0.0),
            metadata=s.get("metadata", {}),
            item_type="snippet",
        )
        for s in rag_snippets
    ]

    result = budgeter.apply_budget(
        route=route,
        system_prompt=system_prompt,
        chat_history=chat_history,
        rag_snippets=snippets,
    )

    return {
        "budget_result": result.to_dict(),
        "processed_system_prompt": system_prompt[: result.system_tokens * 4],
        "processed_history": chat_history,
        "processed_rag_count": len(snippets) - len(result.trimmed_items),
        "warnings": result.warnings,
    }


__all__ = [
    "ContextBudgeter",
    "ContextItem",
    "RouteBudget",
    "Route",
    "BudgetResult",
    "TrimmedItem",
    "estimate_tokens",
    "estimate_message_tokens",
    "apply_context_budget",
]
