"""
RAG-Enabled Base Agent

Extends the existing BaseAgent with RAG injection capability for
syllabus/exam grounded tutoring.

Usage:
    from extensions.knowledge.base_rag_agent import RAGBaseAgent

    class MyAgent(RAGBaseAgent):
        def __init__(self, *args, kb_name="hsc_math_adv", **kwargs):
            super().__init__(*args, kb_name=kb_name, **kwargs)

        async def process(self, user_message: str):
            # Get RAG-enhanced prompts
            rag_context = await self.get_rag_context(user_message)

            if rag_context.kb_empty:
                return {
                    "answer": rag_context.retrieval_result.error or "KB has no coverage",
                    "kb_empty": True,
                }

            # Call LLM with enhanced system prompt
            response = await self.call_llm(
                user_prompt=rag_context.user_prompt,
                system_prompt=rag_context.system_prompt,
            )

            return {"answer": response, "snippets_used": rag_context.snippets_injected}
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.base_agent import BaseAgent
from extensions.knowledge import RAGContext, RAGPromptMixin, inject_rag_context


class RAGBaseAgent(BaseAgent, RAGPromptMixin):
    """
    Base agent with RAG injection capability.

    Extends BaseAgent with the ability to inject knowledge base snippets
    into prompts before LLM calls, ensuring syllabus/exam grounded responses.
    """

    def __init__(
        self,
        module_name: str,
        agent_name: str,
        kb_name: str = "hsc_math_adv",
        rag_enabled: bool = True,
        rag_max_snippets: int = 5,
        rag_top_k: int = 8,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        language: str = "en",
        config: dict[str, Any] | None = None,
        token_tracker: Any | None = None,
        log_dir: str | None = None,
    ):
        """
        Initialize RAG-enabled agent.

        Args:
            module_name: Module name (solve/research/guide/ideagen/co_writer)
            agent_name: Agent name
            kb_name: Knowledge base name (default: hsc_math_adv)
            rag_enabled: Whether RAG injection is enabled
            rag_max_snippets: Maximum snippets to include in prompt
            rag_top_k: Number of results to retrieve
            api_key: API key
            base_url: API endpoint
            model: Model name
            language: Language setting
            config: Optional configuration
            token_tracker: Optional token tracker
            log_dir: Optional log directory
        """
        self.kb_name = kb_name
        self.rag_enabled = rag_enabled
        self.rag_max_snippets = rag_max_snippets
        self.rag_top_k = rag_top_k

        super().__init__(
            module_name=module_name,
            agent_name=agent_name,
            api_key=api_key,
            base_url=base_url,
            model=model,
            language=language,
            config=config,
            token_tracker=token_tracker,
            log_dir=log_dir,
        )

    async def call_with_rag(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        topic: str | None = None,
        response_format: dict[str, str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        verbose: bool = True,
        stage: str | None = None,
    ) -> tuple[str, RAGContext]:
        """
        Call LLM with RAG-enhanced prompts.

        Args:
            user_prompt: User prompt (used for retrieval)
            system_prompt: System prompt (enhanced with RAG context)
            topic: Topic hint for retrieval
            response_format: Response format
            temperature: Temperature
            max_tokens: Max tokens
            model: Model name
            verbose: Verbose output
            stage: Stage marker

        Returns:
            Tuple of (LLM response, RAG context)
        """
        sys_prompt = system_prompt or self.prompts.get("system") if self.prompts else ""

        rag_context = await inject_rag_context(
            kb_name=self.kb_name,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            topic=topic or user_prompt,
            max_snippets=self.rag_max_snippets,
            top_k=self.rag_top_k,
        )

        response = await self.call_llm(
            user_prompt=rag_context.user_prompt,
            system_prompt=rag_context.system_prompt,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            verbose=verbose,
            stage=stage,
        )

        return response, rag_context

    async def get_rag_context(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        topic: str | None = None,
    ) -> RAGContext:
        """
        Get RAG-enhanced prompts without making an LLM call.

        Args:
            user_prompt: User message
            system_prompt: System prompt (uses default if not provided)
            topic: Topic hint for retrieval

        Returns:
            RAGContext with enhanced prompts
        """
        sys_prompt = system_prompt or self.prompts.get("system") if self.prompts else ""

        return await inject_rag_context(
            kb_name=self.kb_name,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            topic=topic or user_prompt,
            max_snippets=self.rag_max_snippets,
            top_k=self.rag_top_k,
        )

    def should_answer_without_rag(self, rag_context: RAGContext) -> bool:
        """
        Check if agent should attempt to answer when KB is empty.

        Override this method to implement module-specific behavior.

        Args:
            rag_context: RAG context from retrieval

        Returns:
            True if agent should proceed without RAG
        """
        return False

    async def process_with_rag(
        self,
        user_message: str,
        system_prompt: str | None = None,
        topic: str | None = None,
        **call_kwargs,
    ) -> dict[str, Any]:
        """
        Complete processing flow with RAG injection.

        Args:
            user_message: User message
            system_prompt: System prompt override
            topic: Topic hint for retrieval
            **call_kwargs: Additional arguments for call_llm

        Returns:
            Dict with answer, metadata, and RAG info
        """
        rag_context = await self.get_rag_context(user_message, system_prompt, topic)

        if rag_context.kb_empty:
            if self.should_answer_without_rag(rag_context):
                response = await self.call_llm(
                    user_prompt=user_message,
                    system_prompt=rag_context.system_prompt,
                    **call_kwargs,
                )
                return {
                    "answer": response,
                    "kb_empty": True,
                    "snippets_used": 0,
                    "warning": rag_context.retrieval_result.error,
                    "rag_context": rag_context.to_dict(),
                }
            else:
                return {
                    "answer": None,
                    "kb_empty": True,
                    "snippets_used": 0,
                    "error": rag_context.retrieval_result.error or "Knowledge base has no coverage",
                    "instruction": (
                        "Please provide a document or URL containing the relevant material, "
                        "or ask about a different topic that is covered in the syllabus."
                    ),
                    "rag_context": rag_context.to_dict(),
                }

        response = await self.call_llm(
            user_prompt=rag_context.user_prompt,
            system_prompt=rag_context.system_prompt,
            **call_kwargs,
        )

        return {
            "answer": response,
            "kb_empty": False,
            "snippets_used": rag_context.snippets_injected,
            "total_chunks": rag_context.retrieval_result.total_chunks,
            "rag_context": rag_context.to_dict(),
        }


__all__ = ["RAGBaseAgent"]
