"""
LLM Client
==========

Unified LLM client for all DeepTutor services.
Supports HybridProviderManager for fallback chains and context budgeting.
"""

from typing import Any, Dict, List, Optional

from src.logging import get_logger

from .config import LLMConfig, get_llm_config

try:
    from extensions.router.provider_manager import (
        HybridProviderManager,
        ProviderConfig,
        get_provider_manager,
        get_provider_response,
        ProviderResult,
    )
    from extensions.utils.context_budgeter import (
        ContextBudgeter,
        ContextItem,
        BudgetResult,
        Route,
    )

    EXTENSIONS_AVAILABLE = True
except ImportError:
    EXTENSIONS_AVAILABLE = False
    HybridProviderManager = None


class LLMClient:
    """
    Unified LLM client for all services.

    Wraps the underlying LLM API (OpenAI-compatible) with a consistent interface.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration. If None, loads from environment.
        """
        self.config = config or get_llm_config()
        self.logger = get_logger("LLMClient")

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call LLM completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            history: Optional conversation history
            **kwargs: Additional arguments passed to the API

        Returns:
            LLM response text
        """
        from lightrag.llm.openai import openai_complete_if_cache

        return await openai_complete_if_cache(
            self.config.model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history or [],
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            **kwargs,
        )

    async def complete_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        preferred_provider: Optional[str] = None,
        allow_fallback: bool = True,
        route: str = "general",
        **kwargs: Any,
    ) -> str:
        """
        Complete request using HybridProviderManager with automatic fallback.

        Uses context budgeting to prevent context collapse and falls back
        through multiple providers on failure.

        Args:
            prompt: User prompt
            system_prompt: Optional system message
            history: Conversation history
            preferred_provider: Preferred provider binding (ollama, vllm, openrouter)
            allow_fallback: Try fallback providers on failure
            route: Route name for context budgeting (solve, chat, research, etc.)
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            LLM response text

        Raises:
            RuntimeError: If no providers configured or all fail
        """
        if not EXTENSIONS_AVAILABLE or HybridProviderManager is None:
            self.logger.warning("Extensions not available, falling back to standard completion")
            return await self.complete(prompt, system_prompt, history, **kwargs)

        try:
            budgeter = ContextBudgeter()

            history_for_budget = history or []
            budget_result = budgeter.apply_budget(
                route=route,
                system_prompt=system_prompt or "",
                chat_history=history_for_budget,
            )

            if budget_result.trimmed_items:
                self.logger.info(
                    f"Context budget applied: {len(budget_result.trimmed_items)} items trimmed, "
                    f"compression={budget_result.compression_ratio:.1%}"
                )

            result = await get_provider_response(
                prompt=prompt,
                system_prompt=budget_result.trimmed_system
                if hasattr(budget_result, "trimmed_system")
                else system_prompt,
                history=history_for_budget,
                preferred_provider=preferred_provider,
                allow_fallback=allow_fallback,
                **kwargs,
            )

            if result.success:
                return result.response
            else:
                actionable_error = self._get_actionable_error(result)
                raise RuntimeError(actionable_error)

        except Exception as e:
            self.logger.error(f"Hybrid completion failed: {e}")
            raise

    def _get_actionable_error(self, result: ProviderResult) -> str:
        """Generate actionable error message from failed provider result."""
        if result.error_type == "connection":
            return (
                f"Cannot connect to {result.provider}.\n"
                f"1. Verify the server is running\n"
                f"2. Check the base URL in configuration\n"
                f"3. For local providers: Is Ollama/vLLM started?\n"
                f"4. Check firewall settings"
            )
        elif result.error_type == "auth":
            return (
                f"Authentication failed for {result.provider}.\n"
                f"1. Check your API key is correct\n"
                f"2. For OpenRouter: Get a key from https://openrouter.ai/keys\n"
                f"3. Ensure the key has not expired"
            )
        elif result.error_type == "rate_limit":
            return (
                f"Rate limited by {result.provider}.\n"
                f"1. Wait a moment and retry\n"
                f"2. Consider upgrading your plan\n"
                f"3. Try a different provider"
            )
        elif result.error_type == "model":
            return (
                f"Model not available for {result.provider}.\n"
                f"1. Check the model name is correct\n"
                f"2. For Ollama: Pull the model with 'ollama pull <model-name>'\n"
                f"3. For vLLM: Ensure the model is loaded"
            )
        else:
            return f"Request failed: {result.error}"

    def complete_with_budget(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        rag_snippets: Optional[List[Dict[str, Any]]] = None,
        route: str = "general",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Complete request with context budgeting applied.

        Returns both the LLM response and budget metadata for monitoring.

        Args:
            prompt: User prompt
            system_prompt: Optional system message
            history: Conversation history
            rag_snippets: RAG snippets with content, token_count, score
            route: Route name for context budgeting
            **kwargs: Additional parameters

        Returns:
            Dict with 'response', 'budget_result', 'stats'
        """
        if not EXTENSIONS_AVAILABLE:
            response = self.complete_sync(prompt, system_prompt, history, **kwargs)
            return {
                "response": response,
                "budget_result": None,
                "stats": {"method": "standard"},
            }

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

        budget_result = budgeter.apply_budget(
            route=route,
            system_prompt=system_prompt or "",
            chat_history=history or [],
            rag_snippets=snippets,
        )

        response = self.complete_sync(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            **kwargs,
        )

        return {
            "response": response,
            "budget_result": budget_result.to_dict(),
            "stats": budgeter.get_stats(),
        }

    def complete_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous wrapper for complete().

        Use this when you need to call from non-async context.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an async context, we need to use a different approach
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self.complete(prompt, system_prompt, history, **kwargs)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.complete(prompt, system_prompt, history, **kwargs)
                )
        except RuntimeError:
            return asyncio.run(self.complete(prompt, system_prompt, history, **kwargs))

    def get_model_func(self):
        """
        Get a function compatible with LightRAG's llm_model_func parameter.

        Returns:
            Callable that can be used as llm_model_func
        """
        from lightrag.llm.openai import openai_complete_if_cache

        def llm_model_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict]] = None,
            **kwargs: Any,
        ):
            return openai_complete_if_cache(
                self.config.model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                **kwargs,
            )

        return llm_model_func

    def get_vision_model_func(self):
        """
        Get a function compatible with RAG-Anything's vision_model_func parameter.

        Returns:
            Callable that can be used as vision_model_func
        """
        from lightrag.llm.openai import openai_complete_if_cache

        def vision_model_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict]] = None,
            image_data: Optional[str] = None,
            messages: Optional[List[Dict]] = None,
            **kwargs: Any,
        ):
            # Handle multimodal messages
            if messages:
                clean_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["messages", "prompt", "system_prompt", "history_messages"]
                }
                return openai_complete_if_cache(
                    self.config.model,
                    prompt="",
                    messages=messages,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    **clean_kwargs,
                )

            # Handle image data
            if image_data:
                # Build image message
                image_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                }
                return openai_complete_if_cache(
                    self.config.model,
                    prompt="",
                    messages=[image_message],
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    **kwargs,
                )

            # Fallback to regular completion
            return openai_complete_if_cache(
                self.config.model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                **kwargs,
            )

        return vision_model_func


# Singleton instance
_client: Optional[LLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """
    Get or create the singleton LLM client.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        LLMClient instance
    """
    global _client
    if _client is None:
        _client = LLMClient(config)
    return _client


def reset_llm_client():
    """Reset the singleton LLM client."""
    global _client
    _client = None
