"""
Hybrid LLM Provider Manager with Fallback Chain

Provides reliable LLM access through ordered fallbacks:
1. Local provider (Ollama/vLLM) - 1 retry on transient error
2. OpenRouter (cloud backup) - 0-1 retry based on config
3. Actionable error - Clear guidance on what to fix

Usage:
    from extensions.router.provider_manager import get_provider_response

    response = await get_provider_response(
        prompt="Hello!",
        system_prompt="You are helpful.",
        preferred_binding="ollama",
        allow_fallback=True,
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from src.logging import get_logger

from .providers.vllm_client import vLLMClient, vLLMConfig

logger = logging.getLogger("ProviderManager")


class ProviderBinding(str, Enum):
    """Supported LLM provider bindings."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    LOLLMS = "lollms"


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    binding: str
    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    priority: int = 0  # Lower = higher priority
    max_retries: int = 1
    timeout: float = 60.0


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""

    max_local_retries: int = 1  # Retries for local providers
    max_cloud_retries: int = 0  # Retries for cloud providers
    local_bindings: List[str] = field(default_factory=lambda: ["ollama", "vllm"])
    cloud_bindings: List[str] = field(default_factory=lambda: ["openrouter", "openai"])
    log_requests: bool = True


@dataclass
class ProviderResult:
    """Result from a provider attempt."""

    success: bool
    response: str = ""
    provider: str = ""
    attempt: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None  # "connection", "auth", "rate_limit", "model", "unknown"
    latency_ms: float = 0.0
    from_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "response": self.response,
            "provider": self.provider,
            "attempt": self.attempt,
            "error": self.error,
            "error_type": self.error_type,
            "latency_ms": self.latency_ms,
            "from_cache": self.from_cache,
        }


class ProviderErrorClassifier:
    """Classifies LLM provider errors into actionable types."""

    @staticmethod
    def classify(error: Exception) -> str:
        """Classify error type from exception."""
        error_str = str(error).lower()

        if any(x in error_str for x in ["connection", "refused", "timeout", "network"]):
            return "connection"
        if any(
            x in error_str for x in ["401", "403", "auth", "api key", "api_key", "unauthorized"]
        ):
            return "auth"
        if any(x in error_str for x in ["429", "rate", "quota", "limit"]):
            return "rate_limit"
        if any(x in error_str for x in ["not found", "does not exist", "no such file"]):
            return "model"
        if any(x in error_str for x in ["model not found", "model does not exist"]):
            return "model"
        return "unknown"


class HybridProviderManager:
    """
    Manages LLM providers with automatic fallback.

    Provider order:
    1. Preferred local provider (Ollama/vLLM) - 1 retry
    2. Cloud backup (OpenRouter/OpenAI) - 0-1 retry
    3. Actionable error
    """

    def __init__(self, fallback_config: Optional[FallbackConfig] = None):
        self.fallback_config = fallback_config or FallbackConfig()
        self._providers: Dict[str, ProviderConfig] = {}
        self._client_cache: Dict[str, Any] = {}

    def add_provider(self, config: ProviderConfig) -> None:
        """Add a provider configuration."""
        config.binding = config.binding.lower()
        self._providers[config.binding] = config
        logger.info(f"Added provider: {config.name} ({config.binding})")

    def get_provider(self, binding: str) -> Optional[ProviderConfig]:
        """Get provider by binding name."""
        return self._providers.get(binding.lower())

    def list_providers(self) -> List[ProviderConfig]:
        """List all configured providers."""
        return list(self._providers.values())

    def get_ordered_providers(
        self,
        preferred: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> List[ProviderConfig]:
        """
        Get providers in fallback order.

        Args:
            preferred: Preferred provider binding
            allow_fallback: Whether to include fallback providers

        Returns:
            Ordered list of provider configs
        """
        providers = []

        local_bindings = self.fallback_config.local_bindings
        cloud_bindings = self.fallback_config.cloud_bindings

        if preferred:
            preferred = preferred.lower()
            if preferred in self._providers:
                providers.append(self._providers[preferred])

        for binding in local_bindings:
            if binding != preferred and binding in self._providers:
                provider = self._providers[binding]
                if provider.enabled:
                    providers.append(provider)

        if allow_fallback:
            for binding in cloud_bindings:
                if binding not in [p.binding for p in providers] and binding in self._providers:
                    provider = self._providers[binding]
                    if provider.enabled:
                        providers.append(provider)

        return providers

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        preferred_provider: Optional[str] = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> ProviderResult:
        """
        Complete request with automatic fallback.

        Args:
            prompt: User prompt
            system_prompt: Optional system message
            history: Conversation history
            preferred_provider: Preferred provider binding
            allow_fallback: Try fallback providers on failure
            **kwargs: Additional parameters

        Returns:
            ProviderResult with response or error
        """
        providers = self.get_ordered_providers(preferred_provider, allow_fallback)

        if not providers:
            return ProviderResult(
                success=False,
                error="No providers configured. Add a provider first.",
                error_type="configuration",
            )

        last_error = None

        for provider in providers:
            max_retries = self._get_retries(provider.binding)

            for attempt in range(max_retries + 1):
                result = await self._try_provider(
                    provider=provider,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=history,
                    attempt=attempt,
                    **kwargs,
                )

                if result.success:
                    self._log_success(result, provider)
                    return result

                last_error = result
                error_type = ProviderErrorClassifier.classify(
                    Exception(result.error or "Unknown error")
                )

                if error_type in ["auth", "model"]:
                    if self.fallback_config.log_requests:
                        logger.warning(
                            f"Provider {provider.binding} error ({error_type}): {result.error}"
                        )
                    break

                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        return ProviderResult(
            success=False,
            error=last_error.error if last_error else "All providers failed",
            error_type=last_error.error_type if last_error else "unknown",
        )

    async def _try_provider(
        self,
        provider: ProviderConfig,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        attempt: int,
        **kwargs: Any,
    ) -> ProviderResult:
        """Try a single provider."""
        import time

        start_time = time.time()

        try:
            if provider.binding == "ollama":
                response = await self._call_ollama(
                    provider, prompt, system_prompt, history, **kwargs
                )
            elif provider.binding == "vllm":
                response = await self._call_vllm(provider, prompt, system_prompt, history, **kwargs)
            elif provider.binding == "openrouter":
                response = await self._call_openrouter(
                    provider, prompt, system_prompt, history, **kwargs
                )
            else:
                response = await self._call_openai_compat(
                    provider, prompt, system_prompt, history, **kwargs
                )

            latency_ms = (time.time() - start_time) * 1000

            return ProviderResult(
                success=True,
                response=response,
                provider=provider.binding,
                attempt=attempt,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_type = ProviderErrorClassifier.classify(e)

            return ProviderResult(
                success=False,
                provider=provider.binding,
                attempt=attempt,
                error=str(e)[:500],
                error_type=error_type,
                latency_ms=latency_ms,
            )

    async def _call_ollama(
        self,
        provider: ProviderConfig,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> str:
        """Call Ollama provider."""
        async with httpx.AsyncClient(timeout=provider.timeout) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            response = await client.post(
                f"{provider.base_url.rstrip('/')}/chat/completions",
                json={
                    "model": provider.model or "llama3.1:8b-instruct",
                    "messages": messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

    async def _call_vllm(
        self,
        provider: ProviderConfig,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> str:
        """Call vLLM provider."""
        async with httpx.AsyncClient(timeout=provider.timeout) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            response = await client.post(
                f"{provider.base_url.rstrip('/')}/chat/completions",
                json={
                    "model": provider.model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                },
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

    async def _call_openrouter(
        self,
        provider: ProviderConfig,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> str:
        """Call OpenRouter provider."""
        async with httpx.AsyncClient(timeout=provider.timeout) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            headers = {"Authorization": f"Bearer {provider.api_key}"}

            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": provider.model or "anthropic/claude-3-haiku",
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                },
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

    async def _call_openai_compat(
        self,
        provider: ProviderConfig,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> str:
        """Call OpenAI-compatible provider."""
        async with httpx.AsyncClient(timeout=provider.timeout) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            headers = {}
            if provider.api_key:
                headers["Authorization"] = f"Bearer {provider.api_key}"

            response = await client.post(
                f"{provider.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": provider.model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                },
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

    def _get_retries(self, binding: str) -> int:
        """Get max retries for a binding."""
        if binding in self.fallback_config.local_bindings:
            return self.fallback_config.max_local_retries
        return self.fallback_config.max_cloud_retries

    def _log_success(self, result: ProviderResult, provider: ProviderConfig) -> None:
        """Log successful request."""
        if self.fallback_config.log_requests:
            logger.info(
                f"LLM request handled by {provider.binding} "
                f"(attempt {result.attempt + 1}, {result.latency_ms:.0f}ms)"
            )

    def get_actionable_error(self, result: ProviderResult) -> str:
        """Generate actionable error message from failed result."""
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
            return (
                f"Request failed: {result.error}\n"
                f"Provider: {result.provider}\n"
                f"Check logs for details and verify provider configuration."
            )


_manager: Optional[HybridProviderManager] = None


def get_provider_manager() -> HybridProviderManager:
    """Get the global provider manager instance."""
    global _manager
    if _manager is None:
        _manager = HybridProviderManager()
        _setup_default_providers(_manager)
    return _manager


def _setup_default_providers(manager: HybridProviderManager) -> None:
    """Set up default providers from environment."""
    import os

    if os.getenv("OLLAMA_BASE_URL"):
        manager.add_provider(
            ProviderConfig(
                binding="ollama",
                name="Local Ollama",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct"),
                priority=0,
            )
        )

    if os.getenv("VLLM_BASE_URL"):
        manager.add_provider(
            ProviderConfig(
                binding="vllm",
                name="Local vLLM",
                base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                model=os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                priority=1,
            )
        )

    if os.getenv("OPENROUTER_API_KEY"):
        manager.add_provider(
            ProviderConfig(
                binding="openrouter",
                name="OpenRouter Cloud",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku"),
                priority=10,
            )
        )

    if os.getenv("LLM_HOST") and os.getenv("LLM_API_KEY"):
        binding = os.getenv("LLM_BINDING", "openai")
        manager.add_provider(
            ProviderConfig(
                binding=binding,
                name=f"Custom {binding}",
                base_url=os.getenv("LLM_HOST", ""),
                api_key=os.getenv("LLM_API_KEY", ""),
                model=os.getenv("LLM_MODEL", ""),
                priority=20 if binding in ["openai", "azure_openai"] else 5,
            )
        )


async def get_provider_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    preferred_provider: Optional[str] = None,
    allow_fallback: bool = True,
    **kwargs: Any,
) -> ProviderResult:
    """
    Convenience function to get LLM response with fallback.

    Args:
        prompt: User prompt
        system_prompt: Optional system message
        history: Conversation history
        preferred_provider: Preferred provider binding
        allow_fallback: Try fallback providers
        **kwargs: Additional parameters

    Returns:
        ProviderResult with response or error
    """
    manager = get_provider_manager()
    return await manager.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        history=history,
        preferred_provider=preferred_provider,
        allow_fallback=allow_fallback,
        **kwargs,
    )


__all__ = [
    "HybridProviderManager",
    "ProviderConfig",
    "ProviderResult",
    "ProviderBinding",
    "FallbackConfig",
    "get_provider_manager",
    "get_provider_response",
]
