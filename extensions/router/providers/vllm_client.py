"""
vLLM Client for DeepTutor

OpenAI-compatible client wrapper for vLLM server endpoints.
Provides unified interface for vLLM with automatic config loading.

Usage:
    from extensions.router.providers.vllm_client import vLLMClient

    # With automatic config from environment
    client = vLLMClient()

    # With explicit config
    client = vLLMClient(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Async completion
    response = await client.acomplete("Hello, how are you?")

    # Sync completion
    response = client.complete("Hello, how are you?")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False


@dataclass
class vLLMConfig:
    """vLLM server configuration."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: str = "sk-no-key-required"  # Dummy key for OpenAI compatibility
    timeout: float = 60.0
    max_tokens: int = 4096
    temperature: float = 0.7


class vLLMClient:
    """
    OpenAI-compatible client for vLLM servers.

    Provides async and sync completion methods with automatic
    config loading from environment variables.
    """

    def __init__(
        self,
        config: Optional[vLLMConfig] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize vLLM client.

        Args:
            config: vLLMConfig object. If provided, takes precedence.
            base_url: vLLM server URL (env: VLLM_BASE_URL)
            model: Model name (env: VLLM_MODEL)
            api_key: API key (env: VLLM_API_KEY)
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for vLLM client. Install with: pip install httpx")

        self.config = config or self._load_config(base_url, model, api_key)
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None

    def _load_config(
        self, base_url: Optional[str], model: Optional[str], api_key: Optional[str]
    ) -> vLLMConfig:
        """Load configuration from environment or defaults."""
        import os

        return vLLMConfig(
            base_url=base_url or os.getenv("VLLM_BASE_URL") or "http://localhost:8000/v1",
            model=model or os.getenv("VLLM_MODEL") or "meta-llama/Llama-3.1-8B-Instruct",
            api_key=api_key or os.getenv("VLLM_API_KEY") or "sk-no-key-required",
        )

    def _get_client(self, async_client: bool = True):
        """Get or create HTTP client."""
        if async_client:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self.config.timeout)
            return self._client
        else:
            if self._sync_client is None:
                self._sync_client = httpx.Client(timeout=self.config.timeout)
            return self._sync_client

    async def _request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to vLLM server."""
        client = self._get_client(async_client=True)
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        if method.upper() == "POST":
            response = await client.post(url, json=data)
        else:
            response = await client.get(url)

        response.raise_for_status()
        return response.json()

    def _request_sync(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make synchronous HTTP request to vLLM server."""
        client = self._get_client(async_client=False)
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        if method.upper() == "POST":
            response = client.post(url, json=data)
        else:
            response = client.get(url)

        response.raise_for_status()
        return response.json()

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from vLLM server."""
        result = await self._request("/models", method="GET")
        return result.get("data", [])

    def list_models_sync(self) -> List[Dict[str, Any]]:
        """List available models (sync)."""
        result = self._request_sync("/models", method="GET")
        return result.get("data", [])

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async completion request to vLLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system message
            history: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        result = await self._request("/chat/completions", method="POST", data=data)

        choices = result.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        message = choice.get("message", {})
        return message.get("content", "")

    def complete_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous completion request to vLLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system message
            history: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        result = self._request_sync("/chat/completions", method="POST", data=data)

        choices = result.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        message = choice.get("message", {})
        return message.get("content", "")

    async def stream_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Async streaming completion.

        Yields:
            Chunks of generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": True,
        }

        client = self._get_client(async_client=True)
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        async with client.stream("POST", url, json=data) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        import json

                        result = json.loads(chunk)
                        choice = result.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        pass

    async def health_check(self) -> Dict[str, Any]:
        """
        Check vLLM server health.

        Returns:
            Health status dictionary
        """
        try:
            models = await self.list_models()
            return {
                "status": "healthy",
                "binding": "vllm",
                "model": self.config.model,
                "base_url": self.config.base_url,
                "message": f"vLLM server responding. Available models: {len(models)}",
            }
        except httpx.ConnectError as e:
            return {
                "status": "unhealthy",
                "binding": "vllm",
                "model": self.config.model,
                "base_url": self.config.base_url,
                "error": f"Cannot connect to vLLM server: {str(e)}",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "binding": "vllm",
                "model": self.config.model,
                "base_url": self.config.base_url,
                "error": str(e),
            }

    async def close(self):
        """Close async client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def close_sync(self):
        """Close sync client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function for quick setup
def get_vllm_client(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> vLLMClient:
    """
    Create vLLM client with environment-based config.

    Args:
        base_url: Server URL (env: VLLM_BASE_URL)
        model: Model name (env: VLLM_MODEL)

    Returns:
        vLLMClient instance
    """
    return vLLMClient(base_url=base_url, model=model)


# Integration helper for DeepTutor's LLM client
def create_vllm_model_func(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Create a model function compatible with DeepTutor's LLM interface.

    Args:
        base_url: vLLM server URL
        model: Model name

    Returns:
        Callable that can be used as llm_model_func
    """
    client = vLLMClient(base_url=base_url, model=model)

    async def model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> str:
        return await client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history_messages,
            **kwargs,
        )

    return model_func


__all__ = [
    "vLLMClient",
    "vLLMConfig",
    "get_vllm_client",
    "create_vllm_model_func",
]
