"""
OpenRouter Provider for DeepTutor

OpenRouter is an OpenAI-compatible API that provides access to multiple models
through a single unified interface.

Configuration:
- OPENROUTER_API_KEY: API key from https://openrouter.ai
- OPENROUTER_BASE_URL: https://openrouter.ai/api/v1
- OPENROUTER_MODEL: Default model (e.g., anthropic/claude-3-haiku)

Usage:
    from extensions.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider()
    response = await provider.complete("Hello, world!")
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================


class OpenRouterConfig(BaseModel):
    """OpenRouter configuration from environment variables."""

    api_key: str = Field(default="", description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="OpenRouter API base URL"
    )
    model: str = Field(default="anthropic/claude-3-haiku", description="Default model to use")
    site_url: str = Field(
        default="https://deeptutor.hku.hk", description="Site URL for OpenRouter tracking"
    )
    app_name: str = Field(default="DeepTutor", description="Application name for OpenRouter")

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku"),
            site_url=os.getenv("OPENROUTER_SITE_URL", "https://deeptutor.hku.hk"),
            app_name=os.getenv("OPENROUTER_APP_NAME", "DeepTutor"),
        )

    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        return bool(self.api_key and self.api_key.strip())

    def validate(self) -> tuple[bool, str]:
        """
        Validate configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.api_key:
            return False, "OPENROUTER_API_KEY is not set"
        if not self.api_key.strip():
            return False, "OPENROUTER_API_KEY is empty"
        if not self.base_url:
            return False, "OPENROUTER_BASE_URL is not set"
        if not self.model:
            return False, "OPENROUTER_MODEL is not set"
        return True, ""


# =============================================================================
# Provider Interface
# =============================================================================


class OpenRouterProvider:
    """
    OpenRouter LLM provider.

    OpenRouter is OpenAI-compatible, meaning it works with the same
    client library and has the same response format as OpenAI.
    """

    BINDING = "openrouter"

    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """
        Initialize OpenRouter provider.

        Args:
            config: Optional configuration. Loads from env if not provided.
        """
        self.config = config or OpenRouterConfig.from_env()

    @property
    def binding(self) -> str:
        """Get provider binding name."""
        return self.BINDING

    @property
    def is_available(self) -> bool:
        """Check if provider is available and configured."""
        return self.config.is_configured()

    def get_config(self) -> Dict[str, Any]:
        """Get provider configuration (without API key)."""
        return {
            "binding": self.BINDING,
            "base_url": self.config.base_url,
            "model": self.config.model,
            "site_url": self.config.site_url,
            "app_name": self.config.app_name,
            "is_available": self.is_available,
        }

    def get_error_message(self, error: Exception) -> str:
        """Generate user-friendly error message."""
        error_str = str(error).lower()

        if "api_key" in error_str or "401" in error_str or "unauthorized" in error_str:
            return (
                "OpenRouter API key is invalid or missing. "
                "Please set OPENROUTER_API_KEY in your environment."
            )
        if "model" in error_str or "404" in error_str:
            return (
                f"Model '{self.config.model}' is not available. "
                "Please check OPENROUTER_MODEL and try a different model."
            )
        if "rate limit" in error_str or "429" in error_str:
            return "OpenRouter rate limit exceeded. Please wait a moment and try again."
        if "connection" in error_str or "network" in error_str:
            return (
                "Cannot connect to OpenRouter. "
                "Please check your internet connection and OPENROUTER_BASE_URL."
            )

        return f"OpenRouter error: {error}"


# =============================================================================
# Factory Integration
# =============================================================================


def get_openrouter_completion_function():
    """
    Get the OpenRouter completion function.

    Returns a function compatible with llm_complete interface.
    """
    import aiohttp

    async def openrouter_complete(
        model: str,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Complete a prompt using OpenRouter.

        Args:
            model: Model name (e.g., 'anthropic/claude-3-haiku')
            prompt: User prompt
            system_prompt: System prompt
            api_key: OpenRouter API key
            base_url: OpenRouter base URL
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            Generated text response
        """
        config = OpenRouterConfig.from_env()

        if api_key is None:
            api_key = config.api_key

        if not api_key:
            raise ValueError(
                "OpenRouter API key is not configured. Set OPENROUTER_API_KEY environment variable."
            )

        url = (base_url or config.base_url).rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": config.site_url,
            "X-Title": config.app_name,
        }

        data = {
            "model": model or config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        # Add OpenRouter-specific parameters
        if "top_p" in kwargs:
            data["top_p"] = kwargs["top_p"]
        if "presence_penalty" in kwargs:
            data["presence_penalty"] = kwargs["presence_penalty"]
        if "frequency_penalty" in kwargs:
            data["frequency_penalty"] = kwargs["frequency_penalty"]

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"OpenRouter API error (status={response.status}): {error_text}"
                    )

                result = await response.json()

                if "choices" not in result or not result["choices"]:
                    raise Exception("Empty response from OpenRouter")

                message = result["choices"][0].get("message", {})
                content = message.get("content", "")

                if not content:
                    # Check for reasoning content (some models)
                    content = message.get("reasoning_content", "") or ""

                return content

    return openrouter_complete


async def fetch_openrouter_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[str]:
    """
    Fetch available models from OpenRouter.

    Args:
        api_key: OpenRouter API key
        base_url: OpenRouter base URL

    Returns:
        List of model identifiers
    """
    import aiohttp

    config = OpenRouterConfig.from_env()

    if api_key is None:
        api_key = config.api_key

    url = (base_url or config.base_url).rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    url = f"{url}/models"

    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return []

            result = await response.json()

            if "data" in result and isinstance(result["data"], list):
                return [m.get("id") for m in result["data"] if m.get("id")]

            return []


# =============================================================================
# Convenience Functions
# =============================================================================


def create_openrouter_provider() -> OpenRouterProvider:
    """Create an OpenRouter provider from environment configuration."""
    return OpenRouterProvider()


def is_openrouter_configured() -> bool:
    """Check if OpenRouter is properly configured."""
    config = OpenRouterConfig.from_env()
    return config.is_configured()


def get_openrouter_status() -> Dict[str, Any]:
    """Get OpenRouter provider status."""
    config = OpenRouterConfig.from_env()
    is_valid, error_msg = config.validate()

    return {
        "configured": is_valid,
        "error": error_msg if not is_valid else None,
        "model": config.model if is_valid else None,
        "binding": OpenRouterProvider.BINDING,
    }
