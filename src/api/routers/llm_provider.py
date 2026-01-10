from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.llm_factory import llm_complete, llm_fetch_models
from src.services.llm.provider import LLMProvider, provider_manager
from src.services.llm.config import get_llm_config, LLMConfig

router = APIRouter()


class LLMHealthResponse(BaseModel):
    """Response model for LLM health check."""

    status: str  # "healthy", "unhealthy", "degraded"
    binding: str
    model: str
    base_url: Optional[str] = None
    message: str = ""
    error: Optional[str] = None


class ProviderHealthResponse(BaseModel):
    """Response model for individual provider health checks."""

    provider: str
    status: str  # "healthy", "unhealthy", "degraded"
    base_url: str
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    message: str = ""
    error: Optional[str] = None


def _sanitize_base_url(base_url: str) -> str:
    """Sanitize base URL for API calls."""
    base_url = base_url.rstrip("/")

    if "/api" in base_url and not base_url.endswith("/v1"):
        if ":11434" in base_url or "ollama" in base_url.lower():
            base_url = base_url.replace("/api", "/v1")

    for suffix in ["/chat/completions", "/completions"]:
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]

    return base_url


@router.get("/health", response_model=LLMHealthResponse)
async def check_llm_health():
    """
    Check LLM service health and connectivity.

    Returns:
        LLMHealthResponse with status, binding, model, and any error message
    """
    try:
        config = get_llm_config()

        if not config.model:
            return LLMHealthResponse(
                status="unhealthy",
                binding="unknown",
                model="",
                error="LLM_MODEL not configured",
            )

        if not config.base_url:
            return LLMHealthResponse(
                status="unhealthy",
                binding=config.binding,
                model=config.model,
                error="LLM_HOST not configured",
            )

        base_url = _sanitize_base_url(config.base_url)

        api_key_for_test = config.api_key or ""

        if config.binding in ("ollama", "lollms"):
            api_key_for_test = "sk-no-key-required"

        try:
            response = await llm_complete(
                model=config.model,
                prompt="Hi",
                system_prompt="Reply with exactly 'ok'.",
                api_key=api_key_for_test,
                base_url=base_url,
                binding=config.binding,
                max_tokens=10,
            )

            if "ok" in response.lower() or not response:
                return LLMHealthResponse(
                    status="healthy",
                    binding=config.binding,
                    model=config.model,
                    base_url=config.base_url,
                    message=f"LLM service is responding. Model: {config.model}",
                )
            else:
                return LLMHealthResponse(
                    status="degraded",
                    binding=config.binding,
                    model=config.model,
                    base_url=config.base_url,
                    message="LLM responded but with unexpected content",
                    error=f"Response: {response[:100]}",
                )

        except Exception as llm_error:
            error_msg = str(llm_error)

            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                return LLMHealthResponse(
                    status="unhealthy",
                    binding=config.binding,
                    model=config.model,
                    base_url=config.base_url,
                    error=f"Cannot connect to LLM service. Is Ollama running?",
                )

            if "model" in error_msg.lower() and (
                "not found" in error_msg.lower() or "no such file" in error_msg.lower()
            ):
                return LLMHealthResponse(
                    status="unhealthy",
                    binding=config.binding,
                    model=config.model,
                    base_url=config.base_url,
                    error=f"Model '{config.model}' not found. Pull it with: ollama pull {config.model}",
                )

            return LLMHealthResponse(
                status="unhealthy",
                binding=config.binding,
                model=config.model,
                base_url=config.base_url,
                error=error_msg[:500],
            )

    except ValueError as ve:
        return LLMHealthResponse(
            status="unhealthy",
            binding="unknown",
            model="",
            error=str(ve),
        )

    except Exception as e:
        return LLMHealthResponse(
            status="unhealthy",
            binding="unknown",
            model="",
            error=f"Unexpected error: {str(e)}",
        )


class TestConnectionRequest(BaseModel):
    binding: str
    base_url: str
    api_key: str
    model: str
    requires_key: bool = True  # Default to True for backward compatibility


@router.get("/", response_model=List[LLMProvider])
async def list_providers():
    """List all configured LLM providers."""
    return provider_manager.list_providers()


@router.post("/", response_model=LLMProvider)
async def add_provider(provider: LLMProvider):
    """Add a new LLM provider."""
    try:
        return provider_manager.add_provider(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{name}/", response_model=LLMProvider)
async def update_provider(name: str, updates: Dict[str, Any]):
    """Update an existing LLM provider."""
    provider = provider_manager.update_provider(name, updates)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider


@router.delete("/")
async def delete_provider_by_query(name: str):
    """Delete an LLM provider (query param version)."""
    success = provider_manager.delete_provider(name)
    if not success:
        raise HTTPException(status_code=404, detail="Provider not found")
    return {"message": "Provider deleted"}


@router.delete("/{name}/")
async def delete_provider(name: str):
    """Delete an LLM provider."""
    success = provider_manager.delete_provider(name)
    if not success:
        raise HTTPException(status_code=404, detail="Provider not found")
    return {"message": "Provider deleted"}


@router.post("/active/", response_model=LLMProvider)
async def set_active_provider(name_payload: Dict[str, str]):
    """Set the active LLM provider."""
    name = name_payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")

    provider = provider_manager.set_active_provider(name)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider


@router.post("/test/", response_model=Dict[str, Any])
async def test_connection(request: TestConnectionRequest):
    """Test connection to an LLM provider."""
    try:
        # Sanitize Base URL
        # Users often paste full endpoints like http://.../v1/chat/completions
        # OpenAI client needs just the base (e.g., http://.../v1)
        base_url = request.base_url.rstrip("/")

        # Special handling for Ollama: if it ends in /api, it's likely wrong for completion but ok for tags
        # But here we want the completion base.
        if "/api" in base_url and not base_url.endswith("/v1"):
            # If user has http://localhost:11434/api -> change to http://localhost:11434/v1
            if ":11434" in base_url or "ollama" in base_url.lower():
                base_url = base_url.replace("/api", "/v1")

        for suffix in ["/chat/completions", "/completions"]:
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]

        # Simple test prompt
        if not request.requires_key and not request.api_key:
            # Inject dummy key if not required and not provided
            # This satisfies the OpenAI client library which demands a key
            api_key_to_use = "sk-no-key-required"
        else:
            api_key_to_use = request.api_key

        response = await llm_complete(
            model=request.model,
            prompt="Hello, are you working?",
            system_prompt="You are a helpful assistant. Reply with 'Yes'.",
            api_key=api_key_to_use,
            base_url=base_url,
            binding=request.binding,
            max_tokens=200,
        )
        return {"success": True, "message": "Connection successful", "response": response}
    except Exception as e:
        return {"success": False, "message": f"Connection failed: {str(e)}"}


@router.post("/models/", response_model=Dict[str, Any])
async def fetch_available_models(request: TestConnectionRequest):
    """Fetch available models from the provider."""
    try:
        # Sanitize Base URL (same as test_connection)
        base_url = request.base_url.rstrip("/")
        if "/api" in base_url and not base_url.endswith("/v1"):
            if ":11434" in base_url or "ollama" in base_url.lower():
                base_url = base_url.replace("/api", "/v1")

        for suffix in ["/chat/completions", "/completions"]:
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]

        models = await llm_fetch_models(
            binding=request.binding,
            base_url=base_url,
            api_key=request.api_key if request.requires_key else None,
        )
        return {"success": True, "models": models}
    except Exception as e:
        return {"success": False, "message": f"Failed to fetch models: {str(e)}"}


@router.get("/health/ollama", response_model=ProviderHealthResponse)
async def check_ollama_health():
    """
    Check Ollama service health and connectivity.

    Returns:
        ProviderHealthResponse with status, latency, and any error message
    """
    import httpx
    import time

    base_url = "http://localhost:11434"
    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/api/tags")
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]

                if models:
                    return ProviderHealthResponse(
                        provider="ollama",
                        status="healthy",
                        base_url=base_url,
                        model=models[0],
                        latency_ms=latency_ms,
                        message=f"Ollama running with {len(models)} models",
                    )
                else:
                    return ProviderHealthResponse(
                        provider="ollama",
                        status="degraded",
                        base_url=base_url,
                        latency_ms=latency_ms,
                        message="Ollama running but no models loaded",
                    )
            else:
                return ProviderHealthResponse(
                    provider="ollama",
                    status="unhealthy",
                    base_url=base_url,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}",
                )

    except httpx.TimeoutException:
        return ProviderHealthResponse(
            provider="ollama",
            status="unhealthy",
            base_url=base_url,
            error="Connection timed out. Is Ollama running?",
        )
    except httpx.ConnectError:
        return ProviderHealthResponse(
            provider="ollama",
            status="unhealthy",
            base_url=base_url,
            error="Connection refused. Is Ollama running on port 11434?",
        )
    except Exception as e:
        return ProviderHealthResponse(
            provider="ollama",
            status="unhealthy",
            base_url=base_url,
            error=str(e)[:200],
        )


@router.get("/health/vllm", response_model=ProviderHealthResponse)
async def check_vllm_health():
    """
    Check vLLM service health and connectivity.

    Returns:
        ProviderHealthResponse with status, latency, and any error message
    """
    import httpx
    import time

    base_url = "http://localhost:8000/v1"
    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url.rstrip('/v1')}/health")
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return ProviderHealthResponse(
                    provider="vllm",
                    status="healthy",
                    base_url=base_url,
                    latency_ms=latency_ms,
                    message="vLLM health check passed",
                )

    except httpx.TimeoutException:
        pass
    except httpx.ConnectError:
        pass
    except Exception:
        pass

    latency_ms = (time.time() - start_time) * 1000

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                model = data.get("model", "unknown")

                return ProviderHealthResponse(
                    provider="vllm",
                    status="healthy",
                    base_url=base_url,
                    model=model,
                    latency_ms=latency_ms,
                    message=f"vLLM responding with model: {model}",
                )
            elif response.status_code == 404:
                return ProviderHealthResponse(
                    provider="vllm",
                    status="unhealthy",
                    base_url=base_url,
                    latency_ms=latency_ms,
                    error="Model 'test' not found. Check available models.",
                )
            else:
                return ProviderHealthResponse(
                    provider="vllm",
                    status="unhealthy",
                    base_url=base_url,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: {response.text[:100]}",
                )

    except httpx.TimeoutException:
        return ProviderHealthResponse(
            provider="vllm",
            status="unhealthy",
            base_url=base_url,
            latency_ms=latency_ms,
            error="Connection timed out. Is vLLM running on port 8000?",
        )
    except httpx.ConnectError:
        return ProviderHealthResponse(
            provider="vllm",
            status="unhealthy",
            base_url=base_url,
            latency_ms=latency_ms,
            error="Connection refused. Is vLLM running on port 8000?",
        )
    except Exception as e:
        return ProviderHealthResponse(
            provider="vllm",
            status="unhealthy",
            base_url=base_url,
            latency_ms=latency_ms,
            error=str(e)[:200],
        )


@router.get("/health/all", response_model=Dict[str, ProviderHealthResponse])
async def check_all_providers_health():
    """
    Check health of all configured LLM providers.

    Returns:
        Dictionary mapping provider names to health responses
    """
    import asyncio

    ollama_task = check_ollama_health()
    vllm_task = check_vllm_health()

    ollama_result, vllm_result = await asyncio.gather(
        ollama_task, vllm_task, return_exceptions=True
    )

    results = {}

    if isinstance(ollama_result, Exception):
        results["ollama"] = ProviderHealthResponse(
            provider="ollama",
            status="unhealthy",
            base_url="http://localhost:11434",
            error=str(ollama_result)[:200],
        )
    else:
        results["ollama"] = ollama_result

    if isinstance(vllm_result, Exception):
        results["vllm"] = ProviderHealthResponse(
            provider="vllm",
            status="unhealthy",
            base_url="http://localhost:8000/v1",
            error=str(vllm_result)[:200],
        )
    else:
        results["vllm"] = vllm_result

    return results
