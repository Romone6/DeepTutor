"""
Degradation Service
===================

Provides graceful degradation when components fail:
- ASR down => text-only input prompt
- TTS down => text-only output
- RAG empty => request missing docs with disclaimer
- OpenRouter down => local-only fallback

Usage:
    from src.services.degradation import get_degradation_service, ComponentStatus

    service = get_degradation_service()
    status = service.get_component_status("asr")
    if status.healthy:
        # Use ASR
    else:
        # Show text-only prompt
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from src.logging import get_logger


class ComponentType(Enum):
    ASR = "asr"
    TTS = "tts"
    RAG = "rag"
    LLM = "llm"
    VOICE_GATEWAY = "voice_gateway"


@dataclass
class ComponentStatus:
    """Status of a component."""

    component: ComponentType
    healthy: bool
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    degradation_mode: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "component": self.component.value,
            "healthy": self.healthy,
            "message": self.message,
            "last_check": self.isoformat(),
            "error": self.error,
            "degradation_mode": self.degradation_mode,
        }

    def isoformat(self) -> str:
        return self.last_check.isoformat()


@dataclass
class DegradationRule:
    """Rule for degrading when a component fails."""

    component: ComponentType
    healthy_check: Callable[[], bool]
    fallback_behavior: str
    user_message: str
    severity: str = "medium"


class DegradationService:
    """
    Central service for managing component health and graceful degradation.

    Tracks the health of critical components and provides degradation strategies
    when they fail. Used by API endpoints to determine available features.
    """

    def __init__(self):
        self.logger = get_logger("DegradationService")
        self._rules: list[DegradationRule] = []
        self._status_cache: dict[ComponentType, ComponentStatus] = {}
        self._check_interval = 30  # seconds
        self._background_task: Optional[asyncio.Task] = None

    def add_rule(self, rule: DegradationRule) -> None:
        """Add a degradation rule."""
        self._rules.append(rule)
        self.logger.info(f"Added degradation rule for {rule.component.value}")

    def get_component_status(self, component: ComponentType) -> ComponentStatus:
        """Get cached status for a component."""
        return self._status_cache.get(
            component, ComponentStatus(component, healthy=True, message="Not checked yet")
        )

    def get_all_status(self) -> list[ComponentStatus]:
        """Get status for all tracked components."""
        return list(self._status_cache.values())

    def get_degradation_status(self) -> dict[str, Any]:
        """Get overall degradation status for UI."""
        status = {
            "overall_healthy": True,
            "degraded_components": [],
            "banners": [],
            "timestamp": datetime.now().isoformat(),
        }

        for component, comp_status in self._status_cache.items():
            if not comp_status.healthy:
                status["overall_healthy"] = False
                status["degraded_components"].append(component.value)
                if comp_status.degradation_mode:
                    status["banners"].append(
                        {
                            "component": component.value,
                            "message": comp_status.message,
                            "severity": "warning",
                            "action": comp_status.degradation_mode,
                        }
                    )

        return status

    async def check_component(self, component: ComponentType) -> ComponentStatus:
        """Check health of a specific component."""
        checkers = {
            ComponentType.ASR: self._check_asr,
            ComponentType.TTS: self._check_tts,
            ComponentType.RAG: self._check_rag,
            ComponentType.LLM: self._check_llm,
            ComponentType.VOICE_GATEWAY: self._check_voice_gateway,
        }

        checker = checkers.get(component)
        if not checker:
            return ComponentStatus(component, healthy=True, message="No checker available")

        try:
            healthy = await asyncio.wait_for(checker(), timeout=5.0)
            return ComponentStatus(
                component=component,
                healthy=healthy,
                message="Healthy" if healthy else "Component unavailable",
                last_check=datetime.now(),
            )
        except asyncio.TimeoutError:
            return ComponentStatus(
                component=component,
                healthy=False,
                message="Check timed out",
                error="timeout",
                last_check=datetime.now(),
            )
        except Exception as e:
            return ComponentStatus(
                component=component,
                healthy=False,
                message=f"Check failed: {str(e)}",
                error=str(e),
                last_check=datetime.now(),
            )

    async def _check_asr(self) -> bool:
        """Check if ASR service is available."""
        try:
            asr_url = os.getenv("VOICE_ASR_URL", "http://localhost:7001")
            async with asyncio.timeout(2):
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{asr_url}/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        return resp.status == 200
        except Exception:
            return False

    async def _check_tts(self) -> bool:
        """Check if TTS service is available."""
        try:
            tts_url = os.getenv("VOICE_TTS_URL", "http://localhost:7001")
            async with asyncio.timeout(2):
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{tts_url}/health/tts", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        return resp.status == 200
        except Exception:
            return False

    async def _check_rag(self) -> bool:
        """Check if RAG service has any knowledge bases."""
        try:
            kb_dir = Path(__file__).parent.parent.parent / "data" / "knowledge_bases"
            if not kb_dir.exists():
                return False
            kbs = [d for d in kb_dir.iterdir() if d.is_dir() and (d / "_indexed").exists()]
            return len(kbs) > 0
        except Exception:
            return False

    async def _check_llm(self) -> bool:
        """Check if LLM provider is available."""
        try:
            from src.services.llm.config import get_llm_config

            config = get_llm_config()
            if not config.api_key and not config.base_url:
                return False
            async with asyncio.timeout(5):
                import aiohttp

                headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
                base_url = config.base_url or "https://api.openai.com/v1"
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url}/chat/completions",
                        json={
                            "model": config.model,
                            "messages": [{"role": "user", "content": "test"}],
                        },
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        return resp.status in (200, 400)
        except Exception:
            return False

    async def _check_voice_gateway(self) -> bool:
        """Check if voice gateway is available."""
        try:
            gateway_url = os.getenv("VOICE_GATEWAY_URL", "http://localhost:7001")
            async with asyncio.timeout(2):
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{gateway_url}/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        return resp.status == 200
        except Exception:
            return False

    async def run_health_check(self) -> dict[str, ComponentStatus]:
        """Run health check on all components."""
        self.logger.info("Running health check on all components")

        components = list(ComponentType)
        tasks = [self.check_component(c) for c in components]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        status_map = {}
        for component, result in zip(components, results):
            if isinstance(result, Exception):
                status = ComponentStatus(
                    component=component,
                    healthy=False,
                    message=f"Check error: {str(result)}",
                    error=str(result),
                )
            else:
                status = result
            status_map[component] = status
            self._status_cache[component] = status

        return status_map

    def start_background_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._background_monitor())

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._background_task)
            except asyncio.CancelledError:
                pass

    async def _background_monitor(self) -> None:
        """Background task that periodically checks component health."""
        while True:
            try:
                await self.run_health_check()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(5)

    def get_fallback_message(self, component: ComponentType) -> str:
        """Get user-facing fallback message for a component."""
        messages = {
            ComponentType.ASR: (
                "Voice input is currently unavailable. Please type your question instead."
            ),
            ComponentType.TTS: (
                "Voice output is currently unavailable. Please read the text response below."
            ),
            ComponentType.RAG: (
                "No knowledge bases are loaded. "
                "Add documents to your knowledge base for contextual answers, "
                "or continue with a general explanation."
            ),
            ComponentType.LLM: (
                "AI service is temporarily unavailable. Please try again in a moment."
            ),
            ComponentType.VOICE_GATEWAY: (
                "Voice services are undergoing maintenance. Please use text input for now."
            ),
        }
        return messages.get(component, "Service temporarily unavailable.")

    def get_rag_disclaimer(self) -> str:
        """Get disclaimer for RAG when no knowledge base is loaded."""
        return (
            "\n\n---\n*Note: No documents were found in your knowledge base. "
            "This response is based on general knowledge. "
            "Add documents to your knowledge base for more accurate, "
            "context-specific answers.*"
        )


# Singleton instance
_degradation_service: Optional[DegradationService] = None


def get_degradation_service() -> DegradationService:
    """Get or create the degradation service singleton."""
    global _degradation_service
    if _degradation_service is None:
        _degradation_service = DegradationService()
        _degradation_service.add_rule(
            DegradationRule(
                component=ComponentType.ASR,
                healthy_check=lambda: False,  # Checked dynamically
                fallback_behavior="text_only",
                user_message="Voice input unavailable. Please type instead.",
            )
        )
        _degradation_service.add_rule(
            DegradationRule(
                component=ComponentType.TTS,
                healthy_check=lambda: False,
                fallback_behavior="text_only",
                user_message="Voice output unavailable. Please read the text response.",
            )
        )
        _degradation_service.add_rule(
            DegradationRule(
                component=ComponentType.RAG,
                healthy_check=lambda: False,
                fallback_behavior="general_explanation",
                user_message="No knowledge base loaded. Response may be less accurate.",
            )
        )
        _degradation_service.add_rule(
            DegradationRule(
                component=ComponentType.LLM,
                healthy_check=lambda: False,
                fallback_behavior="local_only",
                user_message="AI service unavailable. Please try again.",
            )
        )
    return _degradation_service


def reset_degradation_service() -> None:
    """Reset the degradation service (for testing)."""
    global _degradation_service
    if _degradation_service:
        _degradation_service.stop_background_monitoring()
    _degradation_service = None
