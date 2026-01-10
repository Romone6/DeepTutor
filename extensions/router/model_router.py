"""
Model Router for DeepTutor

Selects between local models and OpenRouter based on policy rules.

Features:
- Policy-based routing (holiday, term, exam modes)
- Route-based selection
- Output length estimation
- Context size awareness
- Fallback to local model if OpenRouter disabled

Policy Modes:
- holiday: Relaxed mode, prefer local models
- term: Standard academic mode, balanced
- exam: Intensive mode, prefer best available
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================


class PolicyMode(str, Enum):
    """Policy mode for routing decisions."""

    HOLIDAY = "holiday"  # Relaxed, prefer local
    TERM = "term"  # Standard academic
    EXAM = "exam"  # Intensive, best available


class ModelType(str, Enum):
    """Model type enumeration."""

    LOCAL = "local"
    OPENROUTER = "openrouter"


# =============================================================================
# Pydantic Models
# =============================================================================


class ModelRouterConfig(BaseModel):
    """Configuration for model router."""

    # OpenRouter settings
    openrouter_enabled: bool = True
    openrouter_model: str = "anthropic/claude-3-haiku"
    openrouter_fallback_model: str = "openai/gpt-4o"

    # Local model settings
    local_model: str = "llama3.2"  # Default local model
    local_fallback: str = "mistral"  # Fallback local model

    # Thresholds
    local_max_tokens: int = 2048
    local_max_context: int = 8192
    openrouter_max_tokens: int = 4096
    openrouter_max_context: int = 128000

    # Policy settings
    default_policy: PolicyMode = PolicyMode.TERM

    # Holiday mode - very conservative
    holiday_max_output: int = 500
    holiday_max_complexity: int = 5

    # Term mode - balanced
    term_max_output: int = 2000
    term_max_complexity: int = 8

    # Exam mode - intensive
    exam_max_output: int = 4096
    exam_max_complexity: int = 10

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "openrouter_enabled": True,
                    "openrouter_model": "anthropic/claude-3-haiku",
                    "local_model": "llama3.2",
                    "default_policy": "term",
                }
            ]
        }


class ModelTarget(BaseModel):
    """Model routing decision."""

    model_type: ModelType = ModelType.LOCAL
    model_name: str = "llama3.2"
    reason: str = ""
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    policy_mode: PolicyMode = PolicyMode.TERM
    estimated_tokens: int = 0
    uses_openrouter: bool = False

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "model_type": "local",
                    "model_name": "llama3.2",
                    "reason": "Short response, local model sufficient",
                    "confidence": 0.95,
                    "policy_mode": "term",
                    "estimated_tokens": 150,
                    "uses_openrouter": False,
                }
            ]
        }


class RoutingContext(BaseModel):
    """Context for routing decision."""

    route: str = Field(default="chat", description="Target route/handler")
    user_message: str = Field(..., description="User input message")
    history_length: int = Field(default=0, description="Number of history messages")
    estimated_input_tokens: int = Field(default=0, description="Estimated input token count")
    requires_research: bool = Field(default=False, description="Requires web search")
    requires_coding: bool = Field(default=False, description="Requires code generation")
    requires_long_output: bool = Field(default=False, description="Requires extended response")
    is_follow_up: bool = Field(default=False, description="Is follow-up question")
    policy_mode: Optional[PolicyMode] = Field(default=None, description="Override policy mode")


class ModelRouterResponse(BaseModel):
    """Final model routing response."""

    target: ModelTarget
    context: RoutingContext
    timestamp: str
    version: str = "1.0.0"

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "target": {
                        "model_type": "local",
                        "model_name": "llama3.2",
                        "reason": "Simple question, local model sufficient",
                        "confidence": 0.9,
                        "policy_mode": "term",
                        "estimated_tokens": 200,
                        "uses_openrouter": False,
                    },
                    "context": {
                        "route": "chat",
                        "user_message": "What is 2+2?",
                        "history_length": 0,
                        "requires_research": False,
                    },
                    "timestamp": "2024-01-15T10:30:00Z",
                    "version": "1.0.0",
                }
            ]
        }


# =============================================================================
# Complexity Scorer
# =============================================================================


class ComplexityScorer:
    """Scores message complexity for routing decisions."""

    # Indicators for high complexity
    COMPLEXITY_INDICATORS = [
        (r"\b(analyze|compare|evaluate|synthesize)\b", 3),
        (r"\b(discuss|explain in detail|elaborate)\b", 2),
        (r"\b(long|detailed|comprehensive|thorough)\b", 2),
        (r"\bessay|report|article|analysis\b", 4),
        (r"\bcode|program|function|algorithm\b", 3),
        (r"\bresearch|study|investigate\b", 3),
        (r"\bmathematical|math|equation|formula\b", 2),
        (r"\bstep by step|walk through|guide me\b", 2),
    ]

    # Indicators for simple requests
    SIMPLE_INDICATORS = [
        (r"^\s*[\w\s]+\?\s*$", 1),  # Short question
        (r"\b(what is|who is|where is|when did)\b", 1),
        (r"^[A-Z][^.?!]{0,30}[.?!]?\s*$", 1),  # Short sentence
    ]

    def score(self, message: str, history_length: int = 0) -> tuple[int, list[str]]:
        """
        Calculate complexity score for a message.

        Returns:
            Tuple of (score, matched_indicators)
        """
        score = 0
        matched = []
        message_lower = message.lower()

        # Check for complexity indicators
        for pattern, points in self.COMPLEXITY_INDICATORS:
            if (
                pattern.lower() in message_lower
                if isinstance(pattern, str)
                else pattern.search(message_lower)
            ):
                score += points
                matched.append(
                    f"complex:{pattern.pattern if hasattr(pattern, 'pattern') else pattern}"
                )

        # Check for simple indicators (reduces score)
        for pattern, points in self.SIMPLE_INDICATORS:
            if (
                pattern.lower() in message_lower
                if isinstance(pattern, str)
                else pattern.search(message_lower)
            ):
                score -= points
                matched.append(
                    f"simple:{pattern.pattern if hasattr(pattern, 'pattern') else pattern}"
                )

        # History adds complexity
        score += min(history_length // 3, 5)  # Max 5 from history

        # Length factor
        word_count = len(message.split())
        if word_count < 10:
            score -= 2
        elif word_count > 100:
            score += 2
        elif word_count > 500:
            score += 4

        return max(0, score), matched


# =============================================================================
# Output Estimator
# =============================================================================


class OutputEstimator:
    """Estimates expected output length."""

    def estimate(
        self, message: str, requires_research: bool = False, requires_coding: bool = False
    ) -> int:
        """
        Estimate expected output token count.

        Args:
            message: User message
            requires_research: Whether research is needed
            requires_coding: Whether code generation is needed

        Returns:
            Estimated token count
        """
        base_estimate = len(message.split()) * 1.5  # Rough 1.5x ratio

        # Adjust for message type
        if requires_coding:
            return int(base_estimate * 3)  # Code blocks take more space
        if requires_research:
            return int(base_estimate * 2)  # Research needs more context
        if message.endswith("?"):
            return int(base_estimate * 1.2)  # Questions need answers
        if len(message) < 20:
            return 100  # Short messages need short responses

        return int(base_estimate)


# =============================================================================
# Model Router
# =============================================================================


class ModelRouter:
    """
    Router for selecting between local and OpenRouter models.

    Decision logic:
    1. Check if OpenRouter is disabled → use local
    2. Check policy mode restrictions
    3. Check output length thresholds
    4. Check context size requirements
    5. Check route-specific rules
    6. Fallback to local model
    """

    def __init__(self, config: Optional[ModelRouterConfig] = None):
        """
        Initialize model router.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ModelRouterConfig()
        self.complexity_scorer = ComplexityScorer()
        self.output_estimator = OutputEstimator()

    def _get_effective_policy(self, context: RoutingContext) -> PolicyMode:
        """Get effective policy mode (from context or config)."""
        return context.policy_mode or self.config.default_policy

    def _should_use_openrouter(self, context: RoutingContext) -> tuple[bool, str]:
        """
        Determine if OpenRouter should be used.

        Returns:
            Tuple of (should_use, reason)
        """
        # Check if OpenRouter is disabled
        if not self.config.openrouter_enabled:
            return False, "OpenRouter disabled in configuration"

        policy = self._get_effective_policy(context)
        complexity, _ = self.complexity_scorer.score(context.user_message, context.history_length)
        estimated_output = self.output_estimator.estimate(
            context.user_message, context.requires_research, context.requires_coding
        )

        # Policy-based rules
        if policy == PolicyMode.HOLIDAY:
            # Very conservative: only use OpenRouter for very complex tasks
            if complexity >= self.config.holiday_max_complexity:
                return True, f"High complexity ({complexity}) in holiday mode"
            if estimated_output > self.config.holiday_max_output:
                return True, f"Long output ({estimated_output} tokens) in holiday mode"
            return False, "Holiday mode: prefer local model"

        if policy == PolicyMode.EXAM:
            # Intensive: use OpenRouter for most tasks
            if complexity <= 3 and estimated_output < 500:
                return False, "Simple task in exam mode"
            return True, f"Exam mode: use best model (complexity={complexity})"

        # Term mode (default) - balanced
        # Use OpenRouter for:
        # - Long outputs (> local_max_tokens)
        # - High complexity (> 7)
        # - Research or coding tasks
        if estimated_output > self.config.local_max_tokens:
            return True, f"Output ({estimated_output} tokens) exceeds local limit"

        if complexity > 7:
            return True, f"High complexity ({complexity})"

        if context.requires_coding:
            return True, "Code generation requested"

        if context.requires_research and estimated_output > 500:
            return True, "Research with extended output"

        # Route-specific rules
        if context.route == "co_writer":
            return True, "Co-writing requires advanced model"

        if context.route == "research" and complexity > 5:
            return True, "Complex research task"

        # Default: use local model
        return False, "Local model sufficient for task"

    def _select_model(self, use_openrouter: bool, context: RoutingContext) -> str:
        """Select specific model name."""
        if use_openrouter:
            return self.config.openrouter_model
        return self.config.local_model

    def _create_target(
        self, use_openrouter: bool, reason: str, context: RoutingContext
    ) -> ModelTarget:
        """Create ModelTarget from routing decision."""
        model_name = self._select_model(use_openrouter, context)
        estimated_tokens = self.output_estimator.estimate(
            context.user_message, context.requires_research, context.requires_coding
        )

        return ModelTarget(
            model_type=ModelType.OPENROUTER if use_openrouter else ModelType.LOCAL,
            model_name=model_name,
            reason=reason,
            confidence=0.9 if use_openrouter else 0.85,
            policy_mode=self._get_effective_policy(context),
            estimated_tokens=estimated_tokens,
            uses_openrouter=use_openrouter,
        )

    def route(self, context: RoutingContext) -> ModelRouterResponse:
        """
        Route request to appropriate model.

        Args:
            context: Routing context with request details

        Returns:
            ModelRouterResponse with routing decision
        """
        use_openrouter, reason = self._should_use_openrouter(context)
        target = self._create_target(use_openrouter, reason, context)

        return ModelRouterResponse(
            target=target,
            context=context,
            timestamp=self._get_timestamp(),
        )

    def route_message(
        self,
        message: str,
        route: str = "chat",
        history_length: int = 0,
        requires_research: bool = False,
        requires_coding: bool = False,
        policy_mode: Optional[PolicyMode] = None,
    ) -> ModelRouterResponse:
        """
        Convenience method to route a message.

        Args:
            message: User message
            route: Target route
            history_length: Number of history messages
            requires_research: Whether research is needed
            requires_coding: Whether code generation is needed
            policy_mode: Override policy mode

        Returns:
            ModelRouterResponse with routing decision
        """
        context = RoutingContext(
            route=route,
            user_message=message,
            history_length=history_length,
            estimated_input_tokens=len(message.split()) * 2,
            requires_research=requires_research,
            requires_coding=requires_coding,
            policy_mode=policy_mode,
        )

        return self.route(context)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

    def get_available_models(self) -> dict[str, list[str]]:
        """Get list of available models."""
        models = {
            "local": [self.config.local_model],
            "openrouter": [],
        }

        if self.config.openrouter_enabled:
            models["openrouter"] = [
                self.config.openrouter_model,
                self.config.openrouter_fallback_model,
            ]

        return models

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "openrouter_enabled": self.config.openrouter_enabled,
            "local_model": self.config.local_model,
            "openrouter_model": self.config.openrouter_model,
            "default_policy": self.config.default_policy.value,
            "local_max_tokens": self.config.local_max_tokens,
            "openrouter_max_tokens": self.config.openrouter_max_tokens,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_model_router(
    openrouter_enabled: bool = True,
    local_model: str = "llama3.2",
    openrouter_model: str = "anthropic/claude-3-haiku",
    default_policy: str = "term",
) -> ModelRouter:
    """
    Create a model router with specified configuration.

    Args:
        openrouter_enabled: Enable OpenRouter
        local_model: Local model name
        openrouter_model: OpenRouter model name
        default_policy: Default policy mode (holiday, term, exam)

    Returns:
        Configured ModelRouter instance
    """
    config = ModelRouterConfig(
        openrouter_enabled=openrouter_enabled,
        local_model=local_model,
        openrouter_model=openrouter_model,
        default_policy=PolicyMode(default_policy),
    )
    return ModelRouter(config)


def route_to_model(
    message: str,
    route: str = "chat",
    history_length: int = 0,
    requires_research: bool = False,
    requires_coding: bool = False,
    policy_mode: Optional[str] = None,
) -> ModelRouterResponse:
    """
    Convenience function to route a message to a model.

    Args:
        message: User message
        route: Target route
        history_length: Number of history messages
        requires_research: Whether research is needed
        requires_coding: Whether code generation is needed
        policy_mode: Override policy mode (holiday, term, exam)

    Returns:
        ModelRouterResponse with routing decision
    """
    router = create_model_router()
    pm = PolicyMode(policy_mode) if policy_mode else None
    return router.route_message(
        message=message,
        route=route,
        history_length=history_length,
        requires_research=requires_research,
        requires_coding=requires_coding,
        policy_mode=pm,
    )
